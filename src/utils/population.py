from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PopulationState:
    """EMA estimates of population quality (f_hat) and descriptors (b_hat).

    f_hat: (N,)        - episodic reward-sum EMA per policy
    b_hat: (N, K)      - EMA of episodic behavioral descriptors in [-1, 1]
                         (sum of signed per-step descriptors divided by the
                         episode length, then batch-mean over episodes), per
                         policy
    """

    f_hat: torch.Tensor
    b_hat: torch.Tensor

    @classmethod
    def initialize(cls, N: int, K: int, *, eps_f: float, device: torch.device) -> PopulationState:
        return cls(
            f_hat=torch.full((N,), eps_f, device=device, dtype=torch.float32),
            b_hat=torch.zeros((N, K), device=device, dtype=torch.float32),
        )

    def update(
        self,
        f_ep: torch.Tensor,
        b_ep: torch.Tensor,
        beta_f: float,
        beta_b: float,
        *,
        ema_mask: torch.Tensor | None = None,
    ) -> None:
        """In-place EMA update.

        f_ep: (N,) cumulative episodic rewards per policy (mean over episodes
              finished this batch).
        b_ep: (N, K) per policy: mean over completed episodes this batch of each
              episode's horizon-normalized descriptor sum.
        ema_mask: optional (N,) bool; True = apply EMA for that policy. If False,
                  keep previous f_hat and b_hat (e.g. no completed episode this iter).
        """
        assert torch.isfinite(b_ep).all().item(), "b_ep contains non-finite descriptors"
        assert ((b_ep >= -1.0) & (b_ep <= 1.0)).all().item(), "b_ep must be in [-1, 1]"
        if ema_mask is None:
            self.f_hat.mul_(beta_f).add_(f_ep, alpha=(1.0 - beta_f))
            self.b_hat.mul_(beta_b).add_(b_ep, alpha=(1.0 - beta_b))
        else:
            new_f = self.f_hat.mul(beta_f).add(f_ep, alpha=(1.0 - beta_f))
            self.f_hat.copy_(torch.where(ema_mask, new_f, self.f_hat))
            new_b = self.b_hat.mul(beta_b).add(b_ep, alpha=(1.0 - beta_b))
            self.b_hat.copy_(torch.where(ema_mask.unsqueeze(-1), new_b, self.b_hat))


@dataclass
class DiversityCoefficients:
    """Per-iteration diversity coefficients used to combine advantages.

    c_coeff:   (N, K)   - descriptor diversity coefficients, in [-C_max, C_max]
    b_prime:   (N, K)   - logit-transformed descriptor estimates
    w_k:       (N,)     - task-advantage coefficients (all ones unless use_exact_wk)
    """

    c_coeff: torch.Tensor
    b_prime: torch.Tensor
    w_k: torch.Tensor


def _transform_logit_and_jacobian(
    s: torch.Tensor,
    eps_b: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map [-1, 1] descriptors to logit space and return dx/ds."""
    u = (0.5 * (s + 1.0)).clamp(eps_b, 1.0 - eps_b)  # [0+e, 1-e]
    b_prime = torch.log(u / (1.0 - u))
    s_clip = 2.0 * u - 1.0
    jac = 2.0 / (1.0 - s_clip.square())
    return b_prime, jac


def compute_diversity_coefficients(
    state: PopulationState,
    *,
    K_nn: int,
    gamma_sq: float,
    eps_f: float,
    eps_b: float,
    C_max: float,
    repulsion_mode: str = "symmetric",
    use_exact_wk: bool = False,
) -> DiversityCoefficients:
    """Compute descriptor coefficients from population state.

    repulsion_mode: "symmetric" repels from all K-NN neighbors (exact lower-bound
        gradient); "higher_only" repels only from strictly higher-quality neighbors.
    use_exact_wk: compute the exact task coefficient W_k (restricted to K-NN)
        instead of the fixed W_k = 1.
    The coefficient vector is rescaled so that ||c||_2 <= C_max.
    """
    assert gamma_sq > 0.0, f"gamma_sq must be positive, got {gamma_sq}"
    assert C_max >= 0.0, f"C_max must be non-negative, got {C_max}"
    assert repulsion_mode in ("symmetric", "higher_only"), repulsion_mode
    assert ((state.b_hat >= -1.0) & (state.b_hat <= 1.0)).all().item(), "b_hat must be in [-1, 1]"

    N, K = state.b_hat.shape
    K_nn = min(K_nn, N - 1)
    f_plus = state.f_hat - state.f_hat.min() + eps_f
    b_prime, jac = _transform_logit_and_jacobian(state.b_hat, eps_b)

    if K_nn <= 0:
        return DiversityCoefficients(
            c_coeff=torch.zeros(N, K, device=state.f_hat.device),
            b_prime=b_prime,
            w_k=torch.ones(N, device=state.f_hat.device),
        )

    # K-NN in b_prime space
    diff = b_prime.unsqueeze(1) - b_prime.unsqueeze(0)  # (N, N, K)
    dist_sq = diff.pow(2).sum(dim=-1)  # (N, N)
    dist_sq.fill_diagonal_(float("inf"))  # Don't select self
    nn_dist_sq, nn_idx = torch.topk(dist_sq, K_nn, dim=1, largest=False)  # (N, K_nn)

    # Gather neighbor quantities
    nbr_f_plus = f_plus[nn_idx]  # (N, K_nn)
    weights = torch.exp(-nn_dist_sq / gamma_sq)  # (N, K_nn)

    if use_exact_wk:
        w_k = 1.0 - 0.5 * (torch.sqrt(nbr_f_plus / f_plus.unsqueeze(1)) * weights).sum(dim=1)
    else:
        w_k = torch.ones(N, device=state.f_hat.device)

    # c_n^k
    nbr_diff = b_prime.unsqueeze(1) - b_prime[nn_idx]  # (N, K_nn, K)
    coeff = (2.0 * torch.sqrt(f_plus.unsqueeze(1) * nbr_f_plus) / gamma_sq).unsqueeze(
        -1
    )  # (N, K_nn, 1)
    contrib = coeff * weights.unsqueeze(-1) * nbr_diff  # (N, K_nn, K)
    if repulsion_mode == "higher_only":
        higher = (nbr_f_plus > f_plus.unsqueeze(1)).unsqueeze(-1)  # (N, K_nn, 1)
        contrib = contrib * higher.to(contrib.dtype)
    c_x = contrib.sum(dim=1)
    c_unclamped = c_x * jac  # (N, K)
    norm = c_unclamped.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c_coeff = c_unclamped * (C_max / norm).clamp(max=1.0)

    return DiversityCoefficients(
        c_coeff=c_coeff,
        b_prime=b_prime,
        w_k=w_k,
    )


def vendi_score(descriptors: torch.Tensor, sigma: float = 0.5**0.5) -> float:
    """Vendi score on the population using a Gaussian similarity kernel.

    Vendi(K) = exp(- sum_i lambda_i log lambda_i) where lambda_i are eigenvalues
    of K/N. We use exp(- ||b_i - b_j||^2 / (2 sigma^2)) as the similarity kernel
    on raw descriptors in the bounded behavior space (not transformed).
    """
    N = descriptors.shape[0]
    if N <= 1:
        return float(N)
    diff = descriptors.unsqueeze(1) - descriptors.unsqueeze(0)
    dist_sq = diff.pow(2).sum(dim=-1)
    Kmat = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    Kmat = 0.5 * (Kmat + Kmat.t())
    Kmat = Kmat / N
    eigvals = torch.linalg.eigvalsh(Kmat).clamp_min(1e-12)
    eigvals = eigvals / eigvals.sum()
    H = -(eigvals * eigvals.log()).sum()
    return float(torch.exp(H).item())
