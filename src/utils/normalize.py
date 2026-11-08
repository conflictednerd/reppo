"""Online running mean/variance for observation normalization.

Welford-style update on batched observations. The stats are torch tensors so
they can stay on the rollout device for GPU-backed envs, while NumPy call sites
still receive NumPy arrays from ``normalize``.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """Running mean / variance over a stream of batched observations."""

    def __init__(
        self,
        shape: tuple[int, ...],
        epsilon: float = 1e-4,
        device: torch.device | str | None = None,
    ):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = torch.tensor(float(epsilon), device=self.mean.device, dtype=torch.float64)

    def update(self, x: np.ndarray | torch.Tensor) -> None:
        """In-place parallel-Welford update with a batch."""
        x64 = self._as_tensor(x)
        b_mean = x64.mean(dim=0)
        b_var = x64.var(dim=0, unbiased=False)
        b_n = torch.tensor(x64.shape[0], device=self.mean.device, dtype=torch.float64)
        delta = b_mean - self.mean
        n = self.count + b_n
        self.mean.add_(delta * b_n / n)
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var.copy_((m_a + m_b + delta.square() * self.count * b_n / n) / n)
        self.count.copy_(n)

    def normalize(
        self, x: np.ndarray | torch.Tensor, clip: float = 10.0
    ) -> np.ndarray | torch.Tensor:
        """``(x - mean) / sqrt(var + eps)``, clipped to ``[-clip, clip]``."""
        out = (self._as_tensor(x) - self.mean) / torch.sqrt(self.var + 1e-8)
        out = out.clamp(-clip, clip).to(torch.float32)
        if isinstance(x, torch.Tensor):
            return out
        return out.detach().cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().numpy().copy(),
            "var": self.var.detach().cpu().numpy().copy(),
            "count": float(self.count.detach().cpu().item()),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.mean.copy_(torch.as_tensor(sd["mean"], device=self.mean.device, dtype=torch.float64))
        self.var.copy_(torch.as_tensor(sd["var"], device=self.var.device, dtype=torch.float64))
        self.count.copy_(
            torch.tensor(float(sd["count"]), device=self.count.device, dtype=torch.float64)
        )

    def _as_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(x, device=self.mean.device, dtype=torch.float64)
