"""PPO pieces for LoRA members on a shared base.

Sequences are stored right-padded as prompt + response. Response tensors are (B, R) with R = the longest response.
Position t of a response is predicted from hidden state (prompt_len - 1 + t), which is also the state the critic
values, V(s_t), before the token is emitted.
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Mean of x over the entries where mask is set (over all dims, or along `dim`)."""
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp_min(1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp_min(1)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten x (B, R) over the valid entries (unbiased variance)."""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask) * mask.sum() / (mask.sum() - 1).clamp_min(1)
    out = (x - mean) * torch.rsqrt(var + 1e-8)
    return out if shift_mean else out + mean


def response_positions(
    prompt_lens: torch.Tensor, resp_lens: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hidden-state index for each response token and the (B, R) response mask."""
    R = int(resp_lens.max())
    t = torch.arange(R, device=prompt_lens.device)
    pos = (prompt_lens[:, None] - 1 + t[None, :]).clamp(max=seq_len - 1)
    mask = t[None, :] < resp_lens[:, None]
    return pos, mask


def gather_hidden(hidden: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Hidden states (B, R, H) at the response positions pos (B, R) of hidden (B, L, H)."""
    return hidden.gather(1, pos[..., None].expand(-1, -1, hidden.shape[-1]))


def chunk_logprobs(
    h: torch.Tensor, tokens: torch.Tensor, lm_weight: torch.Tensor, temperature: float, with_entropy: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Log-probs (and entropies) of `tokens` at hidden states h (n, H), with the LM head in fp32."""
    logp = F.log_softmax(F.linear(h.float(), lm_weight) / temperature, dim=-1)
    ent = -(logp.exp() * logp).sum(-1) if with_entropy else None
    return logp.gather(1, tokens[:, None])[:, 0], ent


def token_logprobs(
    lm_weight: torch.Tensor,
    h: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    with_entropy: bool = False,
    chunk: int = 1024,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Log-probs (B, R) of `tokens` from response hidden states h (B, R, H), row by row and `chunk` positions at a
    time, so the fp32 logits never exceed chunk x vocab (with gradients, each chunk's logits are recomputed in the
    backward pass instead of stored); with_entropy also returns the (B, R) entropies.

    The LM head runs in fp32 (lm_weight is an fp32 copy): bf16 logits carry ~0.1 absolute rounding error at typical
    logit magnitudes, which makes the PPO ratio noisy across batch shapes.
    """
    lp = torch.zeros(tokens.shape, device=h.device)
    ent = torch.zeros(tokens.shape, device=h.device) if with_entropy else None
    for i in range(h.shape[0]):
        n = int(mask[i].sum())
        for s in range(0, n, chunk):
            args = (h[i, s : min(n, s + chunk)], tokens[i, s : min(n, s + chunk)], lm_weight, temperature, with_entropy)
            if torch.is_grad_enabled():
                lp_c, ent_c = torch.utils.checkpoint.checkpoint(chunk_logprobs, *args, use_reentrant=False)
            else:
                lp_c, ent_c = chunk_logprobs(*args)
            lp[i, s : s + len(lp_c)] = lp_c
            if with_entropy:
                ent[i, s : s + len(ent_c)] = ent_c
    return (lp, ent) if with_entropy else lp


def gae(
    rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE advantages and returns over the response, per channel. rewards/values: (B, R, C); mask: (B, R). No
    bootstrap after the end."""
    B, R, C = rewards.shape
    m = mask[..., None].float()
    rewards, values = rewards * m, values * m
    last = torch.zeros(B, C, device=rewards.device)
    adv = torch.zeros_like(rewards)
    for t in reversed(range(R)):
        next_v = values[:, t + 1] if t < R - 1 else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    adv = adv * m
    return adv, (adv + values) * m


def policy_loss(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
    n_tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped PPO policy loss summed over the microbatch and divided by n_tokens, the token count of the whole
    minibatch, so the gradient is the minibatch token mean regardless of how it is split into microbatches.
    adv: (B, R) combined advantage."""
    logratio = new_lp - old_lp
    ratio = logratio.exp()
    pg1 = -adv * ratio
    pg2 = -adv * ratio.clamp(1 - clip, 1 + clip)
    loss = (torch.max(pg1, pg2) * mask).sum() / n_tokens
    with torch.no_grad():
        stats = {
            "pg_loss": masked_mean(torch.max(pg1, pg2), mask).item(),
            "approx_kl": masked_mean(0.5 * logratio**2, mask).item(),
            "clipfrac": masked_mean((pg2 > pg1).float(), mask).item(),
            "ratio": masked_mean(ratio, mask).item(),
        }
    return loss, stats


def value_loss(
    vpred: torch.Tensor,
    old_v: torch.Tensor,
    ret: torch.Tensor,
    mask: torch.Tensor,
    vclip: float,
    n_tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped value loss per head (minibatch token mean, as in policy_loss); the task head plus the mean of the
    descriptor heads, as in the Brax RePPO. vpred / old_v / ret: (B, R, C)."""
    vclipped = old_v + (vpred - old_v).clamp(-vclip, vclip)
    vl = 0.5 * torch.max((vpred - ret) ** 2, (vclipped - ret) ** 2)
    m = mask[..., None].float()
    per_head = (vl * m).sum((0, 1)) / n_tokens  # (C,)
    loss = per_head[0] + (per_head[1:].mean() if per_head.numel() > 1 else 0.0)
    with torch.no_grad():
        clipped = ((vclipped - ret) ** 2 > (vpred - ret) ** 2).float()[..., 0]
        v_mean = (vl * m).sum((0, 1)) / m.sum().clamp_min(1)
        stats = {"v_loss": (v_mean[0] + (v_mean[1:].mean() if v_mean.numel() > 1 else 0.0)).item(),
                 "vf_clipfrac": masked_mean(clipped, mask).item()}
    return loss, stats


def token_budget_chunks(idx: torch.Tensor, seq_lens: torch.Tensor, budget: int) -> list[torch.Tensor]:
    """Split idx into microbatches sorted by length (longest first) whose padded size (count x longest) stays
    within `budget` tokens; a single over-budget sequence gets its own microbatch."""
    idx = idx[torch.argsort(seq_lens[idx], descending=True)]
    chunks, start = [], 0
    while start < len(idx):
        width = int(seq_lens[idx[start]])
        end = start + max(1, budget // width)
        chunks.append(idx[start:end])
        start = end
    return chunks
