import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE advantages and returns.

    Shapes (with optional channel dim C):
        rewards / values:  (T, E) or (T, E, C)
        next_value:        (E,)  or (E, C)
        dones / next_done: (T, E) and (E,), shared by all channels, or
                           (T, E, C) and (E, C) for per-channel dones

    Returns (advantages, returns) matching rewards in shape.
    """
    # Append the bootstrap values for the final transition
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
    next_dones = torch.cat([dones[1:], next_done.unsqueeze(0)], dim=0)
    non_terminal = 1.0 - next_dones
    nt = non_terminal.unsqueeze(-1) if non_terminal.dim() < rewards.dim() else non_terminal

    # Bellman residuals and per-step GAE discounts:
    #   advantage_t = delta_t + discount_t * advantage_{t+1}
    deltas = rewards + gamma * next_values * nt - values
    discounts = gamma * gae_lambda * nt

    # Compute the recurrence with an affine prefix scan in O(log T) vectorized passes
    discount_prefix = discounts.flip(0)
    advantage_prefix = deltas.flip(0)
    chunk = 1
    T = deltas.shape[0]
    while chunk < T:
        next_discount_prefix = discount_prefix.clone()
        next_advantage_prefix = advantage_prefix.clone()
        next_advantage_prefix[chunk:] = (
            advantage_prefix[chunk:] + discount_prefix[chunk:] * advantage_prefix[:-chunk]
        )
        next_discount_prefix[chunk:] = discount_prefix[chunk:] * discount_prefix[:-chunk]
        discount_prefix = next_discount_prefix
        advantage_prefix = next_advantage_prefix
        chunk *= 2

    advantages = advantage_prefix.flip(0)
    return advantages, advantages + values
