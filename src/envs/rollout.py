"""Shared rollout batch containers and eval metric aggregation."""

from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig


@dataclass(slots=True)
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    descriptors: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    # Atari: game-over dones for the descriptor channels; None means they share `dones`.
    descriptor_dones: torch.Tensor | None = None


def make_rollout_batch(
    *,
    T: int,
    N: int,
    E: int,
    K: int,
    obs_shape: tuple[int, ...],
    act_shape: tuple[int, ...],
    device: torch.device,
    obs_dtype: torch.dtype = torch.float32,
) -> RolloutBatch:
    return RolloutBatch(
        obs=torch.zeros((T, N, E) + obs_shape, dtype=obs_dtype, device=device),
        actions=torch.zeros((T, N, E) + act_shape, device=device),
        logprobs=torch.zeros((T, N, E), device=device),
        rewards=torch.zeros((T, N, E), device=device),
        descriptors=torch.zeros((T, N, E, K), device=device),
        dones=torch.zeros((T, N, E), device=device),
        values=torch.zeros((T, N, E, 1 + K), device=device),
    )


def make_rollout_collector(
    env_cfg: DictConfig,
    *,
    N: int,
    E: int,
    T: int,
    K: int,
    hidden_dim: int,
    device: torch.device,
    seed: int,
    agent_spec: dict | None = None,
):
    backend = str(env_cfg.backend).lower()
    assert backend in {"gymnasium", "brax", "atari"}
    if agent_spec is not None and backend not in {"brax", "atari"}:
        raise ValueError("pixel policies require the brax or atari backend")
    if backend == "atari":
        from src.envs.atari import AtariRolloutCollector

        return AtariRolloutCollector(env_cfg, N=N, E=E, T=T, K=K, device=device, seed=seed)
    if backend == "brax":
        from src.envs.jax_rollout_collector import JaxRolloutCollector

        return JaxRolloutCollector(
            env_cfg,
            N=N,
            E=E,
            T=T,
            K=K,
            device=device,
            seed=seed,
        )

    from src.envs.gym_rollout_collector import GymRolloutCollector

    return GymRolloutCollector(
        env_cfg,
        N=N,
        E=E,
        T=T,
        K=K,
        hidden_dim=hidden_dim,
        device=device,
        seed=seed,
    )


def make_population_eval_metrics(
    per_policy_returns: list[list[float]],
    per_policy_descs: list[list[np.ndarray]],
    num_episodes_per_policy: int,
    return_dtype=None,
) -> dict:
    means = np.array(
        [np.mean(rs[:num_episodes_per_policy]) for rs in per_policy_returns],
        dtype=return_dtype,
    )
    descs = np.stack(
        [np.mean(np.stack(ds[:num_episodes_per_policy]), axis=0) for ds in per_policy_descs]
    )
    return {
        "eval/return_mean_pop": float(means.mean()),
        "eval/return_max_pop": float(means.max()),
        "eval/return_min_pop": float(means.min()),
        "eval/per_policy_returns": means.tolist(),
        "eval/per_policy_descriptors": descs.tolist(),
    }
