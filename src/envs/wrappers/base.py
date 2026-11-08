"""Base behavioral wrapper exposing signed per-step descriptors via info."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np


class BehavioralWrapper(gym.Wrapper):
    """Base class for behavioral descriptor wrappers.

    Subclasses set ``DESCRIPTOR_NAMES`` and implement
    :meth:`compute_step_descriptor` returning a per-step descriptor vector in
    ``[-1, 1]^K``. The wrapper handles episode behavior descriptors, info-dict
    plumbing, and clipping. ``DESCRIPTOR_AGG`` selects how per-step values
    aggregate into the episode descriptor: ``"mean"`` (time-average, for dense
    cumulants) or ``"sum"`` (for one-shot cumulants that fire on a single
    step; the subclass must keep the episode sum within ``[-1, 1]``).
    """

    DESCRIPTOR_NAMES: ClassVar[list[str]] = []
    DESCRIPTOR_AGG: ClassVar[str] = "mean"
    DESCRIPTOR_EPISODE_BEHAVIOR_KEY: ClassVar[str] = "descriptor_episode_behavior"

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if self.DESCRIPTOR_AGG not in ("mean", "sum"):
            raise ValueError(f"DESCRIPTOR_AGG must be 'mean' or 'sum', got {self.DESCRIPTOR_AGG}")
        self._n_steps = 0
        self._descriptor_sum = np.zeros(len(self.DESCRIPTOR_NAMES), dtype=np.float64)

    @property
    def descriptor_dim(self) -> int:
        return len(self.DESCRIPTOR_NAMES)

    def compute_step_descriptor(self, obs: np.ndarray) -> np.ndarray:
        """Override: return per-step descriptor vector of shape (K,)."""
        raise NotImplementedError

    def _emit_descriptor(self, obs: np.ndarray) -> np.ndarray:
        d = np.asarray(self.compute_step_descriptor(obs), dtype=np.float32)
        if d.shape != (self.descriptor_dim,):
            raise ValueError(
                f"compute_step_descriptor returned shape {d.shape}, expected ({self.descriptor_dim},)"
            )
        if np.any(~np.isfinite(d)):
            raise ValueError(f"compute_step_descriptor returned non-finite values: {d}")
        return np.clip(d, -1.0, 1.0)

    def _reset_descriptor_state(self) -> None:
        self._n_steps = 0
        self._descriptor_sum = np.zeros(self.descriptor_dim, dtype=np.float64)

    def _current_behavior_descriptor(self) -> np.ndarray:
        if self._n_steps <= 0:
            return np.zeros(self.descriptor_dim, dtype=np.float32)
        if self.DESCRIPTOR_AGG == "sum":
            return np.clip(self._descriptor_sum, -1.0, 1.0).astype(np.float32)
        return (self._descriptor_sum / self._n_steps).astype(np.float32)

    def _set_descriptor_info(self, info: dict, d: np.ndarray) -> None:
        info["descriptor_step"] = d.astype(np.float32, copy=False)
        info["descriptor_names"] = self.DESCRIPTOR_NAMES
        info[self.DESCRIPTOR_EPISODE_BEHAVIOR_KEY] = self._current_behavior_descriptor()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_descriptor_state()
        d = np.zeros(self.descriptor_dim, dtype=np.float32)
        self._set_descriptor_info(info, d)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._n_steps += 1
        d = self._emit_descriptor(obs)
        self._descriptor_sum += d
        self._set_descriptor_info(info, d)
        return obs, reward, terminated, truncated, info


def unit_to_signed(x: np.ndarray | float) -> np.ndarray | float:
    """Map descriptor values from [0, 1] to [-1, 1]."""
    return 2.0 * x - 1.0

