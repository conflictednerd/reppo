"""LunarLander behavioral descriptors: one-shot impact position and speed.

Both cumulants fire exactly once, at the first ground contact (either leg
touching down, or the body crashing), and are zero on every other step. With
``DESCRIPTOR_AGG = "sum"`` the episode descriptor equals the impact values
themselves. Episodes that never touch the ground (e.g. drifting off-screen)
keep the zero descriptor.
"""

from __future__ import annotations

import numpy as np

from .base import BehavioralWrapper, unit_to_signed

LUNARLANDER_DESCRIPTOR_NAMES = ["impact_x", "impact_speed"]

# Vertical speed scale in observation units: the free-fall impact speed
# (engines off from the spawn height reaches ~1.93).
IMPACT_SPEED_MAX = 2.0


class LunarLanderBehavioralWrapper(BehavioralWrapper):
    DESCRIPTOR_NAMES = LUNARLANDER_DESCRIPTOR_NAMES
    DESCRIPTOR_AGG = "sum"

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._impact_recorded = False
        self._prev_vy = float(np.asarray(obs).reshape(-1)[3])
        return obs, info

    def compute_step_descriptor(self, obs: np.ndarray) -> np.ndarray:
        if self._impact_recorded:
            return np.zeros(2, dtype=np.float32)
        obs = np.asarray(obs).reshape(-1)
        contact = obs[6] > 0.5 or obs[7] > 0.5 or self.env.unwrapped.game_over
        if not contact:
            self._prev_vy = float(obs[3])
            return np.zeros(2, dtype=np.float32)
        self._impact_recorded = True
        impact_x = float(np.clip(obs[0], -1.0, 1.0))
        # Speed on the step before contact, i.e. before collision damping.
        speed_unit = min(abs(self._prev_vy), IMPACT_SPEED_MAX) / IMPACT_SPEED_MAX
        return np.array([impact_x, unit_to_signed(speed_unit)], dtype=np.float32)
