"""Behavioral wrappers exposing signed per-step descriptors via info['descriptor_step']."""

from __future__ import annotations

from src.envs.atari import ATARI_DESCRIPTORS

from .base import BehavioralWrapper
from .lunar_lander import LUNARLANDER_DESCRIPTOR_NAMES, LunarLanderBehavioralWrapper

WRAPPER_REGISTRY: dict[str, type[BehavioralWrapper]] = {
    "lunarlander_impact": LunarLanderBehavioralWrapper,
}

DESCRIPTOR_NAMES: dict[str, list[str]] = {
    "lunarlander_impact": LUNARLANDER_DESCRIPTOR_NAMES,
    "brax_ant_feet_contact": [
        "front_left_contact",
        "front_right_contact",
        "back_left_contact",
        "back_right_contact",
    ],
    "brax_walker2d_feet_contact": ["right_foot_contact", "left_foot_contact"],
    "brax_halfcheetah_feet_contact": ["front_foot_contact", "back_foot_contact"],
    "brax_humanoid_feet_contact": ["left_shin_contact", "right_shin_contact"],
    **{name: names for name, (names, _) in ATARI_DESCRIPTORS.items()},
}

__all__ = [
    "BehavioralWrapper",
    "WRAPPER_REGISTRY",
    "DESCRIPTOR_NAMES",
    "LunarLanderBehavioralWrapper",
]
