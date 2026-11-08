"""Brax/MJX QDax environments with descriptor logic aligned to RePPO/PPGA."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from brax.envs.base import State
from qdax.tasks.brax import v2 as qdax_brax_v2
from qdax.tasks.brax.v2.wrappers.locomotion_wrappers import FeetContactWrapper

# Current Brax/MJCF leaves Ant ankle bodies unnamed; matches src/envs/brax_backend.py.
FOOT_LINK_FALLBACK_INDICES = {
    "ant": [2, 4, 6, 8],
}

# Same descriptor order as src/envs/brax_backend.py FOOT_LINK_NAMES.
FOOT_LINK_NAMES = {
    "halfcheetah": ["ffoot", "bfoot"],
    "walker2d": ["foot", "foot_left"],
    "humanoid": ["left_shin", "right_shin"],
}

ENV_NAME_MAP = {
    "halfcheetah": "halfcheetah_uni",
    "walker2d": "walker2d_uni",
    "ant": "ant_uni",
    "humanoid": "humanoid_uni",
}


class MjxFeetContactWrapper(FeetContactWrapper):
    """Feet-contact wrapper with MJCF link-index fallback for Ant."""

    def __init__(self, env, env_name: str):
        super().__init__(env, env_name)
        if env_name in FOOT_LINK_NAMES:
            names = list(env.sys.link_names)
            self._feet_idx = jnp.array(
                [names.index(name) for name in FOOT_LINK_NAMES[env_name]], dtype=jnp.int32
            )
        elif env_name in FOOT_LINK_FALLBACK_INDICES and int(self._feet_idx.shape[0]) == 0:
            self._feet_idx = jnp.array(FOOT_LINK_FALLBACK_INDICES[env_name], dtype=jnp.int32)

    def _get_feet_contact(self, state: State) -> jnp.ndarray:
        contact = state.pipeline_state.contact
        if contact is None:
            return jnp.zeros((self.descriptor_length,), dtype=jnp.float32)
        link0, link1 = contact.link_idx
        dist = contact.dist
        feet = self._feet_idx[:, None]
        touches = ((link0[None, :] == feet) | (link1[None, :] == feet)) & (
            dist[None, :] <= 1e-6
        )
        return jnp.any(touches, axis=-1).astype(jnp.float32)


def _patch_feet_wrapper(env_name: str) -> None:
    base = qdax_brax_v2._qdax_custom_envs.get(env_name, {})
    if "FeetContactWrapper" not in [w.__name__ for w in base.get("wrappers", [])]:
        return
    qdax_brax_v2._qdax_custom_envs[env_name] = {
        **base,
        "wrappers": [MjxFeetContactWrapper],
    }


def make_qdax_env(
    env_name: str,
    *,
    episode_length: int = 1000,
    physics_backend: str = "mjx",
    **kwargs: Any,
):
    """Create a QDax Brax env using the shared locomotion descriptor setup."""
    qdax_name = ENV_NAME_MAP.get(env_name.lower(), env_name)
    _patch_feet_wrapper(qdax_name)
    return qdax_brax_v2.create(
        qdax_name,
        episode_length=episode_length,
        backend=physics_backend,
        auto_reset=True,
        **kwargs,
    )


def qdax_env_name(env_name: str) -> str:
    return ENV_NAME_MAP.get(env_name.lower(), env_name)
