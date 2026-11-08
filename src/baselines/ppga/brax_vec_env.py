"""Brax/MJX vector env for PPGA, backed by RePPO's shared Brax backend."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.envs.brax_backend import make_brax_vec_env

_REPO_ROOT = Path(__file__).resolve().parents[3]

_BRAX_ENV_YAML = {
    "halfcheetah": "conf/env/halfcheetah.yaml",
    "walker2d": "conf/env/walker2d.yaml",
    "ant": "conf/env/ant.yaml",
    "humanoid": "conf/env/humanoid.yaml",
}

def load_brax_env_cfg(env_name: str, *, physics_backend: str | None = None) -> DictConfig:
    env_name = env_name.lower()
    if env_name not in _BRAX_ENV_YAML:
        raise ValueError(f"Unsupported PPGA Brax env: {env_name!r}")
    env_cfg = OmegaConf.load(_REPO_ROOT / _BRAX_ENV_YAML[env_name])
    if physics_backend is not None:
        env_cfg.brax.physics_backend = physics_backend
    return env_cfg


class BraxVecEnv:
    def __init__(
        self,
        *,
        env_name: str,
        batch_size: int,
        seed: int,
        num_dims: int,
        physics_backend: str | None = None,
        map_descriptors_neg1_pos1: bool = True,
        clip_obs_rew: bool | None = None,
        clip_rew: bool | None = None,
    ):
        env_cfg = load_brax_env_cfg(env_name, physics_backend=physics_backend)
        # PPGA owns its clipping via the jax-side flags below; disable the
        # torch-side reward clamp the RePPO training env applies.
        env_cfg.clip_reward = float("inf")
        if clip_obs_rew is not None:
            env_cfg.brax.clip_obs_rew = bool(clip_obs_rew)
        if clip_rew is not None:
            env_cfg.brax.clip_rew = bool(clip_rew)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._env = make_brax_vec_env(
            env_cfg,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        self.num_envs = self._env.num_envs
        self.single_observation_space = self._env.single_observation_space
        self.single_action_space = self._env.single_action_space
        self._num_dims = int(num_dims)
        self._device = device
        self._map_descriptors_neg1_pos1 = map_descriptors_neg1_pos1

    def reset(self) -> torch.Tensor:
        return self._env.reset()

    def step(self, actions: torch.Tensor):
        obs, reward, done, infos = self._env.step(actions)
        unit_measures = infos["behavior_descriptors"][:, : self._num_dims]
        if self._map_descriptors_neg1_pos1:
            measures = 2.0 * unit_measures - 1.0
        else:
            measures = unit_measures
        truncation = infos["truncation"]
        return obs, reward, done, {"measures": measures, "truncation": truncation}

    def close(self) -> None:
        self._env.close()


def make_vec_env(cfg) -> BraxVecEnv:
    physics_backend = getattr(cfg, "physics_backend", None)
    map_descriptors_neg1_pos1 = bool(getattr(cfg, "map_descriptors_neg1_pos1", True))
    return BraxVecEnv(
        env_name=cfg.env_name,
        batch_size=int(cfg.env_batch_size),
        seed=int(cfg.seed),
        num_dims=int(cfg.num_dims),
        physics_backend=physics_backend,
        map_descriptors_neg1_pos1=map_descriptors_neg1_pos1,
        clip_obs_rew=getattr(cfg, "clip_obs_rew", None),
        clip_rew=getattr(cfg, "clip_rew", None),
    )
