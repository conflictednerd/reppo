from __future__ import annotations

import logging
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import jax
import numpy as np
import torch
from jax import numpy as jnp
from omegaconf import DictConfig

from src.envs.utils import jax_to_torch, torch_to_jax

warnings.filterwarnings(
    "ignore",
    message="Brax System, piplines and environments are not actively being maintained.*",
    category=UserWarning,
    module="brax.io.mjcf",
)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("mujoco.mjx").setLevel(logging.WARNING)
logging.getLogger("mujoco.mjx._src.io").setLevel(logging.WARNING)


FOOT_LINK_NAMES = {
    "halfcheetah": ["ffoot", "bfoot"],
    "walker2d": ["foot", "foot_left"],
    "humanoid": ["left_shin", "right_shin"],
}

FOOT_LINK_FALLBACK_INDICES = {
    # Current Brax/MJCF leaves the Ant ankle bodies unnamed.
    "ant": [2, 4, 6, 8],
}


def _gym_space(size: int, low: float = -np.inf, high: float = np.inf) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.full((size,), low, dtype=np.float32),
        high=np.full((size,), high, dtype=np.float32),
        dtype=np.float32,
    )


def _resolve_feet_indices(sys, env_name: str) -> list[int]:
    if env_name in FOOT_LINK_FALLBACK_INDICES:
        return FOOT_LINK_FALLBACK_INDICES[env_name]
    names = list(sys.link_names)
    return [names.index(name) for name in FOOT_LINK_NAMES[env_name]]


def _build_brax_env(env_cfg: DictConfig, *, envs, wrapper_cls, training, auto_reset: bool):
    brax_cfg = env_cfg.brax
    env_name = brax_cfg.env_name.lower()

    class FeetContactWrapper(wrapper_cls):
        def __init__(self, env, name: str):
            super().__init__(env)
            self._feet_contact_idx = jnp.asarray(_resolve_feet_indices(env.sys, name))

        def reset(self, rng):
            state = self.env.reset(rng)
            state.info["behavior_descriptors"] = jnp.zeros(
                (self._feet_contact_idx.shape[0],), dtype=jnp.float32
            )
            return state

        def step(self, state, action):
            state = self.env.step(state, action)
            state.info["behavior_descriptors"] = self._feet_contact(state.pipeline_state)
            return state

        def _feet_contact(self, pipeline_state):
            contact = pipeline_state.contact
            if contact is None:
                return jnp.zeros((self._feet_contact_idx.shape[0],), dtype=jnp.float32)
            link0, link1 = contact.link_idx
            dist = contact.dist
            feet = self._feet_contact_idx[:, None]
            touches = ((link0[None, :] == feet) | (link1[None, :] == feet)) & (
                dist[None, :] <= 1e-6
            )
            return jnp.any(touches, axis=1).astype(jnp.float32)

    class ActionClipWrapper(wrapper_cls):
        def step(self, state, action):
            return self.env.step(state, jnp.clip(action, -1.0, 1.0))

    class ObservationClipWrapper(wrapper_cls):
        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=jnp.clip(state.obs, -10.0, 10.0))

        def step(self, state, action):
            state = self.env.step(state, action)
            return state.replace(obs=jnp.clip(state.obs, -10.0, 10.0))

    class RewardClipWrapper(wrapper_cls):
        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(reward=jnp.clip(state.reward, -10.0, 10.0))

        def step(self, state, action):
            state = self.env.step(state, action)
            return state.replace(reward=jnp.clip(state.reward, -10.0, 10.0))

    env = envs.get_environment(
        env_name,
        backend=brax_cfg.physics_backend,
        debug=True,
        **dict(brax_cfg.env_kwargs),
    )
    env = FeetContactWrapper(env, env_name)
    # Clipping inside the jax env, used by PPGA on humanoid (as in the original PPGA repo).
    clip_obs_rew = bool(brax_cfg.get("clip_obs_rew", False))
    clip_rew = bool(brax_cfg.get("clip_rew", clip_obs_rew))
    if clip_obs_rew:
        env = ObservationClipWrapper(env)
    if clip_rew:
        env = RewardClipWrapper(env)
    env = ActionClipWrapper(env)
    env = training.EpisodeWrapper(env, brax_cfg.episode_length, brax_cfg.action_repeat)
    env = training.VmapWrapper(env)
    if auto_reset:
        env = training.AutoResetWrapper(env)
    return env


class BraxVectorEnv:
    """Batched Brax env that returns torch tensors and unit contact descriptors."""

    def __init__(
        self,
        env_cfg: DictConfig,
        *,
        batch_size: int,
        seed: int,
        device: torch.device,
        auto_reset: bool = True,
    ):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from brax import envs
            from brax.envs.base import Wrapper
            from brax.envs.wrappers import training

        env = _build_brax_env(
            env_cfg,
            envs=envs,
            wrapper_cls=Wrapper,
            training=training,
            auto_reset=auto_reset,
        )

        self.env = env
        self.batch_size = batch_size
        self.num_envs = batch_size
        self.device = device
        self.clip_reward = float(env_cfg.get("clip_reward", float("inf")))
        self._key = jax.random.PRNGKey(seed)
        self._reset = jax.jit(env.reset)
        self._step = jax.jit(env.step)
        self._state = None
        self.single_action_space = _gym_space(int(env.action_size), low=-1.0, high=1.0)

        pixels = env_cfg.get("pixels", None)
        self.pixels_enabled = pixels is not None and bool(pixels.get("enabled", False))
        if self.pixels_enabled:
            if not bool(pixels.get("grayscale", True)):
                raise ValueError("brax pixel observations are grayscale-only")
            if bool(env_cfg.normalize_obs):
                raise ValueError("pixels must not be RMS-normalized; the agent scales by 1/255")
            self._env_name = str(env_cfg.brax.env_name).lower()
            self._image_size = int(pixels.image_size)
            self._frame_stack = int(pixels.frame_stack)
            self._supersample = int(pixels.get("supersample", 2))
            self._pix_renderer = None  # built lazily; probing spaces stays CUDA-free
            self._frames: torch.Tensor | None = None
            obs_dim = self._frame_stack * self._image_size * self._image_size
            self.single_observation_space = _gym_space(obs_dim, low=0.0, high=255.0)
        else:
            self.single_observation_space = _gym_space(int(env.observation_size))

    def _render_pixel_frame(self) -> torch.Tensor:
        """Current (batch, H, W) grayscale frame in [0, 255] from pipeline qpos."""
        if self._pix_renderer is None:
            from src.envs.mjwarp_renderer import MJWarpBatchRenderer, build_render_model

            self._pix_renderer = MJWarpBatchRenderer(
                build_render_model(self._env_name),
                nworld=self.batch_size,
                image_size=self._image_size,
                supersample=self._supersample,
                device=self.device,
            )
        qpos = jax_to_torch(self._state.pipeline_state.q, self.device)
        return self._pix_renderer.render_gray(qpos)

    def _pixel_obs_reset(self) -> torch.Tensor:
        frame = self._render_pixel_frame()
        self._frames = frame.unsqueeze(1).repeat(1, self._frame_stack, 1, 1)
        return self._frames.reshape(self.batch_size, -1)

    def _pixel_obs_step(self, done: torch.Tensor) -> torch.Tensor:
        # AutoResetWrapper has already swapped reset states into done slots, so
        # the rendered frame is the new episode's first observation there.
        frame = self._render_pixel_frame()
        frames = torch.roll(self._frames, shifts=-1, dims=1)
        frames[:, -1] = frame
        done_mask = done.bool()
        if done_mask.any():
            frames[done_mask] = frame[done_mask].unsqueeze(1)
        self._frames = frames
        return frames.reshape(self.batch_size, -1)

    def reset(
        self,
        seed: int | None = None,
        seeds: np.ndarray | list[int] | None = None,
    ) -> torch.Tensor:
        if seed is not None and seeds is not None:
            raise ValueError("Pass either seed or seeds, not both")
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        if seeds is None:
            self._key, reset_key = jax.random.split(self._key)
            reset_keys = jax.random.split(reset_key, self.batch_size)
        else:
            seed_arr = jnp.asarray(seeds, dtype=jnp.uint32)
            if seed_arr.shape != (self.batch_size,):
                raise ValueError(
                    f"Expected {self.batch_size} per-env seeds, got shape {seed_arr.shape}"
                )
            reset_keys = jax.vmap(jax.random.PRNGKey)(seed_arr)
        self._state = self._reset(reset_keys)
        if self.pixels_enabled:
            return self._pixel_obs_reset()
        return jax_to_torch(self._state.obs, self.device)

    def step(self, actions: torch.Tensor):
        assert self._state is not None
        self._state = self._step(self._state, torch_to_jax(actions))
        reward = jax_to_torch(self._state.reward, self.device).clamp(
            -self.clip_reward, self.clip_reward
        )
        done = jax_to_torch(self._state.done, self.device)
        if self.pixels_enabled:
            obs = self._pixel_obs_step(done)
        else:
            obs = jax_to_torch(self._state.obs, self.device)
        behavior_descriptors = jax_to_torch(self._state.info["behavior_descriptors"], self.device)
        truncation = jax_to_torch(self._state.info["truncation"], self.device)
        return (
            obs,
            reward,
            done,
            {"behavior_descriptors": behavior_descriptors, "truncation": truncation},
        )

    def close(self) -> None:
        if self.pixels_enabled:
            self._pix_renderer = None
            self._frames = None
        self._state = None


def make_brax_vec_env(
    env_cfg: DictConfig,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    auto_reset: bool = True,
):
    return BraxVectorEnv(
        env_cfg,
        batch_size=batch_size,
        seed=seed,
        device=device,
        auto_reset=auto_reset,
    )


def probe_brax_spaces(env_cfg: DictConfig, *, seed: int) -> tuple[gym.Space, gym.Space]:
    env = make_brax_vec_env(env_cfg, batch_size=1, seed=seed, device=torch.device("cpu"))
    try:
        return env.single_observation_space, env.single_action_space
    finally:
        env.close()
