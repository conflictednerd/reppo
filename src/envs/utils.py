import gymnasium as gym
import jax
import numpy as np
import torch
from omegaconf import DictConfig

from src.envs.wrappers import DESCRIPTOR_NAMES


def jax_to_torch(x, device: torch.device) -> torch.Tensor:
    return torch.from_dlpack(x).to(device=device, dtype=torch.float32)


def torch_to_jax(x: torch.Tensor):
    return jax.dlpack.from_dlpack(x.detach().contiguous())


def same_step_final_infos(infos: dict, done: np.ndarray) -> dict | None:
    done = np.asarray(done, dtype=bool)
    if not done.any():
        return None
    final_mask = np.asarray(infos["_final_info"], dtype=bool)
    if not np.array_equal(final_mask, done):
        raise ValueError(
            f"SAME_STEP final_info mask must match done mask, got final_info={final_mask}, done={done}"
        )
    return infos["final_info"]


def probe_env_spaces(env_cfg: DictConfig, *, seed: int) -> tuple[gym.Space, gym.Space]:
    if str(env_cfg.backend).lower() == "brax":
        from src.envs.brax_backend import probe_brax_spaces

        return probe_brax_spaces(env_cfg, seed=seed)

    from src.envs.gym_backend import probe_gym_spaces

    return probe_gym_spaces(env_cfg, seed=seed)


def descriptor_dim(descriptor_set: str) -> int:
    return len(DESCRIPTOR_NAMES[descriptor_set])


def descriptor_names(descriptor_set: str) -> list[str]:
    return list(DESCRIPTOR_NAMES[descriptor_set])
