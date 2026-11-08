"""Shared re-evaluation machinery: the batched Brax runner and per-method actor loaders.

Every policy is re-evaluated the same way: deterministic actions, fresh random
init states each round, rewards never clipped, unit [0, 1] contact descriptors.
Raw per-episode values are kept.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAX_ENV_NAMES = ("ant", "halfcheetah", "walker2d", "humanoid")


class BraxEvalRunner:
    """Batched deterministic evaluation on a single shared Brax env."""

    def __init__(
        self,
        env_name,
        *,
        actors_per_chunk: int,
        episodes_per_round: int,
        seed: int,
        device: torch.device,
        episode_length: int | None = None,
    ):
        from src.envs import descriptor_dim
        from src.envs.brax_backend import make_brax_vec_env

        if env_name not in BRAX_ENV_NAMES:
            raise ValueError(f"Unknown brax env {env_name!r}, expected one of {BRAX_ENV_NAMES}")

        env_cfg = OmegaConf.load(REPO_ROOT / f"conf/env/{env_name}.yaml")
        overrides: dict = {
            "clip_reward": float("inf"),
            "brax": {"clip_obs_rew": False, "clip_rew": False},
        }
        if episode_length is not None:
            overrides["brax"]["episode_length"] = int(episode_length)
        env_cfg = OmegaConf.merge(env_cfg, OmegaConf.create(overrides))

        self.K = descriptor_dim(env_cfg.descriptor_set)
        self.n_chunk = actors_per_chunk
        self.E = episodes_per_round
        self.episode_length = int(env_cfg.brax.episode_length)
        self.device = device
        self._seed = seed
        self._first_reset = True

        self.env = make_brax_vec_env(
            env_cfg, batch_size=actors_per_chunk * episodes_per_round, seed=seed, device=device
        )
        self.obs_dim = int(self.env.single_observation_space.shape[0])
        self.act_dim = int(self.env.single_action_space.shape[0])

    @torch.no_grad()
    def _run_round(self, actors) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(actors)
        if n > self.n_chunk:
            raise ValueError(f"Got {n} actors but runner was built for chunks of {self.n_chunk}")
        B = self.n_chunk * self.E

        if self._first_reset:
            obs = self.env.reset(seed=self._seed)
            self._first_reset = False
        else:
            obs = self.env.reset()

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        returns = torch.zeros(B, device=self.device)
        lengths = torch.zeros(B, device=self.device)
        desc_sums = torch.zeros(B, self.K, device=self.device)

        for _ in range(self.episode_length):
            actions = torch.empty(B, self.act_dim, device=self.device)
            for i in range(self.n_chunk):
                actor = actors[min(i, n - 1)]
                sl = slice(i * self.E, (i + 1) * self.E)
                actions[sl] = actor(obs[sl])

            obs, reward, done, infos = self.env.step(actions)
            active = ~finished
            returns += reward * active
            lengths += active.float()
            desc_sums += infos["behavior_descriptors"] * active.unsqueeze(1)
            finished |= done.bool()
            if bool(finished.all()):
                break

        assert bool(finished.all())

        descs = desc_sums / lengths.clamp(min=1.0).unsqueeze(1)
        if not torch.isfinite(descs).all() or (descs < 0.0).any() or (descs > 1.0).any():
            raise ValueError(f"Brax unit descriptors must be finite and in [0, 1], got {descs}")

        returns_np = (
            returns.detach().cpu().numpy().astype(np.float64).reshape(self.n_chunk, self.E)[:n]
        )
        descs_np = (
            descs.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(self.n_chunk, self.E, self.K)[:n]
        )
        lengths_np = (
            lengths.detach().cpu().numpy().astype(np.int64).reshape(self.n_chunk, self.E)[:n]
        )
        return returns_np, descs_np, lengths_np

    @torch.no_grad()
    def evaluate(self, actors, n_episodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-episode returns [n, n_episodes], descriptors [n, n_episodes, K],
        and lengths [n, n_episodes]."""
        rounds = math.ceil(n_episodes / self.E)
        ret_chunks = []
        desc_chunks = []
        len_chunks = []
        for _ in range(rounds):
            r, d, ln = self._run_round(actors)
            ret_chunks.append(r)
            desc_chunks.append(d)
            len_chunks.append(ln)
        returns = np.concatenate(ret_chunks, axis=1)[:, :n_episodes]
        descs = np.concatenate(desc_chunks, axis=1)[:, :n_episodes]
        lengths = np.concatenate(len_chunks, axis=1)[:, :n_episodes]
        return returns, descs, lengths

    def close(self):
        self.env.close()


def _make_reppo_actor(agent, rms_state, clip_raw: bool, clip_obs: float, device: torch.device):
    if rms_state is not None:
        mean = torch.as_tensor(rms_state["mean"], dtype=torch.float64, device=device)
        std = torch.sqrt(
            torch.as_tensor(rms_state["var"], dtype=torch.float64, device=device) + 1e-8
        )

    def act(obs):
        x = obs.clamp(-10.0, 10.0) if clip_raw else obs
        if rms_state is not None:
            x = ((x.to(torch.float64) - mean) / std).clamp(-clip_obs, clip_obs).to(torch.float32)
        return agent.act_deterministic(x)

    return act


def load_reppo_actors(run_dir: Path, device: torch.device) -> tuple[list, dict]:
    import yaml

    from src.agents import MultiHeadAgent

    ckpt = torch.load(run_dir / "checkpoints" / "final.pt", map_location=device, weights_only=False)
    env_config = ckpt["env_config"]
    if env_config.get("backend") != "brax":
        raise ValueError(
            f"load_reppo_actors requires a brax-backend checkpoint, got backend="
            f"{env_config.get('backend')!r}"
        )

    env_name = env_config["brax"]["env_name"]
    clip_raw = bool(env_config["brax"].get("clip_obs_rew", False))
    clip_obs = float(ckpt["clip_obs"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    K = int(ckpt["K"])
    hidden_dim = int(ckpt["hidden_dim"])

    actors = []
    for state_dict, rms_state in zip(ckpt["agent_states"], ckpt["obs_rms"], strict=True):
        agent = MultiHeadAgent(obs_dim, act_dim, descriptor_dim=K, hidden_dim=hidden_dim)
        agent.load_state_dict(state_dict)
        agent = agent.to(device).eval()
        actors.append(_make_reppo_actor(agent, rms_state, clip_raw, clip_obs, device))

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    meta = {
        "env_name": env_name,
        "seed": int(cfg["seed"]),
        "total_env_steps": int(cfg["algorithm"]["total_timesteps"]),
    }
    return actors, meta


def _make_ppga_actor(actor, *, clip_raw: bool, normalize_obs: bool):
    if normalize_obs:
        rms = actor.obs_normalizer.obs_rms
        eps = actor.obs_normalizer.epsilon

    def act(obs):
        x = obs.clamp(-10.0, 10.0) if clip_raw else obs
        if normalize_obs:
            x = (x - rms.mean) / torch.sqrt(rms.var + eps)
        return actor(x)

    return act


def load_ppga_actors(
    run_dir: Path, device: torch.device, archive_pkl: Path | None = None
) -> tuple[list, dict]:
    import pickle
    import re

    from src.baselines.ppga.models.actor_critic import Actor

    cfg = json.loads((run_dir / "cfg.json").read_text())
    if archive_pkl is None:
        archive_pkl = sorted(run_dir.glob("checkpoints/cp_*/archive_df_*.pkl"))[-1]
    with open(archive_pkl, "rb") as f:
        archive_df = pickle.load(f)

    env_name = cfg["env_name"]
    clip_raw = bool(cfg.get("clip_obs_rew") or False)
    normalize_obs = bool(cfg.get("normalize_obs", True))
    normalize_returns = bool(cfg.get("normalize_returns", True))

    actors = []
    for _, row in archive_df.iterrows():
        solution = row.filter(regex="^solution_").to_numpy(dtype=np.float64)
        actor = (
            Actor(
                obs_shape=tuple(cfg["obs_shape"]),
                action_shape=tuple(cfg["action_shape"]),
                normalize_obs=normalize_obs,
                normalize_returns=normalize_returns,
            )
            .deserialize(solution)
            .to(device)
        )
        actor.eval()
        if normalize_obs and isinstance(row.get("metadata"), dict):
            obs_norm = row["metadata"].get("obs_normalizer")
            if obs_norm:
                actor.obs_normalizer.load_state_dict(obs_norm)
        actors.append(_make_ppga_actor(actor, clip_raw=clip_raw, normalize_obs=normalize_obs))

    match = re.search(r"(\d+)$", archive_pkl.stem)
    iterations = int(match.group(1)) if match else int(cfg["total_iterations"])
    total_env_steps = int(
        iterations
        * (
            (cfg["calc_gradient_iters"] + cfg["move_mean_iters"])
            * cfg["rollout_length"]
            * cfg["num_envs"]
            + 2 * cfg["env_batch_size"] * 1000
        )
    )

    measures = archive_df.filter(regex="^measure_").to_numpy(dtype=np.float64)
    if bool(cfg.get("map_descriptors_neg1_pos1", True)):
        measures = (measures + 1.0) / 2.0

    meta = {
        "env_name": env_name,
        "seed": int(cfg["seed"]),
        "total_env_steps": total_env_steps,
    }
    return actors, meta


def load_pga_me_actors(
    run_dir: Path, device: torch.device, repertoire_pkl: Path | None = None
) -> tuple[list, dict]:
    import pickle

    cfg = json.loads((run_dir / "cfg.json").read_text())
    if repertoire_pkl is None:
        repertoire_pkl = sorted(run_dir.glob("checkpoints/cp_*/repertoire.pkl"))[-1]
    with open(repertoire_pkl, "rb") as f:
        rep = pickle.load(f)

    fitnesses = np.asarray(rep["fitnesses"]).squeeze()
    active_idx = np.where(fitnesses > -np.inf)[0]

    genotypes = rep["genotypes"]
    dense_keys = sorted(genotypes["params"], key=lambda k: int(k.split("_")[1]))

    # Mirror the QDax MLP from src/baselines/pga_me/train.py: ReLU hidden
    # activations, tanh on the output layer.
    actors = []
    for idx in active_idx:
        idx = int(idx)
        layers: list[torch.nn.Module] = []
        for i, dk in enumerate(dense_keys):
            kernel = np.asarray(genotypes["params"][dk]["kernel"][idx], dtype=np.float32)
            bias = np.asarray(genotypes["params"][dk]["bias"][idx], dtype=np.float32)
            lin = torch.nn.Linear(kernel.shape[0], kernel.shape[1])
            lin.weight.data = torch.from_numpy(kernel.T.copy())
            lin.bias.data = torch.from_numpy(bias.copy())
            layers += [lin, torch.nn.Tanh() if i == len(dense_keys) - 1 else torch.nn.ReLU()]
        actors.append(torch.nn.Sequential(*layers).to(device).eval())

    total_env_steps = int(cfg["total_iterations"] * cfg["env_batch_size"] * cfg["episode_length"])
    meta = {
        "env_name": cfg["env_name"],
        "seed": int(cfg["seed"]),
        "total_env_steps": total_env_steps,
    }
    return actors, meta


def load_dcrl_me_actors(
    run_dir: Path, device: torch.device, repertoire_pkl: Path | None = None
) -> tuple[list, dict]:
    """Archive policies are QDax MLPs, same dump format as PGA-ME."""
    return load_pga_me_actors(run_dir, device, repertoire_pkl)


def load_sep_cma_mae_actors(run_dir: Path, device: torch.device) -> tuple[list, dict]:
    from src.baselines.sep_cma_mae.train import batched_act, unpack_solutions

    cfg = json.loads((run_dir / "config.json").read_text())
    archive_path = run_dir / "archive_final.npz"
    if not archive_path.exists():
        archive_path = sorted(run_dir.glob("archive_*.npz"))[-1]
    archive = np.load(archive_path)
    final_summary = json.loads((run_dir / "final_summary.json").read_text())

    solutions = np.asarray(archive["solution"], dtype=np.float32)
    obs_dim = int(cfg["obs_dim"])
    act_dim = int(cfg["act_dim"])
    weights = unpack_solutions(solutions, obs_dim, act_dim, device)

    actors = []
    for idx in range(len(solutions)):
        actor_weights = tuple(w[idx : idx + 1] for w in weights)

        def act(obs, actor_weights=actor_weights):
            expanded = tuple(w.expand(obs.shape[0], *w.shape[1:]) for w in actor_weights)
            return batched_act(expanded, obs)

        actors.append(act)

    meta = {
        "env_name": cfg["env_name"],
        "seed": int(cfg["seed"]),
        "total_env_steps": int(final_summary["env_steps"]),
    }
    return actors, meta
