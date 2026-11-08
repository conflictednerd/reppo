from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.envs.rollout import make_population_eval_metrics, make_rollout_batch
from src.utils.normalize import RunningMeanStd


class JaxRolloutCollector:
    """Collect RePPO rollouts from a single JAX-batched env."""

    def __init__(
        self,
        env_cfg: DictConfig,
        *,
        N: int,
        E: int,
        T: int,
        K: int,
        device: torch.device,
        seed: int,
    ):
        self.env_cfg = env_cfg
        self.N = N
        self.E = E
        self.T = T
        self.K = K
        self.device = device
        self.seed = seed

        from src.envs.brax_backend import probe_brax_spaces

        observation_space, action_space = probe_brax_spaces(env_cfg, seed=seed)

        self.obs_shape = tuple(observation_space.shape)
        self.act_shape = tuple(action_space.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.act_dim = int(np.prod(self.act_shape))
        self.discrete = False
        self.agent_act_dim = self.act_dim

        pixels = env_cfg.get("pixels", None)
        self.pixels_enabled = pixels is not None and bool(pixels.get("enabled", False))
        self.normalize_obs = bool(env_cfg.normalize_obs)
        if self.pixels_enabled and self.normalize_obs:
            raise ValueError("pixels must not be RMS-normalized; the agent scales by 1/255")
        self.clip_obs = float(env_cfg.clip_obs)
        self.obs_rms: list[RunningMeanStd | None] = [
            RunningMeanStd(self.obs_shape, device=device) if self.normalize_obs else None
            for _ in range(N)
        ]

        # Pixel obs are integral values in [0, 255]; storing the rollout buffer
        # as uint8 quarters its size and round-trips losslessly.
        self.batch = make_rollout_batch(
            T=T,
            N=N,
            E=E,
            K=K,
            obs_shape=self.obs_shape,
            act_shape=self.act_shape,
            device=device,
            obs_dtype=torch.uint8 if self.pixels_enabled else torch.float32,
        )
        self.next_obs = torch.zeros(N, E, *self.obs_shape, device=device)
        self.next_done = torch.zeros(N, E, device=device)
        self.num_policy_workers = 1
        self.num_env_workers = N * E
        self._started = False
        self._eval_envs = {}

    def start(self, agent_pop) -> None:
        del agent_pop
        if self._started:
            return
        from src.envs.brax_backend import make_brax_vec_env

        self.env = make_brax_vec_env(
            self.env_cfg,
            batch_size=self.N * self.E,
            seed=self.seed,
            device=self.device,
        )
        raw_obs = self.env.reset().view(self.N, self.E, *self.obs_shape)
        self.next_obs.copy_(self._normalize_obs(raw_obs, update=True))
        self.next_done.zero_()
        self._ep_returns = torch.zeros(self.N, self.E, device=self.device)
        self._ep_lengths = torch.zeros(self.N, self.E, device=self.device)
        self._ep_desc_sums = torch.zeros(self.N, self.E, self.K, device=self.device)
        self._started = True

    def collect(self, agent_pop) -> dict[str, list]:
        if not self._started:
            raise RuntimeError("JaxRolloutCollector.start() must be called before collect()")

        ep_returns: list[list[float]] = [[] for _ in range(self.N)]
        ep_lengths: list[list[int]] = [[] for _ in range(self.N)]
        ep_behavior_descriptors: list[list[np.ndarray]] = [[] for _ in range(self.N)]

        for t in range(self.T):
            self.batch.obs[t].copy_(self.next_obs)
            self.batch.dones[t].copy_(self.next_done)

            with torch.no_grad():
                action, logprob, _, value = agent_pop.get_action_and_value(self.next_obs)
            self.batch.actions[t].copy_(action)
            self.batch.logprobs[t].copy_(logprob)
            self.batch.values[t].copy_(value)

            raw_obs, reward, done, infos = self.env.step(
                action.reshape(self.N * self.E, self.act_dim)
            )
            raw_obs = raw_obs.view(self.N, self.E, *self.obs_shape)
            reward = reward.view(self.N, self.E)
            done = done.view(self.N, self.E).bool()
            desc_raw = infos["behavior_descriptors"].view(self.N, self.E, self.K)
            desc_step = self._descriptor_step(desc_raw)

            self.batch.rewards[t].copy_(reward)
            self.batch.descriptors[t].copy_(desc_step)
            self._ep_returns.add_(reward)
            self._ep_lengths.add_(1.0)
            self._ep_desc_sums.add_(desc_step)

            for n, e in torch.nonzero(done, as_tuple=False).detach().cpu().tolist():
                length = float(self._ep_lengths[n, e].item())
                ep_returns[n].append(float(self._ep_returns[n, e].item()))
                ep_lengths[n].append(int(length))
                ep_behavior_descriptors[n].append(
                    (self._ep_desc_sums[n, e] / length).detach().cpu().numpy().astype(np.float32)
                )
                self._ep_returns[n, e] = 0.0
                self._ep_lengths[n, e] = 0.0
                self._ep_desc_sums[n, e].zero_()

            self.next_obs.copy_(self._normalize_obs(raw_obs, update=True))
            self.next_done.copy_(done.to(torch.float32))

        return {
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "ep_behavior_descriptors": ep_behavior_descriptors,
        }

    @torch.no_grad()
    def evaluate(self, agent_pop, num_episodes_per_policy: int) -> dict:
        n_per = max(1, min(num_episodes_per_policy, 16))
        if n_per not in self._eval_envs:
            from src.envs.brax_backend import make_brax_vec_env

            eval_env_cfg = OmegaConf.merge(
                self.env_cfg,
                OmegaConf.create({"clip_reward": float("inf")}),
            )
            self._eval_envs[n_per] = make_brax_vec_env(
                eval_env_cfg,
                batch_size=self.N * n_per,
                seed=self.seed + 1000,
                device=self.device,
            )
        env = self._eval_envs[n_per]
        per_policy_returns: list[list[float]] = [[] for _ in range(self.N)]
        per_policy_descs: list[list[np.ndarray]] = [[] for _ in range(self.N)]

        # AutoResetWrapper replays each slot's first init state, so episodes after
        # the first in a slot would be duplicates; run rounds with fresh resets.
        rounds = -(-num_episodes_per_policy // n_per)
        for round_idx in range(rounds):
            if round_idx == 0:
                raw_obs = env.reset(seed=self.seed + 1000)
            else:
                raw_obs = env.reset()
            obs = self._normalize_eval_obs(raw_obs.view(self.N, n_per, *self.obs_shape), n_per)
            running = torch.zeros(self.N, n_per, device=self.device)
            ep_lengths = torch.zeros(self.N, n_per, device=self.device, dtype=torch.int64)
            desc_sums = torch.zeros(self.N, n_per, self.K, device=self.device)
            finished = torch.zeros(self.N, n_per, dtype=torch.bool, device=self.device)

            while not bool(finished.all()):
                actions = agent_pop.act_deterministic(obs).reshape(self.N * n_per, self.act_dim)
                raw_obs_flat, reward_flat, done_flat, infos = env.step(actions)
                raw_obs = raw_obs_flat.view(self.N, n_per, *self.obs_shape)
                reward = reward_flat.view(self.N, n_per)
                done = done_flat.view(self.N, n_per).bool()
                desc_raw = infos["behavior_descriptors"].view(self.N, n_per, self.K)
                desc_step = self._descriptor_step(desc_raw)

                active = ~finished
                running.add_(reward * active)
                ep_lengths.add_(active.to(torch.int64))
                desc_sums.add_(desc_step * active.unsqueeze(-1))

                for n, e in torch.nonzero(done & active, as_tuple=False).detach().cpu().tolist():
                    if len(per_policy_returns[n]) < num_episodes_per_policy:
                        length = max(1, int(ep_lengths[n, e].item()))
                        per_policy_returns[n].append(float(running[n, e].item()))
                        per_policy_descs[n].append(
                            (desc_sums[n, e] / length).detach().cpu().numpy().astype(np.float32)
                        )
                finished |= done
                obs = self._normalize_eval_obs(raw_obs, n_per)

        return make_population_eval_metrics(
            per_policy_returns,
            per_policy_descs,
            num_episodes_per_policy,
            return_dtype=np.float32,
        )

    def _normalize_obs(self, raw_obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if not self.normalize_obs:
            return raw_obs.to(torch.float32)
        out = torch.empty_like(raw_obs, dtype=torch.float32)
        for n, rms in enumerate(self.obs_rms):
            assert rms is not None
            if update:
                rms.update(raw_obs[n])
            out[n] = rms.normalize(raw_obs[n], self.clip_obs)
        return out

    def _normalize_eval_obs(self, raw_obs: torch.Tensor, n_per_policy: int) -> torch.Tensor:
        if not self.normalize_obs:
            return raw_obs.to(torch.float32)
        out = torch.empty((self.N, n_per_policy, self.obs_dim), device=self.device)
        flat = raw_obs.view(self.N, n_per_policy, self.obs_dim)
        for n, rms in enumerate(self.obs_rms):
            assert rms is not None
            out[n] = rms.normalize(flat[n], self.clip_obs)
        return out.view(self.N, n_per_policy, *self.obs_shape)

    def _descriptor_step(self, desc: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(desc).all() or (desc < 0.0).any() or (desc > 1.0).any():
            raise ValueError(f"Brax descriptors must be finite and in [0, 1], got {desc}")
        return 2.0 * desc - 1.0

    def close(self) -> None:
        if hasattr(self, "env"):
            self.env.close()
            del self.env
        self._started = False
        for env in self._eval_envs.values():
            env.close()
        self._eval_envs.clear()
