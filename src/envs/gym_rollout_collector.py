import multiprocessing as mp
import os

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.agents import make_agent
from src.envs.gym_backend import make_gym_vec_env
from src.envs.rollout import make_population_eval_metrics, make_rollout_batch
from src.envs.utils import probe_env_spaces, same_step_final_infos
from src.utils.normalize import RunningMeanStd


def _rollout_worker_loop(
    conn: mp.connection.Connection,
    n: int,
    env_cfg_dict: dict,
    seed: int,
    E: int,
    T: int,
    obs_shape: tuple,
    act_shape: tuple,
    K: int,
    hidden_dim: int,
    obs_dim: int,
    agent_act_dim: int,
    discrete: bool,
    normalize_obs: bool,
    clip_obs: float,
    shared_bufs: dict[str, torch.Tensor],
) -> None:
    """Worker process for one policy's Gym vector env."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)

    env_cfg = OmegaConf.create(env_cfg_dict)
    envs = make_gym_vec_env(env_cfg, num_envs=E, seed=seed + n * E)

    agent = make_agent(
        obs_dim, agent_act_dim, descriptor_dim=K, hidden_dim=hidden_dim, discrete=discrete
    )
    agent.eval()
    obs_rms = RunningMeanStd(shape=obs_shape) if normalize_obs else None
    action_np = np.empty((E,) + act_shape, dtype=np.float32)

    sh_obs = shared_bufs["obs"]
    sh_act = shared_bufs["actions"]
    sh_lp = shared_bufs["logprobs"]
    sh_rew = shared_bufs["rewards"]
    sh_desc = shared_bufs["descriptors"]
    sh_done = shared_bufs["dones"]
    sh_val = shared_bufs["values"]
    sh_next_obs = shared_bufs["next_obs"]
    sh_next_done = shared_bufs["next_done"]

    init_msg = conn.recv()
    agent.load_state_dict(init_msg["agent_state"])
    if obs_rms is not None and init_msg["obs_rms_state"] is not None:
        obs_rms.load_state_dict(init_msg["obs_rms_state"])

    raw_obs, _ = envs.reset(seed=seed + n * E)
    if normalize_obs:
        obs_rms.update(raw_obs)
        norm = obs_rms.normalize(raw_obs, clip=clip_obs).astype(np.float32)
    else:
        norm = raw_obs.astype(np.float32, copy=False)

    next_obs_t = torch.as_tensor(norm, dtype=torch.float32).view(E, *obs_shape)
    next_done_t = torch.zeros(E)
    sh_next_obs.copy_(next_obs_t)
    sh_next_done.copy_(next_done_t)
    conn.send({"obs_rms_state": obs_rms.state_dict() if obs_rms else None})

    while True:
        msg = conn.recv()
        if msg is None:
            break

        agent.load_state_dict(msg["agent_state"])
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ep_behavior_descriptors: list[list[float]] = []

        for t in range(T):
            sh_obs[t].copy_(next_obs_t)
            sh_done[t].copy_(next_done_t)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_t)
            sh_act[t].copy_(action)
            sh_lp[t].copy_(logprob)
            sh_val[t].copy_(value)

            action_np[:] = action.numpy()
            env_action = action_np.reshape(E).astype(np.int64) if discrete else action_np
            raw_obs, reward, term, trunc, infos = envs.step(env_action)
            done = np.logical_or(term, trunc)
            final_infos = same_step_final_infos(infos, done)

            sh_rew[t] = torch.as_tensor(reward, dtype=torch.float32)
            desc_step = np.asarray(infos["descriptor_step"], dtype=np.float32).reshape(E, K)
            if final_infos is not None:
                final_desc_step = np.asarray(
                    final_infos["descriptor_step"], dtype=np.float32
                ).reshape(E, K)
                desc_step[done] = final_desc_step[done]
            if (
                np.any(~np.isfinite(desc_step))
                or np.any(desc_step < -1.0)
                or np.any(desc_step > 1.0)
            ):
                raise ValueError(f"descriptor_step must be finite and in [-1, 1], got {desc_step}")
            sh_desc[t] = torch.as_tensor(
                desc_step,
                dtype=torch.float32,
            )

            if normalize_obs:
                obs_rms.update(raw_obs)
                norm = obs_rms.normalize(raw_obs, clip=clip_obs).astype(np.float32)
            else:
                norm = raw_obs.astype(np.float32, copy=False)
            next_obs_t = torch.as_tensor(norm, dtype=torch.float32).view(E, *obs_shape)
            next_done_t = torch.as_tensor(done, dtype=torch.float32)

            if final_infos is not None:
                mask = np.asarray(final_infos["_episode"], dtype=bool)
                if not np.array_equal(mask, done):
                    raise ValueError(
                        f"SAME_STEP episode mask must match done mask, got episode={mask}, done={done}"
                    )
                returns = np.asarray(final_infos["episode"]["r"])
                lengths = np.asarray(final_infos["episode"]["l"])
                descs = np.asarray(final_infos["descriptor_episode_behavior"], dtype=np.float32)
                if np.any(~np.isfinite(descs)) or np.any(descs < -1.0) or np.any(descs > 1.0):
                    raise ValueError(
                        f"descriptor_episode_behavior must be finite and in [-1, 1], got {descs}"
                    )
                for e in np.where(done)[0]:
                    ep_returns.append(float(returns[e]))
                    ep_lengths.append(int(lengths[e]))
                    ep_behavior_descriptors.append(descs[e].tolist())

        sh_next_obs.copy_(next_obs_t)
        sh_next_done.copy_(next_done_t)
        conn.send(
            {
                "ep_returns": ep_returns,
                "ep_lengths": ep_lengths,
                "ep_behavior_descriptors": ep_behavior_descriptors,
                "obs_rms_state": obs_rms.state_dict() if obs_rms else None,
            }
        )

    envs.close()


class GymRolloutCollector:
    """Collect fixed-length rollouts from Gym envs for a policy population."""

    def __init__(
        self,
        env_cfg: DictConfig,
        *,
        N: int,
        E: int,
        T: int,
        K: int,
        hidden_dim: int,
        device: torch.device,
        seed: int,
    ):
        self.env_cfg = env_cfg
        self.N = N
        self.E = E
        self.T = T
        self.K = K
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed

        observation_space, action_space = probe_env_spaces(env_cfg, seed=seed)
        self.obs_shape = tuple(observation_space.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        if isinstance(action_space, gym.spaces.Discrete):
            # Actions are stored as float indices of shape (1,); the agent's
            # action head has one output per discrete action.
            self.discrete = True
            self.act_shape = (1,)
            self.agent_act_dim = int(action_space.n)
        else:
            assert isinstance(action_space, gym.spaces.Box)
            self.discrete = False
            self.act_shape = tuple(action_space.shape)
            self.agent_act_dim = int(np.prod(self.act_shape))
        self.act_dim = int(np.prod(self.act_shape))

        self.normalize_obs = bool(env_cfg.normalize_obs)
        self.clip_obs = float(env_cfg.clip_obs)
        self.obs_rms: list[RunningMeanStd | None] = [
            RunningMeanStd(shape=self.obs_shape) if self.normalize_obs else None for _ in range(N)
        ]

        self.batch = make_rollout_batch(
            T=T,
            N=N,
            E=E,
            K=K,
            obs_shape=self.obs_shape,
            act_shape=self.act_shape,
            device=device,
        )
        self.next_obs = torch.zeros(N, E, *self.obs_shape, device=device)
        self.next_done = torch.zeros(N, E, device=device)
        self.num_policy_workers = N
        self.num_env_workers = N * E
        self._started = False

    def start(self, agent_pop) -> None:
        if self._started:
            return
        self._start_workers(agent_pop)
        self._started = True

    def collect(self, agent_pop) -> dict[str, list]:
        if not self._started:
            raise RuntimeError("GymRolloutCollector.start() must be called before collect()")
        return self._collect_from_workers(agent_pop)

    @torch.no_grad()
    def evaluate(self, agent_pop, num_episodes_per_policy: int) -> dict:
        """Per-policy deterministic eval in fresh envs."""
        n_per = max(1, min(num_episodes_per_policy, 4))
        n_total = self.N * n_per
        eval_envs = make_gym_vec_env(self.env_cfg, num_envs=n_total, seed=self.seed + 1000)
        obs, _ = eval_envs.reset(seed=self.seed + 1000)
        per_policy_returns: list[list[float]] = [[] for _ in range(self.N)]
        per_policy_descs: list[list[np.ndarray]] = [[] for _ in range(self.N)]
        target = num_episodes_per_policy * self.N

        while sum(len(r) for r in per_policy_returns) < target:
            obs_norm = self._normalize_eval_obs(obs, n_per)
            obs_t = torch.as_tensor(obs_norm, device=self.device, dtype=torch.float32).view(
                self.N, n_per, *self.obs_shape
            )
            actions = (
                agent_pop.act_deterministic(obs_t).reshape(n_total, self.act_dim).cpu().numpy()
            )
            if self.discrete:
                actions = actions.reshape(n_total).astype(np.int64)
            obs, _reward, term, trunc, infos = eval_envs.step(actions)
            done = np.logical_or(term, trunc)
            final_infos = same_step_final_infos(infos, done)

            desc_step = np.asarray(infos["descriptor_step"], dtype=np.float32).reshape(
                n_total, self.K
            )
            if final_infos is not None:
                final_desc_step = np.asarray(
                    final_infos["descriptor_step"], dtype=np.float32
                ).reshape(n_total, self.K)
                desc_step[done] = final_desc_step[done]
            if (
                np.any(~np.isfinite(desc_step))
                or np.any(desc_step < -1.0)
                or np.any(desc_step > 1.0)
            ):
                raise ValueError(f"descriptor_step must be finite and in [-1, 1], got {desc_step}")

            if final_infos is not None:
                mask = np.asarray(final_infos["_episode"], dtype=bool)
                if not np.array_equal(mask, done):
                    raise ValueError(
                        f"SAME_STEP episode mask must match done mask, got episode={mask}, done={done}"
                    )
                returns = np.asarray(final_infos["episode"]["r"])
                episode_desc = np.asarray(
                    final_infos["descriptor_episode_behavior"], dtype=np.float32
                ).reshape(n_total, self.K)
                if (
                    np.any(~np.isfinite(episode_desc))
                    or np.any(episode_desc < -1.0)
                    or np.any(episode_desc > 1.0)
                ):
                    raise ValueError(
                        f"descriptor_episode_behavior must be finite and in [-1, 1], got {episode_desc}"
                    )

                for env_idx in np.where(done)[0]:
                    n = env_idx // n_per
                    if len(per_policy_returns[n]) < num_episodes_per_policy:
                        per_policy_returns[n].append(float(returns[env_idx]))
                        per_policy_descs[n].append(episode_desc[env_idx])
        eval_envs.close()

        return make_population_eval_metrics(
            per_policy_returns,
            per_policy_descs,
            num_episodes_per_policy,
        )

    def _normalize_eval_obs(self, raw_flat: np.ndarray, n_per_policy: int) -> np.ndarray:
        if not self.normalize_obs:
            return raw_flat.astype(np.float32, copy=False)
        out = raw_flat.reshape(self.N, n_per_policy, *self.obs_shape)
        normalized = np.empty_like(out, dtype=np.float32)
        for n, rms in enumerate(self.obs_rms):
            assert rms is not None
            normalized[n] = rms.normalize(out[n], clip=self.clip_obs).astype(np.float32)
        return normalized.reshape(self.N * n_per_policy, *self.obs_shape)

    def _start_workers(self, agent_pop) -> None:
        env_cfg_dict = OmegaConf.to_container(self.env_cfg, resolve=True)
        self._sh_bufs: list[dict[str, torch.Tensor]] = []
        for _ in range(self.N):
            self._sh_bufs.append(
                {
                    "obs": torch.zeros((self.T, self.E) + self.obs_shape).share_memory_(),
                    "actions": torch.zeros((self.T, self.E) + self.act_shape).share_memory_(),
                    "logprobs": torch.zeros(self.T, self.E).share_memory_(),
                    "rewards": torch.zeros(self.T, self.E).share_memory_(),
                    "descriptors": torch.zeros(self.T, self.E, self.K).share_memory_(),
                    "dones": torch.zeros(self.T, self.E).share_memory_(),
                    "values": torch.zeros(self.T, self.E, 1 + self.K).share_memory_(),
                    "next_obs": torch.zeros((self.E,) + self.obs_shape).share_memory_(),
                    "next_done": torch.zeros(self.E).share_memory_(),
                }
            )

        ctx = mp.get_context("fork")
        self._worker_conns: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        for n in range(self.N):
            parent_conn, child_conn = ctx.Pipe()
            worker = ctx.Process(
                target=_rollout_worker_loop,
                args=(
                    child_conn,
                    n,
                    env_cfg_dict,
                    self.seed,
                    self.E,
                    self.T,
                    self.obs_shape,
                    self.act_shape,
                    self.K,
                    self.hidden_dim,
                    self.obs_dim,
                    self.agent_act_dim,
                    self.discrete,
                    self.normalize_obs,
                    self.clip_obs,
                    self._sh_bufs[n],
                ),
                daemon=False,
            )
            worker.start()
            child_conn.close()
            self._worker_conns.append(parent_conn)
            self._workers.append(worker)

        for n, conn in enumerate(self._worker_conns):
            conn.send(
                {
                    "agent_state": agent_pop.policy_state_dict(n),
                    "obs_rms_state": self.obs_rms[n].state_dict()
                    if self.obs_rms[n] is not None
                    else None,
                }
            )

        for n, conn in enumerate(self._worker_conns):
            reply = conn.recv()
            self.next_obs[n] = self._sh_bufs[n]["next_obs"].to(self.device)
            self.next_done[n] = self._sh_bufs[n]["next_done"].to(self.device)
            if self.obs_rms[n] is not None and reply["obs_rms_state"] is not None:
                self.obs_rms[n].load_state_dict(reply["obs_rms_state"])

    def _collect_from_workers(self, agent_pop) -> dict[str, list]:
        for n, conn in enumerate(self._worker_conns):
            conn.send({"agent_state": agent_pop.policy_state_dict(n)})

        ep_returns: list[list[float]] = [[] for _ in range(self.N)]
        ep_lengths: list[list[int]] = [[] for _ in range(self.N)]
        ep_behavior_descriptors: list[list[np.ndarray]] = [[] for _ in range(self.N)]
        for n, conn in enumerate(self._worker_conns):
            result = conn.recv()
            ep_returns[n] = result["ep_returns"]
            ep_lengths[n] = result["ep_lengths"]
            ep_behavior_descriptors[n] = [
                np.asarray(desc, dtype=np.float32) for desc in result["ep_behavior_descriptors"]
            ]
            if self.obs_rms[n] is not None and result["obs_rms_state"] is not None:
                self.obs_rms[n].load_state_dict(result["obs_rms_state"])

        for n, sh in enumerate(self._sh_bufs):
            self.batch.obs[:, n].copy_(sh["obs"])
            self.batch.actions[:, n].copy_(sh["actions"])
            self.batch.logprobs[:, n].copy_(sh["logprobs"])
            self.batch.rewards[:, n].copy_(sh["rewards"])
            self.batch.descriptors[:, n].copy_(sh["descriptors"])
            self.batch.dones[:, n].copy_(sh["dones"])
            self.batch.values[:, n].copy_(sh["values"])
            self.next_obs[n].copy_(sh["next_obs"])
            self.next_done[n].copy_(sh["next_done"])

        return {
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "ep_behavior_descriptors": ep_behavior_descriptors,
        }

    def close(self) -> None:
        if hasattr(self, "_worker_conns"):
            for conn in self._worker_conns:
                try:
                    conn.send(None)
                except Exception:
                    pass
            for worker in self._workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
