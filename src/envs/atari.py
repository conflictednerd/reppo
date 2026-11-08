"""Atari backend on envpool: RAM descriptors, env factory and RePPO rollout collector.

Descriptors read the 128-byte console RAM (envpool ``info["ram"]``). Each function maps
``(ram, prev_ram)``, this and the previous step's RAM (uint8, ``(B, 128)``), to per-step
cumulants of shape ``(B, 2)``. The episodic descriptor is the mean of the cumulants over every
agent step of the full game (all lives), clipped to ``[-1, 1]``; frozen frames (death
animations, ready pauses, level transitions) emit 0 but still count in the mean.

Dense descriptors (Boxing, Asterix, Seaquest) are position-like and clipped per step. The
clearing-order descriptor (Ms. Pac-Man) fires only when a dot is eaten:
``d = scale * w * s`` with ``w = 1 - 2k/(n - 1)`` for the k-th of n dots in the level (so ``w``
sums to 0 over a full level and only the clearing order counts) and ``s`` the sign of the eaten dot's half. ``scale`` puts the range reached by trained
policies near +-1 after averaging over all game steps, so it is not clipped per step.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from src.envs.rollout import make_population_eval_metrics, make_rollout_batch


def boxing(ram: np.ndarray, prev_ram: np.ndarray) -> np.ndarray:
    """(horizontal distance to the opponent, ring height: mean y of both boxers)."""
    # Trained boxers stand 25-55 px apart; 28 px is punching range.
    distance_center, distance_scale = 38.0, 26.0
    height_center, height_scale = 45.0, 44.0
    ram = ram.astype(np.float32)
    distance = (np.abs(ram[:, 32] - ram[:, 33]) - distance_center) / distance_scale
    height = ((ram[:, 34] + ram[:, 35]) / 2.0 - height_center) / height_scale
    return np.clip(np.stack([distance, height], axis=1), -1.0, 1.0)


def asterix(ram: np.ndarray, prev_ram: np.ndarray) -> np.ndarray:
    """(player x, lane 0 top .. 7 bottom); 0 during death animations and wave interstitials."""
    x_center, x_scale = 76.0, 52.0
    lane_center, lane_scale = 3.5, 3.5
    live = (ram[:, 71] == 0) & (ram[:, 2] == 100)
    ram = ram.astype(np.float32)
    x = (ram[:, 41] - x_center) / x_scale
    lane = (ram[:, 39] - lane_center) / lane_scale
    return np.clip(np.stack([x, lane], axis=1), -1.0, 1.0) * live[:, None]


def seaquest(ram: np.ndarray, prev_ram: np.ndarray) -> np.ndarray:
    """(submarine x, depth); 0 while surfaced."""
    x_center, x_scale = 77.5, 56.5
    depth_center, depth_scale = 60.5, 47.5
    surface_y = 13
    submerged = ram[:, 97] > surface_y
    ram = ram.astype(np.float32)
    x = (ram[:, 70] - x_center) / x_scale
    depth = (ram[:, 97] - depth_center) / depth_scale
    return np.clip(np.stack([x, depth], axis=1), -1.0, 1.0) * submerged[:, None]


def mspacman(ram: np.ndarray, prev_ram: np.ndarray) -> np.ndarray:
    """Clearing order (+x: right half first, +y: bottom half first), one term per dot eaten."""
    # Per-axis scales from 10M-step policies: they rarely clear a level, so left/right order is
    # weak, while spawning in the bottom half makes bottom-first the default.
    scale = np.array([30.0, 15.0], dtype=np.float32)
    x_split, y_split = 92, 80
    dots_first_maze, dots_other_mazes = 154, 150
    # ram39 is a phase counter: 255 in play, 1-33 ready pause, 55-71 level clear, 73-82 death.
    phase = ram[:, 39]
    frozen = (phase < 34) | ((phase > 54) & (phase < 255))
    eaten_before = prev_ram[:, 119].astype(np.float32)
    eat = (ram[:, 119] > prev_ram[:, 119]) & ~frozen
    n_dots = np.where(ram[:, 0] == 0, dots_first_maze, dots_other_mazes)
    w = 1.0 - 2.0 * eaten_before / (n_dots - 1)
    side = np.stack(
        [np.where(ram[:, 10] >= x_split, 1.0, -1.0), np.where(ram[:, 16] >= y_split, 1.0, -1.0)],
        axis=1,
    )
    return (eat * w)[:, None] * side * scale


ATARI_DESCRIPTORS = {
    "atari_boxing": (["opponent_distance", "ring_height"], boxing),
    "atari_asterix": (["x", "lane"], asterix),
    "atari_seaquest": (["x", "depth"], seaquest),
    "atari_mspacman_order": (["right_first", "bottom_first"], mspacman),
}


def make_atari_env(env_cfg: DictConfig, *, num_envs: int, seed: int):
    """envpool Atari pool with CleanRL's ``ppo_atari_envpool.py`` settings.

    envpool defaults supply the rest: frame skip 4, 84x84 grayscale, 4-frame stack, up to 30
    no-op starts, and a 27k-step (108k-frame) episode cap.
    """
    import envpool

    return envpool.make(
        str(env_cfg.env_id),
        env_type="gymnasium",
        num_envs=num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=seed,
        repeat_action_probability=float(env_cfg.repeat_action_probability),
        num_threads=int(env_cfg.get("num_threads", 0)),
    )


class AtariRolloutCollector:
    """Collect RePPO rollouts from one envpool Atari pool; member n owns envs [n*E, (n+1)*E).

    The task channel is CleanRL's: sign-clipped reward with dones on every life loss. Fitness is
    the raw score of a full game (all lives), and descriptor channels use game-over dones, so
    losing a life never ends a descriptor episode. The step after a done is envpool's reset
    step (action ignored, reward 0); it counts toward the next life or game, as in CleanRL.
    """

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
        self.descriptor_fn = ATARI_DESCRIPTORS[env_cfg.descriptor_set][1]
        # Rollout CPU work is envpool's own threads; one torch thread keeps concurrent runs
        # from each spawning an intra-op pool the size of the machine.
        torch.set_num_threads(1)

        self.env = make_atari_env(env_cfg, num_envs=N * E, seed=seed)
        self.obs_shape = (int(np.prod(self.env.observation_space.shape)),)
        self.act_shape = (1,)
        self.obs_dim = self.obs_shape[0]
        self.act_dim = 1
        self.discrete = True
        self.agent_act_dim = int(self.env.action_space.n)
        self.clip_obs = None
        self.obs_rms = [None] * N

        self.batch = make_rollout_batch(
            T=T,
            N=N,
            E=E,
            K=K,
            obs_shape=self.obs_shape,
            act_shape=self.act_shape,
            device=device,
            obs_dtype=torch.uint8,
        )
        self.batch.descriptor_dones = torch.zeros(T, N, E, device=device)
        self.next_obs = torch.zeros(N, E, self.obs_dim, device=device)
        self.next_done = torch.zeros(N, E, device=device)
        self.next_descriptor_done = torch.zeros(N, E, device=device)
        self.num_policy_workers = 1
        self.num_env_workers = N * E
        self._started = False
        self._eval_envs = {}

    def start(self, agent_pop) -> None:
        del agent_pop
        if self._started:
            return
        obs, info = self.env.reset()
        self.next_obs.copy_(torch.from_numpy(obs).to(self.device).view(self.N, self.E, -1))
        self.next_done.zero_()
        self.next_descriptor_done.zero_()
        self._prev_ram = info["ram"]
        self._ep_returns = np.zeros(self.N * self.E)
        self._ep_lengths = np.zeros(self.N * self.E, dtype=np.int64)
        self._ep_desc_sums = np.zeros((self.N * self.E, self.K))
        self._started = True

    def collect(self, agent_pop) -> dict[str, list]:
        if not self._started:
            raise RuntimeError("AtariRolloutCollector.start() must be called before collect()")
        N, E, K = self.N, self.E, self.K

        ep_returns: list[list[float]] = [[] for _ in range(N)]
        ep_lengths: list[list[int]] = [[] for _ in range(N)]
        ep_behavior_descriptors: list[list[np.ndarray]] = [[] for _ in range(N)]

        for t in range(self.T):
            self.batch.obs[t].copy_(self.next_obs)
            self.batch.dones[t].copy_(self.next_done)
            self.batch.descriptor_dones[t].copy_(self.next_descriptor_done)

            with torch.no_grad():
                action, logprob, _, value = agent_pop.get_action_and_value(self.next_obs)
            self.batch.actions[t].copy_(action)
            self.batch.logprobs[t].copy_(logprob)
            self.batch.values[t].copy_(value)

            action_np = action.view(N * E).to(torch.int32).cpu().numpy()
            obs, reward, term, trunc, info = self.env.step(action_np)
            game_over = info["terminated"].astype(bool) | trunc
            desc = self.descriptor_fn(info["ram"], self._prev_ram)
            self._prev_ram = info["ram"]

            self.batch.rewards[t].copy_(torch.from_numpy(reward).view(N, E))
            self.batch.descriptors[t].copy_(torch.from_numpy(desc).view(N, E, K))
            self._ep_returns += info["reward"]
            self._ep_lengths += 1
            self._ep_desc_sums += desc

            for i in np.flatnonzero(game_over):
                n = i // E
                ep_returns[n].append(float(self._ep_returns[i]))
                ep_lengths[n].append(int(self._ep_lengths[i]))
                ep_behavior_descriptors[n].append(
                    np.clip(self._ep_desc_sums[i] / self._ep_lengths[i], -1.0, 1.0).astype(
                        np.float32
                    )
                )
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
                self._ep_desc_sums[i] = 0.0

            self.next_obs.copy_(torch.from_numpy(obs).to(self.device).view(N, E, -1))
            self.next_done.copy_(torch.from_numpy(term | trunc).view(N, E))
            self.next_descriptor_done.copy_(torch.from_numpy(game_over).view(N, E))

        return {
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "ep_behavior_descriptors": ep_behavior_descriptors,
        }

    @torch.no_grad()
    def evaluate(self, agent_pop, num_episodes_per_policy: int) -> dict:
        """One full game in each of ``num_episodes_per_policy`` fresh envs per member.

        Actions are sampled from the policy, as in CleanRL's reported returns.
        """
        n_per = num_episodes_per_policy
        if n_per not in self._eval_envs:
            self._eval_envs[n_per] = make_atari_env(
                self.env_cfg, num_envs=self.N * n_per, seed=self.seed + 1000
            )
        env = self._eval_envs[n_per]
        per_policy_returns: list[list[float]] = [[] for _ in range(self.N)]
        per_policy_descs: list[list[np.ndarray]] = [[] for _ in range(self.N)]

        obs, info = env.reset()
        prev_ram = info["ram"]
        returns = np.zeros(self.N * n_per)
        lengths = np.zeros(self.N * n_per, dtype=np.int64)
        desc_sums = np.zeros((self.N * n_per, self.K))
        finished = np.zeros(self.N * n_per, dtype=bool)
        while not finished.all():
            obs_t = torch.from_numpy(obs).to(self.device, torch.float32).view(self.N, n_per, -1)
            action = agent_pop.get_action_and_value(obs_t)[0]
            action_np = action.view(self.N * n_per).to(torch.int32).cpu().numpy()
            obs, _, _, trunc, info = env.step(action_np)
            desc = self.descriptor_fn(info["ram"], prev_ram)
            prev_ram = info["ram"]

            active = ~finished
            returns += info["reward"] * active
            lengths += active
            desc_sums += desc * active[:, None]
            game_over = active & (info["terminated"].astype(bool) | trunc)
            for i in np.flatnonzero(game_over):
                per_policy_returns[i // n_per].append(float(returns[i]))
                per_policy_descs[i // n_per].append(
                    np.clip(desc_sums[i] / lengths[i], -1.0, 1.0).astype(np.float32)
                )
            finished |= game_over

        return make_population_eval_metrics(
            per_policy_returns,
            per_policy_descs,
            num_episodes_per_policy,
            return_dtype=np.float32,
        )

    def close(self) -> None:
        self.env.close()
        self._started = False
        for env in self._eval_envs.values():
            env.close()
        self._eval_envs.clear()
