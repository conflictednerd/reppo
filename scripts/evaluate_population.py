"""Re-evaluate a trained RePPO / Independent PPO population.

Loads ``checkpoints/final.pt``, runs many deterministic episodes per policy
(sampled actions and full games on Atari) in parallel eval envs, and writes
per-policy return + descriptor stats to ``evaluation_results.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.agents import FunctionalPolicyPopulation, make_agent
from src.envs import descriptor_names, make_gym_vec_env, same_step_final_infos
from src.utils.normalize import RunningMeanStd
from src.utils.population import vendi_score


def _load_checkpoint(exp_dir: Path, device: torch.device):
    """Return (agents, obs_rms_list, metadata_dict) from a checkpoint."""
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(exp_dir / "checkpoints" / "final.pt", map_location=device, weights_only=False)
    env_config = ckpt["env_config"]
    env_id = env_config["env_id"]
    ds = env_config["descriptor_set"]
    K = ckpt["K"]
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    discrete = bool(ckpt["discrete"])
    hidden_dim = ckpt["hidden_dim"]
    agent_spec = ckpt["agent_spec"]
    clip_obs = ckpt["clip_obs"]
    state_dicts = ckpt["agent_states"]
    rms_states = ckpt["obs_rms"]
    N = len(state_dicts)

    agents = []
    obs_rms_list = []
    for p in range(N):
        agent = make_agent(
            obs_dim,
            act_dim,
            descriptor_dim=K,
            hidden_dim=hidden_dim,
            discrete=discrete,
            spec=agent_spec,
        ).to(device)
        agent.load_state_dict(state_dicts[p])
        agent.eval()
        agents.append(agent)

        rms = None
        if rms_states[p] is not None:
            rms = RunningMeanStd(shape=(obs_dim,), device=device)
            rms.load_state_dict(rms_states[p])
        obs_rms_list.append(rms)

    meta = dict(
        env_id=env_id,
        descriptor_set=ds,
        env_config=env_config,
        seed=cfg["seed"],
        K=K,
        obs_dim=obs_dim,
        act_dim=act_dim,
        discrete=discrete,
        clip_obs=clip_obs,
        N=N,
    )
    return agents, obs_rms_list, meta


def _summarize_population_results(
    *,
    per_policy_returns: list[list[float]],
    per_policy_descs: list[list[np.ndarray]],
    per_policy_lengths: list[list[int]],
    desc_names: list[str],
    num_episodes: int,
    N: int,
    K: int,
) -> dict:
    results = {
        "num_rollouts_per_policy": num_episodes,
        "descriptor_names": desc_names,
        "N": N,
        "K": K,
        "policies": [],
    }
    all_returns = []
    all_descs = []

    def _std(values: np.ndarray) -> float:
        return float(values.std(ddof=1)) if len(values) > 1 else 0.0

    def _sem(values: np.ndarray) -> float:
        return _std(values) / float(np.sqrt(len(values))) if len(values) > 0 else 0.0

    for n in range(N):
        rets = np.array(per_policy_returns[n][:num_episodes], dtype=np.float64)
        descs = np.stack(per_policy_descs[n][:num_episodes])
        lens = np.array(per_policy_lengths[n][:num_episodes])
        all_returns.append(rets)
        all_descs.append(descs.mean(axis=0))

        rollouts = []
        for idx in range(num_episodes):
            rollout = {
                "rollout_idx": idx,
                "return": float(rets[idx]),
                "length": int(lens[idx]),
                "behavior_descriptor": descs[idx].astype(float).tolist(),
            }
            rollouts.append(rollout)

        policy_result = {
            "policy_idx": n,
            "return_mean": float(rets.mean()),
            "return_std": _std(rets),
            "return_sem": _sem(rets),
            "return_min": float(rets.min()),
            "return_max": float(rets.max()),
            "length_mean": float(lens.mean()),
            "behavior_descriptor_mean": descs.mean(axis=0).astype(float).tolist(),
            "behavior_descriptor_std": descs.std(axis=0, ddof=1).astype(float).tolist()
            if len(descs) > 1
            else np.zeros(K, dtype=float).tolist(),
            "rollouts": rollouts,
        }
        for k_idx, dname in enumerate(desc_names):
            policy_result[f"desc_{dname}_mean"] = float(descs[:, k_idx].mean())
            policy_result[f"desc_{dname}_std"] = (
                float(descs[:, k_idx].std(ddof=1)) if len(descs) > 1 else 0.0
            )
        results["policies"].append(policy_result)

    all_rets_flat = np.concatenate(all_returns)
    pop_descs = np.stack(all_descs)
    policy_return_means = np.array([r.mean() for r in all_returns], dtype=np.float64)
    best_idx = int(np.argmax(policy_return_means))
    best_returns = all_returns[best_idx]

    if N > 1:
        diff = pop_descs[:, None, :] - pop_descs[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        tri = np.triu_indices(N, k=1)
        avg_pairwise_distance = float(dists[tri].mean())
    else:
        avg_pairwise_distance = 0.0

    results["population"] = {
        "best_policy_idx": best_idx,
        "best_policy_return_mean": float(best_returns.mean()),
        "best_policy_return_std": _std(best_returns),
        "best_policy_return_sem": _sem(best_returns),
        "return_mean": float(all_rets_flat.mean()),
        "return_std": _std(all_rets_flat),
        "return_mean_all_rollouts": float(all_rets_flat.mean()),
        "return_std_all_rollouts": _std(all_rets_flat),
        "return_sem_all_rollouts": _sem(all_rets_flat),
        "mean_policy_return_mean": float(policy_return_means.mean()),
        "std_policy_return_mean": _std(policy_return_means),
        "per_policy_return_means": [float(r.mean()) for r in all_returns],
        "descriptor_means": pop_descs.astype(float).tolist(),
        "avg_pairwise_behavior_distance": avg_pairwise_distance,
        "vendi_score": vendi_score(torch.as_tensor(pop_descs, dtype=torch.float32)),
    }
    if N > 1:
        results["population"]["descriptor_spread"] = float(pop_descs.std(axis=0).mean())

    return results


@torch.no_grad()
def _evaluate_brax_population(
    agents: list[nn.Module],
    obs_rms_list: list[RunningMeanStd | None],
    env_cfg,
    meta: dict,
    num_episodes: int,
    envs_per_policy: int,
    device: torch.device,
    seed: int,
) -> dict:
    N = meta["N"]
    K = meta["K"]
    obs_dim = meta["obs_dim"]
    act_dim = meta["act_dim"]
    clip_obs = meta["clip_obs"]
    E = envs_per_policy
    desc_names = descriptor_names(meta["descriptor_set"])

    from omegaconf import OmegaConf

    from src.envs.brax_backend import make_brax_vec_env

    # Fitness is the raw (unclipped) reward sum, as in the baseline archive
    # re-evaluation (src/evaluation.py BraxEvalRunner).
    env_cfg = OmegaConf.merge(
        env_cfg,
        OmegaConf.create(
            {
                "clip_reward": float("inf"),
                "brax": {"clip_obs_rew": False, "clip_rew": False},
            }
        ),
    )

    agent_pop = FunctionalPolicyPopulation(agents).to(device)
    agent_pop.eval()

    def normalize(raw_obs: torch.Tensor) -> torch.Tensor:
        obs = raw_obs.view(N, E, obs_dim)
        normalized = torch.empty_like(obs, dtype=torch.float32)
        for n, rms in enumerate(obs_rms_list):
            if rms is None:
                normalized[n] = obs[n].to(torch.float32)
            else:
                normalized[n] = rms.normalize(obs[n], clip=clip_obs).to(torch.float32)
        return normalized

    env = make_brax_vec_env(env_cfg, batch_size=N * E, seed=seed, device=device)
    try:
        raw_obs = env.reset(seed=seed).view(N, E, obs_dim)
        running_returns = torch.zeros(N, E, device=device)
        ep_lengths = torch.zeros(N, E, device=device, dtype=torch.int64)
        desc_sums = torch.zeros(N, E, K, device=device)

        per_policy_returns: list[list[float]] = [[] for _ in range(N)]
        per_policy_descs: list[list[np.ndarray]] = [[] for _ in range(N)]
        per_policy_lengths: list[list[int]] = [[] for _ in range(N)]
        target = num_episodes * N

        while sum(len(r) for r in per_policy_returns) < target:
            actions = agent_pop.act_deterministic(normalize(raw_obs)).reshape(N * E, act_dim)
            raw_obs_flat, reward_flat, done_flat, infos = env.step(actions)
            raw_obs = raw_obs_flat.view(N, E, obs_dim)
            reward = reward_flat.view(N, E)
            done = done_flat.view(N, E).bool()
            unit_desc = infos["behavior_descriptors"].view(N, E, K)
            desc_step = 2.0 * unit_desc - 1.0

            running_returns.add_(reward)
            ep_lengths.add_(1)
            desc_sums.add_(desc_step)

            for n, e in torch.nonzero(done, as_tuple=False).detach().cpu().tolist():
                if len(per_policy_returns[n]) < num_episodes:
                    length = int(ep_lengths[n, e].item())
                    per_policy_returns[n].append(float(running_returns[n, e].item()))
                    per_policy_descs[n].append(
                        (desc_sums[n, e] / length).detach().cpu().numpy().astype(np.float32)
                    )
                    per_policy_lengths[n].append(length)
                running_returns[n, e] = 0.0
                ep_lengths[n, e] = 0
                desc_sums[n, e].zero_()
    finally:
        env.close()

    return _summarize_population_results(
        per_policy_returns=per_policy_returns,
        per_policy_descs=per_policy_descs,
        per_policy_lengths=per_policy_lengths,
        desc_names=desc_names,
        num_episodes=num_episodes,
        N=N,
        K=K,
    )


@torch.no_grad()
def _evaluate_atari_population(
    agents: list[nn.Module],
    env_cfg,
    meta: dict,
    num_episodes: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Play one full game in each of num_episodes envs per policy on one envpool Atari pool.

    Every env counts its first game only, so long games are not under-sampled. Actions are
    sampled from the policy (CleanRL convention), not the deterministic mean used elsewhere.
    """
    from src.envs.atari import ATARI_DESCRIPTORS, make_atari_env

    N = meta["N"]
    K = meta["K"]
    E = num_episodes
    desc_names = descriptor_names(meta["descriptor_set"])
    descriptor_fn = ATARI_DESCRIPTORS[meta["descriptor_set"]][1]

    torch.manual_seed(seed)
    agent_pop = FunctionalPolicyPopulation(agents).to(device)
    agent_pop.eval()

    env = make_atari_env(env_cfg, num_envs=N * E, seed=seed)
    try:
        obs, info = env.reset()
        prev_ram = info["ram"]
        returns = np.zeros(N * E)
        lengths = np.zeros(N * E, dtype=np.int64)
        desc_sums = np.zeros((N * E, K))
        finished = np.zeros(N * E, dtype=bool)
        while not finished.all():
            obs_t = torch.from_numpy(obs).to(device, torch.float32).view(N, E, -1)
            action = agent_pop.get_action_and_value(obs_t)[0]
            action_np = action.view(N * E).to(torch.int32).cpu().numpy()
            obs, _, _, trunc, info = env.step(action_np)
            desc = descriptor_fn(info["ram"], prev_ram)
            prev_ram = info["ram"]

            active = ~finished
            returns += info["reward"] * active
            lengths += active
            desc_sums += desc * active[:, None]
            finished |= active & (info["terminated"].astype(bool) | trunc)
    finally:
        env.close()

    per_policy_returns = returns.reshape(N, E).tolist()
    per_policy_lengths = lengths.reshape(N, E).tolist()
    descs = np.clip(desc_sums / lengths[:, None], -1.0, 1.0).astype(np.float32).reshape(N, E, K)
    per_policy_descs = [list(d) for d in descs]
    return _summarize_population_results(
        per_policy_returns=per_policy_returns,
        per_policy_descs=per_policy_descs,
        per_policy_lengths=per_policy_lengths,
        desc_names=desc_names,
        num_episodes=num_episodes,
        N=N,
        K=K,
    )


@torch.no_grad()
def evaluate_population(
    agents: list[nn.Module],
    obs_rms_list: list[RunningMeanStd | None],
    meta: dict,
    num_episodes: int,
    envs_per_policy: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Run num_episodes deterministic rollouts per policy and return stats."""
    from omegaconf import OmegaConf

    N = meta["N"]
    K = meta["K"]
    obs_dim = meta["obs_dim"]
    act_dim = meta["act_dim"]
    discrete = meta["discrete"]
    clip_obs = meta["clip_obs"]
    obs_shape = (obs_dim,)
    act_shape = (1,) if discrete else (act_dim,)
    desc_names = descriptor_names(meta["descriptor_set"])

    env_cfg = OmegaConf.merge(
        OmegaConf.create(meta["env_config"]),
        OmegaConf.create({"normalize_obs": False}),
    )
    if str(env_cfg.backend).lower() == "brax":
        return _evaluate_brax_population(
            agents,
            obs_rms_list,
            env_cfg,
            meta,
            num_episodes,
            envs_per_policy,
            device,
            seed,
        )
    if str(env_cfg.backend).lower() == "atari":
        return _evaluate_atari_population(agents, env_cfg, meta, num_episodes, device, seed)

    E = envs_per_policy
    n_total = N * E
    eval_envs = make_gym_vec_env(env_cfg, num_envs=n_total, seed=seed)

    obs, _ = eval_envs.reset(seed=seed)

    per_policy_returns: list[list[float]] = [[] for _ in range(N)]
    per_policy_descs: list[list[np.ndarray]] = [[] for _ in range(N)]
    per_policy_lengths: list[list[int]] = [[] for _ in range(N)]
    running_returns = np.zeros(n_total, dtype=np.float64)
    running_lengths = np.zeros(n_total, dtype=np.int64)
    target = num_episodes * N

    while sum(len(r) for r in per_policy_returns) < target:
        obs_norm = np.empty_like(obs, dtype=np.float32)
        obs_reshaped = obs.reshape(N, E, -1)
        for n in range(N):
            if obs_rms_list[n] is not None:
                obs_norm[n * E : (n + 1) * E] = (
                    obs_rms_list[n].normalize(obs_reshaped[n], clip=clip_obs).astype(np.float32)
                )
            else:
                obs_norm[n * E : (n + 1) * E] = obs_reshaped[n].astype(np.float32)

        obs_t = torch.as_tensor(obs_norm, device=device, dtype=torch.float32).view(N, E, *obs_shape)
        actions = np.zeros((n_total,) + act_shape, dtype=np.float32)
        for n in range(N):
            actions[n * E : (n + 1) * E] = agents[n].act_deterministic(obs_t[n]).cpu().numpy()
        if discrete:
            actions = actions.reshape(n_total).astype(np.int64)

        obs, reward, term, trunc, infos = eval_envs.step(actions)
        done = np.logical_or(term, trunc)
        final_infos = same_step_final_infos(infos, done)
        desc_step = np.asarray(infos["descriptor_step"], dtype=np.float32).reshape(n_total, K)
        if final_infos is not None:
            final_desc_step = np.asarray(final_infos["descriptor_step"], dtype=np.float32).reshape(
                n_total, K
            )
            desc_step[done] = final_desc_step[done]
        if np.any(~np.isfinite(desc_step)) or np.any(desc_step < -1.0) or np.any(desc_step > 1.0):
            raise ValueError(f"descriptor_step must be finite and in [-1, 1], got {desc_step}")

        running_returns += np.asarray(reward, dtype=np.float64)
        running_lengths += 1

        if final_infos is not None:
            mask = np.asarray(final_infos["_episode"], dtype=bool)
            if not np.array_equal(mask, done):
                raise ValueError(
                    f"SAME_STEP episode mask must match done mask, got episode={mask}, done={done}"
                )
            episode_descs = np.asarray(
                final_infos["descriptor_episode_behavior"], dtype=np.float32
            ).reshape(n_total, K)
            done_descs = episode_descs[done]
            if (
                np.any(~np.isfinite(done_descs))
                or np.any(done_descs < -1.0)
                or np.any(done_descs > 1.0)
            ):
                raise ValueError(
                    f"descriptor_episode_behavior must be finite and in [-1, 1], got {done_descs}"
                )
            for env_idx in np.where(done)[0]:
                n = env_idx // E
                if len(per_policy_returns[n]) < num_episodes:
                    per_policy_returns[n].append(float(running_returns[env_idx]))
                    per_policy_descs[n].append(episode_descs[env_idx])
                    per_policy_lengths[n].append(int(running_lengths[env_idx]))
                running_returns[env_idx] = 0.0
                running_lengths[env_idx] = 0

    eval_envs.close()

    return _summarize_population_results(
        per_policy_returns=per_policy_returns,
        per_policy_descs=per_policy_descs,
        per_policy_lengths=per_policy_lengths,
        desc_names=desc_names,
        num_episodes=num_episodes,
        N=N,
        K=K,
    )


def main():
    ap = argparse.ArgumentParser(description="Re-evaluate a trained population")
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--num-episodes", type=int, default=256)
    ap.add_argument("--envs-per-policy", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="Default: checkpoint seed + 1000")
    ap.add_argument("--output-file", type=Path, default=None, help="Default: <exp-dir>/evaluation_results.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    agents, obs_rms_list, meta = _load_checkpoint(args.exp_dir, device)
    eval_seed = args.seed if args.seed is not None else int(meta["seed"]) + 1000
    print(f"Evaluating {args.exp_dir}: {meta['N']} policies x {args.num_episodes} episodes")
    results = evaluate_population(
        agents,
        obs_rms_list,
        meta,
        num_episodes=args.num_episodes,
        envs_per_policy=args.envs_per_policy,
        device=device,
        seed=eval_seed,
    )
    results["exp_dir"] = str(args.exp_dir)
    results["env_id"] = meta["env_id"]
    results["seed"] = eval_seed
    results["envs_per_policy"] = args.envs_per_policy

    out_path = args.output_file or args.exp_dir / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")

    pop = results["population"]
    print(
        f"best policy {pop['best_policy_idx']}: return={pop['best_policy_return_mean']:.1f}"
        f"+/-{pop['best_policy_return_sem']:.1f} SEM"
    )
    print(
        f"population: return_mean={pop['mean_policy_return_mean']:.1f}  "
        f"pairwise_bd={pop['avg_pairwise_behavior_distance']:.4f}  "
        f"vendi={pop['vendi_score']:.3f}"
    )


if __name__ == "__main__":
    main()
