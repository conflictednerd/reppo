"""sep-CMA-MAE baseline via pyribs on Brax/MJX and on Gymnasium (LunarLander)."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from src.agents import make_agent
from src.envs import descriptor_names, probe_env_spaces
from src.envs.gym_backend import make_gym_env_thunk

DIMS = {
    "halfcheetah": [32, 32],
    "walker2d": [32, 32],
    "ant": [6, 6, 6, 6],
    "humanoid": [32, 32],
    "lunarlander": [32, 32],
}

THRESHOLD_MIN = {
    "halfcheetah": -1000.0,
    "walker2d": -1000.0,
    "ant": -500.0,
    "humanoid": 200.0,
    # Return floor of the pyribs LunarLander benchmark (its qd_score_offset).
    "lunarlander": -600.0,
}

# Gym-backend envs use the repo's canonical signed descriptor box; the Brax
# feet-contact descriptors are raw [0, 1] indicators.
GYM_ENVS = {"lunarlander"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train sep-CMA-MAE baseline (pyribs + Brax/MJX or Gymnasium)"
    )
    parser.add_argument("--env_name", type=str, choices=sorted(DIMS), default="ant")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_evals", type=int, default=1_000_000)
    parser.add_argument(
        "--total_env_steps",
        type=int,
        default=None,
        help="Stop once this many env steps are consumed (matched-budget runs)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=3,
        help="Fixed-seed episodes averaged per solution (gym backend)",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Rollout processes (gym backend)"
    )
    parser.add_argument("--num_emitters", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--sigma0", type=float, default=0.02)
    parser.add_argument("--archive_lr", type=float, default=0.001)
    parser.add_argument("--threshold_min", type=float, default=None)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def solution_dim(obs_dim: int, act_dim: int) -> int:
    return obs_dim * 128 + 128 + 128 * 128 + 128 + 128 * act_dim + act_dim


def unpack_solutions(sols, obs_dim: int, act_dim: int, device):
    x = torch.as_tensor(sols, dtype=torch.float32, device=device)
    p = x.shape[0]
    offset = 0

    n = obs_dim * 128
    w1 = x[:, offset : offset + n].reshape(p, obs_dim, 128)
    offset += n
    b1 = x[:, offset : offset + 128]
    offset += 128

    n = 128 * 128
    w2 = x[:, offset : offset + n].reshape(p, 128, 128)
    offset += n
    b2 = x[:, offset : offset + 128]
    offset += 128

    n = 128 * act_dim
    w3 = x[:, offset : offset + n].reshape(p, 128, act_dim)
    offset += n
    b3 = x[:, offset : offset + act_dim]

    return w1, b1, w2, b2, w3, b3


def batched_act(unpacked_weights_tuple, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        w1, b1, w2, b2, w3, b3 = unpacked_weights_tuple
        x = obs.to(dtype=torch.float32)
        h1 = torch.tanh(torch.einsum("pi,pio->po", x, w1) + b1)
        h2 = torch.tanh(torch.einsum("pi,pio->po", h1, w2) + b2)
        return torch.tanh(torch.einsum("pi,pio->po", h2, w3) + b3)


def _initial_solution(obs_dim: int, act_dim: int, seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    w1 = torch.empty(obs_dim, 128, dtype=torch.float32)
    w2 = torch.empty(128, 128, dtype=torch.float32)
    w3 = torch.empty(128, act_dim, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w1, generator=generator)
    torch.nn.init.xavier_uniform_(w2, generator=generator)
    torch.nn.init.xavier_uniform_(w3, generator=generator)
    b1 = torch.zeros(128, dtype=torch.float32)
    b2 = torch.zeros(128, dtype=torch.float32)
    b3 = torch.zeros(act_dim, dtype=torch.float32)
    return torch.cat([w1.flatten(), b1, w2.flatten(), b2, w3.flatten(), b3]).numpy()


def actor_shapes(agent) -> list[tuple[int, ...]]:
    return [tuple(p.shape) for p in agent.actor_logits.parameters()]


_WORKER = {}


def _worker_init(env_cfg_dict: dict, shapes, seed: int) -> None:
    _WORKER["env"] = make_gym_env_thunk(
        OmegaConf.create(env_cfg_dict), seed=seed, idx=os.getpid()
    )()
    _WORKER["shapes"] = shapes


def _worker_eval(task):
    """Roll out a whole solution inside one worker: one IPC round trip, not one per step."""
    sol, n_episodes, base_seed = task
    env, shapes = _WORKER["env"], _WORKER["shapes"]

    params, offset = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        params.append(sol[offset : offset + n].reshape(shape))
        offset += n
    w1, b1, w2, b2, w3, b3 = params

    returns, descs, lengths = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        total, steps = 0.0, 0
        while True:
            # nn.Linear stores (out, in), matching the exported actor_logits.
            h = np.tanh(obs @ w1.T + b1)
            h = np.tanh(h @ w2.T + b2)
            obs, reward, term, trunc, info = env.step(int((h @ w3.T + b3).argmax()))
            total += reward
            steps += 1
            if term or trunc:
                descs.append(np.asarray(info["descriptor_episode_behavior"], dtype=np.float64))
                break
        returns.append(total)
        lengths.append(steps)

    return returns, descs, lengths


def make_rollout_pool(env_cfg, shapes, seed: int, n_workers: int | None = None):
    """Fork a pool of persistent single-env rollout workers."""
    return mp.get_context("fork").Pool(
        int(n_workers or os.cpu_count()),
        initializer=_worker_init,
        initargs=(OmegaConf.to_container(env_cfg, resolve=True), shapes, seed),
    )


def rollout_solutions(pool, sols, n_episodes: int, base_seed: int):
    sols = np.asarray(sols, dtype=np.float32)
    tasks = [(sols[i], n_episodes, base_seed) for i in range(len(sols))]
    return pool.map(_worker_eval, tasks, chunksize=1)


def _evaluate_gym(pool, sols, n_episodes: int, base_seed: int):
    """Mean return and mean episode descriptor over ``n_episodes`` fixed seeds.

    Every solution sees the same initial conditions, so CMA-MAE cannot rank on seed luck.
    """
    results = rollout_solutions(pool, sols, n_episodes, base_seed)
    returns = np.array([np.mean(r[0]) for r in results], dtype=np.float64)
    measures = np.array([np.mean(r[1], axis=0) for r in results], dtype=np.float64)
    steps = int(sum(sum(r[2]) for r in results))
    return returns, measures, steps


def _export_population(outdir: Path, archive, *, args, env_cfg, obs_dim, act_dim, k):
    """Save the whole archive as a RePPO-format checkpoint."""
    data = archive.data()
    solutions = np.asarray(data["solution"], dtype=np.float32)

    state_dicts = []
    for idx in range(len(solutions)):
        agent = make_agent(
            obs_dim, act_dim, descriptor_dim=k, hidden_dim=args.hidden_dim, discrete=True
        )
        vec = torch.as_tensor(solutions[idx], dtype=torch.float32)
        offset = 0
        with torch.no_grad():
            for param in agent.actor_logits.parameters():
                n = param.numel()
                param.copy_(vec[offset : offset + n].view_as(param))
                offset += n
        state_dicts.append(agent.state_dict())

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent_states": state_dicts,
            # The ES searches raw-observation policies: no running normalizer.
            "obs_rms": [None] * len(state_dicts),
            "env_id": env_cfg.env_id,
            "descriptor_set": env_cfg.descriptor_set,
            "env_config": OmegaConf.to_container(env_cfg, resolve=True),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "discrete": True,
            "K": k,
            "N": len(state_dicts),
            "hidden_dim": args.hidden_dim,
            "agent_spec": None,
            "clip_obs": float(env_cfg.clip_obs),
        },
        ckpt_dir / "final.pt",
    )
    with open(outdir / "config.yaml", "w") as f:
        yaml.safe_dump({"seed": int(args.seed), "algorithm": vars(args)}, f)

    return {"exported": int(len(state_dicts))}


def _archive_metrics(archive: GridArchive) -> dict:
    data = archive.data()
    objectives = data["objective"]
    archive_size = int(len(objectives))
    return {
        "coverage": float(np.asarray(archive.stats.coverage)),
        "qd_score": float(np.asarray(objectives).sum()),
        "max_fitness": float(objectives.max()) if archive_size else float("-inf"),
        "archive_size": archive_size,
    }


def _save_archive(path: Path, archive: GridArchive) -> None:
    data = archive.data()
    np.savez_compressed(
        path,
        solution=data["solution"],
        objective=data["objective"],
        measures=data["measures"],
    )


def _append_summary(path: Path, row: dict) -> None:
    fieldnames = [
        "iteration",
        "evals",
        "env_steps",
        "coverage",
        "qd_score",
        "max_fitness",
        "archive_size",
    ]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _evaluate(env, sols, obs_dim: int, act_dim: int, k: int, device: torch.device):
    weights = unpack_solutions(sols, obs_dim, act_dim, device)
    p = len(sols)
    obs = env.reset()
    active = torch.ones(p, dtype=torch.bool, device=device)
    returns = torch.zeros(p, dtype=torch.float32, device=device)
    desc_sums = torch.zeros(p, k, dtype=torch.float32, device=device)
    lengths = torch.zeros(p, dtype=torch.float32, device=device)
    steps_this_iter = 0

    while bool(active.any()):
        actions = batched_act(weights, obs)
        obs, reward, done, info = env.step(actions)
        active_f = active.to(torch.float32)
        returns += reward.to(torch.float32) * active_f
        desc_sums += info["behavior_descriptors"].to(torch.float32) * active_f[:, None]
        lengths += active_f
        steps_this_iter += int(active.sum().item())
        active = active & (~done.bool())

    measures = desc_sums / lengths.clamp(min=1.0)[:, None]
    return (
        returns.detach().cpu().numpy().astype(np.float64),
        measures.detach().cpu().numpy().astype(np.float64),
        steps_this_iter,
    )


def train(args: argparse.Namespace) -> None:
    env_name = args.env_name.lower()
    seed = int(args.seed)
    dims = DIMS[env_name]
    tmin = float(args.threshold_min) if args.threshold_min is not None else THRESHOLD_MIN[env_name]
    device = torch.device(args.device)
    outdir = Path(args.logdir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is non-empty: {outdir}")

    gym_backend = env_name in GYM_ENVS
    batch_total = int(args.num_emitters) * int(args.batch_size)

    if gym_backend:
        # Environment config used verbatim, so the interaction the ES sees is
        # identical to what RePPO and Ind-PPO see on this task.
        cfg = OmegaConf.load(f"conf/env/{env_name}.yaml")
        obs_space, act_space = probe_env_spaces(cfg, seed=seed)
        obs_dim = int(obs_space.shape[0])
        act_dim = int(act_space.n)
        k = len(descriptor_names(cfg.descriptor_set))
        measure_range = (-1.0, 1.0)
        torch.manual_seed(seed)
        proto = make_agent(
            obs_dim, act_dim, descriptor_dim=k, hidden_dim=args.hidden_dim, discrete=True
        )
        x0 = torch.cat([p.detach().reshape(-1) for p in proto.actor_logits.parameters()]).numpy()
        sol_dim = int(x0.size)
        env = make_rollout_pool(cfg, actor_shapes(proto), seed, args.num_workers)
    else:
        from src.envs.brax_backend import make_brax_vec_env

        assert args.hidden_dim == 128, "--hidden_dim only applies to the gym backend"
        base_cfg = OmegaConf.load(f"conf/env/{env_name}.yaml")
        cfg = OmegaConf.merge(
            base_cfg,
            {
                "clip_reward": float("inf"),
                "brax": {
                    "clip_obs_rew": False,
                    "clip_rew": False,
                    "episode_length": int(args.episode_length),
                },
            },
        )
        env = make_brax_vec_env(cfg, batch_size=batch_total, seed=seed, device=device)
        obs_dim = int(env.single_observation_space.shape[0])
        act_dim = int(env.single_action_space.shape[0])
        k = len(dims)
        measure_range = (0.0, 1.0)
        x0 = _initial_solution(obs_dim, act_dim, seed)
        sol_dim = solution_dim(obs_dim, act_dim)

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    run_cfg = vars(args).copy()
    run_cfg.update(
        {
            "env_name": env_name,
            "device": str(device),
            "threshold_min": tmin,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "solution_dim": sol_dim,
            "archive_dims": dims,
        }
    )
    with open(outdir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    archive = GridArchive(
        solution_dim=sol_dim,
        dims=dims,
        ranges=[measure_range] * k,
        learning_rate=args.archive_lr,
        threshold_min=tmin,
        seed=seed,
    )
    result_archive = GridArchive(
        solution_dim=sol_dim,
        dims=dims,
        ranges=[measure_range] * k,
        seed=seed,
    )
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=x0,
            sigma0=args.sigma0,
            ranker="imp",
            es="sep_cma_es",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=args.batch_size,
            seed=seed + i,
        )
        for i in range(int(args.num_emitters))
    ]
    scheduler = Scheduler(archive, emitters, result_archive=result_archive)

    total_iters = int(args.total_evals) // batch_total
    max_env_steps = int(args.total_env_steps) if args.total_env_steps else None
    evals = 0
    env_steps = 0
    itr = 0
    # Soft cap: the budget is checked after a full batch, so the run overshoots slightly.
    budget = f"env_step_target={max_env_steps}" if max_env_steps else f"iters={total_iters}"
    print(
        f"sep-CMA-MAE {env_name} seed={seed} dims={dims} sol_dim={sol_dim} "
        f"emitters={args.num_emitters} batch={args.batch_size} {budget}",
        flush=True,
    )

    try:
        for itr in range(1, total_iters + 1):
            sols = scheduler.ask()
            if gym_backend:
                # Disjoint across run seeds, and clear of the 1000+ eval seeds.
                objectives, measures, steps_this_iter = _evaluate_gym(
                    env, sols, int(args.eval_episodes), seed * int(args.eval_episodes)
                )
            else:
                objectives, measures, steps_this_iter = _evaluate(
                    env, sols, obs_dim, act_dim, k, device
                )
            scheduler.tell(objectives, measures)
            evals += len(sols)
            env_steps += steps_this_iter

            if itr % 10 == 0:
                metrics = _archive_metrics(result_archive)
                _append_summary(
                    summary_path,
                    {
                        "iteration": itr,
                        "evals": evals,
                        "env_steps": env_steps,
                        **metrics,
                    },
                )
            if itr % 50 == 0:
                metrics = _archive_metrics(result_archive)
                print(
                    f"iter {itr} | evals={evals} env_steps={env_steps} "
                    f"cov={metrics['coverage']:.3f} qd={metrics['qd_score']:.1f} "
                    f"max={metrics['max_fitness']:.2f}",
                    flush=True,
                )
            if itr % int(args.checkpoint_every) == 0 or itr == total_iters:
                _save_archive(outdir / f"archive_{itr:06d}.npz", result_archive)
            if max_env_steps is not None and env_steps >= max_env_steps:
                break
    finally:
        env.close()

    _save_archive(outdir / "archive_final.npz", result_archive)
    metrics = _archive_metrics(result_archive)
    final_summary = {
        "total_evals": evals,
        "env_steps": env_steps,
        **metrics,
        "iterations": itr,
    }
    if gym_backend:
        final_summary["export"] = _export_population(
            outdir,
            result_archive,
            args=args,
            env_cfg=cfg,
            obs_dim=obs_dim,
            act_dim=act_dim,
            k=k,
        )
    with open(outdir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
