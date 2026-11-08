#!/usr/bin/env python3
"""Re-evaluate every solution in a trained archive and write evaluation_results.json.

Baseline counterpart of scripts/evaluate_population.py: loads all policies from
a run's archive (PGA-ME / DCRL-ME / PPGA / sep-CMA-MAE, or a RePPO checkpoint),
runs --n-episodes deterministic episodes per policy on the batched Brax eval
env -- or, for gym-backend runs (see GYM_ENVS), in a pool of rollout workers --
and writes per-rollout returns + behavior descriptors in the same JSON structure
as evaluate_population.py.

All episodes of a policy run in parallel (--episodes-per-round, default 256),
with --policy-batch policies per round. If you hit GPU OOM, lower --policy-batch.

Usage:
  uv run python scripts/evaluate_archive.py <run_dir> --method pga_me
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_population import _summarize_population_results

from src.baselines.sep_cma_mae.train import GYM_ENVS
from src.envs import descriptor_names
from src.evaluation import (
    REPO_ROOT,
    BraxEvalRunner,
    load_dcrl_me_actors,
    load_pga_me_actors,
    load_ppga_actors,
    load_reppo_actors,
    load_sep_cma_mae_actors,
)

LOADERS = {
    "reppo": load_reppo_actors,
    "ppga": load_ppga_actors,
    "pga_me": load_pga_me_actors,
    "dcrl_me": load_dcrl_me_actors,
    "sep_cma_mae": load_sep_cma_mae_actors,
}


def _write_results(results, args, *, env_id, eval_seed, envs_per_policy, total_env_steps=None):
    results["exp_dir"] = str(args.run_dir)
    results["env_id"] = env_id
    results["method"] = args.method
    results["seed"] = eval_seed
    results["envs_per_policy"] = envs_per_policy
    if total_env_steps is not None:
        results["total_env_steps"] = int(total_env_steps)

    out_path = args.output_file or (args.run_dir / "evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, separators=(",", ":"))

    pop = results["population"]
    print(
        f"best={pop['best_policy_return_mean']:.1f}  mean={pop['mean_policy_return_mean']:.1f}  "
        f"vendi={pop['vendi_score']:.2f}"
    )
    print(f"Saved -> {out_path}")


def evaluate_gym_archive(args, run_cfg: dict) -> None:
    """Gym-backend archives: re-evaluate every elite in the rollout worker pool.

    Same protocol as the Brax branch: all elites, --n-episodes deterministic episodes each.
    """
    from src.agents import make_agent
    from src.baselines.sep_cma_mae.train import actor_shapes, make_rollout_pool, rollout_solutions

    env_name = run_cfg["env_name"]
    env_cfg = OmegaConf.load(REPO_ROOT / f"conf/env/{env_name}.yaml")
    desc_names = descriptor_names(env_cfg.descriptor_set)
    K = len(desc_names)
    eval_seed = args.eval_seed if args.eval_seed is not None else int(run_cfg["seed"]) + 1000

    archive_path = args.run_dir / "archive_final.npz"
    solutions = np.load(archive_path)["solution"].astype(np.float32)
    if args.max_policies is not None:
        solutions = solutions[: args.max_policies]
    N = len(solutions)

    proto = make_agent(
        int(run_cfg["obs_dim"]),
        int(run_cfg["act_dim"]),
        descriptor_dim=K,
        hidden_dim=int(run_cfg["hidden_dim"]),
        discrete=True,
    )
    pool = make_rollout_pool(env_cfg, actor_shapes(proto), eval_seed, args.num_workers)
    print(f"sep_cma_mae {env_name}: {N} policies x {args.n_episodes} eps", flush=True)

    t0 = time.perf_counter()
    try:
        results = rollout_solutions(pool, solutions, args.n_episodes, eval_seed)
    finally:
        pool.close()
    print(f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    summary = _summarize_population_results(
        per_policy_returns=[[float(x) for x in r[0]] for r in results],
        per_policy_descs=[list(r[1]) for r in results],
        per_policy_lengths=[[int(x) for x in r[2]] for r in results],
        desc_names=desc_names,
        num_episodes=args.n_episodes,
        N=N,
        K=K,
    )
    _write_results(summary, args, env_id=env_cfg.env_id, eval_seed=eval_seed, envs_per_policy=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate every solution in an archive")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--method", required=True, choices=sorted(LOADERS))
    parser.add_argument("--n-episodes", type=int, default=256)
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=256,
        help="Episodes per policy run in parallel (default: all of --n-episodes)",
    )
    parser.add_argument(
        "--policy-batch", type=int, default=32, help="Policies evaluated in parallel"
    )
    parser.add_argument(
        "--eval-seed", type=int, default=None, help="Defaults to the run's seed + 1000"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument(
        "--output-file", type=Path, default=None, help="Default: <run_dir>/evaluation_results.json"
    )
    parser.add_argument(
        "--max-policies", type=int, default=None, help="Debug: evaluate only the first M policies"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Rollout processes (gym-backend runs)"
    )
    args = parser.parse_args()

    # Only sep-CMA-MAE writes config.json; PPGA and PGA-ME write cfg.json.
    if args.method == "sep_cma_mae":
        run_cfg = json.loads((args.run_dir / "config.json").read_text())
        if run_cfg["env_name"] in GYM_ENVS:
            evaluate_gym_archive(args, run_cfg)
            return

    device = torch.device(args.device)
    actors, meta = LOADERS[args.method](args.run_dir, device)
    if args.max_policies is not None:
        actors = actors[: args.max_policies]
    env_name = meta["env_name"]
    eval_seed = args.eval_seed if args.eval_seed is not None else int(meta["seed"]) + 1000

    env_cfg = OmegaConf.load(REPO_ROOT / f"conf/env/{env_name}.yaml")
    desc_names = descriptor_names(env_cfg.descriptor_set)

    N = len(actors)
    chunk = min(args.policy_batch, N)
    runner = BraxEvalRunner(
        env_name,
        actors_per_chunk=chunk,
        episodes_per_round=min(args.episodes_per_round, args.n_episodes),
        seed=eval_seed,
        device=device,
        episode_length=args.episode_length,
    )
    K = runner.K
    print(
        f"{args.method} {env_name}: {N} policies x {args.n_episodes} eps, "
        f"{chunk * runner.E} parallel envs",
        flush=True,
    )

    per_policy_returns: list[list[float]] = []
    per_policy_descs: list[list[np.ndarray]] = []
    per_policy_lengths: list[list[int]] = []
    t0 = time.perf_counter()
    try:
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            rets, descs, lens = runner.evaluate(actors[start:end], args.n_episodes)
            # Descriptors in [-1, 1], as in evaluate_population.py (the runner
            # returns unit [0, 1] contact fractions).
            descs = 2.0 * descs - 1.0
            for i in range(end - start):
                per_policy_returns.append([float(x) for x in rets[i]])
                per_policy_descs.append(list(descs[i]))
                per_policy_lengths.append([int(x) for x in lens[i]])
            elapsed = time.perf_counter() - t0
            eta = elapsed / end * (N - end)
            print(
                f"policies {end}/{N} | {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                flush=True,
            )
    finally:
        runner.close()

    results = _summarize_population_results(
        per_policy_returns=per_policy_returns,
        per_policy_descs=per_policy_descs,
        per_policy_lengths=per_policy_lengths,
        desc_names=desc_names,
        num_episodes=args.n_episodes,
        N=N,
        K=K,
    )
    _write_results(
        results,
        args,
        env_id=env_name,
        eval_seed=eval_seed,
        envs_per_policy=runner.E,
        total_env_steps=meta.get("total_env_steps"),
    )


if __name__ == "__main__":
    main()
