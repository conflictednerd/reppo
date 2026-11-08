"""PGA-ME (Policy Gradient Assisted MAP-Elites) baseline via QDax on Brax/MJX."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from src.baselines.pga_me import qdax_compat  # noqa: F401  (patches QDax v1 imports)

# isort: split
import jax
import jax.numpy as jnp
import numpy as np
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig
from qdax.core.emitters.qpg_emitter import QualityPGEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax import v2 as qdax_brax_v2
from qdax.tasks.brax.v2.env_creators import create_brax_scoring_fn
from qdax.utils.metrics import CSVLogger, default_qd_metrics

from src.baselines.pga_me.env import make_qdax_env, qdax_env_name

REPO_ROOT = Path(__file__).resolve().parents[3]

ENV_DEFAULTS = {
    "halfcheetah": {
        "grid_size": 50,
        "num_dims": 2,
        "iso_sigma": 1.0,
        "line_sigma": 0.01,
    },
    "walker2d": {
        "grid_size": 50,
        "num_dims": 2,
        "iso_sigma": 1.0,
        "line_sigma": 0.01,
    },
    "ant": {
        "grid_size": 10,
        "num_dims": 4,
        "iso_sigma": 3.0,
        "line_sigma": 0.01,
    },
    "humanoid": {
        "grid_size": 50,
        "num_dims": 2,
        "iso_sigma": 0.5,
        "line_sigma": 0.01,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PGA-ME baseline (QDax + Brax/MJX)")
    parser.add_argument("--env_name", type=str, default="ant")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--total_iterations", type=int, default=2000)
    parser.add_argument("--env_batch_size", type=int, default=300)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--physics_backend", type=str, default="mjx")
    parser.add_argument("--policy_hidden_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--critic_hidden_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--proportion_mutation_ga", type=float, default=0.5)
    parser.add_argument("--num_pg_training_steps", type=int, default=100)
    parser.add_argument("--num_critic_training_steps", type=int, default=300)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--num_dims", type=int, default=None)
    parser.add_argument("--iso_sigma", type=float, default=None)
    parser.add_argument("--line_sigma", type=float, default=None)
    parser.add_argument("--descriptor_low", type=float, default=0.0)
    parser.add_argument("--descriptor_high", type=float, default=1.0)
    parser.add_argument("--expdir", type=str, required=True)
    return parser.parse_args()


def _variation_fn(x1, x2, key, *, iso_sigma: float, line_sigma: float):
    return isoline_variation(x1, x2, key, iso_sigma, line_sigma)


def _build_pgame_emitter(
    config: PGAMEConfig,
    policy_network,
    env,
    *,
    iso_sigma: float,
    line_sigma: float,
):
    """PGAME = QualityPGEmitter + GA MixingEmitter (fixed fn signatures for QDax 0.5)."""
    ga_batch_size = int(config.proportion_mutation_ga * config.env_batch_size)
    qpg_batch_size = config.env_batch_size - ga_batch_size

    from qdax.core.emitters.qpg_emitter import QualityPGConfig

    qpg_config = QualityPGConfig(
        env_batch_size=qpg_batch_size,
        num_critic_training_steps=config.num_critic_training_steps,
        num_pg_training_steps=config.num_pg_training_steps,
        replay_buffer_size=config.replay_buffer_size,
        critic_hidden_layer_size=config.critic_hidden_layer_size,
        critic_learning_rate=config.critic_learning_rate,
        actor_learning_rate=config.greedy_learning_rate,
        policy_learning_rate=config.policy_learning_rate,
        noise_clip=config.noise_clip,
        policy_noise=config.policy_noise,
        discount=config.discount,
        reward_scaling=config.reward_scaling,
        batch_size=config.batch_size,
        soft_tau_update=config.soft_tau_update,
        policy_delay=config.policy_delay,
    )
    q_emitter = QualityPGEmitter(config=qpg_config, policy_network=policy_network, env=env)
    def variation(x1, x2, k):
        return _variation_fn(x1, x2, k, iso_sigma=iso_sigma, line_sigma=line_sigma)

    ga_emitter = MixingEmitter(
        mutation_fn=lambda x, _key: x,
        variation_fn=variation,
        variation_percentage=1.0,
        batch_size=ga_batch_size,
    )
    from qdax.core.emitters.multi_emitter import MultiEmitter

    return MultiEmitter(emitters=(q_emitter, ga_emitter))


def _save_cfg(outdir: Path, cfg: dict) -> None:
    with open(outdir / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _save_repertoire(outdir: Path, repertoire: MapElitesRepertoire, iteration: int) -> Path:
    ckpt_dir = outdir / "checkpoints" / f"cp_{iteration:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _to_numpy(tree):
        return jax.tree.map(lambda x: np.asarray(x), tree)

    payload = {
        "genotypes": _to_numpy(repertoire.genotypes),
        "fitnesses": np.asarray(repertoire.fitnesses),
        "descriptors": np.asarray(repertoire.descriptors),
        "centroids": np.asarray(repertoire.centroids),
    }
    path = ckpt_dir / "repertoire.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def train_pga_me(cfg: dict) -> None:
    env_name = cfg["env_name"].lower()
    defaults = ENV_DEFAULTS[env_name]
    num_dims = int(cfg.get("num_dims") or defaults["num_dims"])
    grid_size = int(cfg.get("grid_size") or defaults["grid_size"])
    iso_sigma = float(cfg.get("iso_sigma") if cfg.get("iso_sigma") is not None else defaults["iso_sigma"])
    line_sigma = float(
        cfg.get("line_sigma") if cfg.get("line_sigma") is not None else defaults["line_sigma"]
    )

    seed = int(cfg["seed"])
    key = jax.random.PRNGKey(seed)

    env = make_qdax_env(
        env_name,
        episode_length=int(cfg["episode_length"]),
        physics_backend=cfg.get("physics_backend", "mjx"),
    )
    qdax_name = qdax_env_name(env_name)

    policy_layer_sizes = tuple(cfg["policy_hidden_sizes"]) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    key, subkey = jax.random.split(key)
    scoring_fn = create_brax_scoring_fn(
        env,
        policy_network,
        qdax_brax_v2.descriptor_extractor[qdax_name],
        subkey,
        episode_length=int(cfg["episode_length"]),
        deterministic=True,
    )

    grid_shape = tuple([grid_size] * num_dims)
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=cfg["descriptor_low"],
        maxval=cfg["descriptor_high"],
    )

    batch_size = int(cfg["env_batch_size"])
    key, subkey = jax.random.split(key)
    init_keys = jax.random.split(subkey, batch_size)
    fake_batch = jnp.zeros((batch_size, env.observation_size))
    init_genotypes = jax.vmap(policy_network.init)(init_keys, fake_batch)

    pgame_config = PGAMEConfig(
        env_batch_size=batch_size,
        proportion_mutation_ga=float(cfg["proportion_mutation_ga"]),
        num_critic_training_steps=int(cfg["num_critic_training_steps"]),
        num_pg_training_steps=int(cfg["num_pg_training_steps"]),
        critic_hidden_layer_size=tuple(cfg["critic_hidden_sizes"]),
    )
    emitter = _build_pgame_emitter(
        pgame_config,
        policy_network,
        env,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
    )

    def metrics_fn(rep):
        return default_qd_metrics(rep, 0.0)

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = map_elites.init(init_genotypes, centroids, subkey)

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    _save_cfg(outdir, cfg)

    summary_path = outdir / "summary.csv"
    logger = CSVLogger(
        str(summary_path),
        ["iteration", "qd_score", "coverage", "max_fitness"],
    )

    total_iters = int(cfg["total_iterations"])
    print(
        f"PGA-ME {env_name} seed={seed} backend={cfg.get('physics_backend')} "
        f"grid={grid_shape} batch={batch_size} iters={total_iters}",
        flush=True,
    )

    t0 = time.perf_counter()
    for itr in range(1, total_iters + 1):
        iter_t0 = time.perf_counter()
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = map_elites.update(repertoire, emitter_state, subkey)
        metrics_np = jax.tree.map(lambda x: float(x), metrics)
        logger.log(
            {
                "iteration": itr,
                "qd_score": metrics_np["qd_score"],
                "coverage": metrics_np["coverage"] / 100.0,
                "max_fitness": metrics_np["max_fitness"],
            }
        )
        elapsed = time.perf_counter() - iter_t0
        print(
            f"iter {itr}/{total_iters} | QD={metrics_np['qd_score']:.1f} cov={metrics_np['coverage']:.1f}% "
            f"max={metrics_np['max_fitness']:.2f} | {elapsed:.1f}s",
            flush=True,
        )
        if itr == total_iters or itr % max(1, cfg.get("log_arch_freq", 100)) == 0:
            _save_repertoire(outdir, repertoire, itr)

    total_elapsed = time.perf_counter() - t0
    print(f"Done in {total_elapsed / 60:.1f} min. Output: {outdir}", flush=True)


def main() -> None:
    args = parse_args()
    env_name = args.env_name.lower()
    defaults = ENV_DEFAULTS[env_name]
    outdir = Path(args.expdir)

    cfg = {
        "method": "pga_me",
        "env_name": env_name,
        "seed": args.seed,
        "total_iterations": args.total_iterations,
        "env_batch_size": args.env_batch_size,
        "episode_length": args.episode_length,
        "physics_backend": args.physics_backend,
        "policy_hidden_sizes": args.policy_hidden_sizes,
        "critic_hidden_sizes": args.critic_hidden_sizes,
        "proportion_mutation_ga": args.proportion_mutation_ga,
        "num_pg_training_steps": args.num_pg_training_steps,
        "num_critic_training_steps": args.num_critic_training_steps,
        "grid_size": args.grid_size or defaults["grid_size"],
        "num_dims": args.num_dims or defaults["num_dims"],
        "iso_sigma": args.iso_sigma if args.iso_sigma is not None else defaults["iso_sigma"],
        "line_sigma": args.line_sigma if args.line_sigma is not None else defaults["line_sigma"],
        "descriptor_low": args.descriptor_low,
        "descriptor_high": args.descriptor_high,
        "outdir": str(outdir),
        "obs_shape": None,
        "action_shape": None,
        "log_arch_freq": 100,
    }

    # Fill obs/action shapes after env creation in train.
    env = make_qdax_env(env_name, episode_length=args.episode_length, physics_backend=args.physics_backend)
    cfg["obs_shape"] = [env.observation_size]
    cfg["action_shape"] = [env.action_size]

    train_pga_me(cfg)


if __name__ == "__main__":
    main()
