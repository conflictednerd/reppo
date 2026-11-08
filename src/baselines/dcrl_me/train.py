"""DCRL-ME baseline via pip QDax on this paper's Brax/MJX tasks.

Uses QDax's DCRLMEEmitter and DCRLMEConfig as shipped. The env is our MJX
stack (feet-contact descriptors, 1000-step episodes). The only intentional
algorithm change is training length so the env-step budget matches RePPO.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from functools import partial
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
    compute_cvt_centroids,
)
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.tasks.brax import v2 as qdax_brax_v2
from qdax.tasks.brax.v2.env_creators import create_brax_scoring_fn
from qdax.tasks.brax.v2.wrappers.reward_wrappers import ClipRewardWrapper, OffsetRewardWrapper
from qdax.utils.metrics import CSVLogger, default_qd_metrics

from src.baselines.pga_me.env import make_qdax_env, qdax_env_name

# Faldor et al. DCRL-ME Table 2 (and official dcrl_me.yaml). QDax's
# DCRLMEConfig default for policy_learning_rate is 1e-3; the paper is 5e-3.
DCRL_HPARAMS = {
    "ga_batch_size": 128,
    "dcrl_batch_size": 64,
    "ai_batch_size": 64,
    "lengthscale": 0.1,
    "iso_sigma": 0.005,
    "line_sigma": 0.05,
    "num_critic_training_steps": 3000,
    "policy_delay": 2,
    "num_pg_training_steps": 150,
    "td3_batch_size": 100,
    "replay_buffer_size": 1_000_000,
    "discount": 0.99,
    "reward_scaling": 1.0,
    "critic_learning_rate": 3e-4,
    "actor_learning_rate": 3e-4,
    "policy_learning_rate": 5e-3,
    "noise_clip": 0.5,
    "policy_noise": 0.2,
    "soft_tau_update": 0.005,
    "num_centroids": 1024,
    "num_init_cvt_samples": 50_000,
}

# RePPO budget: 8000 * 256 * 1000 = 2_048_000_000 env steps.
DEFAULT_TOTAL_ITERATIONS = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCRL-ME baseline (QDax + Brax/MJX)")
    parser.add_argument("--env_name", type=str, default="ant")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_iterations", type=int, default=DEFAULT_TOTAL_ITERATIONS)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--physics_backend", type=str, default="mjx")
    parser.add_argument("--policy_hidden_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--critic_hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--ga_batch_size", type=int, default=DCRL_HPARAMS["ga_batch_size"])
    parser.add_argument("--dcrl_batch_size", type=int, default=DCRL_HPARAMS["dcrl_batch_size"])
    parser.add_argument("--ai_batch_size", type=int, default=DCRL_HPARAMS["ai_batch_size"])
    parser.add_argument("--lengthscale", type=float, default=DCRL_HPARAMS["lengthscale"])
    parser.add_argument(
        "--num_pg_training_steps", type=int, default=DCRL_HPARAMS["num_pg_training_steps"]
    )
    parser.add_argument(
        "--num_critic_training_steps",
        type=int,
        default=DCRL_HPARAMS["num_critic_training_steps"],
    )
    parser.add_argument("--policy_delay", type=int, default=DCRL_HPARAMS["policy_delay"])
    parser.add_argument(
        "--policy_learning_rate", type=float, default=DCRL_HPARAMS["policy_learning_rate"]
    )
    parser.add_argument(
        "--actor_learning_rate", type=float, default=DCRL_HPARAMS["actor_learning_rate"]
    )
    parser.add_argument(
        "--td3_batch_size", type=int, default=DCRL_HPARAMS["td3_batch_size"]
    )
    parser.add_argument(
        "--replay_buffer_size", type=int, default=DCRL_HPARAMS["replay_buffer_size"]
    )
    parser.add_argument("--num_centroids", type=int, default=DCRL_HPARAMS["num_centroids"])
    parser.add_argument(
        "--num_init_cvt_samples",
        type=int,
        default=DCRL_HPARAMS["num_init_cvt_samples"],
    )
    parser.add_argument("--iso_sigma", type=float, default=DCRL_HPARAMS["iso_sigma"])
    parser.add_argument("--line_sigma", type=float, default=DCRL_HPARAMS["line_sigma"])
    parser.add_argument(
        "--descriptor_low",
        type=float,
        default=None,
        help="Override BD lower bound; default is the env's descriptor_limits",
    )
    parser.add_argument(
        "--descriptor_high",
        type=float,
        default=None,
        help="Override BD upper bound; default is the env's descriptor_limits",
    )
    parser.add_argument("--expdir", type=str, required=True)
    return parser.parse_args()


def _as_dcrl_transition(transition, descriptor_dim: int) -> DCRLTransition:
    """QDax Brax scoring returns QDTransition; DCRLEmitter needs desc fields."""
    zeros = jnp.zeros((*transition.obs.shape[:-1], descriptor_dim))
    return DCRLTransition(
        obs=transition.obs,
        next_obs=transition.next_obs,
        rewards=transition.rewards,
        dones=transition.dones,
        truncations=transition.truncations,
        actions=transition.actions,
        state_desc=transition.state_desc,
        next_state_desc=transition.next_state_desc,
        desc=zeros,
        desc_prime=zeros,
    )


def _with_dcrl_transitions(scoring_fn, descriptor_dim: int):
    def wrapped(genotypes, key):
        fitnesses, descriptors, extra_scores = scoring_fn(genotypes, key)
        extra_scores = {
            **extra_scores,
            "transitions": _as_dcrl_transition(extra_scores["transitions"], descriptor_dim),
        }
        return fitnesses, descriptors, extra_scores

    return wrapped


def _build_dcrl_emitter(
    config: DCRLMEConfig,
    policy_network,
    actor_network,
    env,
    *,
    iso_sigma: float,
    line_sigma: float,
):
    """QDax DCRLMEEmitter with the official isoline variation operator."""
    return DCRLMEEmitter(
        config=config,
        policy_network=policy_network,
        actor_network=actor_network,
        env=env,
        variation_fn=partial(isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma),
    )


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


def train_dcrl_me(cfg: dict) -> None:
    env_name = cfg["env_name"].lower()
    iso_sigma = float(cfg["iso_sigma"])
    line_sigma = float(cfg["line_sigma"])

    seed = int(cfg["seed"])
    key = jax.random.PRNGKey(seed)

    env = make_qdax_env(
        env_name,
        episode_length=int(cfg["episode_length"]),
        physics_backend=cfg.get("physics_backend", "mjx"),
    )
    qdax_name = qdax_env_name(env_name)
    # Original DCRL training scheme: offset then clip at 0 so TD3 sees
    # non-negative rewards. Eval still uses the shared unclipped runner.
    reward_offset = float(qdax_brax_v2.reward_offset[qdax_name])
    env = ClipRewardWrapper(OffsetRewardWrapper(env, offset=reward_offset))
    cfg["train_reward_offset"] = reward_offset
    cfg["train_clip_reward_min"] = 0.0
    env_min, env_max = env.descriptor_limits
    minval = cfg["descriptor_low"] if cfg["descriptor_low"] is not None else env_min
    maxval = cfg["descriptor_high"] if cfg["descriptor_high"] is not None else env_max
    cfg["descriptor_low"] = np.asarray(minval).tolist()
    cfg["descriptor_high"] = np.asarray(maxval).tolist()

    policy_layer_sizes = tuple(cfg["policy_hidden_sizes"]) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    # Descriptor-conditioned actor: concat(obs, desc) in the first layer.
    # Actor injection folds desc into bias so archive genotypes stay MLP-shaped.
    actor_network = MLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    key, subkey = jax.random.split(key)
    scoring_fn = _with_dcrl_transitions(
        create_brax_scoring_fn(
            env,
            policy_network,
            qdax_brax_v2.descriptor_extractor[qdax_name],
            subkey,
            episode_length=int(cfg["episode_length"]),
            deterministic=True,
        ),
        descriptor_dim=int(env.descriptor_length),
    )

    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=int(env.descriptor_length),
        num_init_cvt_samples=int(cfg["num_init_cvt_samples"]),
        num_centroids=int(cfg["num_centroids"]),
        minval=minval,
        maxval=maxval,
        key=subkey,
    )

    batch_size = int(cfg["env_batch_size"])
    key, subkey = jax.random.split(key)
    init_keys = jax.random.split(subkey, batch_size)
    fake_batch = jnp.zeros((batch_size, env.observation_size))
    init_genotypes = jax.vmap(policy_network.init)(init_keys, fake_batch)

    dcrl_config = DCRLMEConfig(
        ga_batch_size=int(cfg["ga_batch_size"]),
        dcrl_batch_size=int(cfg["dcrl_batch_size"]),
        ai_batch_size=int(cfg["ai_batch_size"]),
        lengthscale=float(cfg["lengthscale"]),
        critic_hidden_layer_size=tuple(cfg["critic_hidden_sizes"]),
        num_critic_training_steps=int(cfg["num_critic_training_steps"]),
        num_pg_training_steps=int(cfg["num_pg_training_steps"]),
        batch_size=int(cfg["td3_batch_size"]),
        replay_buffer_size=int(cfg["replay_buffer_size"]),
        discount=float(cfg["discount"]),
        reward_scaling=float(cfg["reward_scaling"]),
        critic_learning_rate=float(cfg["critic_learning_rate"]),
        actor_learning_rate=float(cfg["actor_learning_rate"]),
        policy_learning_rate=float(cfg["policy_learning_rate"]),
        noise_clip=float(cfg["noise_clip"]),
        policy_noise=float(cfg["policy_noise"]),
        soft_tau_update=float(cfg["soft_tau_update"]),
        policy_delay=int(cfg["policy_delay"]),
    )
    emitter = _build_dcrl_emitter(
        dcrl_config,
        policy_network,
        actor_network,
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
        f"DCRL-ME {env_name} seed={seed} backend={cfg.get('physics_backend')} "
        f"cvt={cfg['num_centroids']} batch={batch_size} "
        f"(ga={cfg['ga_batch_size']} dcrl={cfg['dcrl_batch_size']} ai={cfg['ai_batch_size']}) "
        f"iters={total_iters} env_steps={cfg['total_env_steps']}",
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
    outdir = Path(args.expdir)
    env_batch_size = args.ga_batch_size + args.dcrl_batch_size + args.ai_batch_size
    total_env_steps = args.total_iterations * env_batch_size * args.episode_length

    cfg = {
        "method": "dcrl_me",
        "env_name": env_name,
        "seed": args.seed,
        "total_iterations": args.total_iterations,
        "env_batch_size": env_batch_size,
        "ga_batch_size": args.ga_batch_size,
        "dcrl_batch_size": args.dcrl_batch_size,
        "ai_batch_size": args.ai_batch_size,
        "episode_length": args.episode_length,
        "physics_backend": args.physics_backend,
        "policy_hidden_sizes": args.policy_hidden_sizes,
        "critic_hidden_sizes": args.critic_hidden_sizes,
        "lengthscale": args.lengthscale,
        "num_pg_training_steps": args.num_pg_training_steps,
        "num_critic_training_steps": args.num_critic_training_steps,
        "policy_delay": args.policy_delay,
        "td3_batch_size": args.td3_batch_size,
        "replay_buffer_size": args.replay_buffer_size,
        "discount": DCRL_HPARAMS["discount"],
        "reward_scaling": DCRL_HPARAMS["reward_scaling"],
        "critic_learning_rate": DCRL_HPARAMS["critic_learning_rate"],
        "actor_learning_rate": args.actor_learning_rate,
        "policy_learning_rate": args.policy_learning_rate,
        "noise_clip": DCRL_HPARAMS["noise_clip"],
        "policy_noise": DCRL_HPARAMS["policy_noise"],
        "soft_tau_update": DCRL_HPARAMS["soft_tau_update"],
        "archive": "cvt",
        "num_centroids": args.num_centroids,
        "num_init_cvt_samples": args.num_init_cvt_samples,
        "iso_sigma": args.iso_sigma,
        "line_sigma": args.line_sigma,
        "descriptor_low": args.descriptor_low,
        "descriptor_high": args.descriptor_high,
        "total_env_steps": total_env_steps,
        "outdir": str(outdir),
        "obs_shape": None,
        "action_shape": None,
        "log_arch_freq": 100,
    }

    env = make_qdax_env(
        env_name, episode_length=args.episode_length, physics_backend=args.physics_backend
    )
    cfg["obs_shape"] = [env.observation_size]
    cfg["action_shape"] = [env.action_size]

    train_dcrl_me(cfg)


if __name__ == "__main__":
    main()
