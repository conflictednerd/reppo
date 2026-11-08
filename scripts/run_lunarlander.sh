#!/usr/bin/env bash
# LunarLander (gymnasium, CPU), seeds 1-5: RePPO (C_max=0.4), Independent PPO
# and sep-CMA-MAE, all with the same env config, 2x64 tanh categorical actor and
# 65.536M env-step budget, each followed by the 256-episode re-evaluation.
# PPO settings follow the SB3-Zoo LunarLander recipe (gamma 0.999, lambda 0.98,
# lr 1e-3 annealed over the run, ent 0.01, 4 epochs x 32 minibatches).

set -euo pipefail
cd "$(dirname "$0")/.."

PPO_ARGS="env=lunarlander device=cpu algorithm.gamma=0.999 algorithm.gae_lambda=0.98
    algorithm.learning_rate=1e-3 algorithm.ent_coef=0.01 algorithm.lr_schedule_timesteps=131072000
    algorithm.num_envs_per_policy=8 algorithm.num_steps=256 algorithm.hidden_dim=64
    algorithm.total_timesteps=65536000 algorithm.update_epochs=4 algorithm.num_minibatches=32"

# RePPO
for SEED in 1 2 3 4 5; do
    OUT="outputs/lunarlander/reppo/lunarlander_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -m src.main $PPO_ARGS seed="$SEED" log_dir="$OUT" algorithm.C_max=0.4
    uv run python scripts/evaluate_population.py --exp-dir "$OUT"
done

# Independent PPO
for SEED in 1 2 3 4 5; do
    OUT="outputs/lunarlander/ind_ppo/lunarlander_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -m src.main $PPO_ARGS seed="$SEED" log_dir="$OUT" algorithm.force_c_zero=true
    uv run python scripts/evaluate_population.py --exp-dir "$OUT"
done

# sep-CMA-MAE (locomotion CMA-MAE settings: 5x40 emitters, sigma0=0.02, archive_lr=0.001)
for SEED in 1 2 3 4 5; do
    OUT="outputs/lunarlander/sep_cma_mae/lunarlander_s${SEED}"
    uv run python -u -m src.baselines.sep_cma_mae.train --env_name=lunarlander --seed="$SEED" \
        --device=cpu --hidden_dim=64 --eval_episodes=10 --total_env_steps=65536000 \
        --total_evals=2000000 --checkpoint_every=100 --logdir="$OUT"
    uv run python scripts/evaluate_archive.py "$OUT" --method sep_cma_mae
done
