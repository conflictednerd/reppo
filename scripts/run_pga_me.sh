#!/usr/bin/env bash
# PGA-ME (QDax, Brax/MJX) on the four Brax tasks, seeds 1, 2, 42, 96, 114,
# 6827 iterations x 300 evaluations (2.05B env steps), followed by the 256-episode
# archive re-evaluation. Needs `uv sync --extra pga_me`.

set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PGA_ME_ARGS="--env_batch_size=300 --physics_backend=mjx --policy_hidden_sizes 128 128
    --critic_hidden_sizes 128 128 --proportion_mutation_ga=0.5 --total_iterations=6827
    --num_pg_training_steps=100 --num_critic_training_steps=300"

for SEED in 1 2 42 96 114; do
    OUT="outputs/pga_me/ant_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.pga_me.train --env_name=ant --seed="$SEED" --expdir="$OUT" $PGA_ME_ARGS \
        --grid_size=10 --num_dims=4 --iso_sigma=3.0 --line_sigma=0.01
    uv run python scripts/evaluate_archive.py "$OUT" --method pga_me
done

for SEED in 1 2 42 96 114; do
    OUT="outputs/pga_me/halfcheetah_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.pga_me.train --env_name=halfcheetah --seed="$SEED" --expdir="$OUT" $PGA_ME_ARGS \
        --grid_size=50 --num_dims=2 --iso_sigma=1.0 --line_sigma=0.01
    uv run python scripts/evaluate_archive.py "$OUT" --method pga_me
done

for SEED in 1 2 42 96 114; do
    OUT="outputs/pga_me/humanoid_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.pga_me.train --env_name=humanoid --seed="$SEED" --expdir="$OUT" $PGA_ME_ARGS \
        --grid_size=50 --num_dims=2 --iso_sigma=0.5 --line_sigma=0.01
    uv run python scripts/evaluate_archive.py "$OUT" --method pga_me
done

for SEED in 1 2 42 96 114; do
    OUT="outputs/pga_me/walker2d_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.pga_me.train --env_name=walker2d --seed="$SEED" --expdir="$OUT" $PGA_ME_ARGS \
        --grid_size=50 --num_dims=2 --iso_sigma=1.0 --line_sigma=0.01
    uv run python scripts/evaluate_archive.py "$OUT" --method pga_me
done
