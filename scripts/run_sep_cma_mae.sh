#!/usr/bin/env bash
# sep-CMA-MAE (pyribs) on the four Brax tasks, seeds 1, 2, 42, 96, 114,
# 2.048M evaluations capped at 2.048B env steps, followed by the 256-episode archive re-evaluation.

set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

for ENV in ant halfcheetah humanoid walker2d; do
    for SEED in 1 2 42 96 114; do
        OUT="outputs/sep_cma_mae/${ENV}_s${SEED}"
        uv run python -u -m src.baselines.sep_cma_mae.train --env_name="$ENV" --seed="$SEED" \
            --total_evals=2048000 --total_env_steps=2048000000 --logdir="$OUT"
        uv run python scripts/evaluate_archive.py "$OUT" --method sep_cma_mae
    done
done
