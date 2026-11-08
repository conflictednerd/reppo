#!/usr/bin/env bash
# DCRL-ME (QDax DCRLMEEmitter as shipped, Brax/MJX) on the four Brax tasks,
# seeds 1, 2, 42, 96, 114, 8000 iterations x 256 evaluations (2.048B env steps), followed by
# the 256-episode archive re-evaluation. Needs `uv sync --extra pga_me`.

set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

for ENV in ant halfcheetah humanoid walker2d; do
    for SEED in 1 2 42 96 114; do
        OUT="outputs/dcrl_me/${ENV}_s${SEED}"
        uv run python -u -m src.baselines.dcrl_me.train --env_name="$ENV" --seed="$SEED" \
            --total_iterations=8000 --expdir="$OUT"
        uv run python scripts/evaluate_archive.py "$OUT" --method dcrl_me
    done
done
