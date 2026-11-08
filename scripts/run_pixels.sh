#!/usr/bin/env bash
# RePPO and Independent PPO from pixels (MuJoCo Warp rendering of the Brax
# tasks): 3 envs x seeds 1-3, N=16. Each run is followed by the 256-episode
# population re-evaluation.

set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"

# RePPO
for ENV in halfcheetah_pixels humanoid_pixels walker2d_pixels; do
    for SEED in 1 2 3; do
        OUT="outputs/pixels/reppo/${ENV}_s${SEED}"
        uv run python -m src.main env="$ENV" algorithm=reppo_pixels seed="$SEED" log_dir="$OUT"
        uv run python scripts/evaluate_population.py --exp-dir "$OUT"
    done
done

# Independent PPO
for ENV in halfcheetah_pixels humanoid_pixels walker2d_pixels; do
    for SEED in 1 2 3; do
        OUT="outputs/pixels/ind_ppo/${ENV}_s${SEED}"
        uv run python -m src.main env="$ENV" algorithm=reppo_pixels seed="$SEED" log_dir="$OUT" \
            algorithm.force_c_zero=true
        uv run python scripts/evaluate_population.py --exp-dir "$OUT"
    done
done
