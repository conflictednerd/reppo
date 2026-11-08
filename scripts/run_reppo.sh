#!/usr/bin/env bash
# RePPO on the four Brax tasks, N=32, seeds 1-5. Each run is followed by the
# 256-episode population re-evaluation. HalfCheetah, Humanoid and Walker2D use
# lr 1e-3 annealed to 5% at the end of the run, no value-loss clipping and no
# entropy bonus; HalfCheetah also gamma_sq=0.5, Walker2D 5 update epochs. Ant
# uses the config defaults.

set -euo pipefail
cd "$(dirname "$0")/.."

# Ant
for SEED in 1 2 3 4 5; do
    OUT="outputs/reppo/ant_s${SEED}"
    uv run python -m src.main env=ant seed="$SEED" log_dir="$OUT"
    uv run python scripts/evaluate_population.py --exp-dir "$OUT"
done

# HalfCheetah
for SEED in 1 2 3 4 5; do
    OUT="outputs/reppo/halfcheetah_s${SEED}"
    uv run python -m src.main env=halfcheetah seed="$SEED" log_dir="$OUT" \
        algorithm.learning_rate=1e-3 algorithm.lr_schedule_timesteps=4096000000 \
        algorithm.clip_vloss=false algorithm.ent_coef=0.0 algorithm.gamma_sq=0.5
    uv run python scripts/evaluate_population.py --exp-dir "$OUT"
done

# Humanoid
for SEED in 1 2 3 4 5; do
    OUT="outputs/reppo/humanoid_s${SEED}"
    uv run python -m src.main env=humanoid seed="$SEED" log_dir="$OUT" \
        algorithm.learning_rate=1e-3 algorithm.lr_schedule_timesteps=4096000000 \
        algorithm.clip_vloss=false algorithm.ent_coef=0.0
    uv run python scripts/evaluate_population.py --exp-dir "$OUT"
done

# Walker2D
for SEED in 1 2 3 4 5; do
    OUT="outputs/reppo/walker2d_s${SEED}"
    uv run python -m src.main env=walker2d seed="$SEED" log_dir="$OUT" \
        algorithm.learning_rate=1e-3 algorithm.lr_schedule_timesteps=4096000000 \
        algorithm.clip_vloss=false algorithm.ent_coef=0.0 algorithm.update_epochs=5
    uv run python scripts/evaluate_population.py --exp-dir "$OUT"
done
