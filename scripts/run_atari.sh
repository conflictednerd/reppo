#!/usr/bin/env bash
# RePPO and Independent PPO on Atari (envpool, sticky actions 0.25): 4 games x
# seeds 1-5, N=8, 10M agent steps per policy. Each run is followed by the
# 32-game population evaluation. Needs Python 3.12 and `uv sync --extra atari`.

set -euo pipefail
cd "$(dirname "$0")/.."

# RePPO
for GAME in boxing asterix seaquest mspacman; do
    for SEED in 1 2 3 4 5; do
        OUT="outputs/atari/reppo/${GAME}_s${SEED}"
        uv run python -m src.main env="atari_${GAME}" algorithm=reppo_atari seed="$SEED" log_dir="$OUT"
        uv run python scripts/evaluate_population.py --exp-dir "$OUT" --num-episodes 32
    done
done

# Independent PPO
for GAME in boxing asterix seaquest mspacman; do
    for SEED in 1 2 3 4 5; do
        OUT="outputs/atari/ind_ppo/${GAME}_s${SEED}"
        uv run python -m src.main env="atari_${GAME}" algorithm=reppo_atari seed="$SEED" log_dir="$OUT" \
            algorithm.force_c_zero=true
        uv run python scripts/evaluate_population.py --exp-dir "$OUT" --num-episodes 32
    done
done
