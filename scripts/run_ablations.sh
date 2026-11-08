#!/usr/bin/env bash
# RePPO ablations on Humanoid, seeds 1-3, with the Humanoid PPO settings of
# scripts/run_reppo.sh (which provides the default arm):
# C_max {0.1, 0.3, 0.5, 2, 4, 8} (default 1.0), gamma_sq {0.02, 0.05, 0.5, 1, 2, 5}
# (default 0.1), K_nn {4, 16, 31} (default 8), population size {8, 16, 64} at
# fixed total compute (default 32), asymmetric repulsion, exact W_k.

set -euo pipefail
cd "$(dirname "$0")/.."

for ARM in \
    "cmax0p1 algorithm.C_max=0.1" \
    "cmax0p3 algorithm.C_max=0.3" \
    "cmax0p5 algorithm.C_max=0.5" \
    "cmax2p0 algorithm.C_max=2.0" \
    "cmax4p0 algorithm.C_max=4.0" \
    "cmax8p0 algorithm.C_max=8.0" \
    "gsq0p02 algorithm.gamma_sq=0.02" \
    "gsq0p05 algorithm.gamma_sq=0.05" \
    "gsq0p5 algorithm.gamma_sq=0.5" \
    "gsq1p0 algorithm.gamma_sq=1.0" \
    "gsq2p0 algorithm.gamma_sq=2.0" \
    "gsq5p0 algorithm.gamma_sq=5.0" \
    "knn4 algorithm.K_nn=4" \
    "knn16 algorithm.K_nn=16" \
    "knn31 algorithm.K_nn=31" \
    "pop8 algorithm.N=8" \
    "pop16 algorithm.N=16" \
    "pop64 algorithm.N=64" \
    "asym algorithm.repulsion_mode=higher_only" \
    "exactwk algorithm.use_exact_wk=true"; do
    NAME="${ARM%% *}"
    OVERRIDE="${ARM#* }"
    for SEED in 1 2 3; do
        OUT="outputs/ablations/${NAME}/humanoid_s${SEED}"
        uv run python -m src.main env=humanoid seed="$SEED" log_dir="$OUT" "$OVERRIDE" \
            algorithm.learning_rate=1e-3 algorithm.lr_schedule_timesteps=4096000000 \
            algorithm.clip_vloss=false algorithm.ent_coef=0.0
        uv run python scripts/evaluate_population.py --exp-dir "$OUT"
    done
done
