#!/usr/bin/env bash
# PPGA (Brax/MJX port, src/baselines/ppga) on the four Brax tasks, seeds 1-5,
# 200 iterations (2.74B env steps), followed by the 256-episode archive
# re-evaluation. Needs `uv sync --extra ppga`.

set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PPGA_ARGS="--rollout_length=128 --num_minibatches=8 --update_epochs=4 --normalize_obs=True
    --normalize_returns=True --popsize=300 --env_batch_size=3000 --vf_coef=2
    --entropy_coef=0.0 --target_kl=0.008 --max_grad_norm=1 --calc_gradient_iters=10
    --move_mean_iters=10 --physics_backend=mjx --total_iterations=200 --log_arch_freq=100
    --restart_rule=no_improvement"

for SEED in 1 2 3 4 5; do
    OUT="outputs/ppga/ant_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.ppga.train --env_name=ant --seed="$SEED" --expdir="$OUT" $PPGA_ARGS \
        --grid_size=10 --num_dims=4 --archive_lr=0.1 --sigma0=3.0 --threshold_min=-500 --learning_rate=0.001 \
        --map_descriptors_neg1_pos1=False --archive_descriptor_low=0.0
    uv run python scripts/evaluate_archive.py "$OUT" --method ppga
done

for SEED in 1 2 3 4 5; do
    OUT="outputs/ppga/halfcheetah_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.ppga.train --env_name=halfcheetah --seed="$SEED" --expdir="$OUT" $PPGA_ARGS \
        --grid_size=50 --num_dims=2 --archive_lr=0.5 --sigma0=1.0 --threshold_min=200 --learning_rate=0.001
    uv run python scripts/evaluate_archive.py "$OUT" --method ppga
done

for SEED in 1 2 3 4 5; do
    OUT="outputs/ppga/humanoid_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.ppga.train --env_name=humanoid --seed="$SEED" --expdir="$OUT" $PPGA_ARGS \
        --grid_size=50 --num_dims=2 --archive_lr=0.1 --sigma0=0.5 --threshold_min=200 --learning_rate=0.0003 \
        --adaptive_stddev=False --clip_obs_rew=True --map_descriptors_neg1_pos1=False --archive_descriptor_low=0.0
    uv run python scripts/evaluate_archive.py "$OUT" --method ppga
done

for SEED in 1 2 3 4 5; do
    OUT="outputs/ppga/walker2d_s${SEED}"
    # shellcheck disable=SC2086
    uv run python -u -m src.baselines.ppga.train --env_name=walker2d --seed="$SEED" --expdir="$OUT" $PPGA_ARGS \
        --grid_size=50 --num_dims=2 --archive_lr=0.5 --sigma0=1.0 --threshold_min=200 --learning_rate=0.001
    uv run python scripts/evaluate_archive.py "$OUT" --method ppga
done
