#!/usr/bin/env bash
# RefQD (Ref-DQN-ME) on four Atari games x seeds 1-5, each run followed by the re-evaluation of its archive with the
# common protocol (every filled cell plays 32 full games, no sticky actions). Run from the root of the RefQD checkout
# with refqd.patch applied. One run needs about 25 GB of RAM and under 2 GB of GPU memory.
# Writes runs/<game>_s<seed>/{metrics.csv,checkpoint.pkl.xz,evaluation_results.json}.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for s in 1 2 3 4 5; do
    .venv/bin/python -m refqd task=Boxing-v5 seed=$s emitter=Ref-DQN-ME task.bd=ours task.full_game=true task.uint8_obs_buffer=true task.total_steps=80000000 wandb=False pdb=False debug_nans=False fork_final=False hydra.run.dir=runs/boxing_s$s
    .venv/bin/python -m refqd.export_eval --run-dir runs/boxing_s$s
done

for s in 1 2 3 4 5; do
    .venv/bin/python -m refqd task=Asterix-v5 seed=$s emitter=Ref-DQN-ME task.bd=ours task.full_game=true task.uint8_obs_buffer=true task.total_steps=80000000 wandb=False pdb=False debug_nans=False fork_final=False hydra.run.dir=runs/asterix_s$s
    .venv/bin/python -m refqd.export_eval --run-dir runs/asterix_s$s
done

for s in 1 2 3 4 5; do
    .venv/bin/python -m refqd task=Seaquest-v5 seed=$s emitter=Ref-DQN-ME task.bd=ours task.full_game=true task.uint8_obs_buffer=true task.total_steps=80000000 wandb=False pdb=False debug_nans=False fork_final=False emitter.replay_buffer_size=300000 hydra.run.dir=runs/seaquest_s$s
    .venv/bin/python -m refqd.export_eval --run-dir runs/seaquest_s$s
done

for s in 1 2 3 4 5; do
    .venv/bin/python -m refqd task=MsPacman-v5 seed=$s emitter=Ref-DQN-ME task.bd=ours task.full_game=true task.uint8_obs_buffer=true task.total_steps=80000000 wandb=False pdb=False debug_nans=False fork_final=False hydra.run.dir=runs/mspacman_s$s
    .venv/bin/python -m refqd.export_eval --run-dir runs/mspacman_s$s
done
