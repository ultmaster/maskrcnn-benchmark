#!/usr/bin/env sh

mkdir -p logs checkpoints
srun --partition HA_3D --mpi=pmi2 --gres=gpu:1 -n1 --job-name=COCO --kill-on-bad-exit=1 \
    python tools/train_net.py \
        --config-file "configs/test_coco.yaml" 2>&1 &
