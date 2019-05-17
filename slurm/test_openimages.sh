#!/usr/bin/env sh

export NGPUS=8
mkdir -p logs checkpoints
srun --partition HA_3D --mpi=pmi2 --gres=gpu:8 -n1 --job-name=COCO --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
        --config-file "configs/test_openimages.yaml" 2>&1 &
