export NGPUS=8
#!/usr/bin/env sh

mkdir -p logs checkpoints
srun --partition HA_3D --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=COCO --kill-on-bad-exit=1 \
    python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py \
        --config-file "configs/test_coco_multiple.yaml" 2>&1
