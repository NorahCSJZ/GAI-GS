#!/bin/bash
# DDP 训练启动脚本
# 使用方法: ./run_ddp.sh [NUM_GPUS]

NUM_GPUS=${1:-2}  # 默认使用 2 个 GPU
shift $(( $# > 0 ? 1 : 0 ))

echo "Starting DDP training with ${NUM_GPUS} GPUs..."

# 使用 torchrun 启动 DDP 训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train.py \
    --use_ddp \
    --dataset_type rfid \
    --disable_viewer \
    "$@"
