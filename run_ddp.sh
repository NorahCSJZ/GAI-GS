#!/bin/bash
# DDP training launcher
# Usage: ./run_ddp.sh [NUM_GPUS]

NUM_GPUS=${1:-2}
shift $(( $# > 0 ? 1 : 0 ))

echo "Starting DDP training with ${NUM_GPUS} GPUs..."

torchrun     --nproc_per_node=${NUM_GPUS}     --master_port=29500     train.py     --dataset_type rfid     --disable_viewer     "$@"
