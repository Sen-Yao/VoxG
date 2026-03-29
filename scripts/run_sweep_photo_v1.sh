#!/bin/bash
# Photo Sweep V1
# Sweep ID: 47lfdg9i

echo "=== Photo Sweep V1 ==="
echo "开始时间: $(date)"

# 使用空闲 GPU
for GPU in 2 4 5; do
    echo "启动 agent on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python -m wandb agent Senyao/VoxG/47lfdg9i &
done

wait
echo "=== Sweep 完成 ==="
echo "结束时间: $(date)"
