#!/bin/bash
# Focal Loss 对比实验 - Reddit & Elliptic

# GPU 1: Reddit + Focal Loss
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --dataset reddit \
    --use_focal_loss True \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --seed 789 \
    --num_epoch 200 \
    --batch_size 512 \
    > logs/focal_reddit_gpu1.log 2>&1 &
echo "GPU1: Reddit + Focal Loss (PID $!)"

# GPU 2: Elliptic + Focal Loss
CUDA_VISIBLE_DEVICES=2 nohup python run.py \
    --dataset elliptic \
    --use_focal_loss True \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --seed 42 \
    --num_epoch 200 \
    --batch_size 512 \
    > logs/focal_elliptic_gpu2.log 2>&1 &
echo "GPU2: Elliptic + Focal Loss (PID $!)"

# GPU 3: Reddit + BCE (baseline)
CUDA_VISIBLE_DEVICES=3 nohup python run.py \
    --dataset reddit \
    --seed 789 \
    --num_epoch 200 \
    --batch_size 512 \
    > logs/bce_reddit_gpu3.log 2>&1 &
echo "GPU3: Reddit + BCE baseline (PID $!)"

# GPU 4: Elliptic + BCE (baseline)
CUDA_VISIBLE_DEVICES=4 nohup python run.py \
    --dataset elliptic \
    --seed 42 \
    --num_epoch 200 \
    --batch_size 512 \
    > logs/bce_elliptic_gpu4.log 2>&1 &
echo "GPU4: Elliptic + BCE baseline (PID $!)"

echo "=== 所有实验已启动 ==="
