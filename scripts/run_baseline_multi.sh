#!/bin/bash
# Baseline 实验 - 使用 MatrixGAD 复现超参数
# Original 模式 + 5 seed

echo "=== Baseline 实验 (MatrixGAD 超参数) ==="
echo "开始时间: $(date)"

# Amazon on GPU 2
echo "--- Amazon (GPU 2) ---"
for SEED in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=2 python run.py \
        --batch_size=1024 \
        --dataset=Amazon \
        --end_lr=0.0001 \
        --num_epoch=100 \
        --peak_lr=0.0003 \
        --pp_k=5 \
        --progregate_alpha=0.4 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --seed=$SEED \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=VoxGFormer \
        --token_mode=original \
        2>&1 | tee logs/baseline_amazon_seed${SEED}.log &
done

# Photo on GPU 4
echo "--- Photo (GPU 4) ---"
for SEED in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=4 python run.py \
        --batch_size=128 \
        --dataset=photo \
        --end_lr=0.0001 \
        --num_epoch=200 \
        --peak_lr=0.0005 \
        --pp_k=6 \
        --progregate_alpha=0.2 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --seed=$SEED \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=VoxGFormer \
        --token_mode=original \
        2>&1 | tee logs/baseline_photo_seed${SEED}.log &
done

# Reddit on GPU 5
echo "--- Reddit (GPU 5) ---"
for SEED in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=5 python run.py \
        --batch_size=32768 \
        --dataset=reddit \
        --end_lr=0.0003 \
        --num_epoch=150 \
        --outlier_beta=0.3 \
        --peak_lr=0.0005 \
        --pp_k=6 \
        --progregate_alpha=0.6 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --seed=$SEED \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=VoxGFormer \
        --token_mode=original \
        2>&1 | tee logs/baseline_reddit_seed${SEED}.log &
done

wait
echo ""
echo "=== 所有实验完成 ==="
echo "结束时间: $(date)"
