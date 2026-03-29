#!/bin/bash
# Delta Vector 实验 - Elliptic 数据集
# 使用 Sweep 最佳超参数

DATASET=elliptic
ALPHA=0.7
PP_K=6
REC_W=2.0
LR=0.001
BCE_W=2.0
TRAIN_RATE=0.1
NUM_EPOCH=100
BATCH_SIZE=32768
TOKEN_MODE=concat

echo "=== Delta Vector 实验 - Elliptic ==="
echo "配置: alpha=$ALPHA, pp_k=$PP_K, rec_w=$REC_W, lr=$LR, token_mode=$TOKEN_MODE"
echo "开始时间: $(date)"

for SEED in 0 1 2 3 4; do
    echo ""
    echo "--- Seed $SEED ---"
    CUDA_VISIBLE_DEVICES=5 python run.py \
        --dataset=$DATASET \
        --model_type=VoxGFormer \
        --train_rate=$TRAIN_RATE \
        --num_epoch=$NUM_EPOCH \
        --batch_size=$BATCH_SIZE \
        --progregate_alpha=$ALPHA \
        --pp_k=$PP_K \
        --rec_loss_weight=$REC_W \
        --peak_lr=$LR \
        --bce_loss_weight=$BCE_W \
        --token_mode=$TOKEN_MODE \
        --seed=$SEED \
        --warmup_updates=50 \
        --warmup_epoch=20 \
        --end_lr=0.0003 \
        2>&1 | tee logs/delta_elliptic_seed${SEED}.log
done

echo ""
echo "=== 完成 ==="
echo "结束时间: $(date)"
