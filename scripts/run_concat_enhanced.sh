#!/bin/bash
# 方案A: 增加模型容量 - Concat 模式增强实验
# embedding_dim: 256→512, GT_num_heads: 2→4, epoch: 100→200, lr: 0.001→0.0005

DATASET=elliptic
ALPHA=0.7
PP_K=6
REC_W=2.0
LR=0.0005
BCE_W=2.0
TRAIN_RATE=0.1
NUM_EPOCH=200
BATCH_SIZE=32768
TOKEN_MODE=concat
EMBEDDING_DIM=512
GT_NUM_HEADS=4
GT_FFN_DIM=512

echo "=== 方案A: Concat增强实验 - Elliptic ==="
echo "配置: embedding_dim=$EMBEDDING_DIM, GT_num_heads=$GT_NUM_HEADS, GT_ffn_dim=$GT_FFN_DIM"
echo "epoch=$NUM_EPOCH, lr=$LR"
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
        --embedding_dim=$EMBEDDING_DIM \
        --GT_num_heads=$GT_NUM_HEADS \
        --GT_ffn_dim=$GT_FFN_DIM \
        --seed=$SEED \
        --warmup_updates=100 \
        --warmup_epoch=30 \
        --end_lr=0.0001 \
        2>&1 | tee logs/concat_enhanced_seed${SEED}.log
done

echo ""
echo "=== 完成 ==="
echo "结束时间: $(date)"
