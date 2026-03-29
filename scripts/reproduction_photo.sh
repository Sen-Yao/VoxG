#!/bin/bash
# Photo 数据集复现脚本
# 最佳配置来自 Sweep V1 (2026-03-26)
# AUC: 0.8966 (超过 SOTA 0.8960)

set -e

echo "=========================================="
echo "Photo 数据集复现"
echo "AUC 目标: 0.8966 (超过 SOTA 0.8960)"
echo "=========================================="
echo ""

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
echo "使用 GPU: $GPU_ID"
echo ""

# 最佳超参数
DATASET="photo"
ALPHA=0.4
PP_K=5
BATCH_SIZE=1024
PEAK_LR=0.0003
REC_W=1.0
NUM_EPOCH=200
TRAIN_RATE=0.05

echo "配置:"
echo "  progregate_alpha: $ALPHA"
echo "  pp_k: $PP_K"
echo "  batch_size: $BATCH_SIZE"
echo "  peak_lr: $PEAK_LR"
echo "  rec_loss_weight: $REC_W"
echo "  num_epoch: $NUM_EPOCH"
echo "  train_rate: $TRAIN_RATE"
echo ""

# 单次运行
echo "--- 单次运行 ---"
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
    --dataset=$DATASET \
    --model_type=VoxGFormer \
    --progregate_alpha=$ALPHA \
    --pp_k=$PP_K \
    --batch_size=$BATCH_SIZE \
    --peak_lr=$PEAK_LR \
    --rec_loss_weight=$REC_W \
    --num_epoch=$NUM_EPOCH \
    --train_rate=$TRAIN_RATE \
    --token_mode=original

echo ""
echo "完成！"
