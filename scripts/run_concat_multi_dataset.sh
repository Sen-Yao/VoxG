#!/bin/bash
# 选项A: 多数据集验证 concat_enhanced (dim=512)
# 在 Photo, Amazon, Reddit 上测试

EMBEDDING_DIM=512
GT_NUM_HEADS=4
GT_FFN_DIM=512
NUM_EPOCH=200
LR=0.0005
PP_K=6
ALPHA=0.7
REC_W=2.0
BCE_W=2.0
TRAIN_RATE=0.05  # 5% 半监督
BATCH_SIZE=32768

echo "=== 多数据集验证: concat_enhanced (dim=512) ==="
echo "开始时间: $(date)"

# Photo on GPU 2
echo "--- Photo (GPU 2) ---"
CUDA_VISIBLE_DEVICES=2 python run.py \
    --dataset=photo \
    --model_type=VoxGFormer \
    --train_rate=$TRAIN_RATE \
    --num_epoch=$NUM_EPOCH \
    --batch_size=$BATCH_SIZE \
    --progregate_alpha=$ALPHA \
    --pp_k=$PP_K \
    --rec_loss_weight=$REC_W \
    --peak_lr=$LR \
    --bce_loss_weight=$BCE_W \
    --token_mode=concat \
    --embedding_dim=$EMBEDDING_DIM \
    --GT_num_heads=$GT_NUM_HEADS \
    --GT_ffn_dim=$GT_FFN_DIM \
    --seed=0 \
    --warmup_updates=100 \
    --warmup_epoch=30 \
    2>&1 | tee logs/concat_photo_seed0.log &

# Amazon on GPU 4  
echo "--- Amazon (GPU 4) ---"
CUDA_VISIBLE_DEVICES=4 python run.py \
    --dataset=Amazon \
    --model_type=VoxGFormer \
    --train_rate=$TRAIN_RATE \
    --num_epoch=$NUM_EPOCH \
    --batch_size=$BATCH_SIZE \
    --progregate_alpha=$ALPHA \
    --pp_k=$PP_K \
    --rec_loss_weight=$REC_W \
    --peak_lr=$LR \
    --bce_loss_weight=$BCE_W \
    --token_mode=concat \
    --embedding_dim=$EMBEDDING_DIM \
    --GT_num_heads=$GT_NUM_HEADS \
    --GT_ffn_dim=$GT_FFN_DIM \
    --seed=0 \
    --warmup_updates=100 \
    --warmup_epoch=30 \
    2>&1 | tee logs/concat_amazon_seed0.log &

# Reddit on GPU 5
echo "--- Reddit (GPU 5) ---"
CUDA_VISIBLE_DEVICES=5 python run.py \
    --dataset=reddit \
    --model_type=VoxGFormer \
    --train_rate=$TRAIN_RATE \
    --num_epoch=$NUM_EPOCH \
    --batch_size=$BATCH_SIZE \
    --progregate_alpha=$ALPHA \
    --pp_k=$PP_K \
    --rec_loss_weight=$REC_W \
    --peak_lr=$LR \
    --bce_loss_weight=$BCE_W \
    --token_mode=concat \
    --embedding_dim=$EMBEDDING_DIM \
    --GT_num_heads=$GT_NUM_HEADS \
    --GT_ffn_dim=$GT_FFN_DIM \
    --seed=0 \
    --warmup_updates=100 \
    --warmup_epoch=30 \
    2>&1 | tee logs/concat_reddit_seed0.log &

wait
echo ""
echo "=== 所有数据集完成 ==="
echo "结束时间: $(date)"
