#!/bin/bash
# Cosine Positional Encoding Experiment
# 派遣时间: 2026-03-19 15:14
# 目标: 验证 Cosine 位置编码对图 Transformer 的改进效果

set -e

cd /home/linziyao/VoxG

# 实验 1: Cosine PE vs Baseline on Photo (快速验证)
echo "=========================================="
echo "实验 1: Photo 数据集 - Cosine PE"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=2
python run.py \
    --dataset=photo \
    --train_rate=0.05 \
    --seed=0 \
    --num_epoch=100 \
    --batch_size=4096 \
    --peak_lr=0.0005 \
    --use_pe \
    --pe_type=cosine \
    --wandb_project=voxg-innovation \
    --wandb_name=photo_cosine_pe

# 实验 2: Reddit - Cosine PE
echo "=========================================="
echo "实验 2: Reddit 数据集 - Cosine PE"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=4
python run.py \
    --dataset=reddit \
    --train_rate=0.05 \
    --seed=789 \
    --num_epoch=100 \
    --batch_size=512 \
    --peak_lr=0.0005 \
    --use_pe \
    --pe_type=cosine \
    --wandb_project=voxg-innovation \
    --wandb_name=reddit_cosine_pe

# 实验 3: Learnable PE for comparison
echo "=========================================="
echo "实验 3: Photo 数据集 - Learnable PE"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=5
python run.py \
    --dataset=photo \
    --train_rate=0.05 \
    --seed=0 \
    --num_epoch=100 \
    --batch_size=4096 \
    --peak_lr=0.0005 \
    --use_pe \
    --pe_type=learnable \
    --wandb_project=voxg-innovation \
    --wandb_name=photo_learnable_pe

echo "=========================================="
echo "所有实验完成！"
echo "=========================================="