#!/bin/bash
# 对比学习超参数调优实验
# 在 Photo 和 Reddit 数据集上并行测试不同配置

cd /root/gpufree-data/linziyao/VoxG

echo "=== 启动对比学习调优实验 ==="
echo "时间: $(date)"

# Photo 数据集实验
CUDA_VISIBLE_DEVICES=1 nohup python run.py --dataset Photo --use_contrastive True \
    --contrastive_temp 0.05 --contrastive_weight 0.1 \
    --num_epoch 200 --seed 42 \
    > logs/contrastive_photo_t005_w01.log 2>&1 &
echo "GPU 1: Photo temp=0.05 weight=0.1 PID=$!"

CUDA_VISIBLE_DEVICES=2 nohup python run.py --dataset Photo --use_contrastive True \
    --contrastive_temp 0.1 --contrastive_weight 0.3 \
    --num_epoch 200 --seed 42 \
    > logs/contrastive_photo_t01_w03.log 2>&1 &
echo "GPU 2: Photo temp=0.1 weight=0.3 PID=$!"

CUDA_VISIBLE_DEVICES=3 nohup python run.py --dataset Photo --use_contrastive True \
    --contrastive_temp 0.2 --contrastive_weight 0.5 \
    --num_epoch 200 --seed 42 \
    > logs/contrastive_photo_t02_w05.log 2>&1 &
echo "GPU 3: Photo temp=0.2 weight=0.5 PID=$!"

# Reddit 数据集实验
CUDA_VISIBLE_DEVICES=4 nohup python run.py --dataset Reddit --use_contrastive True \
    --contrastive_temp 0.05 --contrastive_weight 0.1 \
    --num_epoch 200 --seed 789 \
    > logs/contrastive_reddit_t005_w01.log 2>&1 &
echo "GPU 4: Reddit temp=0.05 weight=0.1 PID=$!"

CUDA_VISIBLE_DEVICES=5 nohup python run.py --dataset Reddit --use_contrastive True \
    --contrastive_temp 0.1 --contrastive_weight 0.3 \
    --num_epoch 200 --seed 789 \
    > logs/contrastive_reddit_t01_w03.log 2>&1 &
echo "GPU 5: Reddit temp=0.1 weight=0.3 PID=$!"

echo "=== 所有实验已启动 ==="
