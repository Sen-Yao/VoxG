#!/bin/bash
# VoxG 软正交化消融实验 - 并行版本
# 6 个实验同时在 6 张 GPU 上运行

set -e

echo "=========================================="
echo "VoxG 软正交化消融实验 - 并行版本"
echo "=========================================="
echo ""

# 检查 GPU 可用性
echo "📊 GPU 状态:"
nvidia-smi --query-gpu=index,memory.used --format=csv
echo ""

# 创建日志目录
mkdir -p ~/VoxG/logs/experiments/parallel

echo "🚀 同时启动 6 个实验..."
echo ""

# ============================================
# 实验 1: 基线（无正交化）- GPU 1
# ============================================
echo "📍 实验 1: 基线 → GPU 1"
cd ~/VoxG
CUDA_VISIBLE_DEVICES=1 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=False \
    > logs/experiments/parallel/01_baseline.log 2>&1 &
PID1=$!
echo "   PID: $PID1"

# ============================================
# 实验 2: 软正交化 beta=0.5 - GPU 2
# ============================================
echo "📍 实验 2: 软正交化 beta=0.5 → GPU 2"
CUDA_VISIBLE_DEVICES=2 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=True \
    --orthogonal_beta=0.5 \
    > logs/experiments/parallel/02_soft_beta50.log 2>&1 &
PID2=$!
echo "   PID: $PID2"

# ============================================
# 实验 3: 软正交化 beta=0.3 - GPU 3
# ============================================
echo "📍 实验 3: 软正交化 beta=0.3 → GPU 3"
CUDA_VISIBLE_DEVICES=3 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=True \
    --orthogonal_beta=0.3 \
    > logs/experiments/parallel/03_soft_beta30.log 2>&1 &
PID3=$!
echo "   PID: $PID3"

# ============================================
# 实验 4: 正交正则化 lambda=0.01 - GPU 4
# ============================================
echo "📍 实验 4: 正交正则化 lambda=0.01 → GPU 4"
CUDA_VISIBLE_DEVICES=4 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=False \
    --lambda_orthogonal=0.01 \
    > logs/experiments/parallel/04_orth_reg_001.log 2>&1 &
PID4=$!
echo "   PID: $PID4"

# ============================================
# 实验 5: 正交正则化 lambda=0.1 - GPU 5
# ============================================
echo "📍 实验 5: 正交正则化 lambda=0.1 → GPU 5"
CUDA_VISIBLE_DEVICES=5 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=False \
    --lambda_orthogonal=0.1 \
    > logs/experiments/parallel/05_orth_reg_01.log 2>&1 &
PID5=$!
echo "   PID: $PID5"

# ============================================
# 实验 6: 组合方案（软正交化 + 正则化）- GPU 6
# ============================================
echo "📍 实验 6: 组合方案（beta=0.3 + lambda=0.01）→ GPU 6"
CUDA_VISIBLE_DEVICES=6 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=True \
    --orthogonal_beta=0.3 --lambda_orthogonal=0.01 \
    > logs/experiments/parallel/06_combined.log 2>&1 &
PID6=$!
echo "   PID: $PID6"

echo ""
echo "=========================================="
echo "✅ 所有实验已启动！"
echo "=========================================="
echo ""
echo "📊 实验列表:"
echo "  实验 1 (PID $PID1): 基线 → GPU 1"
echo "  实验 2 (PID $PID2): 软正交化 beta=0.5 → GPU 2"
echo "  实验 3 (PID $PID3): 软正交化 beta=0.3 → GPU 3"
echo "  实验 4 (PID $PID4): 正交正则化 lambda=0.01 → GPU 4"
echo "  实验 5 (PID $PID5): 正交正则化 lambda=0.1 → GPU 5"
echo "  实验 6 (PID $PID6): 组合方案 → GPU 6"
echo ""
echo "⏱️  预计完成时间：~3 分钟（所有实验并行）"
echo ""
echo "📁 日志位置：~/VoxG/logs/experiments/parallel/"
echo ""
echo "📊 WandB 追踪：https://wandb.ai/HCCS/VoxG"
echo ""

# 等待所有实验完成
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

echo ""
echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
