#!/bin/bash
# 正交正则化重新测试 - 并行版本（使用正确的 lambda 值）

set -e

echo "=========================================="
echo "正交正则化重新测试 - 并行版本"
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p ~/VoxG/logs/experiments/orth_reg_v2

echo "🚀 同时启动 4 个实验..."
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
    > logs/experiments/orth_reg_v2/01_baseline.log 2>&1 &
PID1=$!
echo "   PID: $PID1"

# ============================================
# 实验 2: 正交正则化 lambda=1.0 - GPU 2
# ============================================
echo "📍 实验 2: 正交正则化 lambda=1.0 → GPU 2"
CUDA_VISIBLE_DEVICES=2 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=False \
    --lambda_orthogonal=1.0 \
    > logs/experiments/orth_reg_v2/02_orth_reg_1.log 2>&1 &
PID2=$!
echo "   PID: $PID2"

# ============================================
# 实验 3: 正交正则化 lambda=10.0 - GPU 3
# ============================================
echo "📍 实验 3: 正交正则化 lambda=10.0 → GPU 3"
CUDA_VISIBLE_DEVICES=3 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=False \
    --lambda_orthogonal=10.0 \
    > logs/experiments/orth_reg_v2/03_orth_reg_10.log 2>&1 &
PID3=$!
echo "   PID: $PID3"

# ============================================
# 实验 4: 组合方案（软正交化 beta=0.3 + 正则化 lambda=1.0）- GPU 4
# ============================================
echo "📍 实验 4: 组合方案（beta=0.3 + lambda=1.0）→ GPU 4"
CUDA_VISIBLE_DEVICES=4 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=True \
    --orthogonal_beta=0.3 --lambda_orthogonal=1.0 \
    > logs/experiments/orth_reg_v2/04_combined_v2.log 2>&1 &
PID4=$!
echo "   PID: $PID4"

echo ""
echo "=========================================="
echo "✅ 所有实验已启动！"
echo "=========================================="
echo ""
echo "📊 实验列表:"
echo "  实验 1 (PID $PID1): 基线 → GPU 1"
echo "  实验 2 (PID $PID2): 正交正则化 lambda=1.0 → GPU 2"
echo "  实验 3 (PID $PID3): 正交正则化 lambda=10.0 → GPU 3"
echo "  实验 4 (PID $PID4): 组合方案 → GPU 4"
echo ""
echo "⏱️  预计完成时间：~3 分钟（所有实验并行）"
echo ""
echo "📁 日志位置：~/VoxG/logs/experiments/orth_reg_v2/"
echo ""
echo "📊 WandB 追踪：https://wandb.ai/HCCS/VoxG"
echo ""

# 等待所有实验完成
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
