#!/bin/bash
# SPSE MVP 快速验证脚本
# 目标：<1 小时内完成验证

set -e

echo "=========================================="
echo "SPSE MVP 快速验证"
echo "=========================================="
echo ""

# 在 Tolokers（最小数据集）上快速验证
echo "📍 数据集：Tolokers（最小，最快）"
echo "📍 实验：基线 vs SPSE MVP"
echo ""

cd ~/VoxG

# 实验 1: 基线
echo "=========================================="
echo "实验 1: 基线（无 SPSE）"
echo "=========================================="
python3 run.py \
    --batch_size=1024 --dataset=tolokers --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=50 --peak_lr=0.0003 \
    --pp_k=3 --progregate_alpha=0.3 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer \
    --use_spse_mvp=False \
    2>&1 | tee logs/spse_mvp_baseline.log | grep -E 'AUC|AP|Best'

echo ""

# 实验 2: SPSE MVP
echo "=========================================="
echo "实验 2: SPSE MVP（+三角形计数）"
echo "=========================================="
python3 run.py \
    --batch_size=1024 --dataset=tolokers --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=50 --peak_lr=0.0003 \
    --pp_k=3 --progregate_alpha=0.3 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer \
    --use_spse_mvp=True \
    2>&1 | tee logs/spse_mvp_enabled.log | grep -E 'AUC|AP|Best'

echo ""
echo "=========================================="
echo "✅ MVP 验证完成！"
echo "=========================================="
echo ""
echo "📊 对比结果："
echo "  基线：grep 'AUC.max' logs/spse_mvp_baseline.log"
echo "  SPSE:  grep 'AUC.max' logs/spse_mvp_enabled.log"
echo ""
echo "🎯 决策标准："
echo "  AUC 提升>2% → 继续完整实现 SPSE"
echo "  AUC 提升<2% → 放弃 SPSE"
echo ""
