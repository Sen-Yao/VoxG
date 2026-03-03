#!/bin/bash
# 测试正交正则化是否生效

set -e

echo "=========================================="
echo "测试正交正则化代码是否生效"
echo "=========================================="
echo ""

# 测试 1: lambda=1.0（应该有明显影响）
echo "📍 测试 1: lambda=1.0（大值，应该有影响）"
CUDA_VISIBLE_DEVICES=7 python run.py \
    --batch_size=1024 --dataset=Amazon --end_lr=0.0001 \
    --lambda_rec_emb=0.1 --num_epoch=20 --peak_lr=0.0003 \
    --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 \
    --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 \
    --seed=0 --train_rate=0.05 --warmup_updates=50 \
    --model_type=GGADFormer --orthogonalize_tokens=False \
    --lambda_orthogonal=1.0 \
    2>&1 | tee logs/test_orth_lambda1.log | grep -E '正交正则化调试|AUC|AP' | head -20

echo ""
echo "✅ 测试完成！查看日志：logs/test_orth_lambda1.log"
echo ""
echo "如果看到 '正交正则化调试：loss_orth=X.XXXXXX' 说明代码生效"
echo "如果 AUC/AP 与基线不同，说明正则化有影响"
