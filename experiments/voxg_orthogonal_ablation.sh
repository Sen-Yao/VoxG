#!/bin/bash
# VoxG 正交化消融实验
# 验证 Gram-Schmidt 正交化在 Amazon 数据集上的效果

set -e

echo "=========================================="
echo "VoxG 正交化消融实验"
echo "=========================================="
echo ""

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
echo "📍 使用 GPU: $GPU_ID"
echo ""

# ============================================
# 实验 1: Amazon 基线（VecGAD 原始配置）
# ============================================
run_amazon_baseline() {
    echo "=========================================="
    echo "📊 实验 1: Amazon 基线（VecGAD 原始）"
    echo "=========================================="
    python run.py \
        --batch_size=1024 \
        --dataset=Amazon \
        --end_lr=0.0001 \
        --lambda_rec_emb=0.1 \
        --num_epoch=100 \
        --peak_lr=0.0003 \
        --pp_k=5 \
        --progregate_alpha=0.4 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=1 \
        --seed=0 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer \
        --orthogonalize_tokens=False
}

# ============================================
# 实验 2: Amazon + 正交化
# ============================================
run_amazon_orthogonal() {
    echo "=========================================="
    echo "📊 实验 2: Amazon + Gram-Schmidt 正交化"
    echo "=========================================="
    python run.py \
        --batch_size=1024 \
        --dataset=Amazon \
        --end_lr=0.0001 \
        --lambda_rec_emb=0.1 \
        --num_epoch=100 \
        --peak_lr=0.0003 \
        --pp_k=5 \
        --progregate_alpha=0.4 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=1 \
        --seed=0 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer \
        --orthogonalize_tokens=True
}

# ============================================
# 主函数
# ============================================
case "${1:-both}" in
    baseline)
        run_amazon_baseline
        ;;
    orthogonal)
        run_amazon_orthogonal
        ;;
    both)
        echo "⚠️  运行完整消融实验（基线 + 正交化）"
        echo ""
        run_amazon_baseline
        echo ""
        run_amazon_orthogonal
        ;;
    *)
        echo "用法：$0 {baseline|orthogonal|both}"
        echo ""
        echo "示例:"
        echo "  $0 baseline    # 只运行基线"
        echo "  $0 orthogonal  # 只运行正交化"
        echo "  $0 both        # 运行完整消融实验"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 实验完成！"
echo "=========================================="
echo ""
echo "📊 查看 WandB 结果："
echo "   https://wandb.ai/HCCS/VoxG"
echo ""
