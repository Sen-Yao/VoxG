#!/bin/bash
# VoxG 软正交化 + 正交正则化消融实验

set -e

echo "=========================================="
echo "VoxG 软正交化 + 正交正则化消融实验"
echo "=========================================="
echo ""

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
echo "📍 使用 GPU: $GPU_ID"
echo ""

# ============================================
# 实验 1: 基线（无正交化）
# ============================================
run_baseline() {
    echo "=========================================="
    echo "📊 实验 1: 基线（无正交化）"
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
# 实验 2: 软正交化 (beta=0.5)
# ============================================
run_soft_orth_beta50() {
    echo "=========================================="
    echo "📊 实验 2: 软正交化 (beta=0.5)"
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
        --orthogonalize_tokens=True \
        --orthogonal_beta=0.5
}

# ============================================
# 实验 3: 软正交化 (beta=0.3)
# ============================================
run_soft_orth_beta30() {
    echo "=========================================="
    echo "📊 实验 3: 软正交化 (beta=0.3)"
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
        --orthogonalize_tokens=True \
        --orthogonal_beta=0.3
}

# ============================================
# 实验 4: 正交正则化 (lambda=0.01)
# ============================================
run_orth_reg_001() {
    echo "=========================================="
    echo "📊 实验 4: 正交正则化 (lambda=0.01)"
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
        --orthogonalize_tokens=False \
        --lambda_orthogonal=0.01
}

# ============================================
# 实验 5: 正交正则化 (lambda=0.1)
# ============================================
run_orth_reg_01() {
    echo "=========================================="
    echo "📊 实验 5: 正交正则化 (lambda=0.1)"
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
        --orthogonalize_tokens=False \
        --lambda_orthogonal=0.1
}

# ============================================
# 实验 6: 组合方案（软正交化 + 正则化）
# ============================================
run_combined() {
    echo "=========================================="
    echo "📊 实验 6: 组合方案（软正交化 beta=0.3 + 正则化 lambda=0.01）"
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
        --orthogonalize_tokens=True \
        --orthogonal_beta=0.3 \
        --lambda_orthogonal=0.01
}

# ============================================
# 主函数
# ============================================
case "${1:-all}" in
    baseline)
        run_baseline
        ;;
    soft50)
        run_soft_orth_beta50
        ;;
    soft30)
        run_soft_orth_beta30
        ;;
    reg001)
        run_orth_reg_001
        ;;
    reg01)
        run_orth_reg_01
        ;;
    combined)
        run_combined
        ;;
    all)
        echo "⚠️  运行完整消融实验（6 个实验）"
        echo ""
        run_baseline
        echo ""
        run_soft_orth_beta50
        echo ""
        run_soft_orth_beta30
        echo ""
        run_orth_reg_001
        echo ""
        run_orth_reg_01
        echo ""
        run_combined
        ;;
    *)
        echo "用法：$0 {baseline|soft50|soft30|reg001|reg01|combined|all}"
        echo ""
        echo "示例:"
        echo "  $0 baseline    # 只运行基线"
        echo "  $0 soft50      # 软正交化 beta=0.5"
        echo "  $0 all         # 运行完整消融实验"
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
