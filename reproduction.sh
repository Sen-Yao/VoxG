#!/bin/bash
# VoxG Reproduction Script
# 基于 MatrixGAD 超参数配置，针对 VecGAD/VoxG 优化

set -e

echo "=========================================="
echo "VoxG 性能复现脚本"
echo "基于 MatrixGAD 超参数配置"
echo "=========================================="
echo ""

# 默认使用 GPU 0，可通过环境变量覆盖
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
echo "📍 使用 GPU: $GPU_ID"
echo ""

# ============================================
# Amazon 数据集 - 目标 AUC: 0.94
# ============================================
run_amazon() {
    echo "=========================================="
    echo "📊 运行 Amazon 数据集 (目标 AUC: 0.94)"
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
        --model_type=GGADFormer
}

# ============================================
# Reddit 数据集 - 目标 AUC: 0.95
# ============================================
run_reddit() {
    echo "=========================================="
    echo "📊 运行 Reddit 数据集 (目标 AUC: 0.95)"
    echo "=========================================="
    python run.py \
        --batch_size=32768 \
        --dataset=reddit \
        --end_lr=0.0003 \
        --lambda_rec_emb=2 \
        --num_epoch=150 \
        --outlier_beta=0.3 \
        --peak_lr=0.0005 \
        --pp_k=6 \
        --progregate_alpha=0.6 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=20 \
        --seed=1 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# Photo 数据集 - 目标 AUC: 0.91
# ============================================
run_photo() {
    echo "=========================================="
    echo "📊 运行 Photo 数据集 (目标 AUC: 0.91)"
    echo "=========================================="
    python run.py \
        --batch_size=128 \
        --dataset=photo \
        --end_lr=1e-4 \
        --lambda_rec_emb=0.1 \
        --num_epoch=200 \
        --peak_lr=5e-4 \
        --pp_k=6 \
        --progregate_alpha=0.2 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=1 \
        --seed=2 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# Elliptic 数据集 - 目标 AUC: 0.95
# ============================================
run_elliptic() {
    echo "=========================================="
    echo "📊 运行 Elliptic 数据集 (目标 AUC: 0.95)"
    echo "=========================================="
    python run.py \
        --batch_size=32768 \
        --dataset=elliptic \
        --end_lr=0.0003 \
        --lambda_rec_emb=2 \
        --num_epoch=150 \
        --outlier_beta=0.3 \
        --peak_lr=0.0005 \
        --pp_k=7 \
        --progregate_alpha=0.6 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=20 \
        --seed=0 \
        --train_rate=0.1 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# T-Finance 数据集 - 目标 AUC: 0.85
# ============================================
run_tfinance() {
    echo "=========================================="
    echo "📊 运行 T-Finance 数据集 (目标 AUC: 0.85)"
    echo "=========================================="
    python run.py \
        --batch_size=8192 \
        --dataset=t_finance \
        --end_lr=0.0001 \
        --lambda_rec_emb=0.1 \
        --num_epoch=40 \
        --outlier_beta=0.3 \
        --peak_lr=0.0005 \
        --pp_k=7 \
        --progregate_alpha=0.3 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.5 \
        --ring_loss_weight=1 \
        --seed=0 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# Tolokers 数据集 - 目标 AUC: 0.98
# ============================================
run_tolokers() {
    echo "=========================================="
    echo "📊 运行 Tolokers 数据集 (目标 AUC: 0.98)"
    echo "=========================================="
    python run.py \
        --batch_size=1024 \
        --dataset=tolokers \
        --end_lr=0.0001 \
        --lambda_rec_emb=0.5 \
        --num_epoch=70 \
        --outlier_beta=0.3 \
        --peak_lr=0.0001 \
        --pp_k=3 \
        --progregate_alpha=0.3 \
        --rec_loss_weight=0.1 \
        --ring_R_max=0.5 \
        --ring_R_min=0.5 \
        --ring_loss_weight=20 \
        --seed=0 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# Reddit 数据集 - 目标 AUC: 0.5782
# ============================================
run_reddit() {
    echo "=========================================="
    echo "📊 运行 Reddit 数据集 (目标 AUC: 0.5782)"
    echo "=========================================="
    python run.py \
        --batch_size=32768 \
        --dataset=reddit \
        --end_lr=0.0003 \
        --lambda_rec_emb=2 \
        --num_epoch=150 \
        --outlier_beta=0.3 \
        --peak_lr=0.0005 \
        --pp_k=6 \
        --progregate_alpha=0.6 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=20 \
        --seed=1 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# Elliptic 数据集 - 目标 AUC: 0.7475
# ============================================
run_elliptic() {
    echo "=========================================="
    echo "📊 运行 Elliptic 数据集 (目标 AUC: 0.7475)"
    echo "=========================================="
    python run.py \
        --batch_size=32768 \
        --dataset=elliptic \
        --end_lr=0.0003 \
        --lambda_rec_emb=2 \
        --num_epoch=150 \
        --outlier_beta=0.3 \
        --peak_lr=0.0005 \
        --pp_k=7 \
        --progregate_alpha=0.6 \
        --rec_loss_weight=1 \
        --ring_R_max=1 \
        --ring_R_min=0.3 \
        --ring_loss_weight=20 \
        --seed=0 \
        --train_rate=0.1 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# Questions 数据集 - 目标 AUC: 0.5842
# ============================================
run_questions() {
    echo "=========================================="
    echo "📊 运行 Questions 数据集 (目标 AUC: 0.5842)"
    echo "=========================================="
    python run.py \
        --batch_size=1024 \
        --dataset=questions \
        --end_lr=0.0001 \
        --lambda_rec_emb=0.5 \
        --num_epoch=70 \
        --outlier_beta=0.3 \
        --peak_lr=0.0001 \
        --pp_k=3 \
        --progregate_alpha=0.3 \
        --rec_loss_weight=0.1 \
        --ring_R_max=0.5 \
        --ring_R_min=0.5 \
        --ring_loss_weight=20 \
        --seed=0 \
        --train_rate=0.05 \
        --warmup_updates=50 \
        --model_type=GGADFormer
}

# ============================================
# 主函数 - 根据参数运行指定数据集
# ============================================
case "${1:-all}" in
    amazon)
        run_amazon
        ;;
    reddit)
        run_reddit
        ;;
    photo)
        run_photo
        ;;
    elliptic)
        run_elliptic
        ;;
    t_finance)
        run_tfinance
        ;;
    tolokers)
        run_tolokers
        ;;
    questions)
        run_questions
        ;;
    all)
        echo "⚠️  运行所有数据集（可能需要数小时）"
        echo "💡 提示：可以指定单个数据集，如 ./reproduction.sh amazon"
        echo ""
        run_amazon
        echo ""
        run_photo
        echo ""
        run_tfinance
        echo ""
        run_tolokers
        echo ""
        run_questions
        echo ""
        # Reddit 和 Elliptic 需要大量显存，按需启用
        echo "⚠️  Reddit 和 Elliptic 需要大量显存，跳过。如需运行请手动执行"
        ;;
    full_all)
        echo "⚠️  运行完整所有 7 个数据集（可能需要 10+ 分钟）"
        run_amazon
        echo ""
        run_reddit
        echo ""
        run_photo
        echo ""
        run_elliptic
        echo ""
        run_tfinance
        echo ""
        run_tolokers
        echo ""
        run_questions
        ;;
    *)
        echo "用法：$0 {amazon|reddit|photo|elliptic|t_finance|tolokers|questions|all|full_all}"
        echo ""
        echo "示例:"
        echo "  $0 amazon     # 只运行 Amazon"
        echo "  $0 photo      # 只运行 Photo"
        echo "  $0 all        # 运行 5 个轻量数据集"
        echo "  $0 full_all   # 运行所有 7 个数据集"
        echo ""
        echo "环境变量:"
        echo "  CUDA_VISIBLE_DEVICES - 指定 GPU (默认：0)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 运行完成！"
echo "=========================================="
echo ""
echo "📊 查看 WandB 结果："
echo "   https://wandb.ai/HCCS/GGADFormer"
echo ""
