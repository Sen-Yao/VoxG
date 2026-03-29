#!/bin/bash
# Photo 数据集 5-seed 验证
# 使用 Sweep V1 最佳配置

set -e

echo "=========================================="
echo "Photo 5-seed 验证"
echo "配置: alpha=0.4, pp_k=5, batch=1024, lr=0.0003"
echo "=========================================="

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
RESULTS=()

for SEED in 0 1 2 3 4; do
    echo ""
    echo "--- Seed $SEED ---"
    CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
        --dataset=photo \
        --model_type=VoxGFormer \
        --progregate_alpha=0.4 \
        --pp_k=5 \
        --batch_size=1024 \
        --peak_lr=0.0003 \
        --rec_loss_weight=1.0 \
        --num_epoch=200 \
        --train_rate=0.05 \
        --seed=$SEED \
        --token_mode=original \
        2>&1 | tee /tmp/photo_seed${SEED}.log
    
    # 提取 AUC
    AUC=$(grep "AUC" /tmp/photo_seed${SEED}.log | tail -1 | grep -oP "AUC.*?=\s*\K[\d.]+")
    RESULTS+=($AUC)
    echo "Seed $SEED AUC: $AUC"
done

echo ""
echo "=========================================="
echo "5-seed 结果汇总"
echo "=========================================="
for i in 0 1 2 3 4; do
    echo "Seed $i: ${RESULTS[$i]}"
done

# 计算 mean±std
python3 -c "
import numpy as np
results = [${RESULTS[0]}, ${RESULTS[1]}, ${RESULTS[2]}, ${RESULTS[3]}, ${RESULTS[4]}]
print(f"Mean±Std: {np.mean(results):.4f} ± {np.std(results):.4f}")
"
