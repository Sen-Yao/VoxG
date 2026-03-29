#!/bin/bash

# V1 最佳配置 5-seed 验证
# 参数来源: nexus/benchmarks/photo.md

for SEED in 0 1 2 3 4; do
    echo "=== Seed $SEED ==="
    CUDA_VISIBLE_DEVICES=0 python run.py         --dataset=photo         --model_type=VoxGFormer         --progregate_alpha=0.4         --pp_k=5         --batch_size=1024         --peak_lr=0.0003         --rec_loss_weight=1.0         --num_epoch=200         --train_rate=0.05         --seed=$SEED         2>&1 | tee logs/photo_v1_5seed_s${SEED}.log
    echo ""
done

echo "=== 完成 ==="
