#!/bin/bash
# 实验脚本：测试位置编码创新
# 在 GPU 1-5 上并行运行

cd /root/gpufree-data/linziyao/VoxG

echo "=========================================="
echo "Nexus 创新实验 - Cosine 位置编码"
echo "时间: $(date)"
echo "=========================================="

# 实验配置
DATASETS=("photo" "amazon" "questions")
DEVICES=(1 2 3)
PE_TYPES=("cosine" "learnable" "none")

# 运行基线实验 (无位置编码)
echo ""
echo "--- 基线实验 (无位置编码) ---"
for i in ${!DATASETS[@]}; do
    DATASET=${DATASETS[$i]}
    DEVICE=${DEVICES[$i]}
    echo "运行 $DATASET 在 GPU $DEVICE (基线)"
    CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
        --dataset $DATASET \
        --device 0 \
        --num_epoch 100 \
        --seed 0 \
        --use_pe False \
        > logs/exp_pe_baseline_${DATASET}.log 2>&1 &
done
wait

# 运行 Cosine 位置编码实验
echo ""
echo "--- Cosine 位置编码实验 ---"
for i in ${!DATASETS[@]}; do
    DATASET=${DATASETS[$i]}
    DEVICE=${DEVICES[$i]}
    echo "运行 $DATASET 在 GPU $DEVICE (Cosine PE)"
    CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
        --dataset $DATASET \
        --device 0 \
        --num_epoch 100 \
        --seed 0 \
        --use_pe True \
        --pe_type cosine \
        > logs/exp_pe_cosine_${DATASET}.log 2>&1 &
done
wait

# 运行可学习位置编码实验
echo ""
echo "--- 可学习位置编码实验 ---"
for i in ${!DATASETS[@]}; do
    DATASET=${DATASETS[$i]}
    DEVICE=${DEVICES[$i]}
    echo "运行 $DATASET 在 GPU $DEVICE (Learnable PE)"
    CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
        --dataset $DATASET \
        --device 0 \
        --num_epoch 100 \
        --seed 0 \
        --use_pe True \
        --pe_type learnable \
        > logs/exp_pe_learnable_${DATASET}.log 2>&1 &
done
wait

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "结果在 logs/exp_pe_*.log 中"
echo "=========================================="

# 打印结果摘要
echo ""
echo "--- 结果摘要 ---"
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "数据集: $DATASET"
    for TYPE in baseline cosine learnable; do
        LOG_FILE="logs/exp_pe_${TYPE}_${DATASET}.log"
        if [ -f "$LOG_FILE" ]; then
            AUC=$(grep "Best AUC" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            if [ -n "$AUC" ]; then
                echo "  $TYPE: AUC = $AUC"
            else
                echo "  $TYPE: 运行中或失败"
            fi
        fi
    done
done