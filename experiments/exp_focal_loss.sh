#!/bin/bash
# 实验脚本：测试 Focal Loss 效果
# 在多个数据集上对比 BCE vs Focal Loss

cd /root/gpufree-data/linziyao/VoxG

echo "=========================================="
echo "Focal Loss 对比实验"
echo "时间: $(date)"
echo "=========================================="

# 创建 losses 目录
mkdir -p losses

# 实验配置
DATASETS=("photo" "reddit" "elliptic")
DEVICES=(1 2 3)

# 运行基线实验 (BCE Loss)
echo ""
echo "--- 基线实验 (BCE Loss) ---"
for i in ${!DATASETS[@]}; do
    DATASET=${DATASETS[$i]}
    DEVICE=${DEVICES[$i]}
    echo "运行 $DATASET 在 GPU $DEVICE (基线 BCE)"
    CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
        --dataset $DATASET \
        --device 0 \
        --num_epoch 100 \
        --seed 0 \
        --batch_size 128 \
        --train_rate 0.05 \
        --use_focal_loss False \
        > logs/exp_focal_baseline_${DATASET}.log 2>&1 &
done
wait

echo ""
echo "基线实验已启动，等待完成..."
sleep 5

# 打印基线结果
for DATASET in "${DATASETS[@]}"; do
    LOG_FILE="logs/exp_focal_baseline_${DATASET}.log"
    if [ -f "$LOG_FILE" ]; then
        AUC=$(grep "Best AUC" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -n "$AUC" ]; then
            echo "  $DATASET (BCE): Best AUC = $AUC"
        else
            echo "  $DATASET (BCE): 运行中..."
        fi
    fi
done

echo ""
echo "=========================================="
echo "实验说明:"
echo "1. 基线使用标准 BCE Loss"
echo "2. Focal Loss 需要修改 run.py 添加 --use_focal_loss 参数"
echo "3. 查看 logs/exp_focal_*.log 获取详细结果"
echo "=========================================="