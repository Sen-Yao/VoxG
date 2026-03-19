#!/bin/bash
# Cosine Positional Encoding Experiment - Parallel Version
# 派遣时间: 2026-03-19 15:17
# 目标: 并行验证 Cosine 位置编码对图 Transformer 的改进效果

set -e
cd ~/VoxG

# 创建日志目录
mkdir -p logs

# 实验 1: Photo + Cosine PE (GPU 2)
echo "=========================================="
echo "实验 1: Photo 数据集 - Cosine PE (GPU 2)"
echo "=========================================="
nohup bash -c 'export CUDA_VISIBLE_DEVICES=2; python run.py \
    --dataset=photo \
    --train_rate=0.05 \
    --seed=0 \
    --num_epoch=100 \
    --batch_size=4096 \
    --peak_lr=0.0005 \
    --device=0 \
    --use_pe True \
    --pe_type=cosine' > logs/photo_cosine_pe.log 2>&1 &
PID1=$!
echo "PID1=$PID1"

# 实验 2: Reddit + Cosine PE (GPU 4)
echo "=========================================="
echo "实验 2: Reddit 数据集 - Cosine PE (GPU 4)"
echo "=========================================="
nohup bash -c 'export CUDA_VISIBLE_DEVICES=4; python run.py \
    --dataset=reddit \
    --train_rate=0.05 \
    --seed=789 \
    --num_epoch=100 \
    --batch_size=512 \
    --peak_lr=0.0005 \
    --device=0 \
    --use_pe True \
    --pe_type=cosine' > logs/reddit_cosine_pe.log 2>&1 &
PID2=$!
echo "PID2=$PID2"

# 实验 3: Photo + Learnable PE (GPU 5)
echo "=========================================="
echo "实验 3: Photo 数据集 - Learnable PE (GPU 5)"
echo "=========================================="
nohup bash -c 'export CUDA_VISIBLE_DEVICES=5; python run.py \
    --dataset=photo \
    --train_rate=0.05 \
    --seed=0 \
    --num_epoch=100 \
    --batch_size=4096 \
    --peak_lr=0.0005 \
    --device=0 \
    --use_pe True \
    --pe_type=learnable' > logs/photo_learnable_pe.log 2>&1 &
PID3=$!
echo "PID3=$PID3"

echo "=========================================="
echo "所有实验已启动！"
echo "PID1=$PID1 (Photo+Cosine, GPU 2)"
echo "PID2=$PID2 (Reddit+Cosine, GPU 4)"
echo "PID3=$PID3 (Photo+Learnable, GPU 5)"
echo "查看日志: tail -f logs/*.log"
echo "=========================================="

# 等待所有进程完成
wait $PID1 $PID2 $PID3
echo "所有实验完成！"