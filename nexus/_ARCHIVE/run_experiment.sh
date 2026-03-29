#!/bin/bash
# Nexus 实验启动脚本
# 用法: ./run_experiment.sh <seed> <gpu> [tokens] [init] [dropout] [warmup] [lr]

SEED=${1:-0}
GPU=${2:-1}
TOKENS=${3:-1}
INIT=${4:-layer_aware}
DROPOUT=${5:-0.1}
WARMUP=${6:-50}
LR=${7:-5e-4}

cd /root/gpufree-data/linziyao/VoxG

CUDA_VISIBLE_DEVICES=$GPU nohup python run.py \
  --dataset reddit \
  --train_rate 0.05 \
  --model_type VoxGFormer \
  --use_deep_prompt True \
  --deep_prompt_layers 0,1,2 \
  --deep_prompt_tokens_per_layer $TOKENS \
  --deep_prompt_init $INIT \
  --deep_prompt_dropout $DROPOUT \
  --warmup_updates $WARMUP \
  --peak_lr $LR \
  --seed $SEED \
  --num_epoch 200 \
  --batch_size 512 \
  > logs/nexus_t${TOKENS}_${INIT}_d${DROPOUT}_w${WARMUP}_lr${LR}_s${SEED}.log 2>&1 &

echo "Started: tokens=$TOKENS, init=$INIT, dropout=$DROPOUT, warmup=$WARMUP, lr=$LR, seed=$SEED, GPU=$GPU"
