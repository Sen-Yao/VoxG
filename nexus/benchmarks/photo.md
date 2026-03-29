# Photo 基准测试

## 数据集信息

| 属性 | 值 |
|------|-----|
| 节点数 | 7,535 |
| 边数 | 119,081 |
| 平均度 | **31.6** (密集图) |
| 特征维度 | **745** (高维) |
| 特征稀疏性 | **65.1%** |
| 异常比例 | 9.26% |

---

## Sweep V1 结果 (2026-03-26)

### 最佳配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **AUC** | **0.8966** | 超过 SOTA 0.8960 |
| progregate_alpha | 0.4 | 密集图用更大 alpha |
| pp_k | 5 | 密集图用较少层 |
| batch_size | 1024 | 高维稀疏特征用大 batch |
| peak_lr | 0.0003 | - |
| rec_loss_weight | 1.0 | - |

### 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --dataset=photo \
    --model_type=VoxGFormer \
    --progregate_alpha=0.4 \
    --pp_k=5 \
    --batch_size=1024 \
    --peak_lr=0.0003 \
    --rec_loss_weight=1.0 \
    --num_epoch=200 \
    --train_rate=0.05
```

### 5-seed 验证

```bash
# 运行 5 seed 验证
for SEED in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --dataset=photo \
        --progregate_alpha=0.4 \
        --pp_k=5 \
        --batch_size=1024 \
        --peak_lr=0.0003 \
        --rec_loss_weight=1.0 \
        --num_epoch=200 \
        --train_rate=0.05 \
        --seed=$SEED
done
```

---

## SOTA 对比

| 方法 | AUC | 设置 |
|------|-----|------|
| **VoxG (Sweep V1)** | **0.8966** | 5% 半监督 |
| VecGAD | 0.8960 | 5% 半监督 |

---

## 关键发现

1. **密集图需要更大 alpha**: alpha=0.4 > 0.2
2. **密集图用较少 pp_k**: pp_k=5 < 6
3. **高维稀疏特征用大 batch**: batch=1024 > 128

---

_最后更新: 2026-03-26_
