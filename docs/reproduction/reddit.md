# Reddit 数据集复现报告

## 实验信息

- **实验日期**: 2026-03-03
- **服务器**: HCCS86 (NVIDIA L40, 44.4 GB)
- **模型**: GGADFormer
- **数据集**: Reddit

## 超参数配置

由于原始配置 (batch_size=32768) 导致显存不足，调整为 batch_size=2048：

```bash
--batch_size=2048
--dataset=reddit
--end_lr=0.0003
--lambda_rec_emb=2
--num_epoch=150
--outlier_beta=0.3
--peak_lr=0.0005
--pp_k=6
--progregate_alpha=0.6
--rec_loss_weight=1
--ring_R_max=1
--ring_R_min=0.3
--seed=1
--train_rate=0.05
--warmup_updates=50
--model_type=GGADFormer
```

## 实验结果

### 最终性能指标

| 指标 | 复现结果 | 论文目标 | 差异 |
|------|----------|----------|------|
| **AUC** | 0.5654 | 0.5782 | -0.0128 (-2.2%) |
| **AP** | 0.0465 | 0.0441 | +0.0024 (+5.4%) |

### 训练过程

- **总训练时间**: 54.62 秒
- **总 Epochs**: 150
- **训练集**: Counter({0.0: 530, 1.0: 19})
- **测试集**: Counter({0.0: 9026, 1.0: 311})

### WandB Run

- **URL**: https://wandb.ai/HCCS/GGADFormer/runs/umnu5u77

## 验证结论

### AUC 指标
- **状态**: ⚠️ 略低于目标
- 复现结果 0.5654 与论文目标 0.5782 相差 2.2%
- 可能原因：batch_size 从 32768 降低到 2048，影响了梯度估计的稳定性

### AP 指标
- **状态**: ✅ 超过目标
- 复现结果 0.0465 优于论文目标 0.0441
- 提升约 5.4%

### 总体评估
- **部分通过验证** ✅
- AP 指标达到并超过论文目标
- AUC 指标接近目标（差距在可接受范围内）
- batch_size 调整可能是性能差异的主要原因

## 备注

原始 batch_size=32768 配置在 44.4 GB 显存的 NVIDIA L40 上仍出现 OOM 错误，因此使用 batch_size=2048 完成训练。建议使用梯度累积或其他内存优化技术来复现原始配置。
