# Questions 数据集复现报告

## 实验信息

- **实验日期**: 2026-03-03
- **服务器**: HCCS86 (NVIDIA L40, 44.4 GB)
- **模型**: GGADFormer
- **数据集**: Questions

## 超参数配置

```bash
--batch_size=1024
--dataset=questions
--end_lr=0.0001
--lambda_rec_emb=0.5
--num_epoch=70
--outlier_beta=0.3
--peak_lr=0.0001
--pp_k=3
--progregate_alpha=0.3
--rec_loss_weight=0.1
--ring_R_max=0.5
--ring_R_min=0.5
--seed=0
--train_rate=0.05
--warmup_updates=50
--model_type=GGADFormer
```

## 实验结果

### 最终性能指标

| 指标 | 复现结果 | 论文目标 | 差异 |
|------|----------|----------|------|
| **AUC** | 0.5912 | 0.5842 | +0.0070 (+1.2%) |
| **AP** | 0.0433 | 0.0396 | +0.0037 (+9.3%) |

### 训练过程

- **总训练时间**: 157.65 秒
- **总 Epochs**: 70
- **训练集**: Counter({0: 2378, 1: 68})
- **测试集**: Counter({0: 40342, 1: 1241})
- **异常率**: 2.98%

### WandB Run

- **URL**: https://wandb.ai/HCCS/VoxG/runs/oxj61593

## 验证结论

### AUC 指标
- **状态**: ✅ 超过目标
- 复现结果 0.5912 优于论文目标 0.5842
- 提升约 1.2%

### AP 指标
- **状态**: ✅ 超过目标
- 复现结果 0.0433 优于论文目标 0.0396
- 提升约 9.3%

### 总体评估
- **通过验证** ✅
- AUC 和 AP 指标均达到并超过论文目标
- 最佳性能出现在训练早期（约 epoch 2），之后有所下降
- 这可能表明需要调整学习率调度或增加正则化

## 备注

Questions 数据集从 Yandex Research 的 heterophilous-graphs 仓库下载，并转换为 .mat 格式以兼容 VoxG 框架。数据集包含 48,921 个节点和 153,540 条边，节点特征维度为 301。
