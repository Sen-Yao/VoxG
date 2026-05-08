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

## ✅ 当前最佳：delta 模式 (2026-03-30)

### 最佳配置 (5-seed 验证)

| 参数 | 值 |
|------|-----|
| **AUC** | **0.8653±0.0268** |
| **AP** | 0.5415±0.0401 |
| token_mode | **delta** |
| progregate_alpha | 0.1 |
| pp_k | 6 |
| batch_size | 256 |
| peak_lr | 0.0003 |
| rec_loss_weight | 2.0 |

### 代码版本

- Commit: ce43e7d
- Message: 📊 补充 Photo/Amazon benchmark 的 Git Commit 信息（审计）
- Time: 2026-03-29 15:30:52 +0800

### WandB

- Sweep ID: 4sqj9r3i
- URL: https://wandb.ai/Senyao/VoxG/sweeps/4sqj9r3i

---

## original 模式 (2026-03-29)

### 最佳配置 (5-seed 验证)

| 参数 | 值 |
|------|-----|
| **AUC** | **0.8316±0.0132** |
| **AP** | 0.5138±0.0329 |
| token_mode | original |
| progregate_alpha | 0.1 |
| pp_k | 6 |
| batch_size | 256 |
| peak_lr | 0.0003 |
| rec_loss_weight | 1.0 |

### 代码版本

- Commit: d73ba8e
- Message: 🔧 清理项目结构 + 修复 import 错误 + V3 Sweep 完成
- Time: 2026-03-29 11:09:12 +0800

---

## SOTA 对比

| 方法 | AUC | 差距 |
|------|-----|------|
| **VoxG (delta)** | **0.8653±0.0268** | - |
| VecGAD (reproduced, 5-seed) | **0.9104±0.0218** | SOTA reference |
| VoxG (original) | 0.8316±0.0132 | -3.4% vs delta |

### token_mode 对比

| token_mode | AUC | 说明 |
|------------|-----|------|
| original | 0.8316±0.0132 | 基线 |
| **delta** | **0.8653±0.0268** | **+3.4%** ✨ |
| concat | - | 待测试 |

---

## 历史记录 (已更正)

### V1 Sweep (2026-03-26)

| 指标 | 记录值 | 实际值 |
|------|--------|--------|
| 最高 AUC | ~~0.8966~~ | **0.8727** (单 seed) |
| 数据集 | ~~Photo~~ | Amazon (记录错误) |

**注意**: V1 记录存在错误，0.8966 实际来自 Amazon 数据集。

---

## 待改进

- [x] delta 模式比 original 提升 3.4%
- [x] 已在 VecGAD 基线上复现 `0.9104±0.0218`（2026-04-23）
- [ ] 测试 concat 模式
- [ ] 尝试更小的 train_rate（5%）
- [ ] 调整 embedding_dim

---

_最后更新: 2026-03-30 09:45_
_状态: 有效 (delta 模式最佳)_


## VecGAD reproduction (2026-04-23)

### 正确复现超参数

| 参数 | 值 |
|------|-----|
| batch_size | 128 |
| num_epoch | 200 |
| peak_lr | 0.0005 |
| end_lr | 0.0001 |
| pp_k | 6 |
| progregate_alpha | 0.05 |
| train_rate | 0.05 |
| warmup_updates | 50 |
| lambda_rec_emb | 0.1 |
| rec_loss_weight | 1 |
| ring_R_min | 0.3 |
| ring_R_max | 1 |
| ring_loss_weight | 1 |
| seed | [0,1,2,3,4] |

### 复现结果

| 指标 | 数值 |
|------|------|
| **AUC** | **0.9104±0.0218** |
| **AP** | **0.6509±0.0706** |
| 最佳单 seed AUC | 0.9454 (seed=1) |
| Sweep ID | `pmlnea98` |

### 说明

此前错误 baseline 使用了不匹配的超参数（尤其是 `batch_size=8192`、`peak_lr=0.001`、`progregate_alpha=0.1`），导致结果只有 `0.6946±0.0534`。按 `reproduction.sh` 中 photo 配置重跑后，已成功复现到 0.89+ 水平。


### baseline 入口

- VecGAD baseline reproduction script: `scripts/reproduce_vecgad_baseline.sh`
