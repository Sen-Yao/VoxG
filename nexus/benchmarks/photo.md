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
| VecGAD | 0.8960 | **-3.1%** ⚠️ |
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
- [ ] 当前性能距 SOTA 差 3.1%
- [ ] 测试 concat 模式
- [ ] 尝试更小的 train_rate（5%）
- [ ] 调整 embedding_dim

---

_最后更新: 2026-03-30 09:45_
_状态: 有效 (delta 模式最佳)_
