# Delta Vector + 自适应滤波：重新审视平滑性

## 核心想法

Delta Vector 本质上是 ISP（Individual Smoothing Patterns）的简化版本。SmoothGNN 证明异常节点 ISP 更高（更难平滑），但 Delta Vector + 简单方法失败的原因是：

1. **方向信息丢失**：马氏距离只考虑幅度，不考虑方向
2. **没有自适应**：不同数据集/节点需要不同的频率响应
3. **缺少学习**：简单方法无法学习复杂的正常模式

**核心假设**：如果将 Delta Vector 与自适应谱滤波结合，可能达到 RHO 级别的性能，同时保持简单性。

## 直觉来源

1. **今天的实验**：Delta Vector + 马氏距离在 Elliptic 上 AUC=0.34（极差），而 RHO 达到 0.85
2. **SmoothGNN 论文**：异常节点在大多数 hop 上 ISP 更高（Figure 1, 2）
3. **RHO 论文**：AdaFreq 可以自适应学习频率响应，处理多样 homophily

## 潜在价值

- [ ] 能解决什么问题？→ 简化异常检测流程，减少超参数
- [ ] 为什么比现有方法更好？→ 保持简单性的同时达到 SOTA 性能
- [ ] 预期收益是什么？→ 更快、更稳定的训练，更好的泛化

## 理论可行性

| 问题 | 状态 | 说明 |
|------|------|------|
| 有理论支撑吗？ | ✅ | SmoothGNN 证明 ISP 与异常相关 |
| 与现有理论冲突吗？ | ❌ | 与谱方法一致 |
| 需要什么假设？ | ❓ | 假设异常节点的 Delta Vector 有可学习的模式 |

## 实施可行性

| 问题 | 状态 | 说明 |
|------|------|------|
| 技术上能实现吗？ | ✅ | Delta Vector 已实现，只需添加自适应滤波 |
| 数据/资源够吗？ | ✅ | 已有 6 个数据集 + GPU |
| 时间成本合理吗？ | ✅ | 预计 1-2 周实现原型 |

## 关键洞察

### 1. Delta Vector = 简化的 ISP

```
ISP: ‖(P^t - P^∞)x‖²     # 节点与收敛状态的距离
Delta Vector: X_p^{k+1} - X_p^k   # 相邻 hop 的差异
```

**关系**：Delta Vector 是 ISP 的"局部版本"，衡量的是传播过程中的变化。

### 2. RHO 成功的关键

```python
# RHO 的 AdaFreq
g(λ) = 1 - kλ   # k 可学习

# 当 k > 0：低通滤波（正常节点 = 低频）
# 当 k < 0：高通滤波（异常节点 = 高频？）
```

**关键**：自适应学习不同的频率响应！

### 3. 为什么 Delta Vector + 马氏距离失败

| 问题 | 说明 |
|------|------|
| 无自适应 | 所有数据集用同样的马氏距离 |
| 无学习 | 没有学习什么是"正常" |
| 方向丢失 | 只看幅度，不看变化方向 |

## 潜在方案

### 方案 A：Delta Vector + AdaFreq

```python
# 1. 计算 Delta Vectors
delta = [X_k+1 - X_k for k in range(K)]

# 2. 应用 AdaFreq（学习 k 参数）
filtered_delta = adaptive_filter(delta, learnable_k)

# 3. 正常性学习
normal_score = one_class_loss(filtered_delta)
```

### 方案 B：Delta Vector 作为 ISP 的代理

```python
# 直接用 Delta Vector 近似 ISP
ISP_approx = sum(||delta_k|| for k in range(K))

# 异常分数 = ISP（越高越异常）
anomaly_score = ISP_approx
```

### 方案 C：Delta Vector + 双视图学习

```python
# Cross-channel view
delta_cross = compute_delta(X)

# Channel-wise view
delta_channel = compute_delta(X_per_channel)

# GNA 对齐
consistency = align(delta_cross, delta_channel)
```

## 相关文献

- SmoothGNN (WWW 2025) - ISP/NSP 定义
- RHO (NeurIPS 2025) - AdaFreq 自适应滤波
- BWGNN - 谱方法在 GAD 的应用

## 关联 Idea

- Delta Token Ablation（已完成，concat 效果最好）

## 状态

- [x] 💡 火花（刚提出）
- [ ] 🔬 探索中（正在验证）
- [ ] 📋 可行方案（已验证，待实施）
- [ ] 🚀 实施中
- [ ] ✅ 完成
- [ ] ❌ 放弃（记录原因）

---

## 记录

### 2026-03-26 讨论

**发现的问题**：
- Delta Vector + 简单方法在 Elliptic 上 AUC=0.34（极差）
- VoxGFormer 在 Elliptic 上 AUC=0.72（未达 SOTA 0.85）
- 正常节点和异常节点在嵌入空间高度重叠

**新的洞察**：
- SmoothGNN 证明异常节点 ISP 更高 = 更难平滑
- RHO 的成功关键：自适应滤波 + 双视图学习
- Delta Vector 本质上是 ISP 的简化版本

**下一步**：
1. 阅读更多谱方法论文（BWGNN, AMNet）
2. 实现 AdaFreq 的简化版本
3. 测试 Delta Vector + AdaFreq 的效果

---

_创建时间: 2026-03-26_