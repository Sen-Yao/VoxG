# VecGAD: Vectorized Discrepancy-Guided Graph Anomaly Detection

> **论文来源**: KDD '26 (under review)
> **本地笔记**: knowledge/Research/Papers/2026/anonymous2026vecgad/

---

## 核心思想

**从标量重构误差到向量化的偏离方向信号**

传统重构方法将重构误差压缩为标量分数，丢失了方向信息。VecGAD 发现：

> **重构误差向量 = 方向性的异常偏离指南针**

它指明了节点在哪个方向上、以何种方式偏离了"正常性流形"。

---

## 方法架构

### 1. 结构感知图 Tokenization

```
X_p^k = Â^k · X_0           # k-hop 传播
X_p^{k+1} = (1-α)X_p^k + αX_0   # 残差保留原始特征
```

每个节点得到 Token 序列：`[X_0, X_p^1, X_p^2, ..., X_p^K]`

### 2. 重构差异向量 (RDV)

```python
# 自编码器重构
h_i = Encoder(T_i)           # 压缩嵌入
T̂_i = Decoder(h_i)           # 重构 Token

# 重构差异向量
R_i = T_i - T̂_i              # 方向性偏离信号
```

### 3. 伪异常生成

```python
# 投影到嵌入空间
p_i = P(flatten(R_i))        # 可学习投影层

# 沿偏离方向扰动
h̃_i = h_i + β · p_i          # β 控制异常强度
```

### 4. 超球壳约束 (HSC)

将伪异常约束在距离质心 `[R_min, R_max]` 的球壳内：

```
L_HSC = max(0, R_min - ||h̃ - c||) + max(0, ||h̃ - c|| - R_max)
```

**目的**: 确保生成的伪异常是"有挑战性的硬负样本"，而非无意义离群点。

---

## 与其他方法对比

| 方法 | 伪异常来源 | 标签依赖 | 标签稀缺时表现 |
|------|-----------|---------|---------------|
| **GGAD** | 已标注正常节点的邻居结构 | 高 | < 5% 标签时显著退化 |
| **RHO** | 在已标注集上优化谱滤波器 | 高 | < 5% 标签时显著退化 |
| **VecGAD** | 重构差异向量（从所有节点学习） | 低 | 稳定 |

**关键优势**: 重构过程从**所有节点**学习，不依赖标签数量。

---

## 性能表现

### Tolokers 数据集（不同标签比例）

| 标签比例 | GGAD | RHO | VecGAD |
|---------|------|-----|--------|
| 15% | 较好 | 较好 | **最优** |
| 5% | 退化 | 退化 | **稳定** |
| 1% | 失效 | 失效 | **稳定** |

---

## VoxG 项目关联

**VoxG = VecGAD + PromptGAD 改进**

### 已验证的发现

1. **Delta Vector 效果** (insights/2026-03-23-delta-vector-findings.md)
   - Delta 向量（相邻 Hop 特征差）在 photo 上 AUC=1.0
   - 与 VecGAD 的 RDV 思想一致：方向信息 > 标量范数

2. **Token 相似度问题** (insights/2026-03-23-delta-ablation-*.md)
   - 原版 VecGAD Token 相似度 > 0.95
   - Prompt-GAD 引入 Signed Attention 解决

### 待探索

- [ ] Delta Vector 与 RDV 的关系？
- [ ] 能否用 Delta Vector 替代/增强 RDV？
- [ ] HSC 约束 vs 简化版中心对齐？

---

## 参考文献

```bibtex
@inproceedings{anonymous2026vecgad,
  title={Leveraging Vectorized Discrepancy for Label-Efficient Graph Anomaly Detection},
  author={Anonymous Authors},
  booktitle={KDD},
  year={2026}
}
```

---

_整合自: knowledge/Research/Papers/2026/anonymous2026vecgad/ 及项目实践_