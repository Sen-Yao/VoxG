# 创新探索计划 - VoxG 项目

**创建时间**: 2026-03-19 14:20
**阶段**: 创新探索期

---

## 一、当前架构分析

### 1.1 Tokenization (nagphormer_tokenization)
- **方法**: Personalized PageRank (PPR) 多跳聚合
- **参数**: `pp_k=6` (6跳), `progregate_alpha=0.2`
- **输出**: `[N, pp_k+1, feature_dim]` 的 token 序列

### 1.2 Transformer Encoder
- **层数**: 3 层
- **注意力头**: 2 头
- **隐藏维度**: 256
- **FFN 维度**: 256
- **Dropout**: 0.4

### 1.3 损失函数
- BCE Loss (分类)
- Reconstruction Loss (重构)
- Ring Loss (超球壳约束)

---

## 二、创新方向

### 方向 A: Tokenization 创新 [优先级: HIGH]

#### A1. 动态 Token 长度
- **问题**: 当前所有节点使用固定 7 个 token (pp_k+1)
- **思路**: 根据节点度数或重要性动态调整 token 数量
- **预期**: 高度节点获得更多上下文，低度节点减少噪声

#### A2. 可学习位置编码
- **问题**: 当前没有显式的位置编码
- **思路**: 为每个 hop 添加可学习的位置 embedding
- **预期**: 模型能更好地区分不同 hop 的信息

#### A3. 注意力引导的邻居选择
- **问题**: PPR 是静态的，无法适应不同节点
- **思路**: 用轻量级注意力网络选择重要邻居
- **预期**: 更精准的邻居信息聚合

### 方向 B: 注意力机制创新 [优先级: HIGH]

#### B1. 稀疏注意力
- **问题**: 全注意力复杂度 O(N²)
- **思路**: 引入稀疏注意力模式 (如 Longformer)
- **预期**: 支持更大规模图

#### B2. 图结构感知注意力
- **问题**: 当前注意力与图结构解耦
- **思路**: 将边信息融入注意力计算
- **预期**: 更好地利用图拓扑信息

### 方向 C: 损失函数创新 [优先级: MEDIUM]

#### C1. 对比学习损失
- **思路**: 增强正常节点与伪异常节点的区分度
- **方法**: InfoNCE 或 triplet loss

#### C2. 焦点损失
- **问题**: 类别不平衡 (异常节点稀少)
- **思路**: Focal Loss 增加难分类样本权重

---

## 三、实验计划

### 实验 1: 可学习位置编码
**修改文件**: `GGADFormer.py`
**预期工作量**: 小
**评估数据集**: Photo (快速验证)

### 实验 2: 动态 Token 长度
**修改文件**: `utils.py`, `GGADFormer.py`
**预期工作量**: 中
**评估数据集**: Photo + Amazon

### 实验 3: 对比学习损失
**修改文件**: `run.py`, `GGADFormer.py`
**预期工作量**: 中
**评估数据集**: Photo + Reddit (类别不平衡)

---

## 四、文献调研结果 (2026-03-19)

### 关键论文

| 论文 | 会议 | 创新点 | 适用性 |
|------|------|--------|--------|
| **Exphormer** | ICML 2023 | 稀疏注意力 + 虚拟全局节点，O(n) 复杂度 | ⭐⭐⭐⭐⭐ |
| **SFi-Former** | ICLR 2025 | 网络流优化的稀疏注意力，长程依赖 | ⭐⭐⭐⭐⭐ |
| **TransGAD** | 2024 | Transformer 自编码器，Cosine 位置编码 | ⭐⭐⭐⭐⭐ 直接相关！ |
| **TSAD** | 2025 | 半监督 + 自适应记忆库 + 对比学习 | ⭐⭐⭐⭐ |
| **NodePiece** | - | 固定大小 token 序列，参数高效 | ⭐⭐⭐⭐ |

### 推荐实验路线

**短期 (1-2周)**: Exphormer 稀疏注意力替换
```python
# 用 Exphormer 替换标准 Self-Attention
ExphormerLayer(
  local_attention,      # 原图邻接矩阵
  global_nodes=4,       # 虚拟全局节点
  expander_degree=3     # 扩展图度数
)
```

**中期 (2-4周)**: TransGAD 节点 tokenization
```python
# TransGAD 式节点 tokenization
node_tokens = NodeAsSequence(
  center_node=h_i,
  neighbor_tokens=[h_j for j in N(i)],
  positional_encoding=cosine_pe
)
```

**长期 (1-2月)**: TSAD 半监督学习模块
```python
# 自适应记忆库
memory_bank = AdaptiveMemory(normal_prototypes=K=64)
# 对比学习
contrastive_loss = SemiSupContrastive(threshold=0.8)
```

---

## 五、下一步行动

1. **[完成]** 文献调研 - 已完成
2. **[执行]** 实现 Exphormer 稀疏注意力实验
3. **[执行]** 添加 Cosine 位置编码

---

*计划由 Nexus 生成 | 文献调研完成 2026-03-19*