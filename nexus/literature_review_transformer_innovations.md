# 图 Transformer 架构创新文献调研报告

**项目**: VoxG (图异常检测)  
**创建时间**: 2026-03-19  
**调研范围**: 注意力机制、Tokenization、损失函数创新

---

## 目录

1. [优先级 1：注意力机制改进](#优先级-1注意力机制改进)
   - 1.1 Sparse Attention
   - 1.2 RWPE (Random Walk Position Encoding)
   - 1.3 Graph-Structured Attention
2. [优先级 2：Tokenization 创新](#优先级-2tokenization-创新)
   - 2.1 Node-level Tokenization
   - 2.2 Subgraph Tokenization
   - 2.3 Dynamic Token Length
3. [优先级 3：损失函数创新](#优先级-3损失函数创新)
   - 3.1 Contrastive Learning
   - 3.2 Focal Loss
   - 3.3 Self-supervised Objectives
4. [综合建议与实施路线图](#综合建议与实施路线图)

---

## 优先级 1：注意力机制改进

### 1.1 Sparse Attention (稀疏注意力)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **Exphormer: Sparse Transformers for Graphs** | ICML 2023 | 基于扩展图(Expander Graph)和虚拟全局节点的稀疏注意力，O(n)复杂度 | [arXiv:2303.06147](https://arxiv.org/abs/2303.06147) |
| **SFi-Former: Sparse Flow Induced Attention** | ICLR 2025 | 网络流优化的稀疏注意力，通过 ℓ1-norm 正则化学习稀疏模式 | [OpenReview](https://openreview.net/forum?id=bWc6O8QSyp) |
| **GraphGPS: Recipe for General, Powerful, Scalable Graph Transformer** | NeurIPS 2022 | 解耦局部消息传递与全局注意力，O(N+E)线性复杂度 | [arXiv:2205.12454](https://arxiv.org/abs/2205.12454) |

#### 核心技术详解

**Exphormer** 的稀疏注意力机制包含三个关键组件：
1. **Local Attention**: 仅在原图边上进行注意力计算
2. **Global Nodes**: 虚拟全局节点连接所有节点，实现全局信息传递
3. **Expander Edges**: 基于扩展图的随机边，保证图的高连通性和谱扩展性质

```python
# Exphormer 核心结构示意
ExphormerLayer(
    local_attention,      # 原图邻接矩阵上的注意力
    global_nodes=4,       # 虚拟全局节点数量
    expander_degree=3     # 扩展图度数
)
```

**SFi-Former** 创新点：
- 将稀疏注意力问题建模为网络流优化问题
- 通过最小化能量函数学习最优稀疏模式
- 特别适合长程依赖关系的图数据

#### 可行性评估：⭐⭐⭐⭐⭐ (高度可行)

**适配优势**：
- VoxG 当前使用标准 Self-Attention，可直接替换
- 稀疏注意力与 PPR tokenization 的多跳结构天然兼容
- 无需修改 tokenization 模块

**技术挑战**：
- 需要实现虚拟全局节点与扩展图边的构建
- 超参数调优（global_nodes 数量、expander_degree）

#### 预期收益

| 指标 | 当前状态 | 预期改进 |
|------|---------|---------|
| 计算复杂度 | O(N²) | O(N) 或 O(N log N) |
| 大图可扩展性 | 受限 | 支持 10x-100x 规模 |
| 长程依赖建模 | 弱 | 显著增强 |

#### 实现复杂度：中等

**修改范围**：
- `GGADFormer.py`: 替换 TransformerEncoder 为 ExphormerLayer
- 新增模块: `sparse_attention.py`（Exphormer/SFi-Former 实现）
- 代码量估计: 200-400 行

---

### 1.2 RWPE (Random Walk Position Encoding)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **Graph Neural Networks with Learnable Structural and Positional Representations** | NeurIPS 2021 | 分离结构编码与位置编码，使用 RWPE 作为位置信息 | [arXiv:2110.07875](https://arxiv.org/abs/2110.07875) |
| **Graph Positional Encoding via Random Feature Propagation** | ICML 2023 | 随机特征传播实现高效位置编码 | [PMLR](https://proceedings.mlr.press/v202/eliasof23a/) |
| **Benchmarking Positional Encodings for GNNs and Graph Transformers** | arXiv 2024 | 系统对比多种位置编码方法 | [arXiv:2411.12732](https://arxiv.org/abs/2411.12732) |

#### 核心技术详解

**RWPE 计算公式**：
$$RW = A \cdot D^{-1}$$
$$PE_i = [RW^1_{ii}, RW^2_{ii}, ..., RW^k_{ii}]$$

其中 $A$ 为邻接矩阵，$D$ 为度矩阵，$RW^k$ 表示随机游走矩阵的 k 次幂。

**Laplacian PE 对比**：
| 特性 | RWPE | Laplacian PE |
|------|------|--------------|
| 计算复杂度 | O(k·E) | O(N³) 特征分解 |
| 可扩展性 | 高 | 低 |
| 表达能力 | 随机游走结构 | 全局拓扑 |
| 符号稳定性 | 稳定 | 需要处理符号模糊 |

#### 可行性评估：⭐⭐⭐⭐ (可行)

**适配优势**：
- VoxG 当前使用 PPR (Personalized PageRank)，与 RWPE 计算相似
- 可复用现有的邻居聚合代码框架
- 计算开销相对较小

**技术挑战**：
- RWPE 与当前 PPR tokenization 的协同设计
- 位置编码与 hop-level token 的融合策略

#### 预期收益

| 指标 | 预期改进 |
|------|---------|
| 位置感知能力 | +5-10% AUROC |
| 训练稳定性 | 改善（消除位置模糊） |
| 推理速度 | 轻微下降（额外编码计算） |

#### 实现复杂度：低-中等

**修改范围**：
- `utils.py`: 添加 RWPE 计算函数
- `GGADFormer.py`: 在 token embedding 中加入位置编码
- 代码量估计: 50-100 行

---

### 1.3 Graph-Structured Attention (图结构感知注意力)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **TransGAD: A Transformer-Based Autoencoder for Graph Anomaly Detection** | DASFAA 2024 | 将节点视为序列，邻居作为 token，余弦位置编码 | [OpenReview](https://openreview.net/forum?id=x9zTk330hJ) |
| **GTAT: Empowering GNNs with Cross Attention** | Scientific Reports 2025 | 拓扑特征与节点特征的交叉注意力 | [Nature](https://www.nature.com/articles/s41598-025-88993-3) |
| **Graph Attention Networks** | ICLR 2018 | 基于图结构的注意力权重分配 | [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) |

#### 核心技术详解

**TransGAD 的节点序列化方法**：
```python
# TransGAD 风格的节点 tokenization
node_sequence = [
    center_node_embedding,           # 中心节点
    neighbor_1_embedding,            # 一跳邻居
    neighbor_2_embedding,            # 二跳邻居
    ...
]
positional_encoding = cosine_pe(len(node_sequence))
```

**关键创新**：
1. **Cosine Positional Encoding**: 使用余弦函数编码序列位置
2. **Masking Strategy**: 随机遮蔽部分邻居，增强鲁棒性
3. **Reconstruction Error**: 用于异常检测评分

#### 可行性评估：⭐⭐⭐⭐⭐ (高度可行)

**适配优势**：
- TransGAD 直接针对图异常检测设计
- 与 VoxG 的 autoencoder 架构高度兼容
- Cosine PE 实现简单

**技术挑战**：
- 需要设计有效的 masking 策略
- 与现有重构损失的结合方式

#### 预期收益

| 指标 | 当前状态 | 预期改进 |
|------|---------|---------|
| AUROC | 基准 | +2-5% |
| 过平滑问题 | 存在 | 显著缓解 |
| 异常区分度 | 中等 | 增强 |

#### 实现复杂度：中等

**修改范围**：
- `nagphormer_tokenization.py`: 借鉴 TransGAD 的 tokenization
- `GGADFormer.py`: 添加 Cosine PE 和 masking
- 代码量估计: 100-200 行

---

## 优先级 2：Tokenization 创新

### 2.1 Node-level Tokenization (节点级分词)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **NAGphormer: A Tokenized Graph Transformer** | ICLR 2023 | Hop2Token 模块，多跳邻居聚合为 token 序列 | [arXiv:2206.04910](https://arxiv.org/abs/2206.04910) |
| **NTFormer: A Composite Node Tokenized Graph Transformer** | arXiv 2024 | Node2Par 模块，多种 token 元素组合 | [arXiv:2406.19249](https://arxiv.org/abs/2406.19249) |
| **NodePiece: Tokenizing Knowledge Graphs** | ICLR 2022 | 锚点+距离+关系的组合 tokenization | [GitHub](https://migalkin.github.io/posts/2021/06/24/post/) |

#### 核心技术详解

**NAGphormer Hop2Token**：
```python
# 当前 VoxG 使用类似方法 (PPR)
token_sequence = [
    h_0,              # 0-hop (自身)
    h_1,              # 1-hop 聚合
    h_2,              # 2-hop 聚合
    ...
    h_k               # k-hop 聚合
]
# shape: [N, k+1, feature_dim]
```

**NTFormer Node2Par**：
- 多视角 token 生成
- 不同 token 元素组合（特征、结构、位置）
- 无需图特定修改的 Transformer backbone

**NodePiece 优势**：
- 参数高效：固定大小的 token 序列
- 归纳式设计：可处理未见节点
- 适用于大规模知识图谱

#### 可行性评估：⭐⭐⭐⭐⭐ (高度可行)

**适配优势**：
- VoxG 已实现类似 NAGphormer 的 PPR tokenization
- NTFormer 的 Node2Par 可直接借鉴增强

**技术挑战**：
- 多种 token 元素的融合策略
- 固定 token 数量 vs 动态调整

#### 预期收益

| 指标 | 预期改进 |
|------|---------|
| 特征表达能力 | +3-8% AUROC |
| 异构图处理 | 增强 |
| 参数效率 | 提升 |

#### 实现复杂度：低

**修改范围**：
- 主要是增强现有 tokenization 模块
- 代码量估计: 50-100 行

---

### 2.2 Subgraph Tokenization (子图级分词)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **Rethinking Tokenizer and Decoder in Masked Graph Modeling** | NeurIPS 2023 | 子图级 token 重构优于节点级 | [arXiv:2310.14753](https://arxiv.org/abs/2310.14753) |
| **Subgraph Neural Networks** | NeurIPS 2020 | 子图级别的表示学习 | [Harvard](https://zitniklab.hms.harvard.edu/projects/SubGNN/) |
| **TurboGAE: Subgraph-Optimized Graph Autoencoder** | JCIM 2025 | 子图优化的图自编码器 | [ACS](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02536) |

#### 核心技术详解

**子图级 Tokenization 策略**：
1. **Motif-based**: 基于图模体（三角形、环等）划分子图
2. **Functional Group**: 化学分子中的官能团作为子图单元
3. **Random Walk Subgraph**: 随机游走采样生成子图

**重构目标对比**：
| Token 类型 | 表达能力 | 计算开销 | 适用场景 |
|-----------|---------|---------|---------|
| Node | 低 | 低 | 简单图 |
| Edge | 中 | 中 | 边重要场景 |
| Subgraph | 高 | 高 | 复杂结构图 |

#### 可行性评估：⭐⭐⭐ (中等可行)

**适配挑战**：
- 异常检测关注节点级异常，子图级可能过度平滑
- 需要设计子图到节点的映射机制
- 计算开销增加

**潜在应用**：
- 结构异常检测（异常子结构）
- 作为辅助信号增强节点表示

#### 预期收益

| 指标 | 预期改进 |
|------|---------|
| 结构异常检测 | +5-10% AUROC |
| 计算开销 | +30-50% |

#### 实现复杂度：高

**修改范围**：
- 新增子图采样/分割模块
- 修改 tokenization 流程
- 代码量估计: 300-500 行

---

### 2.3 Dynamic Token Length (动态 Token 长度)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **GLFormer: Adaptive Token Mixing for Dynamic Link Prediction** | arXiv 2025 | 自适应 token mixer，无注意力架构 | [arXiv:2511.12442](https://arxiv.org/abs/2511.12442) |
| **Efficient Transformers with Dynamic Token Pooling** | ACL 2023 | 动态 token 池化，变长序列处理 | [ACL](https://aclanthology.org/2023.acl-long.353.pdf) |
| **Dynamic Tokenization** | EmergentMind | 动态分词综述 | [EmergentMind](https://www.emergentmind.com/topics/dynamic-tokenization) |

#### 核心技术详解

**动态 Token 长度策略**：
1. **基于节点度数**：高度节点使用更多 token
2. **基于重要性**：重要节点获得更长序列
3. **自适应池化**：动态合并/分割 token

```python
# 动态 token 长度示意
def dynamic_token_length(node_degree, max_tokens=10):
    """根据节点度数动态调整 token 数量"""
    return min(max(node_degree // 5, 3), max_tokens)
```

#### 可行性评估：⭐⭐⭐ (中等可行)

**适配挑战**：
- 需要 padding/masking 处理变长序列
- batch 化处理复杂度增加
- 与现有 mini-batch 训练流程兼容性问题

**潜在收益**：
- 高度节点获得更丰富的上下文
- 低度节点减少噪声干扰

#### 预期收益

| 指标 | 预期改进 |
|------|---------|
| 高度节点 AUROC | +3-5% |
| 低度节点 AUROC | +2-3% |
| 计算效率 | 可能下降 |

#### 实现复杂度：中-高

**修改范围**：
- tokenization 模块：动态序列生成
- 训练流程：padding/masking 逻辑
- 代码量估计: 200-400 行

---

## 优先级 3：损失函数创新

### 3.1 Contrastive Learning (对比学习)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **CVGAD: Rethinking Contrastive Learning in GAD** | arXiv 2025 | 清洁视图增强，渐进式净化模块 | [arXiv:2505.18002](https://arxiv.org/abs/2505.18002) |
| **AD-GCL: Revisiting Graph Contrastive Learning on Anomaly Detection** | AAAI 2025 | 结构不平衡视角下的对比学习 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33415) |
| **Deep Graph Level Anomaly Detection with Contrastive Learning** | Scientific Reports 2022 | 图级对比学习异常检测 | [Nature](https://www.nature.com/articles/s41598-022-22086-3) |

#### 核心技术详解

**CVGAD 的清洁视图框架**：
```python
# 清洁视图增强对比学习
class CVGAD:
    def forward(self, graph):
        # 1. 多尺度异常感知模块
        interference_edges = self.anomaly_awareness(graph)
        
        # 2. 渐进式净化
        clean_graph = self.progressive_purification(graph, interference_edges)
        
        # 3. 对比学习
        pos_pairs = [(node, clean_subgraph)]
        neg_pairs = [(node, perturbed_subgraph)]
        loss = contrastive_loss(pos_pairs, neg_pairs)
```

**AD-GCL 的结构不平衡处理**：
- 区分头部节点和尾部节点的对比学习策略
- 自适应采样平衡不同度数节点

#### 可行性评估：⭐⭐⭐⭐ (可行)

**适配优势**：
- 与 VoxG 的重构损失可并行使用
- 半监督设置下可利用对比信号
- 增强正常/异常样本区分度

**技术挑战**：
- 干扰边的识别与去除
- 对比样本的构建策略
- 与现有损失函数的权重平衡

#### 预期收益

| 指标 | 预期改进 |
|------|---------|
| AUROC | +2-5% |
| 类别不平衡处理 | 改善 |
| 表征质量 | 提升 |

#### 实现复杂度：中等

**修改范围**：
- `run.py`: 添加对比损失
- `GGADFormer.py`: 对比学习模块
- 代码量估计: 150-250 行

---

### 3.2 Focal Loss (焦点损失)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **Focal Loss for Dense Object Detection** | ICCV 2017 | 降低易分类样本权重，聚焦难分类样本 | 原始论文 |
| **Batch-balanced Focal Loss** | PMC 2023 | 批次平衡的 Focal Loss | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/) |
| **Imbalanced Graph Learning via Mixed Entropy Minimization** | Scientific Reports 2024 | 图不平衡学习的混合熵最小化 | [Nature](https://www.nature.com/articles/s41598-024-75999-6) |

#### 核心技术详解

**Focal Loss 公式**：
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中：
- $p_t$: 模型对真实类别的预测概率
- $\alpha_t$: 类别平衡权重
- $\gamma$: 聚焦参数（通常取 2）

**对 BCE Loss 的改进**：
```python
# 标准 BCE Loss
bce_loss = -y * log(p) - (1-y) * log(1-p)

# Focal Loss
focal_loss = -alpha * (1-p)^gamma * y * log(p) 
           - (1-alpha) * p^gamma * (1-y) * log(1-p)
```

#### 可行性评估：⭐⭐⭐⭐⭐ (高度可行)

**适配优势**：
- 实现极其简单（几行代码）
- 即插即用，无需架构修改
- 对类别不平衡问题直接有效

**技术挑战**：
- $\alpha$ 和 $\gamma$ 参数需要调优
- 与其他损失的权重平衡

#### 预期收益

| 数据集类型 | 预期改进 |
|-----------|---------|
| 高度不平衡（如 Reddit） | +3-8% AUROC |
| 中度不平衡 | +1-3% AUROC |
| 平衡数据集 | 轻微影响 |

#### 实现复杂度：极低

**修改范围**：
- `run.py`: 替换 BCE Loss 为 Focal Loss
- 代码量估计: 10-20 行

---

### 3.3 Self-supervised Objectives (自监督目标)

#### 关键论文

| 论文 | 会议/期刊 | 核心思想 | 链接 |
|------|----------|---------|------|
| **Towards Automated SSL for Unsupervised GAD** | DAMI 2025 | SSL 策略自动选择，避免标签泄露 | [arXiv:2501.14694](https://arxiv.org/abs/2501.14694) |
| **APF: Anomaly-Aware Pre-Training and Fine-Tuning** | arXiv 2025 | 异常感知预训练框架 | [arXiv:2504.14250](https://arxiv.org/abs/2504.14250) |
| **Federated GAD via Contrastive SSL** | TNNLS 2024 | 联邦对比自监督学习 | [GitHub](https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection) |

#### 核心技术详解

**自监督策略分类**：

| 类型 | 方法 | 适用场景 |
|------|------|---------|
| **生成式** | GraphMAE, Graph AutoEncoder | 结构/属性重构 |
| **对比式** | InfoNCE, Triplet Loss | 表征学习 |
| **预测式** | 属性掩码、边预测 | 自监督预训练 |

**APF 框架**：
```python
# 异常感知预训练
pretrain_loss = (
    reconstruction_loss +      # 重构损失
    contrastive_loss +         # 对比损失
    anomaly_awareness_loss     # 异常感知损失
)

# 微调阶段
finetune_loss = supervised_loss + pseudo_anomaly_loss
```

**关键洞察**：
- SSL 策略选择、超参数调优、组合权重是三大关键因素
- 避免标签信息泄露导致的性能高估
- 使用内部评估策略进行超参数选择

#### 可行性评估：⭐⭐⭐⭐ (可行)

**适配优势**：
- VoxG 已有 autoencoder 重构损失
- 可添加多种自监督辅助任务
- 与半监督设置天然契合

**技术挑战**：
- 多任务的权重平衡
- 预训练与微调的迁移策略
- 计算开销增加

#### 预期收益

| 指标 | 预期改进 |
|------|---------|
| 少样本场景 AUROC | +5-15% |
| 标签效率 | 提升 |
| 泛化能力 | 增强 |

#### 实现复杂度：中等

**修改范围**：
- `GGADFormer.py`: 添加自监督任务头
- `run.py`: 多任务训练逻辑
- 代码量估计: 200-350 行

---

## 综合建议与实施路线图

### 推荐优先级

| 优先级 | 方向 | 预期收益 | 实现复杂度 | 推荐指数 |
|--------|------|---------|-----------|---------|
| 🥇 | Focal Loss | 高（不平衡数据） | 极低 | ⭐⭐⭐⭐⭐ |
| 🥈 | Exphormer 稀疏注意力 | 中-高 | 中等 | ⭐⭐⭐⭐⭐ |
| 🥉 | Cosine PE (TransGAD) | 中 | 低 | ⭐⭐⭐⭐ |
| 4 | RWPE 位置编码 | 中 | 低-中 | ⭐⭐⭐⭐ |
| 5 | 对比学习损失 | 中 | 中等 | ⭐⭐⭐⭐ |
| 6 | Node2Par tokenization | 中 | 低 | ⭐⭐⭐ |
| 7 | 自监督预训练 | 高（少样本） | 中等 | ⭐⭐⭐ |
| 8 | 子图 tokenization | 中 | 高 | ⭐⭐ |
| 9 | 动态 token 长度 | 中 | 高 | ⭐⭐ |

### 实施路线图

#### Phase 1: 快速收益 (1-2 周)

```python
# 1. Focal Loss 替换 BCE
# 2. 添加 Cosine Positional Encoding
# 3. 可学习位置编码实验
```

**预期收益**: +2-5% AUROC（不平衡数据集）

#### Phase 2: 架构升级 (2-4 周)

```python
# 1. Exphormer 稀疏注意力替换
# 2. RWPE 位置编码集成
# 3. TransGAD 风格 tokenization
```

**预期收益**: +3-8% AUROC, 支持大规模图

#### Phase 3: 损失函数增强 (2-3 周)

```python
# 1. 对比学习损失
# 2. 自监督辅助任务
# 3. 多任务权重自适应调整
```

**预期收益**: +2-5% AUROC, 增强鲁棒性

### 代码修改清单

| 文件 | 修改内容 | 预计代码量 |
|------|---------|-----------|
| `GGADFormer.py` | Exphormer 层、Cosine PE、对比学习模块 | 300-500 行 |
| `utils.py` | RWPE 计算、动态 token 长度 | 100-200 行 |
| `run.py` | Focal Loss、多任务训练逻辑 | 50-100 行 |
| 新增 `sparse_attention.py` | Exphormer/SFi-Former 实现 | 200-400 行 |

### 实验验证计划

1. **基准测试**: 在 Photo, Amazon 上验证各改进独立效果
2. **消融实验**: 分析各组件贡献
3. **极限测试**: 在 Reddit, Elliptic 等困难数据集验证
4. **消融分析**: 各改进的交互效应

---

## 参考文献

### 注意力机制

1. Shirzad, H., et al. "Exphormer: Sparse Transformers for Graphs." ICML 2023.
2. SFi-Former: Sparse Flow Induced Attention for Graph Transformer. ICLR 2025.
3. Rampášek, L., et al. "Recipe for a General, Powerful, Scalable Graph Transformer." NeurIPS 2022.

### Position Encoding

4. Dwivedi, V. P., et al. "Graph Neural Networks with Learnable Structural and Positional Representations." NeurIPS 2021.
5. Eliasof, M., et al. "Graph Positional Encoding via Random Feature Propagation." ICML 2023.

### Tokenization

6. Chen, J., et al. "NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs." ICLR 2023.
7. Chen, J., et al. "NTFormer: A Composite Node Tokenized Graph Transformer." arXiv 2024.
8. Galkin, M., et al. "NodePiece: Tokenizing Knowledge Graphs." ICLR 2022.

### 异常检测

9. TransGAD: A Transformer-Based Autoencoder for Graph Anomaly Detection. DASFAA 2024.
10. VecGAD: Leveraging Vectorized Discrepancy for Label-Efficient Graph Anomaly Detection. KDD 2026 (under review).

### 损失函数与自监督

11. Cao, J., et al. "Rethinking Contrastive Learning in Graph Anomaly Detection." arXiv 2025.
12. Li, Z., et al. "Towards Automated SSL for Unsupervised GAD." DAMI 2025.
13. Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.

---

*报告生成时间: 2026-03-19*  
*调研范围: 2017-2025 年图 Transformer 相关文献*  
*文献数量: 30+ 篇核心论文*