# VoxG 项目细节文档

> 由 Nexus 科研代理自动生成 | 最后更新：2026-03-18

---

## 1. 项目概述

**项目名称**：VoxG (原名 GGADFormer / VecFormer / VecGAD)

**核心目标**：设计基于图 Transformer 的半监督图异常检测方法，在主流数据集上达到 SOTA

**核心创新**：
- 解耦式输入编码（图拓扑与节点表征分离）
- 生成式伪异常样本合成（解决标签稀缺）
- Transformer 架构（避免 GNN 过平滑问题）
- 仅需 5% 标注率实现优异性能

---

## 2. 架构详解

### 2.1 核心模型：GGADFormer



**关键组件**：

1. **nagphormer_tokenization()**
   - 将图结构编码为 token 序列
   - 使用 Personalized PageRank (PPR) 采样邻居
   - 输出形状：[N, pp_k+1, feature_dim]

2. **MultiHeadAttention**
   - num_heads = 2 (默认)
   - 输入维度: embedding_dim (256)
   - 注意力 dropout: 0.4

3. **FeedForwardNetwork**
   - hidden_size → ffn_size → hidden_size
   - 激活函数：GELU
   - ffn_dim = 256 (默认)

4. **Discriminator Head**
   - FC1: embedding_dim → embedding_dim/2
   - FC2: embedding_dim/2 → embedding_dim/4
   - FC3: embedding_dim/4 → 1 (logits)

### 2.2 损失函数组合

| 损失 | 权重 | 作用 |
|------|------|------|
| BCE Loss | 1.0 | 二分类异常检测 |
| Reconstruction Loss | 1.0 | 输入重构约束 |
| Ring Loss | 1.0 | 中心点对齐 |
| Contrastive Loss | 0.1 | 对比学习增强 |

### 2.3 训练流程



---

## 3. 数据集支持

### 已支持数据集

| 数据集 | 节点数 | 特征维度 | 异常比例 | 说明 |
|--------|--------|----------|----------|------|
| photo | 7,650 | 745 | ~1% | Amazon Photo |
| amazon | 11,944 | 100 | ~2% | Amazon Computer |
| reddit | 22,167 | 604 | ~2% | Reddit Social |
| elliptic | 4,640 | 93 | ~30% | 比特币交易 |
| t_finance | 39,357 | 79 | ~4% | 金融交易 |
| tolokers | 11,758 | 10 | ~1% | 众包平台 |
| questions | 39,721 | 10 | ~1% | 问答平台 |
| dgraph | 3,694,482 | 61 | ~2% | 大规模图 |

### 数据加载流程

1.  - 加载 .mat 格式数据集
2.  - 加载大规模 DGraph 数据集
3. 数据预处理：
   - 特征归一化
   - 邻接矩阵处理
   - 标签分割 (train/val/test)

---

## 4. 超参数配置

### 默认配置（推荐）



### 超参搜索建议

| 参数 | 搜索范围 | 优先级 |
|------|----------|--------|
| peak_lr | [1e-5, 5e-4] | 高 |
| embedding_dim | [128, 256, 512] | 中 |
| GT_num_layers | [2, 3, 4] | 中 |
| GT_dropout | [0.2, 0.4, 0.6] | 低 |

---

## 5. 基线方法

### GGAD 目录下实现

| 方法 | 论文/来源 | 文件 |
|------|-----------|------|
| GGAD | NeurIPS 2024 |  |
| TAM | 亲和力方法 |  |
| AEGIS | 对抗式 |  |
| AnomalyDAE | 自编码器 |  |
| DOMINANT | 图自编码器 |  |
| GAAN | 生成对抗 |  |
| OCGNN | 一类分类 GNN |  |

---

## 6. 待优化项

### 高优先级
- [ ] 多 GPU 训练支持 (DataParallel / DDP)
- [ ] 大规模数据集内存优化

### 中优先级
- [ ] 混合精度训练 (AMP)
- [ ] 动态损失权重调整

### 低优先级
- [ ] 分布式数据加载
- [ ] 模型压缩/量化

---

## 7. 关键文件索引



---

## 8. 常见问题解答

### Q: 训练时 CUDA OOM 怎么办？
A: 减小 （推荐 64-256），或使用  监控内存

### Q: AUC 很低怎么办？
A: 检查：
1. 数据集异常比例是否合理
2. 学习率是否过高（尝试 1e-5）
3. epoch 是否足够

### Q: 如何添加新数据集？
A: 在  中添加新的  函数，确保返回格式与  一致

---

*本文档由 Nexus 自动维护，如有更新请同步修改*
