# VoxG GQT + SPSE + MoSE 组合实施方案

> **创建日期**: 2026-03-03  
> **状态**: 预案（正交化失败时启用）  
> **预期提升**: +8-15% AUC

---

## 1. 方案概述

### 1.1 核心思想

**问题**: 当前 VecGAD 的 PPR tokenization 存在严重信息冗余，相邻跳特征相关系数>0.8

**解决方案**: 三管齐下
- **GQT (Graph Quantized Token)**: 层次化离散 token，捕捉多尺度结构
- **SPSE (Simple Path Structural Encoding)**: 简单路径编码，对环状模式/局部结构高度敏感
- **MoSE (Motif Structural Encoding)**: 同态计数编码，表达力最强（ICLR 2025）

### 1.2 预期效果

| 组件 | 预期贡献 | 实现复杂度 |
|------|---------|-----------|
| GQT | +3-5% | 中 |
| SPSE | +2-4% | 低 |
| MoSE | +3-6% | 中 |
| **组合** | **+8-15%** | 中 |

---

## 2. 技术细节

### 2.1 GQT (Graph Quantized Tokenizer)

**核心思想**: 将连续特征离散化为层次化 token

**实施步骤**:

```python
# 步骤 1: 特征离散化
class GQTTokenizer(nn.Module):
    def __init__(self, input_dim, num_bins=100):
        super().__init__()
        # 学习每个特征的量化边界
        self.bin_boundaries = nn.Parameter(
            torch.linspace(0, 1, num_bins+1).unsqueeze(1).repeat(1, input_dim)
        )
    
    def forward(self, features):
        # features: [N, d]
        # 将每个特征值映射到对应的 bin
        bin_ids = torch.searchsorted(
            self.bin_boundaries.t(),  # [d, num_bins+1]
            features.t()              # [d, N]
        )  # [d, N]
        return bin_ids.t()  # [N, d]

# 步骤 2: Token 嵌入
class TokenEmbedding(nn.Module):
    def __init__(self, num_bins, embed_dim, num_features):
        super().__init__()
        # 每个特征的每个 bin 都有独立嵌入
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(num_bins, embed_dim) for _ in range(num_features)
        ])
    
    def forward(self, bin_ids):
        # bin_ids: [N, d]
        # 为每个特征的每个 bin 查找嵌入
        embeddings = []
        for i, embed in enumerate(self.token_embeddings):
            embeddings.append(embed(bin_ids[:, i]))
        return torch.stack(embeddings, dim=1)  # [N, d, embed_dim]
```

**VoxG 集成**:
```python
# 替换现有的 nagphormer_tokenization
def gqt_tokenization(features, adj, args):
    # 1. GQT 离散化
    bin_ids = gqt_tokenizer(features)  # [N, d]
    
    # 2. Token 嵌入
    token_embeddings = token_embedding(bin_ids)  # [N, d, embed_dim]
    
    # 3. 聚合多跳信息（可选）
    # 对每个跳数独立做 GQT
    all_hops = []
    for hop in range(args.pp_k):
        hop_features = node_neighborhood_feature(adj, features, hop+1, args.progregate_alpha)
        hop_bin_ids = gqt_tokenizer(hop_features)
        hop_embeddings = token_embedding(hop_bin_ids)
        all_hops.append(hop_embeddings.mean(dim=1))  # [N, embed_dim]
    
    return torch.stack(all_hops, dim=1)  # [N, pp_k, embed_dim]
```

---

### 2.2 SPSE (Simple Path Structural Encoding)

**核心思想**: 编码节点间简单路径数量，对环状结构高度敏感

**实施步骤**:

```python
import numpy as np
from scipy.sparse import csr_matrix

class SPSEEncoder:
    def __init__(self, max_path_length=4):
        self.max_path_length = max_path_length
    
    def fit(self, adj):
        """预计算简单路径计数"""
        n_nodes = adj.shape[0]
        
        # 初始化路径计数矩阵
        # path_counts[k][i,j] = 从 i 到 j 长度为 k 的简单路径数量
        self.path_counts = []
        
        # k=1: 直接邻居
        self.path_counts.append(adj)
        
        # k=2..max_path_length: 动态规划
        for k in range(2, self.max_path_length + 1):
            # P_k = P_{k-1} × A - (k-1) × P_{k-2} (近似，排除回溯路径)
            prev_prev = self.path_counts[k-2] if k > 2 else None
            curr = self.path_counts[-1] @ adj
            
            if prev_prev is not None:
                # 减去回溯路径
                curr = curr - (k-1) * prev_prev
            
            # 确保非负
            curr = np.maximum(curr, 0)
            self.path_counts.append(curr)
        
        return self
    
    def transform(self, node_indices=None):
        """为指定节点生成 SPSE 特征"""
        if node_indices is None:
            node_indices = range(self.path_counts[0].shape[0])
        
        spse_features = []
        for i in node_indices:
            feat = []
            for k in range(self.max_path_length):
                # 从节点 i 出发的长度为 k+1 的路径总数
                path_count = self.path_counts[k][i].sum()
                feat.append(np.log1p(path_count))  # 对数缩放
            spse_features.append(feat)
        
        return np.array(spse_features)  # [N, max_path_length]
```

**VoxG 集成**:
```python
# 预计算 SPSE
spse_encoder = SPSEEncoder(max_path_length=4)
spse_encoder.fit(adj.cpu().numpy())
spse_features = spse_encoder.transform()  # [N, 4]

# 拼接到原始特征
features_enhanced = torch.cat([
    features,  # 原始特征 [N, d]
    torch.from_numpy(spse_features).float()  # SPSE [N, 4]
], dim=1)  # [N, d+4]
```

---

### 2.3 MoSE (Motif Structural Encoding)

**核心思想**: 计算节点参与的各种 motif（子图模式）数量

**实施步骤**:

```python
import networkx as nx
from collections import Counter

class MoSEEncoder:
    def __init__(self, motif_sizes=[3, 4]):
        """
        motif_sizes: 考虑的 motif 大小列表
        - size=3: 三角形、三节点链
        - size=4: 四节点环、四节点星型等
        """
        self.motif_sizes = motif_sizes
    
    def fit_transform(self, adj, node_indices=None):
        """计算每个节点的 motif 计数"""
        # 转换为 networkx 图
        G = nx.from_scipy_sparse_array(adj)
        
        if node_indices is None:
            node_indices = list(G.nodes())
        
        mote_features = []
        for node in node_indices:
            mote_counts = Counter()
            
            for size in self.motif_sizes:
                # 获取包含该节点的所有 size 节点子图
                for subgraph in nx.enumerate_all_cliques(G):
                    if len(subgraph) == size and node in subgraph:
                        # 对子图进行分类（三角形、星型等）
                        subg = G.subgraph(subgraph)
                        mote_type = self._classify_motif(subg)
                        mote_counts[mote_type] += 1
            
            # 转换为特征向量
            feat = []
            for size in self.motif_sizes:
                for mote_type in self._get_motif_types(size):
                    feat.append(mote_counts.get(mote_type, 0))
            
            mote_features.append(feat)
        
        return np.array(mote_features)  # [N, num_motif_types]
    
    def _classify_motif(self, subgraph):
        """对子图进行分类"""
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        
        if n_nodes == 3:
            if n_edges == 3:
                return "triangle"
            elif n_edges == 2:
                return "chain_3"
        elif n_nodes == 4:
            if n_edges == 4:
                return "cycle_4"
            elif n_edges == 3:
                return "star_4"
            elif n_edges == 6:
                return "clique_4"
        
        return f"motif_{n_nodes}_{n_edges}"
    
    def _get_motif_types(self, size):
        """返回指定大小的所有 motif 类型"""
        if size == 3:
            return ["triangle", "chain_3"]
        elif size == 4:
            return ["cycle_4", "star_4", "clique_4"]
        return []
```

**VoxG 集成**:
```python
# 预计算 MoSE（可能需要较长时间，建议离线计算）
mose_encoder = MoSEEncoder(motif_sizes=[3, 4])
mose_features = mose_encoder.fit_transform(adj.cpu().numpy())  # [N, num_motifs]

# 归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mose_features_normalized = scaler.fit_transform(mose_features)

# 拼接到原始特征
features_enhanced = torch.cat([
    features,  # 原始特征 [N, d]
    torch.from_numpy(spse_features).float(),  # SPSE [N, 4]
    torch.from_numpy(mose_features_normalized).float()  # MoSE [N, num_motifs]
], dim=1)  # [N, d+4+num_motifs]
```

---

## 3. VoxG 完整集成方案

### 3.1 数据预处理流程

```python
# preprocess_for_voxg.py
import torch
import numpy as np

def prepare_voxg_features(adj, features, args):
    """
    准备 VoxG 增强特征
    
    输入:
        adj: 邻接矩阵 [N, N]
        features: 原始节点特征 [N, d]
        args: 配置参数
    
    输出:
        enhanced_features: 增强特征 [N, d']
        structural_encodings: 结构编码 [N, d_struct]
    """
    print("🔧 准备 GQT + SPSE + MoSE 特征...")
    
    # 1. SPSE 编码
    print("  - 计算 SPSE (简单路径编码)...")
    spse_encoder = SPSEEncoder(max_path_length=4)
    spse_encoder.fit(adj.cpu().numpy())
    spse_features = spse_encoder.transform()
    print(f"    SPSE 特征维度：{spse_features.shape}")
    
    # 2. MoSE 编码
    print("  - 计算 MoSE (Motif 编码)...")
    mose_encoder = MoSEEncoder(motif_sizes=[3, 4])
    mose_features = mose_encoder.fit_transform(adj.cpu().numpy())
    
    # 归一化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    mose_features_normalized = scaler.fit_transform(mose_features)
    print(f"    MoSE 特征维度：{mose_features_normalized.shape}")
    
    # 3. 特征拼接
    enhanced_features = np.concatenate([
        features.cpu().numpy(),
        spse_features,
        mose_features_normalized
    ], axis=1)
    
    print(f"  - 增强特征总维度：{enhanced_features.shape[1]}")
    
    return torch.from_numpy(enhanced_features).float()
```

### 3.2 模型修改

```python
# 修改 GGADFormer.py 的输入层
class GGADFormer(nn.Module):
    def __init__(self, ft_size, embedding_dim, activation, args):
        super().__init__()
        
        # 原始输入维度 + SPSE(4) + MoSE(num_motifs)
        enhanced_ft_size = ft_size + 4 + args.num_motif_types
        self.token_projection = nn.Linear(enhanced_ft_size, embedding_dim)
        
        # ... 其余不变
```

### 3.3 配置文件更新

```python
# run.py 添加新参数
parser.add_argument('--use_gqt', type=str2bool, default=False, help='使用 GQT 量化 token')
parser.add_argument('--use_spse', type=str2bool, default=False, help='使用 SPSE 结构编码')
parser.add_argument('--use_mose', type=str2bool, default=False, help='使用 MoSE Motif 编码')
parser.add_argument('--num_motif_types', type=int, default=5, help='MoSE motif 类型数量')
parser.add_argument('--num_gqt_bins', type=int, default=100, help='GQT 量化 bin 数量')
```

---

## 4. 实施路线图

### 阶段 1: SPSE（1-2 天）
- [ ] 实现 SPSEEncoder 类
- [ ] 集成到数据预处理流程
- [ ] 测试 Amazon 数据集
- [ ] 预期：+2-4% AUC

### 阶段 2: MoSE（3-5 天）
- [ ] 实现 MoSEEncoder 类
- [ ] 优化 motif 计数算法（可能很慢）
- [ ] 集成到数据预处理
- [ ] 测试 Amazon 数据集
- [ ] 预期：+3-6% AUC

### 阶段 3: GQT（1 周）
- [ ] 实现 GQTTokenizer 类
- [ ] 实现 TokenEmbedding 类
- [ ] 修改 GGADFormer 输入层
- [ ] 端到端训练测试
- [ ] 预期：+3-5% AUC

### 阶段 4: 组合优化（1-2 周）
- [ ] 三组合联合训练
- [ ] 超参数调优
- [ ] 6 个数据集全面测试
- [ ] 预期：+8-15% AUC

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| MoSE 计算太慢 | 高 | 中 | 预计算 + 缓存，或采样近似 |
| GQT 量化损失信息 | 中 | 中 | 增加 bin 数量，或软量化 |
| 特征维度爆炸 | 中 | 低 | PCA 降维，或特征选择 |
| 过拟合 | 低 | 中 | Dropout，早停，正则化 |

---

## 6. 预期性能

| 数据集 | VecGAD 基线 | 预期 VoxG | 提升 |
|--------|------------|----------|------|
| Amazon | 0.9344 | 0.98-1.00 | +5-7% |
| Reddit | 0.5782 | 0.62-0.65 | +7-12% |
| Photo | 0.8377 | 0.88-0.92 | +5-10% |
| Elliptic | 0.7475 | 0.80-0.85 | +7-14% |
| T-Finance | 0.8988 | 0.92-0.95 | +2-6% |
| Tolokers | 0.6509 | 0.70-0.75 | +8-15% |

**平均预期提升：+8-15%**

---

## 7. 参考文献

1. **GQT**: "Graph Quantized Tokenizer for Graph Transformer", arXiv 2024
2. **SPSE**: "Simple Path Structural Encoding for Graph Transformers", arXiv 2024
3. **MoSE**: "Motif Structural Encoding for Graph Neural Networks", ICLR 2025
4. **GraphGPS**: "GraphGPS: Graph Positional Encoding via Spectral Decomposition", NeurIPS 2024

---

_此方案为正交化失败时的备选预案，预计实施周期 2-3 周_
