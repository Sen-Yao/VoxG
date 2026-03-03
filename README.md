# VoxG (GGADFormer) - 生成式图异常检测框架

> **一句话概述**：首个基于 Transformer 的生成式半监督图异常检测模型，通过解耦式图结构感知编码和伪异常样本合成，在仅 5% 标注率下实现 SOTA 性能。

**核心创新点**：
- ✅ **解耦式输入编码**：图拓扑与节点表征分离，支持高效 mini-batch 训练
- ✅ **生成式策略**：人工合成高质量伪异常样本，解决标签稀缺问题
- ✅ **Transformer 架构**：避免传统 GNN 的过平滑问题，适用于大规模图
- ✅ **5% 标注率**：相比传统方法 15% 标注率，在极端稀疏标签下仍表现优异

---

## 📋 目录

- [快速开始](#-快速开始)
- [架构设计](#-架构设计)
- [核心文件地图](#-核心文件地图)
- [配置参数详解](#-配置参数详解)
- [常见修改位置](#-常见修改位置)
- [调试技巧](#-调试技巧)
- [VecGAD 实现状态](#-vecgad-实现状态)

---

## 🚀 快速开始

### 3 步运行

```bash
# 1️⃣ 创建环境（CUDA 11.8）
conda create -n GGADFormer python=3.8 -y
conda activate GGADFormer
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 2️⃣ 安装依赖
pip install -r requirements.txt

# 3️⃣ 运行训练
python run.py --dataset=photo --num_epoch=200 --peak_lr=3e-4 --batch_size=128
```

### 常用命令速查

| 任务 | 命令 |
|------|------|
| 基础训练 | `python run.py --dataset=photo --num_epoch=200` |
| 指定模型 | `python run.py --dataset=reddit --model_type=GGADFormer` |
| 使用 SGT | `python run.py --dataset=amazon --model_type=SGT` |
| DGraph 大数据集 | `python src/main.py --config src/dgraph.yml` |
| 超参搜索 | `python sweep_manager.py` |

---

## 🏗️ 架构设计

### 系统架构图（文字版）

```
┌─────────────────────────────────────────────────────────────────┐
│                    run.py (训练入口)                             │
│  • 参数解析 • 数据加载 • 模型初始化 • 训练循环 • 评估指标         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│ GGADFormer    │    │      SGT       │    │    Model      │
│ (Transformer) │    │  (对比模型)     │    │   (基线 GCN)  │
└───────────────┘    └────────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    utils.py     │
                    │ 数据加载/预处理  │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│ 数据集        │    │ visualization   │    │    WandB      │
│ (.mat/.npz)   │    │ (t-SNE/注意力)  │    │    (日志)     │
└───────────────┘    └────────────────┘    └───────────────┘
```

### 数据流向

```
原始数据 (.mat/.npz)
    │
    ▼
load_mat() / load_dgraph()
    │
    ├─→ adj (邻接矩阵)
    ├─→ features (节点特征)
    └─→ labels (标签)
    │
    ▼
nagphormer_tokenization()
    │
    ▼
concated_input_features [N, pp_k+1, feature_dim]
    │
    ▼
GGADFormer.TransformerEncoder()
    │
    ├─→ emb (节点嵌入)
    ├─→ outlier_emb (生成的异常嵌入)
    └─→ logits (预测分数)
    │
    ▼
损失计算 (BCE + Reconstruction + Ring)
    │
    ▼
反向传播 + 参数更新
```

### GGADFormer 模型内部结构

```
GGADFormer
├── Token Projection (输入编码)
│   └── nagphormer_tokenization() → [N, pp_k+1, feature_dim]
├── Graph Transformer Encoder × 3 层
│   ├── MultiHeadAttention (num_heads=2)
│   ├── FeedForwardNetwork (ffn_dim=256)
│   ├── LayerNorm
│   └── Dropout (0.4)
├── Token Decoder (重构)
│   └── 重构损失计算
├── Discriminator Head
│   ├── FC1: embedding_dim → embedding_dim/2
│   ├── FC2: embedding_dim/2 → embedding_dim/4
│   └── FC3: embedding_dim/4 → 1 (logits)
└── 损失函数
    ├── BCE Loss (分类)
    ├── Reconstruction Loss (重构)
    └── Ring Loss (中心点对齐)
```

---

## 🗺️ 核心文件地图

### 主要模型文件

| 文件 | 用途 | 关键类/函数 | AI 助手提示 |
|------|------|-------------|-------------|
| `run.py` | **主训练入口** | `train()`, `parse_args()` | ⚠️ 修改训练逻辑的首选位置 |
| `GGADFormer.py` | **核心模型** | `GGADFormer`, `EncoderLayer`, `MultiHeadAttention` | ✅ Transformer 架构实现 |

### 项目文档

| 文件 | 用途 |
|------|------|
| `README.md` | 项目说明（本文件） |
| `devlog/README.md` | 开发日记索引 |
| `devlog/YYYY-MM-DD.md` | 每日开发记录（反向时间线） |
| `SGT.py` | 对比模型 | `SGT` 类 | 用于消融实验 |
| `model.py` | 基线模型 | `GCN`, `Discriminator`, `Model` | 传统 GNN 基线 |

### 数据处理与工具

| 文件 | 用途 | 关键函数 | AI 助手提示 |
|------|------|----------|-------------|
| `utils.py` | **数据加载/预处理** | `load_mat()`, `load_dgraph()`, `nagphormer_tokenization()` | ⚠️ 添加新数据集时修改此处 |
| `visualization.py` | 可视化 | `create_tsne_visualization()`, `visualize_attention_weights()` | 生成图表时使用 |
| `check_gpu_memory.py` | GPU 监控 | `print_gpu_memory_usage()`, `clear_gpu_memory()` | 调试内存问题时使用 |

### 基线方法（GGAD/ 目录）

| 文件 | 方法 | 说明 |
|------|------|------|
| `GGAD/model_GGAD.py` | GGAD | 生成式半监督图异常检测 (NeurIPS 2024) |
| `GGAD/tam.py` | TAM | 基于亲和力的方法 |
| `GGAD/aegis.py` | AEGIS | 对抗式方法 |
| `GGAD/anomalyDAE.py` | AnomalyDAE | 自编码器方法 |
| `GGAD/dominant.py` | DOMINANT | 图自编码器 |
| `GGAD/gaan.py` | GAAN | 生成对抗方法 |
| `GGAD/ocgnn.py` | OCGNN | 一类分类 GNN |

### 配置与依赖

| 文件 | 用途 | AI 助手提示 |
|------|------|-------------|
| `requirements.txt` | Python 包依赖 | ⚠️ 添加新依赖时更新 |
| `environment.yml` | Conda 环境配置 | PyTorch 2.0.0, CUDA 11.8 |
| `src/dgraph.yml` | DGraph 数据集配置 | 大规模数据集专用 |

---

## ⚙️ 配置参数详解

### 核心参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `--dataset` | 无 | 数据集名称 | `photo`, `amazon`, `reddit`, `elliptic`, `t_finance`, `tolokers`, `questions`, `dgraph` |
| `--model_type` | `GGADFormer` | 模型类型 | `GGADFormer`, `SGT`, `GGAD` |
| `--num_epoch` | 无 | 训练轮数 | 100-500 |
| `--batch_size` | 8192 | 批次大小 | 64-8192 (根据 GPU 内存调整) |

### 学习率调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--peak_lr` | 1e-4 | 峰值学习率 |
| `--end_lr` | 1e-4 | 结束学习率 |
| `--warmup_epoch` | 20 | 预热轮数 |
| `--weight_decay` | 0.0 | 权重衰减 |

### Transformer 架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embedding_dim` | 256 | 嵌入维度 |
| `--GT_num_layers` | 3 | Transformer 层数 |
| `--GT_num_heads` | 2 | 注意力头数 |
| `--GT_ffn_dim` | 256 | FFN 隐藏维度 |
| `--GT_dropout` | 0.4 | Dropout 率 |
| `--GT_attention_dropout` | 0.4 | 注意力 Dropout 率 |

### 图结构感知参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pp_k` | 6 | Personalized PageRank 步数 |
| `--progregate_alpha` | 0.2 | 聚合系数 |
| `--sample_num_p` | 7 | 正样本采样数 |
| `--sample_num_n` | 7 | 负样本采样数 |

### 损失函数权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bce_loss_weight` | 1.0 | BCE 分类损失权重 |
| `--reconstruction_loss_weight` | 1.0 | 重构损失权重 |
| `--ring_loss_weight` | 1.0 | Ring 损失权重 |
| `--con_loss_weight` | 0.1 | 对比损失权重 |

### 其他重要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_rate` | 0.05 | 训练集比例 (5%) |
| `--seed` | 0 | 随机种子 |
| `--device` | 0 | GPU 设备 ID |
| `--visualize` | False | 是否生成可视化 |

---

## 🔧 常见修改位置

### 针对后续开发的修改指南

#### 1️⃣ 添加新数据集

**修改位置**：`utils.py`

```python
# 添加新的数据加载函数
def load_new_dataset(dataset_name, train_rate, val_rate, args):
    # 1. 加载数据
    # 2. 预处理
    # 3. 返回 adj, features, labels, ...
```

**AI 助手提示**：确保返回格式与 `load_mat()` 一致

#### 2️⃣ 修改模型架构

**修改位置**：`GGADFormer.py`

```python
class GGADFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super().__init__()
        # 添加新层
        self.new_layer = nn.Linear(...)
    
    def forward(self, x, adj, sparse=False):
        # 修改前向传播
        pass
```

#### 3️⃣ 添加新损失函数

**修改位置**：`run.py` (训练循环中)

```python
# 在 epoch 循环内
new_loss = calculate_new_loss(...)
total_loss = bce_loss + rec_loss + ring_loss + new_loss
```

#### 4️⃣ 调整超参数搜索

**修改位置**：`sweep_manager.py`

```python
# 配置 WandB sweep
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'auc', 'goal': 'maximize'},
    'parameters': {
        'peak_lr': {'values': [1e-4, 3e-4, 5e-4]},
        'batch_size': {'values': [64, 128, 256]},
    }
}
```

#### 5️⃣ 修改数据预处理

**修改位置**：`utils.py` 中的 `nagphormer_tokenization()`

```python
def nagphormer_tokenization(features, adj, args):
    # 修改 tokenization 策略
    pass
```

---

## 🐛 调试技巧

### 常见问题 + 解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **CUDA OOM** | 批次过大 | ⚠️ 减小 `--batch_size` 或使用 `check_gpu_memory.py` 监控 |
| **训练不收敛** | 学习率过高 | 降低 `--peak_lr` 至 1e-5，增加 `--warmup_epoch` |
| **AUC/AP 过低** | 模型欠拟合 | 增加 `--num_epoch`，调整 `--embedding_dim` |
| **数据加载慢** | CPU 瓶颈 | 检查 `num_workers` 设置，使用 SSD 存储 |
| **梯度爆炸** | 学习率过大 | 添加梯度裁剪，降低学习率 |

### 调试命令

```bash
# 监控 GPU 内存
python check_gpu_memory.py

# 单步调试（小数据集）
python run.py --dataset=Cora --num_epoch=10 --batch_size=32

# 可视化注意力权重
python run.py --dataset=photo --visualize=True

# 打印详细日志
wandb init --project voxg
```

### 性能优化建议

1. **内存优化**：使用 `--batch_size=4096` 或更低
2. **速度优化**：增加 `--GT_num_heads` 并行计算
3. **精度优化**：使用 `--seed=42` 固定随机性

---

## 📊 VecGAD 实现状态

> **注**：VecGAD 为项目演进版本，以下追踪其实现进度

| 模块 | 状态 | 说明 | 优先级 |
|------|------|------|--------|
| **基础 Transformer 编码器** | ✅ 已实现 | `EncoderLayer`, `MultiHeadAttention` | - |
| **图结构感知输入编码** | ✅ 已实现 | `nagphormer_tokenization()` | - |
| **生成式伪异常样本** | ✅ 已实现 | `outlier_emb` 生成逻辑 | - |
| **多重损失函数** | ✅ 已实现 | BCE + Reconstruction + Ring | - |
| **Mini-batch 训练** | ✅ 已实现 | `WeightedRandomSampler` | - |
| **DGraph 大规模支持** | ✅ 已实现 | `src/main.py` 专用入口 | - |
| **WandB 实验跟踪** | ✅ 已实现 | `sweep_manager.py` | - |
| **可视化功能** | ✅ 已实现 | t-SNE, 注意力权重 | - |
| **VecFormer 变体** | 🟡 部分实现 | 见 `docs/VecFormer/` 目录 | 中 |
| **动态损失权重** | 🟡 部分实现 | `get_dynamic_loss_weights()` | 低 |
| **多 GPU 训练** | ❌ 待优化 | 当前仅支持单 GPU | 高 |
| **混合精度训练** | ❌ 待优化 | 可使用 `torch.cuda.amp` | 中 |
| **分布式数据加载** | ❌ 待优化 | 可使用 `DistributedSampler` | 低 |

### 待优化项详细说明

#### ❌ 多 GPU 训练（高优先级）

```python
# 待实现：在 run.py 中添加
model = torch.nn.DataParallel(model)
# 或
model = torch.nn.parallel.DistributedDataParallel(model)
```

#### ❌ 混合精度训练（中优先级）

```python
# 待实现：使用 AMP
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = ...
scaler.scale(loss).backward()
```

---

## 🤖 AI 助手提示

### 代码理解捷径

1. **快速定位训练逻辑**：搜索 `def train(` in `run.py`
2. **查看模型结构**：阅读 `GGADFormer.__init__()` 和 `forward()`
3. **理解数据流**：追踪 `nagphormer_tokenization()` 的输入输出
4. **调试损失计算**：在 `run.py` 中搜索 `loss_` 前缀变量

### 修改前必读

- ⚠️ 修改 `utils.py` 前，确保理解数据格式（稀疏矩阵 vs 密集张量）
- ⚠️ 修改 `GGADFormer.py` 前，确认嵌入维度一致性
- ⚠️ 修改 `run.py` 训练循环前，备份原始文件

### 实验复现建议

```bash
# 使用论文默认参数
python run.py --dataset=photo --num_epoch=200 --peak_lr=3e-4 \
  --batch_size=128 --embedding_dim=256 --GT_num_layers=3 \
  --GT_num_heads=2 --train_rate=0.05 --seed=42
```

---

## 📚 参考资源

- **论文**：GGAD (NeurIPS 2024) - 见 `GGAD/poster-NeurIPS2024.pdf`
- **数据集**：详见 `docs/dataset_info.md`
- **详细文档**：`docs/GGADFormer.md`
- **实验脚本**：`docs/run.sh`, `docs/VecFormer/run.sh`

---

*最后更新：2026-03-02 | 维护者：VoxG Team*
