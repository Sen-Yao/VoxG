# VoxG 项目结构分析报告

## 1. 文件树结构

```
VoxG/
├── .git/                          # Git 版本控制
├── .gitignore                     # Git 忽略配置
├── README.md                      # 项目说明文档
├── requirements.txt               # Python 依赖包列表
├── environment.yml                # Conda 环境配置
├── run.py                         # 主训练入口脚本
├── model.py                       # 基础模型定义（GCN、判别器等）
├── utils.py                       # 工具函数（数据加载、预处理等）
├── GGADFormer.py                  # GGADFormer 模型实现（Transformer 架构）
├── SGT.py                         # SGT 模型实现
├── visualization.py               # 可视化功能（t-SNE、注意力权重）
├── sweep_manager.py               # WandB 超参数搜索管理器
├── check_gpu_memory.py            # GPU 内存监控工具
├── utils_tam.py                   # TAM 方法相关工具
├── GGAD/                          # GGAD 基线方法实现
│   ├── README.md                  # GGAD 论文说明
│   ├── requirements.txt           # GGAD 依赖
│   ├── run.py                     # GGAD 运行脚本
│   ├── model.py                   # GGAD 模型定义
│   ├── utils.py                   # GGAD 工具函数
│   ├── utils_GGAD.py              # GGAD 专用工具
│   ├── utils_tam.py               # TAM 工具
│   ├── tam.py                     # TAM 方法实现
│   ├── aegis.py                   # AEGIS 基线
│   ├── anomalyDAE.py              # AnomalyDAE 基线
│   ├── dominant.py                # DOMINANT 基线
│   ├── gaan.py                    # GAAN 基线
│   ├── ocgnn.py                   # OCGNN 基线
│   ├── model_*.py                 # 各基线模型实现
│   ├── poster-NeurIPS2024.pdf     # NeurIPS 2024 海报
│   ├── framework.png              # GGAD 框架图
│   └── src/                       # DGraph 数据集相关代码
│       ├── main.py                # DGraph 主入口
│       ├── model.py               # DGraph 模型
│       ├── model_handler.py       # 模型训练处理器
│       ├── layers.py              # 网络层定义
│       ├── utils.py               # 工具函数
│       ├── graphsage*.py          # GraphSAGE 变体
│       └── dgraph.yml             # DGraph 配置
├── src/                           # 主项目源代码目录（与 GGAD/src 类似）
│   ├── main.py                    # 主入口
│   ├── model.py                   # 模型定义
│   ├── model_handler.py           # 模型处理器
│   ├── layers.py                  # 网络层
│   ├── utils.py                   # 工具函数
│   ├── graphsage*.py              # GraphSAGE 变体
│   └── dgraph.yml                 # 配置文件
├── docs/                          # 文档目录
│   ├── dataset_info.md            # 数据集信息
│   ├── GGADFormer.md              # GGADFormer 文档
│   ├── run.sh                     # 运行脚本
│   └── VecFormer/                 # VecFormer 实验文档
│       ├── 15rate.md
│       ├── experiments.md
│       ├── paper.md
│       └── run.sh
└── figs/                          # 实验结果图表
    ├── degradation/               # 性能退化分析
    ├── propagation_steps/         # 传播步数分析
    ├── residual_weight/           # 残差权重分析
    └── train_ratio/               # 训练比例分析
```

## 2. 核心文件清单及用途

### 2.1 主要模型文件

| 文件 | 用途 | 关键组件 |
|------|------|----------|
| `GGADFormer.py` | 核心模型 - 基于 Transformer 的图异常检测框架 | `GGADFormer` 类、`EncoderLayer`、`MultiHeadAttention`、`FeedForwardNetwork` |
| `SGT.py` | SGT 模型实现（对比模型） | `SGT` 类、交叉学习机制 |
| `model.py` | 基础模型组件 | `GCN`、`Discriminator`、`Model`（基线模型） |
| `run.py` | 主训练入口 | 训练循环、数据加载、损失计算、评估指标 |

### 2.2 数据处理与工具

| 文件 | 用途 | 关键函数 |
|------|------|----------|
| `utils.py` | 数据加载与预处理 | `load_mat()`, `load_dgraph()`, `nagphormer_tokenization()`, `preprocess_sample_features()` |
| `visualization.py` | 可视化功能 | `create_tsne_visualization()`, `visualize_attention_weights()` |
| `check_gpu_memory.py` | GPU 内存监控 | 内存使用打印、清理功能 |

### 2.3 基线方法（GGAD 目录）

| 文件 | 方法 | 说明 |
|------|------|------|
| `GGAD/model_GGAD.py` | GGAD | 生成式半监督图异常检测（NeurIPS 2024） |
| `tam.py` | TAM | 基于亲和力的方法 |
| `aegis.py` | AEGIS | 对抗式方法 |
| `anomalyDAE.py` | AnomalyDAE | 自编码器方法 |
| `dominant.py` | DOMINANT | 图自编码器 |
| `gaan.py` | GAAN | 生成对抗方法 |
| `ocgnn.py` | OCGNN | 一类分类 GNN |

### 2.4 配置与依赖

| 文件 | 用途 |
|------|------|
| `requirements.txt` | Python 包依赖 |
| `environment.yml` | Conda 环境配置（PyTorch 2.0.0, CUDA 11.8） |
| `src/dgraph.yml` | DGraph 数据集配置 |

## 3. 依赖包列表

### 3.1 核心依赖（requirements.txt）

```yaml
numpy==1.21.6
scipy==1.7.3
scikit-learn==1.0.2
networkx==2.6.3
pandas==1.3.5
matplotlib==3.7.0
tqdm==4.65.0
dgl==1.1.3              # Deep Graph Library
torchdata==0.6.0        # PyTorch 数据加载
torch-geometric==2.3.1  # PyTorch Geometric
wandb                   # 实验跟踪
```

### 3.2 环境依赖（environment.yml）

```yaml
python=3.8
pytorch=2.0.0
torchvision=0.15.0
torchaudio=2.0.0
pytorch-cuda=11.8
```

### 3.3 依赖关系图

```
GGADFormer
├── PyTorch 2.0.0 (核心深度学习框架)
├── DGL 1.1.3 (图神经网络库)
├── PyTorch Geometric 2.3.1 (图神经网络)
├── TorchData 0.6.0 (数据加载)
├── NumPy/SciPy (科学计算)
├── Scikit-learn (机器学习工具)
├── NetworkX (图分析)
├── Pandas (数据处理)
├── Matplotlib (可视化)
└── WandB (实验跟踪)
```

## 4. 模块依赖关系图

### 4.1 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        run.py (训练入口)                      │
├─────────────────────────────────────────────────────────────┤
│  • 参数解析 (argparse)                                       │
│  • 数据加载 (utils.py)                                       │
│  • 模型初始化 (GGADFormer.py / SGT.py / model.py)            │
│  • 训练循环                                                   │
│  • 评估指标 (AUC, AP)                                        │
│  • WandB 日志记录                                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│  GGADFormer   │    │      SGT       │    │    Model      │
│   (Transformer│    │ (对比模型)      │    │  (基线 GCN)   │
│    架构)       │    │                │    │               │
└───────────────┘    └────────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │     utils.py    │
                    ├─────────────────┤
                    │ • load_mat()    │
                    │ • load_dgraph() │
                    │ • tokenization  │
                    │ • 数据预处理     │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│  datasets/    │    │  visualization │    │   WandB       │
│  (.mat, .npz) │    │  (t-SNE, 注意力)│    │   (日志)      │
└───────────────┘    └────────────────┘    └───────────────┘
```

### 4.2 GGADFormer 模型内部结构

```
GGADFormer
├── Token Projection (输入编码)
│   └── nagphormer_tokenization() → [N, pp_k+1, feature_dim]
├── Graph Transformer Encoder
│   ├── MultiHeadAttention (多头注意力)
│   ├── FeedForwardNetwork (前馈网络)
│   ├── LayerNorm (层归一化)
│   └── EncoderLayer × GT_num_layers (默认 3 层)
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

### 4.3 数据流

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
损失计算 + 反向传播
```

### 4.4 GGAD 基线方法架构

```
GGAD/src/main.py
    │
    └─→ ModelHandler (model_handler.py)
            │
            ├─→ 数据加载 (utils.py)
            │   ├─→ load_data()
            │   └─→ 数据集划分
            │
            └─→ 模型选择
                ├─→ PCGNN (model.py + layers.py)
                ├─→ GraphSAGE (graphsage.py)
                └─→ GCN (graphsage.py)
```

## 5. 关键设计特点

### 5.1 半监督学习设定
- 仅使用 5% 的节点作为训练集
- 训练时仅能访问正常节点的标签
- 通过生成伪异常样本进行对比学习

### 5.2 核心创新点
1. **解耦式图结构感知输入编码** - 将图拓扑与节点表征学习分离
2. **生成式策略** - 人工合成高质量伪异常样本
3. **Transformer 架构** - 首个基于 Transformer 的生成式半监督图异常检测模型
4. **Mini-batch 训练** - 支持大规模图的可扩展训练

### 5.3 支持的 datasets
- Amazon (Co-review)
- Reddit (Social Media)
- Photo (Co-purchase)
- Elliptic (Bitcoin Transaction)
- T-Finance (Transaction)
- DGraph (Financial Networks) - 大规模数据集 (3.7M 节点)

## 6. 运行命令示例

```bash
# 基础训练
python run.py --dataset=photo --num_epoch=200 --peak_lr=3e-4 --batch_size=128

# 指定模型类型
python run.py --dataset=reddit --model_type=GGADFormer

# 使用 SGT 模型
python run.py --dataset=amazon --model_type=SGT

# DGraph 数据集
python src/main.py --config src/dgraph.yml
```

---

*分析报告生成时间：2026-03-02*
