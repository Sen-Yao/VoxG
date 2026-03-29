# Channel-wise Token + VoxGFormer 结合方案

## 一、VoxGFormer 核心流程

```
输入 Token [N, K+1, D]
    ↓
Prompt 提取 [N, num_prompts, D]  ← 可学习的 Prompt 查询 hop
    ↓
CLS Token 编码 [N, D]
    ↓
伪异常生成（温度调整、采样）
    ↓
BCE 分类（正常 vs 伪异常）
```

## 二、Channel-wise Token 核心流程

```
输入 Token [N, K+1, D]
    ↓
Channel-wise Tokenizer [N, D+1, H]  ← 每个通道独立 Token
    ↓
Transformer 编码 [N, H]
    ↓
CLS Token 输出 [N, H]
```

## 三、结合方案

### 方案 A：Channel-wise Token 替换输入

**核心思想**：将 Channel-wise Token 作为 VoxGFormer 的新输入。

```
原始 Token [N, K+1, D]
    ↓
Channel-wise Tokenizer [N, D+1, H]  ← 新增
    ↓
VoxGFormer Prompt 提取 [N, num_prompts, H]  ← Prompt 从通道维度查询
    ↓
CLS Token 编码 [N, H]
    ↓
伪异常生成 + BCE
```

**关键改动**：
- Prompt 查询的维度从 hop 变为 channel
- 每个 Prompt 学习"哪些通道组合重要"

**伪异常生成**：
- 对 Channel-wise Token 添加通道级别的扰动
- 例如：随机 mask 某些通道、通道 shuffle

---

### 方案 B：Channel-wise Token 替换 Prompt 提取

**核心思想**：用 Channel-wise Transformer 替换 Prompt 提取。

```
原始 Token [N, K+1, D]
    ↓
Channel-wise Transformer [N, H]  ← 替换 Prompt 提取
    ↓
伪异常生成（通道级别）
    ↓
BCE 分类
```

**关键改动**：
- 移除可学习的 Prompt
- 用 Transformer 注意力学习通道重要性
- 更直接，更可解释

**伪异常生成**：
```python
def generate_channel_pseudo_anomaly(channel_tokens, temperature=2.0):
    """
    通道级别伪异常生成
    
    方式 1：通道 mask
    - 随机 mask 部分通道（模拟信息缺失）
    
    方式 2：通道 shuffle
    - 打乱通道顺序（模拟结构异常）
    
    方式 3：通道噪声
    - 添加通道特定的噪声（模拟特征异常）
    """
    N, D, H = channel_tokens.shape
    
    # 方式 1：通道 mask
    mask = torch.rand(N, D, 1, device=channel_tokens.device) > 0.3
    masked_tokens = channel_tokens * mask
    
    # 方式 2：通道 shuffle
    perm = torch.randperm(D)
    shuffled_tokens = channel_tokens[:, perm, :]
    
    # 方式 3：通道噪声
    noise = torch.randn_like(channel_tokens) * 0.1
    noised_tokens = channel_tokens + noise
    
    # 组合
    pseudo_anomaly = masked_tokens + noise
    
    return pseudo_anomaly
```

---

### 方案 C：双分支融合（推荐 ⭐）

**核心思想**：同时保留 hop-level 和 channel-level 信息。

```
原始 Token [N, K+1, D]
    ├──→ Hop 分支（VoxGFormer 原版）
    │    Prompt 提取 → CLS Token [N, H]
    │
    └──→ Channel 分支（Channel-wise）
         Channel-wise Tokenizer → CLS Token [N, H]
              ↓
         特征融合 [N, 2H]
              ↓
         伪异常生成（双分支）
              ↓
         BCE 分类
```

**优势**：
- 保留原有 VoxGFormer 的优势
- 新增 Channel-wise 的通道特异性
- 双分支可以互补

**融合方式**：
```python
class DualBranchVoxGFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hops, args):
        super().__init__()
        
        # Hop 分支（原版）
        self.hop_prompts = nn.Parameter(torch.randn(1, num_prompts, input_dim))
        self.hop_encoder = TransformerEncoder(...)
        
        # Channel 分支
        self.channel_tokenizer = ChannelWiseTokenizer(num_hops, input_dim, hidden_dim)
        self.channel_encoder = TransformerEncoder(...)
        
        # 融合层
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, tokens, train_flag, normal_idx):
        # Hop 分支
        hop_features = self.extract_with_prompts(tokens)  # [N, H]
        
        # Channel 分支
        channel_tokens = self.channel_tokenizer(tokens)  # [N, D+1, H]
        channel_features = self.channel_encoder(channel_tokens)[:, 0, :]  # [N, H]
        
        # 融合
        combined = torch.cat([hop_features, channel_features], dim=-1)  # [N, 2H]
        fused = self.fusion(combined)  # [N, H]
        
        if train_flag:
            # 伪异常生成
            pseudo_anomaly = self.generate_dual_pseudo_anomaly(hop_features, channel_features)
            
            # BCE 损失
            labels = torch.cat([
                torch.zeros(len(normal_idx)),  # 正常
                torch.ones(pseudo_anomaly.shape[0])  # 伪异常
            ])
            
            logits = self.classifier(torch.cat([fused[normal_idx], pseudo_anomaly]))
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            return fused, loss
        
        return fused
```

---

## 四、伪异常生成策略

### 4.1 Hop-level 伪异常（原版）

```python
def generate_hop_pseudo_anomaly(tokens, temperature=2.0):
    """
    Hop 级别伪异常
    
    方式：升温部分 Token
    """
    # 对前 K//2 个 hop 升温
    high_temp_tokens = tokens[:, :K//2, :] * temperature
    normal_temp_tokens = tokens[:, K//2:, :]
    
    pseudo_anomaly_tokens = torch.cat([high_temp_tokens, normal_temp_tokens], dim=1)
    return pseudo_anomaly_tokens
```

### 4.2 Channel-level 伪异常（新增）

```python
def generate_channel_pseudo_anomaly(tokens, channel_tokens, strategy='mask'):
    """
    Channel 级别伪异常
    
    策略：
    1. mask: 随机 mask 通道
    2. noise: 添加通道噪声
    3. outlier: 选择离群通道
    """
    N, D = tokens.shape[:2]
    
    if strategy == 'mask':
        # 随机 mask 30% 通道
        mask = torch.rand(N, D, 1, device=tokens.device) > 0.3
        pseudo = channel_tokens * mask
    
    elif strategy == 'noise':
        # 添加高斯噪声
        noise = torch.randn_like(channel_tokens) * 0.2
        pseudo = channel_tokens + noise
    
    elif strategy == 'outlier':
        # 选择注意力权重最低的通道（离群通道）
        attention = compute_channel_attention(channel_tokens)
        outlier_idx = torch.argsort(attention, dim=1)[:, :int(D * 0.3)]
        
        pseudo = channel_tokens.clone()
        for i in range(N):
            pseudo[i, outlier_idx[i], :] += torch.randn_like(pseudo[i, outlier_idx[i], :]) * 0.5
    
    return pseudo
```

---

## 五、实验设计

### 5.1 对比方法

| 方法 | Token 类型 | 伪异常方式 |
|------|-----------|-----------|
| VoxGFormer（原版） | Hop-level | 温度调整 |
| Channel-wise Only | Channel-level | 通道 mask/noise |
| **Dual-Branch** | Hop + Channel | 双分支融合 |

### 5.2 评估指标

- **AUC-ROC**：主要指标
- **AUC-AP**：辅助指标
- **注意力可视化**：通道重要性分析

### 5.3 预期结果

| 数据集 | 特征维度 | 预期最优方法 |
|--------|----------|-------------|
| Photo | 745（高维） | Dual-Branch（hop 主导） |
| Amazon | 25（低维） | Channel-wise Only |
| Elliptic | 93（中维） | Dual-Branch（平衡） |

---

## 六、实现文件位置

| 文件 | 位置 |
|------|------|
| Channel-wise Tokenizer | `models/channel_wise_token.py` |
| Dual-Branch VoxGFormer | `models/dual_branch_voxg.py`（新建）|
| 实验脚本 | `experiments/dual_branch_experiment.py`（新建）|

---

## 七、总结

### 推荐方案：双分支融合（方案 C）

**理由**：
1. 保留原 VoxGFormer 的优势
2. 新增 Channel-wise 的通道特异性
3. 双分支互补，适应不同数据集
4. 可解释性好（可视化两个分支的注意力）

**下一步**：
1. 实现 `DualBranchVoxGFormer`
2. 设计双分支伪异常生成策略
3. 在三个数据集上验证

---

_方案设计时间: 2026-03-27 15:15_
_作者: Nexus_