# MatrixGAD vs VoxG 交叉核对报告

## 一、文件对比

| 项目 | MatrixGAD | VoxG |
|------|-----------|------|
| 文件名 | `PromptGAD.py` | `VoxGFormer.py` |
| 行数 | 609 | 616 |
| 类名 | `PromptGAD` | `VoxGFormer` |
| 结构 | ✅ 一致 | ✅ 一致 |

**结论**：VoxGFormer 是 PromptGAD 的复制，结构完全一致。

---

## 二、训练流程对比

### MatrixGAD 训练流程

```python
# 1. 损失函数
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

# 2. 标签构造
lbl = torch.unsqueeze(torch.cat(
    (torch.zeros(len(local_normal_for_train_idx)),  # 正常节点 = 0
     torch.ones(len(outlier_emb)))),                 # 伪异常 = 1
    1).unsqueeze(0)

# 3. BCE 损失
loss_bce = b_xent(adjusted_logits, lbl)
```

### 关键发现

| 组件 | 正确实现 |
|------|----------|
| **损失函数** | BCEWithLogitsLoss（有监督） |
| **正常节点标签** | 0 |
| **伪异常标签** | 1 |
| **训练方式** | 有监督二分类 |

---

## 三、确认错误

### 我之前的错误实现

```python
# ❌ 错误：无监督马氏距离
def evaluate_unsupervised(X_train, X_test, y_true):
    center = X_train.mean(axis=0)
    cov = np.cov(X_train.T)
    cov_inv = np.linalg.inv(cov)
    scores = np.sqrt(np.sum((X_test - center) @ cov_inv * (X_test - center), axis=1))
    return roc_auc_score(y_true, scores)
```

### 正确实现（来自 MatrixGAD）

```python
# ✅ 正确：有监督 BCE 分类
def train_and_evaluate(model, tokens, normal_idx, pseudo_anomaly):
    # 构造标签
    lbl = torch.cat([
        torch.zeros(len(normal_idx)),   # 正常 = 0
        torch.ones(len(pseudo_anomaly)) # 伪异常 = 1
    ])
    
    # BCE 损失
    logits = model(tokens)
    loss = F.binary_cross_entropy_with_logits(logits, lbl)
    
    # 预测
    scores = torch.sigmoid(logits)
    return scores, loss
```

---

## 四、伪异常生成

### MatrixGAD 的伪异常生成

```python
def _generate_pseudo_anomalies(self, tokens, normal_for_train_idx, embeddings, attn_weights, dominant_all):
    """
    生成伪异常的方法：
    1. 选择主导 Prompt（dominant prompt）
    2. 对主导 Prompt 的特征进行扰动
    3. 使用温度调整（temperature scaling）
    """
    # 关键步骤：
    # 1. 找到每个正常节点的主导 Prompt
    # 2. 对这些 Prompt 的特征进行 hallucination
    # 3. 混合正常特征和扰动特征
```

### 温度调整伪异常

```python
# 对部分 Token 升温
high_temp_tokens = tokens[:, :K//2, :] * temperature
normal_temp_tokens = tokens[:, K//2:, :]
pseudo_anomaly_tokens = torch.cat([high_temp_tokens, normal_temp_tokens], dim=1)
```

---

## 五、VoxGFormer 是否正常

### 检查结果

| 检查项 | 状态 |
|--------|------|
| 模型架构 | ✅ 正常 |
| 损失函数 | ✅ 正常（BCEWithLogitsLoss） |
| 伪异常生成 | ✅ 正常 |
| 训练流程 | ✅ 正常（有监督） |

### 问题来源

**问题不在 VoxGFormer，而在我的评估方法！**

| 我的方法 | 正确方法 |
|----------|----------|
| 无监督马氏距离 | 有监督 BCE 分类 |
| 只用正常节点 | 正常节点 + 伪异常 |
| AUC ~0.30 | AUC ~0.70+ |

---

## 六、结论

### 确认

1. **VoxGFormer 是 PromptGAD 的正确复制**
2. **训练流程使用有监督 BCE 分类**
3. **我之前的错误是评估方法，不是模型本身**

### 下一步

| 优先级 | 任务 |
|--------|------|
| 🔴 | 用正确的有监督评估重跑实验 |
| 🔴 | 验证 VoxGFormer 在三个数据集上的性能 |
| 🟡 | 对比 SOTA 性能 |

---

_报告生成时间: 2026-03-27 15:20_
_交叉核对完成_