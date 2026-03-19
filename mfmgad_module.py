"""
MFMGAD Module: Masked Frequency Modeling for Graph Anomaly Detection

核心思想：
1. 使用可学习的频域滤波器 (Prompt) 提取节点的多频域视角特征
2. 通过 Masked Frequency Prediction 学习跨频一致性
3. 正常节点一致性强，异常节点一致性弱
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConsistencyPredictor(nn.Module):
    """
    一致性预测器：预测被 mask 的频域 token
    
    输入：可见的频域 token + 正常基线
    输出：对被 mask token 的预测
    """
    
    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super(ConsistencyPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, visible_tokens, baseline_tokens):
        """
        Args:
            visible_tokens: 可见的频域 token [batch_size, num_prompts, hidden_dim]
            baseline_tokens: 正常基线 token [batch_size, num_prompts, hidden_dim]
        
        Returns:
            predicted_tokens: 对所有位置的预测 [batch_size, num_prompts, hidden_dim]
        """
        # 使用交叉注意力：baseline 作为 query，visible tokens 作为 key/value
        y = self.layer_norm1(baseline_tokens)
        y, _ = self.cross_attention(y, visible_tokens, visible_tokens)
        
        # 残差连接
        predicted = baseline_tokens + y
        
        # MLP 预测
        y = self.layer_norm2(predicted)
        predicted = predicted + self.predictor(y)
        
        return predicted


class MFMGADModule(nn.Module):
    """
    Masked Frequency Modeling 模块
    
    可以插入到任何 Transformer 模型中，用于异常检测
    """
    
    def __init__(self, hidden_dim, num_prompts=8, mask_ratio=0.25, num_heads=2, dropout=0.1):
        super(MFMGADModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts
        self.mask_ratio = mask_ratio
        
        # 可学习的频域滤波器 (Prompt)
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, hidden_dim))
        
        # 符号注意力的投影层
        self.sign_query = nn.Linear(hidden_dim, hidden_dim)
        self.sign_key = nn.Linear(hidden_dim, hidden_dim)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 一致性预测器
        self.consistency_predictor = ConsistencyPredictor(hidden_dim, num_heads, dropout)
        
        # 正常基线（动态更新）
        self.register_buffer('prompt_baselines', torch.zeros(num_prompts, hidden_dim))
        self.register_buffer('baseline_initialized', torch.tensor(False))
        
    def extract_frequency_tokens(self, node_tokens, temp=1.0):
        """
        使用频域滤波器提取节点的多频域视角特征
        
        Args:
            node_tokens: 节点特征序列 [batch_size, seq_len, hidden_dim]
            temp: 温度参数
        
        Returns:
            frequency_tokens: 频域 token [batch_size, num_prompts, hidden_dim]
            attn_weights: 注意力权重 [batch_size, num_prompts, seq_len]
        """
        batch_size = node_tokens.size(0)
        seq_len = node_tokens.size(1)
        
        # 扩展频域滤波器
        queries = self.prompts.expand(batch_size, -1, -1)  # [batch_size, num_prompts, hidden_dim]
        keys = node_tokens
        values = node_tokens
        
        # 计算幅值注意力分数
        score_mag = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        
        # 计算方向注意力分数
        score_sign = torch.matmul(self.sign_query(queries), self.sign_key(keys).transpose(-1, -2))
        
        # 合并注意力的幅值和方向
        magnitude = F.softmax(score_mag / temp, dim=-1)
        sign = torch.tanh(score_sign / temp)
        
        attn_weights = magnitude * sign
        frequency_tokens = torch.matmul(attn_weights, values)
        frequency_tokens = self.layer_norm(frequency_tokens)
        
        return frequency_tokens, attn_weights
    
    def apply_frequency_mask(self, frequency_tokens):
        """
        随机 mask 若干频域 token
        
        Args:
            frequency_tokens: 频域 token [batch_size, num_prompts, hidden_dim]
        
        Returns:
            masked_tokens: mask 后的 token
            mask: 布尔掩码，True 表示被 mask
            visible_mask: 布尔掩码，True 表示可见
        """
        batch_size, num_prompts, _ = frequency_tokens.shape
        device = frequency_tokens.device
        
        # 向量化生成 mask
        num_mask = max(1, int(num_prompts * self.mask_ratio))
        
        # 生成随机分数
        rand_scores = torch.rand(batch_size, num_prompts, device=device)
        
        # 获取每行最小的 num_mask 个位置的索引
        _, mask_indices = torch.topk(rand_scores, num_mask, dim=1, largest=False)
        
        # 创建 mask
        mask = torch.zeros(batch_size, num_prompts, dtype=torch.bool, device=device)
        mask.scatter_(1, mask_indices, True)
        
        visible_mask = ~mask
        
        # 创建 masked tokens
        masked_tokens = frequency_tokens * visible_mask.unsqueeze(-1).float()
        
        return masked_tokens, mask, visible_mask
    
    def update_baselines(self, frequency_tokens, normal_idx, momentum=0.9):
        """
        更新正常基线
        
        Args:
            frequency_tokens: 频域 token [num_nodes, num_prompts, hidden_dim]
            normal_idx: 正常节点索引
            momentum: 动量参数
        """
        with torch.no_grad():
            normal_tokens = frequency_tokens[normal_idx]
            new_baselines = normal_tokens.mean(dim=0)
            
            if self.baseline_initialized:
                self.prompt_baselines = momentum * self.prompt_baselines + (1 - momentum) * new_baselines
            else:
                self.prompt_baselines = new_baselines
                self.baseline_initialized = torch.tensor(True, device=frequency_tokens.device)
    
    def compute_consistency_loss(self, frequency_tokens, mask, predicted_tokens):
        """
        计算跨频一致性损失
        
        Args:
            frequency_tokens: 原始频域 token
            mask: 布尔掩码
            predicted_tokens: 预测的频域 token
        
        Returns:
            consistency_loss: 一致性损失
            prediction_errors: 每个样本的预测误差
        """
        # 只计算被 mask 位置的误差
        mask_expanded = mask.unsqueeze(-1).expand_as(frequency_tokens)
        
        # 预测误差
        prediction_errors = (predicted_tokens - frequency_tokens) ** 2
        masked_errors = prediction_errors * mask_expanded.float()
        
        # 每个样本的误差
        sample_errors = masked_errors.sum(dim=(1, 2)) / (mask.sum(dim=1, keepdim=True).float() * self.hidden_dim + 1e-8)
        
        # 平均一致性损失
        consistency_loss = masked_errors.sum() / (mask.sum().float() * self.hidden_dim + 1e-8)
        
        return consistency_loss, sample_errors
    
    def forward(self, node_tokens, normal_idx=None, update_baseline=True):
        """
        前向传播
        
        Args:
            node_tokens: 节点特征序列 [batch_size, seq_len, hidden_dim]
            normal_idx: 正常节点索引（用于更新基线）
            update_baseline: 是否更新基线
        
        Returns:
            frequency_tokens: 频域 token
            consistency_loss: 一致性损失
            prediction_errors: 预测误差（用于异常分数）
        """
        # 1. 提取频域特征
        frequency_tokens, attn_weights = self.extract_frequency_tokens(node_tokens)
        
        # 2. 更新基线（训练时）
        if update_baseline and normal_idx is not None:
            self.update_baselines(frequency_tokens, normal_idx)
        
        # 3. 随机 mask
        masked_tokens, mask, visible_mask = self.apply_frequency_mask(frequency_tokens)
        
        # 4. 预测被 mask 的部分
        batch_size = frequency_tokens.size(0)
        baseline_tokens = self.prompt_baselines.unsqueeze(0).expand(batch_size, -1, -1)
        predicted_tokens = self.consistency_predictor(masked_tokens, baseline_tokens)
        
        # 5. 计算一致性损失
        consistency_loss, prediction_errors = self.compute_consistency_loss(
            frequency_tokens, mask, predicted_tokens
        )
        
        return frequency_tokens, consistency_loss, prediction_errors
