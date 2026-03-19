"""
Focal Loss 实现 - 处理类别不平衡

论文: "Focal Loss for Dense Object Detection" (ICCV 2017)
公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

核心思想:
- 降低易分类样本的权重
- 聚焦于难分类样本
- 通过 α 平衡正负样本

预期收益:
- 不平衡数据集 (Reddit, Elliptic): +3-8% AUROC
- 实现复杂度: 极低 (10-20 行)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    
    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = model(x)  # shape: [N, 1]
        >>> labels = torch.tensor([0, 1, 0, 1])  # shape: [N]
        >>> loss = loss_fn(logits, labels)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):
        """
        Args:
            logits: Model output, shape [N, 1] or [N]
            labels: Ground truth labels, shape [N], values in {0, 1}
        
        Returns:
            Focal loss value
        """
        # Flatten logits
        logits = logits.view(-1)
        labels = labels.view(-1)
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # p_t = p if y=1, else 1-p
        p_t = probs * labels + (1 - probs) * (1 - labels)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha term
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        
        # Cross entropy: -log(p_t)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        # Focal loss
        focal_loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss with class-aware alpha
    
    Automatically adjusts alpha based on class imbalance ratio.
    Useful for extreme imbalance scenarios (e.g., Reddit with 3.5% positive).
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', None)
    
    def set_alpha(self, labels):
        """Compute alpha based on class distribution"""
        num_pos = labels.sum().item()
        num_neg = labels.size(0) - num_pos
        
        if num_pos == 0 or num_neg == 0:
            self.alpha = torch.tensor(0.5)
        else:
            # Alpha for positive class
            self.alpha = torch.tensor(num_neg / (num_pos + num_neg))
    
    def forward(self, logits, labels, auto_alpha=True):
        """
        Args:
            logits: Model output
            labels: Ground truth labels
            auto_alpha: If True, compute alpha from current batch
        """
        if auto_alpha:
            self.set_alpha(labels)
        
        logits = logits.view(-1)
        labels = labels.view(-1).float()
        
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_term = (1 - p_t) ** self.gamma
        
        alpha = self.alpha.to(logits.device)
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        focal_loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels, beta=0.999):
    """
    Compute effective number of samples weights
    
    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    
    Args:
        labels: Ground truth labels
        beta: Hyperparameter (default: 0.999)
    
    Returns:
        weight_per_class: [weight_neg, weight_pos]
    """
    num_pos = labels.sum().item()
    num_neg = labels.size(0) - num_pos
    
    # Effective number of samples
    effective_num_pos = 1 - beta ** num_pos
    effective_num_neg = 1 - beta ** num_neg
    
    # Weights
    weight_pos = (1 - beta) / effective_num_pos if num_pos > 0 else 0
    weight_neg = (1 - beta) / effective_num_neg if num_neg > 0 else 0
    
    # Normalize
    total = weight_pos + weight_neg
    weight_pos /= total
    weight_neg /= total
    
    return torch.tensor([weight_neg, weight_pos])


# ============ 使用示例 ============

if __name__ == '__main__':
    # 测试代码
    torch.manual_seed(42)
    
    # 模拟不平衡数据 (90% 负样本)
    n_samples = 1000
    labels = torch.cat([
        torch.zeros(900),  # 90% 负样本
        torch.ones(100)    # 10% 正样本
    ])
    
    # 模拟模型输出
    logits = torch.randn(n_samples)
    
    # 标准 BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
    print(f"BCE Loss: {bce_loss.item():.4f}")
    
    # Focal Loss
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(logits, labels)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Adaptive Focal Loss
    adaptive_focal = AdaptiveFocalLoss(gamma=2.0)
    adaptive_loss = adaptive_focal(logits, labels, auto_alpha=True)
    print(f"Adaptive Focal Loss: {adaptive_loss.item():.4f}")
    
    # Class weights
    weights = compute_class_weights(labels)
    print(f"Class weights: neg={weights[0]:.4f}, pos={weights[1]:.4f}")