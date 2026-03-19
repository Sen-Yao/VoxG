"""
结构重构模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_degree_loss(h, adj, node_indices=None):
    """
    度重构损失 (支持批量节点)
    
    Args:
        h: 节点表示 [num_nodes, hidden_dim] 或 [1, num_nodes, hidden_dim]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        node_indices: 批量节点的索引 (可选)
    
    Returns:
        度重构损失
    """
    # 处理输入形状
    if h.dim() == 3:
        h = h.squeeze(0)  # [1, N, D] -> [N, D]
    
    num_nodes = h.size(0)
    
    # 计算真实度
    if adj.is_sparse:
        degrees = torch.sparse.sum(adj, dim=1).to_dense().float()
    else:
        if adj.dim() == 3:
            adj = adj.squeeze(0)
        degrees = adj.sum(dim=1).float()
    
    # 确保 degrees 是 1D
    degrees = degrees.view(-1)
    
    # 确保节点数量匹配
    if degrees.size(0) != num_nodes:
        # 调整 h 以匹配度数向量
        h = h[:degrees.size(0)]
        num_nodes = h.size(0)
    
    # 预测度（基于表示的模长）
    degree_pred = torch.norm(h, dim=1).view(-1)  # [num_nodes]
    
    # 归一化
    degrees_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-10)
    pred_norm = (degree_pred - degree_pred.min()) / (degree_pred.max() - degree_pred.min() + 1e-10)
    
    # MSE 损失
    loss = F.mse_loss(pred_norm, degrees_norm, reduction='mean')
    
    return loss

def compute_structure_loss(h, adj, sample_ratio=0.01):
    """
    边重构损失（采样版）
    """
    # 处理输入形状
    if h.dim() == 3:
        h = h.squeeze(0)
    
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    num_nodes = h.size(0)
    device = h.device
    
    if num_nodes < 2:
        return torch.tensor(0.0, device=device)
    
    # 采样节点对
    num_samples = max(int(num_nodes * sample_ratio), 100)
    num_samples = min(num_samples, num_nodes * (num_nodes - 1) // 2)
    
    if num_samples < 1:
        return torch.tensor(0.0, device=device)
    
    # 随机采样节点对
    idx_i = torch.randint(0, num_nodes, (num_samples,), device=device)
    idx_j = torch.randint(0, num_nodes, (num_samples,), device=device)
    
    # 确保不采样自己
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(idx_i) == 0:
        return torch.tensor(0.0, device=device)
    
    # 计算预测的边存在概率
    h_i = h[idx_i]
    h_j = h[idx_j]
    
    pred = F.cosine_similarity(h_i, h_j, dim=1)
    pred = (pred + 1) / 2
    
    # 获取真实边标签
    if adj.is_sparse:
        adj_dense = adj.to_dense()
        target = adj_dense[idx_i, idx_j].float()
    else:
        target = adj[idx_i, idx_j].float()
    
    loss = F.binary_cross_entropy(pred, target, reduction='mean')
    
    return loss
