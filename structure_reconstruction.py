"""
结构重构模块 (修复版 v2 - 正确处理设备)
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
        node_indices: 批量节点的全局索引 (可选，用于 batch 训练)
    
    Returns:
        度重构损失
    """
    # 处理输入形状
    if h.dim() == 3:
        h = h.squeeze(0)  # [1, N, D] -> [N, D]
    
    device = h.device
    num_nodes = h.size(0)
    
    # 处理邻接矩阵
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    # 确保邻接矩阵在正确的设备上
    if adj.device != device:
        adj = adj.to(device)
    
    # 计算真实度
    if adj.is_sparse:
        degrees = torch.sparse.sum(adj, dim=1).to_dense().float()
    else:
        degrees = adj.sum(dim=1).float()
    
    # 确保 degrees 是 1D
    degrees = degrees.view(-1)
    
    # 如果提供了 node_indices，只计算 batch 中节点的损失
    if node_indices is not None:
        # 确保 node_indices 在正确的设备上
        node_indices = node_indices.to(device)
        
        # 确保 node_indices 在有效范围内
        max_nodes = min(num_nodes, degrees.size(0))
        valid_mask = node_indices < max_nodes
        valid_indices = node_indices[valid_mask]
        
        if valid_indices.size(0) == 0:
            return torch.tensor(0.0, device=device)
        
        h_batch = h[valid_indices]
        degrees_batch = degrees[valid_indices]
    else:
        # 调整 h 以匹配度数向量
        min_nodes = min(num_nodes, degrees.size(0))
        h_batch = h[:min_nodes]
        degrees_batch = degrees[:min_nodes]
    
    # 预测度（基于表示的模长）
    degree_pred = torch.norm(h_batch, dim=1).view(-1)  # [batch_nodes]
    
    # 归一化
    if degrees_batch.size(0) > 1:
        degrees_norm = (degrees_batch - degrees_batch.min()) / (degrees_batch.max() - degrees_batch.min() + 1e-10)
        pred_norm = (degree_pred - degree_pred.min()) / (degree_pred.max() - degree_pred.min() + 1e-10)
        loss = F.mse_loss(pred_norm, degrees_norm, reduction='mean')
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss

def compute_structure_loss(h, adj, node_indices=None, sample_ratio=0.01):
    """
    边重构损失（采样版，支持 batch 训练）
    
    Args:
        h: 节点表示 [num_nodes, hidden_dim]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        node_indices: 批量节点的全局索引 (可选)
        sample_ratio: 采样比例
    """
    # 处理输入形状
    if h.dim() == 3:
        h = h.squeeze(0)
    
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    device = h.device
    num_nodes = h.size(0)
    
    # 确保邻接矩阵在正确的设备上
    if adj.device != device:
        adj = adj.to(device)
    
    if num_nodes < 2:
        return torch.tensor(0.0, device=device)
    
    # 如果提供了 node_indices，提取 batch 子图
    if node_indices is not None:
        node_indices = node_indices.to(device)
        valid_mask = node_indices < num_nodes
        valid_indices = node_indices[valid_mask]
        
        if valid_indices.size(0) < 2:
            return torch.tensor(0.0, device=device)
        
        # 提取子图邻接矩阵
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        batch_adj = adj_dense[valid_indices][:, valid_indices]
        h_batch = h[valid_indices]
    else:
        h_batch = h
        if adj.is_sparse:
            batch_adj = adj.to_dense()
        else:
            batch_adj = adj
    
    batch_size = h_batch.size(0)
    
    # 采样节点对
    num_samples = max(int(batch_size * sample_ratio), 50)
    num_samples = min(num_samples, batch_size * (batch_size - 1) // 2)
    
    if num_samples < 1:
        return torch.tensor(0.0, device=device)
    
    # 随机采样节点对
    idx_i = torch.randint(0, batch_size, (num_samples,), device=device)
    idx_j = torch.randint(0, batch_size, (num_samples,), device=device)
    
    # 确保不采样自己
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(idx_i) == 0:
        return torch.tensor(0.0, device=device)
    
    # 计算预测的边存在概率
    h_i = h_batch[idx_i]
    h_j = h_batch[idx_j]
    
    pred = F.cosine_similarity(h_i, h_j, dim=1)
    pred = (pred + 1) / 2
    
    # 获取真实边标签
    target = batch_adj[idx_i, idx_j].float()
    
    loss = F.binary_cross_entropy(pred, target, reduction='mean')
    
    return loss