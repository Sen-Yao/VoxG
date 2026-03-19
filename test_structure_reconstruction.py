"""
测试结构重构效果
独立脚本，不修改 run.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

# 导入模型
from VoxGFormer import VoxGFormer
from structure_reconstruction import compute_degree_loss

def load_data(dataset_name):
    """加载数据"""
    data_path = f'/root/gpufree-data/linziyao/GGADFormer/GGAD/dataset/{dataset_name}.mat'
    data = scio.loadmat(data_path)
    
    features = torch.FloatTensor(data['Attributes'].todense() if hasattr(data['Attributes'], 'todense') else data['Attributes'])
    adj = data['Network']
    if hasattr(adj, 'todense'):
        adj = torch.FloatTensor(adj.todense())
    else:
        adj = torch.FloatTensor(adj)
    labels = data['Label'].flatten()
    
    return features, adj, labels

def test_structure_reconstruction(dataset='reddit', structure_weight=0.1, num_epochs=100):
    """
    测试结构重构效果
    
    对比：
    1. 基线（无结构重构）
    2. 结构重构（degree loss）
    """
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset}")
    print(f"结构损失权重: {structure_weight}")
    print(f"{'='*60}")
    
    # 加载数据
    features, adj, labels = load_data(dataset)
    num_nodes = features.shape[0]
    
    print(f"节点数: {num_nodes}, 特征维度: {features.shape[1]}")
    print(f"异常比例: {labels.mean():.2%}")
    
    # 创建模型参数
    class Args:
        embedding_dim = 128
        pp_k = 4
        device = 0
        num_epoch = num_epochs
        batch_size = 512
        var = 0.1
        mean = 0.0
        sample_rate = 0.5
        outlier_beta = 0.1
        ring_R_min = 0.5
        ring_R_max = 2.0
    
    args = Args()
    device = torch.device('cuda:0')
    
    # 训练基线
    print(f"\n--- 基线训练 ---")
    torch.manual_seed(42)
    model_baseline = VoxGFormer(features.shape[1], args.embedding_dim, 'prelu', args).to(device)
    features_dev = features.to(device)
    adj_dev = adj.to(device)
    
    # 简单训练循环
    optimizer = torch.optim.Adam(model_baseline.parameters(), lr=0.001)
    
    # 使用全部数据训练
    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    train_idx = torch.where(train_mask)[0]
    normal_idx = torch.where(torch.from_numpy(labels) == 0)[0]
    
    best_auc_baseline = 0
    for epoch in range(num_epochs):
        model_baseline.train()
        optimizer.zero_grad()
        
        # 前向传播
        emb, emb_combine, logits, outlier_emb, noised_emb, loss_rec, loss_ring, _ = model_baseline(
            features_dev, adj_dev, train_idx, normal_idx[:int(len(normal_idx)*0.5)], True, args
        )
        
        # 简单损失
        loss = loss_rec + 0.1 * loss_ring
        
        loss.backward()
        optimizer.step()
        
        # 评估
        if (epoch + 1) % 10 == 0:
            model_baseline.eval()
            with torch.no_grad():
                emb_out, _, _, _, _, _, _, _ = model_baseline(features_dev, adj_dev, None, None, False, args)
                # 简单异常分数
                scores = torch.norm(emb_out.squeeze(0), dim=1).cpu().numpy()
                auc = roc_auc_score(labels, scores)
                best_auc_baseline = max(best_auc_baseline, auc)
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, AUC={auc:.4f}")
    
    print(f"\n基线最佳 AUC: {best_auc_baseline:.4f}")
    
    # 训练带结构重构的模型
    print(f"\n--- 结构重构训练 ---")
    torch.manual_seed(42)
    model_struct = VoxGFormer(features.shape[1], args.embedding_dim, 'prelu', args).to(device)
    
    optimizer = torch.optim.Adam(model_struct.parameters(), lr=0.001)
    
    best_auc_struct = 0
    for epoch in range(num_epochs):
        model_struct.train()
        optimizer.zero_grad()
        
        # 前向传播
        emb, emb_combine, logits, outlier_emb, noised_emb, loss_rec, loss_ring, _ = model_struct(
            features_dev, adj_dev, train_idx, normal_idx[:int(len(normal_idx)*0.5)], True, args
        )
        
        # 计算结构重构损失
        node_repr = emb.squeeze(0)
        loss_struct = compute_degree_loss(node_repr, adj_dev)
        
        # 总损失
        loss = loss_rec + 0.1 * loss_ring + structure_weight * loss_struct
        
        loss.backward()
        optimizer.step()
        
        # 评估
        if (epoch + 1) % 10 == 0:
            model_struct.eval()
            with torch.no_grad():
                emb_out, _, _, _, _, _, _, _ = model_struct(features_dev, adj_dev, None, None, False, args)
                scores = torch.norm(emb_out.squeeze(0), dim=1).cpu().numpy()
                auc = roc_auc_score(labels, scores)
                best_auc_struct = max(best_auc_struct, auc)
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, StructLoss={loss_struct.item():.4f}, AUC={auc:.4f}")
    
    print(f"\n结构重构最佳 AUC: {best_auc_struct:.4f}")
    
    # 对比结果
    improvement = (best_auc_struct - best_auc_baseline) / best_auc_baseline * 100
    print(f"\n{'='*60}")
    print(f"结果对比:")
    print(f"  基线 AUC: {best_auc_baseline:.4f}")
    print(f"  结构重构 AUC: {best_auc_struct:.4f}")
    print(f"  提升: {improvement:+.2f}%")
    print(f"{'='*60}")
    
    return best_auc_baseline, best_auc_struct

if __name__ == '__main__':
    test_structure_reconstruction('reddit', structure_weight=0.1, num_epochs=50)
