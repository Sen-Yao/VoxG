"""
诊断模块：分析特征重构 vs 结构重构
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_feature_reconstruction_error(model, features, adj, args):
    """
    计算特征重构误差
    
    返回：每个节点的特征重构误差
    """
    model.eval()
    with torch.no_grad():
        # 获取模型输出
        hidden, recon_x = model(features, adj, args)
        
        # 计算特征重构误差 (MSE)
        feat_error = torch.mean((recon_x - features) ** 2, dim=1)
        
    return feat_error.cpu().numpy()

def compute_structure_reconstruction_error(features, adj, ppr_matrix=None):
    """
    计算结构重构误差（基于 PPR 或邻接矩阵）
    
    返回：每个节点的结构异常分数
    """
    # 方法 1：基于度的异常分数
    degrees = np.array(adj.sum(axis=1)).flatten()
    
    # 方法 2：基于局部聚类系数
    # 简化版：使用邻居的平均度
    adj_np = adj.cpu().numpy() if torch.is_tensor(adj) else adj
    neighbor_avg_degree = []
    for i in range(adj_np.shape[0]):
        neighbors = adj_np[i].nonzero()[1] if hasattr(adj_np[i], 'nonzero') else np.where(adj_np[i] > 0)[0]
        if len(neighbors) > 0:
            neighbor_avg_degree.append(degrees[neighbors].mean())
        else:
            neighbor_avg_degree.append(0)
    
    # 结构异常分数：与邻居平均度的差异
    struct_score = np.abs(degrees - np.array(neighbor_avg_degree))
    
    return struct_score

def diagnose_structure_reconstruction(model, features, adj, labels, args, verbose=True):
    """
    诊断结构重构效果
    
    分析：
    1. 特征重构误差 vs 结构重构误差的相关性
    2. 分别评估两类异常检测效果
    3. 分析最优组合权重
    """
    # 计算两类误差
    feat_error = compute_feature_reconstruction_error(model, features, adj, args)
    struct_error = compute_structure_reconstruction_error(features, adj)
    
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    # 1. 相关性分析
    correlation = np.corrcoef(feat_error, struct_error)[0, 1]
    
    # 2. 分别评估 AUC
    feat_auc = roc_auc_score(labels_np, feat_error)
    struct_auc = roc_auc_score(labels_np, struct_error)
    
    # 3. 寻找最优组合权重
    best_auc = 0
    best_lambda = 0
    for lam in np.arange(0, 2.1, 0.1):
        combined = feat_error + lam * struct_error
        try:
            auc = roc_auc_score(labels_np, combined)
            if auc > best_auc:
                best_auc = auc
                best_lambda = lam
        except:
            pass
    
    if verbose:
        print("=" * 50)
        print("结构重构诊断报告")
        print("=" * 50)
        print(f"特征-结构误差相关性: {correlation:.4f}")
        print(f"")
        print(f"单独评估:")
        print(f"  仅特征重构 AUC: {feat_auc:.4f}")
        print(f"  仅结构重构 AUC: {struct_auc:.4f}")
        print(f"")
        print(f"最优组合:")
        print(f"  最优 λ: {best_lambda:.2f}")
        print(f"  组合 AUC: {best_auc:.4f}")
        print("=" * 50)
    
    return {
        'correlation': correlation,
        'feat_auc': feat_auc,
        'struct_auc': struct_auc,
        'best_lambda': best_lambda,
        'best_auc': best_auc,
        'feat_error': feat_error,
        'struct_error': struct_error
    }

def analyze_dataset_characteristics(features, adj, labels):
    """
    分析数据集特性，判断结构/特征异常的主导性
    """
    # 特征维度
    feat_dim = features.shape[1]
    
    # 图结构特性
    adj_np = adj.cpu().numpy() if torch.is_tensor(adj) else adj
    num_nodes = adj_np.shape[0]
    num_edges = adj_np.sum() // 2
    
    # 异常比例
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    anomaly_ratio = labels_np.mean()
    
    # 平均度
    degrees = np.array(adj.sum(axis=1)).flatten()
    avg_degree = degrees.mean()
    
    # 度分布偏度
    degree_skew = ((degrees - avg_degree) ** 3).mean() / (degrees.std() ** 3) if degrees.std() > 0 else 0
    
    print("\n数据集特性分析:")
    print(f"  节点数: {num_nodes}")
    print(f"  边数: {int(num_edges)}")
    print(f"  特征维度: {feat_dim}")
    print(f"  异常比例: {anomaly_ratio:.2%}")
    print(f"  平均度: {avg_degree:.2f}")
    print(f"  度分布偏度: {degree_skew:.2f}")
    
    # 判断哪种异常可能更主导
    if degree_skew > 1.0:
        print(f"  → 疑似**结构异常**主导（度分布右偏）")
    elif feat_dim > 100:
        print(f"  → 疑似**特征异常**主导（高维特征）")
    else:
        print(f"  → 两类异常可能都重要")
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'feat_dim': feat_dim,
        'anomaly_ratio': anomaly_ratio,
        'avg_degree': avg_degree,
        'degree_skew': degree_skew
    }
