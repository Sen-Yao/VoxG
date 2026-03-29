#!/usr/bin/env python3
"""
对比谱域 Channel-wise k vs 特征域 Channel-wise k

关键区别：
- 特征域版本（错误）：对每个特征维度 d 有独立 k_d → H[:, d] = X[:, d] - k_d * L_X[:, d]
- 谱域版本（RHO）：对每个特征值 λ_i 有独立 k_i → g(λ_i) = 1 - k_i * λ_i

RHO 论文的谱域自适应滤波：
1. 谱分解：L = U Λ U^T
2. 自适应滤波：g(λ_i) = 1 - k_i * λ_i
3. 谱域应用：Ĥ = U diag(g(λ_1), ..., g(λ_n)) U^T X

用法：
    python compare_spectral_vs_feature_adafreq.py --dataset photo
    python compare_spectral_vs_feature_adafreq.py --dataset elliptic --num_eigenvalues 50
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from datetime import datetime
import time

from utils import load_mat, normalize_adj, preprocess_features


# ============================================================================
# 核心模块定义
# ============================================================================

class FeatureAdaFreq(nn.Module):
    """
    特征域 Channel-wise k（之前错误实现）
    
    对每个特征维度 d 有独立 k_d：
    H[:, d] = X[:, d] - k_d * L_X[:, d]
    """
    
    def __init__(self, num_features, k_init=0.5):
        super().__init__()
        self.num_features = num_features
        # 对每个特征维度有独立的 k
        self.k = nn.Parameter(torch.ones(num_features) * k_init)
    
    def forward(self, X, L_norm):
        """
        Args:
            X: [N, D] 节点特征
            L_norm: [N, N] 归一化拉普拉斯矩阵
        Returns:
            H: [N, D] 滤波后的特征
        """
        # L @ X
        L_X = torch.matmul(L_norm, X)  # [N, D]
        # H = X - k * L_X (对每个特征维度独立)
        H = X - self.k.unsqueeze(0) * L_X  # [N, D]
        return H
    
    def get_k(self):
        return self.k.detach().cpu().numpy()


class SpectralAdaFreq(nn.Module):
    """
    谱域 Channel-wise k（RHO 论文正确方式）
    
    对每个特征值 λ_i 有独立 k_i：
    g(λ_i) = 1 - k_i * λ_i
    Ĥ = U diag(g(λ_1), ..., g(λ_K)) U^T X
    """
    
    def __init__(self, num_eigenvalues, k_init=0.5):
        super().__init__()
        self.num_eigenvalues = num_eigenvalues
        # 对每个特征值有独立的 k
        self.k = nn.Parameter(torch.ones(num_eigenvalues) * k_init)
        
        # 谱基（固定，预计算）
        self.register_buffer('U', None)  # 特征向量 [N, K]
        self.register_buffer('eigenvalues', None)  # 特征值 [K]
    
    def set_spectral_basis(self, L_sparse):
        """
        预计算谱基（只计算一次）
        
        Args:
            L_sparse: scipy 稀疏拉普拉斯矩阵
        """
        print(f"  计算前 {self.num_eigenvalues} 个最小特征值...")
        start_time = time.time()
        
        # 使用稀疏矩阵的特征分解
        # which='SM' 表示最小特征值
        eigenvalues, eigenvectors = eigsh(L_sparse, k=self.num_eigenvalues, which='SM')
        
        elapsed = time.time() - start_time
        print(f"  特征分解完成，耗时 {elapsed:.2f}s")
        print(f"  特征值范围: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        
        self.U = torch.FloatTensor(eigenvectors)
        self.eigenvalues = torch.FloatTensor(eigenvalues)
    
    def forward(self, X):
        """
        Args:
            X: [N, D] 节点特征
        Returns:
            H: [N, D] 滤波后的特征
        """
        assert self.U is not None, "必须先调用 set_spectral_basis()"
        
        # 1. 变换到谱域
        # X_spectral = U^T @ X = [K, N] @ [N, D] = [K, D]
        X_spectral = self.U.T @ X
        
        # 2. 应用自适应滤波 g(λ) = 1 - k * λ
        # 对每个特征值有独立的 k
        g = 1 - self.k * self.eigenvalues  # [K]
        X_filtered = g.unsqueeze(1) * X_spectral  # [K, D]
        
        # 3. 变换回空间域
        # H = U @ X_filtered = [N, K] @ [K, D] = [N, D]
        H = self.U @ X_filtered
        
        return H
    
    def get_k(self):
        return self.k.detach().cpu().numpy()
    
    def get_filter_response(self):
        """获取滤波器响应函数 g(λ)"""
        with torch.no_grad():
            g = 1 - self.k * self.eigenvalues
        return g.cpu().numpy()


class OneClassClassifier(nn.Module):
    """单类别分类器"""
    
    def __init__(self, input_dim, hidden_dim=128, rep_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, rep_dim)
        )
        
        self.center = None
        self.rep_dim = rep_dim
    
    def forward(self, X):
        return self.encoder(X)
    
    def compute_loss(self, Z, normal_idx):
        if self.center is None:
            self.center = Z[normal_idx].mean(dim=0).detach()
        
        dist = torch.norm(Z - self.center, p=2, dim=1)
        normal_loss = dist[normal_idx].mean()
        
        return normal_loss
    
    def compute_anomaly_score(self, Z):
        if self.center is None:
            return torch.zeros(Z.size(0), device=Z.device)
        return torch.norm(Z - self.center, p=2, dim=1)


# ============================================================================
# 辅助函数
# ============================================================================

def compute_laplacian_sparse(adj):
    """
    计算稀疏归一化拉普拉斯矩阵 L = I - D^{-1/2} A D^{-1/2}
    返回 scipy 稀疏矩阵用于 eigsh
    """
    adj = sp.csr_matrix(adj)
    degree = np.array(adj.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    
    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    L = sp.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    return L.tocsr()


def compute_laplacian_dense(adj):
    """计算稠密归一化拉普拉斯矩阵"""
    adj = torch.FloatTensor(np.array(adj.todense() if sp.issparse(adj) else adj))
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    L = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    return L


def compute_gcn_filter(adj):
    """计算 GCN 滤波器"""
    adj = torch.FloatTensor(np.array(adj.todense() if sp.issparse(adj) else adj))
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return A_norm


def compute_metrics(y_true, y_score, threshold=None):
    """计算评估指标"""
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    if threshold is None:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        threshold = thresholds[np.argmax(j_scores)]
    
    y_pred = (y_score >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }


def print_metrics(metrics, prefix=""):
    """打印评估指标"""
    print(f"{prefix}AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f} | "
          f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
          f"F1: {metrics['f1']:.4f}")


# ============================================================================
# 实验函数
# ============================================================================

def train_feature_adafreq(X, L_norm, normal_idx, labels, test_idx, args):
    """
    训练特征域 AdaFreq（特征维度 k）
    """
    print("\n" + "=" * 70)
    print("特征域 Channel-wise k（特征维度 k_d）")
    print("=" * 70)
    print("公式: H[:, d] = X[:, d] - k_d * L_X[:, d]")
    print("特点: 对每个特征维度 d 有独立 k_d")
    
    input_dim = X.size(1)
    
    model = FeatureAdaFreq(input_dim, k_init=args.k_init)
    classifier = OneClassClassifier(input_dim, hidden_dim=args.hidden_dim)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr
    )
    
    print(f"\n参数量: 特征维度 k = {input_dim} 个")
    print(f"训练中...")
    
    best_auc = 0
    best_metrics = None
    training_time = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        optimizer.zero_grad()
        
        H = model(X, L_norm)
        Z = classifier(H)
        loss = classifier.compute_loss(Z, normal_idx)
        
        loss.backward()
        optimizer.step()
        
        training_time += time.time() - start_time
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            classifier.eval()
            with torch.no_grad():
                H = model(X, L_norm)
                Z = classifier(H)
                classifier.center = Z[normal_idx].mean(dim=0)
                scores = classifier.compute_anomaly_score(Z).numpy()
            
            metrics = compute_metrics(labels[test_idx], scores[test_idx])
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_metrics = metrics
            
            if (epoch + 1) % 50 == 0:
                k_values = model.get_k()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"k_mean={np.mean(k_values):.4f}, k_std={np.std(k_values):.4f}, "
                      f"AUC={metrics['auc']:.4f}")
            
            model.train()
            classifier.train()
    
    # 最终评估
    model.eval()
    classifier.eval()
    with torch.no_grad():
        H = model(X, L_norm)
        Z = classifier(H)
        classifier.center = Z[normal_idx].mean(dim=0)
        scores = classifier.compute_anomaly_score(Z).numpy()
    
    final_metrics = compute_metrics(labels[test_idx], scores[test_idx])
    k_values = model.get_k()
    
    print(f"\n最终结果:")
    print_metrics(final_metrics, "  ")
    print(f"  k 统计: mean={np.mean(k_values):.4f}, std={np.std(k_values):.4f}, "
          f"min={np.min(k_values):.4f}, max={np.max(k_values):.4f}")
    print(f"  训练时间: {training_time:.2f}s")
    
    return {
        'metrics': final_metrics,
        'k_values': k_values,
        'training_time': training_time
    }


def train_spectral_adafreq(X, L_sparse, normal_idx, labels, test_idx, args):
    """
    训练谱域 AdaFreq（谱域频率 k_i）
    """
    print("\n" + "=" * 70)
    print("谱域 Channel-wise k（RHO 论文方式）")
    print("=" * 70)
    print("公式: g(λ_i) = 1 - k_i * λ_i")
    print("特点: 对每个特征值 λ_i 有独立 k_i")
    
    input_dim = X.size(1)
    
    # 创建模型
    model = SpectralAdaFreq(args.num_eigenvalues, k_init=args.k_init)
    
    # 预计算谱基（这是最耗时的部分）
    print("\n预计算谱基...")
    model.set_spectral_basis(L_sparse)
    
    classifier = OneClassClassifier(input_dim, hidden_dim=args.hidden_dim)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr
    )
    
    print(f"\n参数量: 特征值 k = {args.num_eigenvalues} 个")
    print(f"训练中...")
    
    best_auc = 0
    best_metrics = None
    training_time = 0
    spectral_time = 0  # 谱变换时间
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # 谱域滤波
        spectral_start = time.time()
        H = model(X)
        spectral_time += time.time() - spectral_start
        
        Z = classifier(H)
        loss = classifier.compute_loss(Z, normal_idx)
        
        loss.backward()
        optimizer.step()
        
        training_time += time.time() - start_time
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            classifier.eval()
            with torch.no_grad():
                H = model(X)
                Z = classifier(H)
                classifier.center = Z[normal_idx].mean(dim=0)
                scores = classifier.compute_anomaly_score(Z).numpy()
            
            metrics = compute_metrics(labels[test_idx], scores[test_idx])
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_metrics = metrics
            
            if (epoch + 1) % 50 == 0:
                k_values = model.get_k()
                g = model.get_filter_response()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"k_mean={np.mean(k_values):.4f}, k_std={np.std(k_values):.4f}, "
                      f"AUC={metrics['auc']:.4f}")
            
            model.train()
            classifier.train()
    
    # 最终评估
    model.eval()
    classifier.eval()
    with torch.no_grad():
        H = model(X)
        Z = classifier(H)
        classifier.center = Z[normal_idx].mean(dim=0)
        scores = classifier.compute_anomaly_score(Z).numpy()
    
    final_metrics = compute_metrics(labels[test_idx], scores[test_idx])
    k_values = model.get_k()
    g = model.get_filter_response()
    eigenvalues = model.eigenvalues.cpu().numpy()
    
    print(f"\n最终结果:")
    print_metrics(final_metrics, "  ")
    print(f"  k 统计: mean={np.mean(k_values):.4f}, std={np.std(k_values):.4f}, "
          f"min={np.min(k_values):.4f}, max={np.max(k_values):.4f}")
    print(f"  滤波器响应 g(λ): mean={np.mean(g):.4f}, std={np.std(g):.4f}")
    print(f"  训练时间: {training_time:.2f}s (谱变换: {spectral_time:.2f}s)")
    
    return {
        'metrics': final_metrics,
        'k_values': k_values,
        'eigenvalues': eigenvalues,
        'filter_response': g,
        'training_time': training_time,
        'spectral_time': spectral_time
    }


def analyze_k_distributions(feature_k, spectral_k, eigenvalues):
    """分析两种 k 值分布的差异"""
    print("\n" + "=" * 70)
    print("k 值分布对比分析")
    print("=" * 70)
    
    # 特征域 k 分析
    print("\n【特征域 k】（对每个特征维度）")
    print(f"  参数量: {len(feature_k)}")
    print(f"  均值: {np.mean(feature_k):.4f}")
    print(f"  标准差: {np.std(feature_k):.4f}")
    print(f"  最小值: {np.min(feature_k):.4f}")
    print(f"  最大值: {np.max(feature_k):.4f}")
    print(f"  中位数: {np.median(feature_k):.4f}")
    
    # 谱域 k 分析
    print("\n【谱域 k】（对每个特征值）")
    print(f"  参数量: {len(spectral_k)}")
    print(f"  均值: {np.mean(spectral_k):.4f}")
    print(f"  标准差: {np.std(spectral_k):.4f}")
    print(f"  最小值: {np.min(spectral_k):.4f}")
    print(f"  最大值: {np.max(spectral_k):.4f}")
    print(f"  中位数: {np.median(spectral_k):.4f}")
    
    # 滤波器响应分析
    g = 1 - spectral_k * eigenvalues
    print("\n【滤波器响应 g(λ) = 1 - k * λ】")
    print(f"  均值: {np.mean(g):.4f}")
    print(f"  标准差: {np.std(g):.4f}")
    print(f"  最小值: {np.min(g):.4f}")
    print(f"  最大值: {np.max(g):.4f}")
    
    # 分析不同频率的滤波行为
    print("\n【不同频率的滤波行为】")
    
    # 按特征值排序（小特征值 = 低频）
    sorted_idx = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_k = spectral_k[sorted_idx]
    sorted_g = g[sorted_idx]
    
    # 分成低、中、高频
    n = len(eigenvalues)
    low_idx = sorted_idx[:n//3]
    mid_idx = sorted_idx[n//3:2*n//3]
    high_idx = sorted_idx[2*n//3:]
    
    print(f"\n  低频分量（最小特征值）:")
    print(f"    特征值范围: [{eigenvalues[low_idx].min():.6f}, {eigenvalues[low_idx].max():.6f}]")
    print(f"    k 均值: {np.mean(spectral_k[low_idx]):.4f}")
    print(f"    g(λ) 均值: {np.mean(g[low_idx]):.4f} ({'低通' if np.mean(g[low_idx]) > 0.5 else '高通'})")
    
    print(f"\n  中频分量:")
    print(f"    特征值范围: [{eigenvalues[mid_idx].min():.6f}, {eigenvalues[mid_idx].max():.6f}]")
    print(f"    k 均值: {np.mean(spectral_k[mid_idx]):.4f}")
    print(f"    g(λ) 均值: {np.mean(g[mid_idx]):.4f}")
    
    print(f"\n  高频分量（最大特征值）:")
    print(f"    特征值范围: [{eigenvalues[high_idx].min():.6f}, {eigenvalues[high_idx].max():.6f}]")
    print(f"    k 均值: {np.mean(spectral_k[high_idx]):.4f}")
    print(f"    g(λ) 均值: {np.mean(g[high_idx]):.4f} ({'低通' if np.mean(g[high_idx]) > 0.5 else '高通'})")
    
    return {
        'feature_k_stats': {
            'mean': np.mean(feature_k),
            'std': np.std(feature_k),
            'min': np.min(feature_k),
            'max': np.max(feature_k)
        },
        'spectral_k_stats': {
            'mean': np.mean(spectral_k),
            'std': np.std(spectral_k),
            'min': np.min(spectral_k),
            'max': np.max(spectral_k)
        },
        'filter_response_stats': {
            'mean': np.mean(g),
            'std': np.std(g),
            'min': np.min(g),
            'max': np.max(g)
        }
    }


def run_comparison(args):
    """运行完整对比实验"""
    print("=" * 70)
    print("谱域 vs 特征域 Channel-wise k 对比实验")
    print("=" * 70)
    print(f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {args.dataset}")
    print(f"特征值数量: {args.num_eigenvalues}")
    print(f"训练比例: {args.train_rate}")
    print(f"随机种子: {args.seed}")
    
    print("\n" + "-" * 70)
    print("核心区别说明")
    print("-" * 70)
    print("特征域版本（错误）: H[:, d] = X[:, d] - k_d * L_X[:, d]")
    print("  - 对每个特征维度 d 有独立 k_d")
    print("  - 不同特征列有不同滤波强度")
    print("")
    print("谱域版本（RHO 正确）: g(λ_i) = 1 - k_i * λ_i")
    print("  - 对每个特征值 λ_i 有独立 k_i")
    print("  - 不同频率分量有不同滤波强度")
    print("  - 低频（小 λ）vs 高频（大 λ）的不同处理")
    
    # 加载数据
    print(f"\n{'=' * 70}")
    print("加载数据...")
    print("=" * 70)
    
    run_args = argparse.Namespace(
        dataset=args.dataset,
        train_rate=args.train_rate,
        seed=args.seed,
        data_split_seed=args.seed,
        sample_rate=0.15
    )
    
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, \
    normal_for_train_idx, normal_for_generation_idx = load_mat(
        args.dataset, args.train_rate, 0.1, run_args
    )
    
    # 预处理
    if args.dataset.lower() in ['amazon', 'tf_finace', 'tfinance', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    
    adj = normalize_adj(adj)
    adj_dense = (adj + sp.eye(adj.shape[0])).todense()
    
    features = torch.FloatTensor(features)
    labels = np.squeeze(np.array(ano_label))
    
    # 计算矩阵
    L_sparse = compute_laplacian_sparse(adj_dense)  # 稀疏拉普拉斯（用于特征分解）
    L_dense = compute_laplacian_dense(adj_dense)  # 稠密拉普拉斯（用于特征域版本）
    
    # 数据集信息
    num_anomaly = np.sum(labels[idx_test] == 1)
    dataset_info = {
        'num_nodes': features.size(0),
        'num_features': features.size(1),
        'num_train_normal': len(normal_for_train_idx),
        'num_test': len(idx_test),
        'anomaly_ratio': num_anomaly / len(idx_test)
    }
    
    print(f"节点数: {dataset_info['num_nodes']}")
    print(f"特征维度: {dataset_info['num_features']}")
    print(f"训练集（正常节点）: {dataset_info['num_train_normal']}")
    print(f"测试集: {dataset_info['num_test']}")
    print(f"测试集异常比例: {dataset_info['anomaly_ratio']:.2%}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results = {}
    
    # 实验 1: 特征域 AdaFreq
    results['feature'] = train_feature_adafreq(
        features, L_dense, normal_for_train_idx, labels, idx_test, args
    )
    
    # 重置随机种子确保公平对比
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 实验 2: 谱域 AdaFreq
    results['spectral'] = train_spectral_adafreq(
        features, L_sparse, normal_for_train_idx, labels, idx_test, args
    )
    
    # 分析 k 值分布
    results['k_analysis'] = analyze_k_distributions(
        results['feature']['k_values'],
        results['spectral']['k_values'],
        results['spectral']['eigenvalues']
    )
    
    # 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    
    feature_auc = results['feature']['metrics']['auc']
    spectral_auc = results['spectral']['metrics']['auc']
    improvement = (spectral_auc - feature_auc) / feature_auc * 100
    
    print(f"\n{'方法':<30} {'AUC':<10} {'AP':<10} {'F1':<10} {'训练时间':<15}")
    print("-" * 75)
    print(f"{'特征域 k（特征维度）':<30} {feature_auc:<10.4f} {results['feature']['metrics']['ap']:<10.4f} "
          f"{results['feature']['metrics']['f1']:<10.4f} {results['feature']['training_time']:<15.2f}s")
    print(f"{'谱域 k（RHO 论文）':<30} {spectral_auc:<10.4f} {results['spectral']['metrics']['ap']:<10.4f} "
          f"{results['spectral']['metrics']['f1']:<10.4f} {results['spectral']['training_time']:<15.2f}s")
    
    print(f"\n改进: {improvement:+.2f}%")
    
    if spectral_auc > feature_auc:
        print("✅ 谱域 Channel-wise k 优于特征域版本")
    else:
        print("⚠️ 谱域 Channel-wise k 未优于特征域版本")
    
    print("\n" + "-" * 70)
    print("关键发现")
    print("-" * 70)
    
    findings = []
    
    # 发现 1: 性能对比
    if spectral_auc > feature_auc:
        findings.append(f"谱域 k 比 特征域 k 提升 {improvement:.2f}% AUC")
    else:
        findings.append(f"谱域 k 比 特征域 k 降低 {-improvement:.2f}% AUC")
    
    # 发现 2: 参数量对比
    findings.append(f"参数量: 特征域 {features.size(1)} vs 谱域 {args.num_eigenvalues}")
    
    # 发现 3: k 值分布
    feature_k_std = results['k_analysis']['feature_k_stats']['std']
    spectral_k_std = results['k_analysis']['spectral_k_stats']['std']
    
    if spectral_k_std > feature_k_std:
        findings.append(f"谱域 k 变异性更大 ({spectral_k_std:.4f} vs {feature_k_std:.4f})")
    else:
        findings.append(f"特征域 k 变异性更大 ({feature_k_std:.4f} vs {spectral_k_std:.4f})")
    
    # 发现 4: 滤波器响应
    g_mean = results['k_analysis']['filter_response_stats']['mean']
    if g_mean > 0.5:
        findings.append(f"滤波器整体呈低通特性 (g(λ) 均值 = {g_mean:.4f})")
    else:
        findings.append(f"滤波器整体呈高通特性 (g(λ) 均值 = {g_mean:.4f})")
    
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")
    
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='谱域 vs 特征域 Channel-wise k 对比')
    parser.add_argument('--dataset', type=str, default='photo',
                        choices=['photo', 'amazon', 'reddit', 'elliptic', 'tfinance', 'tolokers'])
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--num_eigenvalues', type=int, default=50,
                        help='谱域版本使用的特征值数量')
    parser.add_argument('--k_init', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_comparison(args)


if __name__ == "__main__":
    main()