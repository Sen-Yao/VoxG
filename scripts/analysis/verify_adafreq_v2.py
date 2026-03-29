#!/usr/bin/env python3
"""
验证 AdaFreq 自适应滤波的全面可行性 (v2)

基于 RHO 论文的核心思想：
- g(λ) = 1 - kλ
- k 是可学习参数
- channel-wise k: 每个特征通道有不同的 k（RHO 的核心创新）

验证内容：
1. 多层 AdaFreq（K 层，每层 k 可学习）
2. Channel-wise k vs Single k（核心对比）
3. 与 RHO 论文一致的评估方式
4. 更多基线对比（不同频率的固定滤波器）
5. 详细分析报告

用法：
    python verify_adafreq_v2.py --dataset photo
    python verify_adafreq_v2.py --dataset elliptic
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

from utils import load_mat, normalize_adj, preprocess_features
import scipy.sparse as sp


# ============================================================================
# 核心模块定义
# ============================================================================

class AdaFreqFilter(nn.Module):
    """
    单层 AdaFreq 自适应滤波器
    
    核心公式：g(λ) = 1 - kλ
    实现：H = (I - k * L) @ X
    """
    
    def __init__(self, in_dim, k_init=0.5, channel_wise=False):
        super().__init__()
        self.channel_wise = channel_wise
        self.in_dim = in_dim
        
        if channel_wise:
            # Channel-wise k: 每个特征通道有不同的 k
            self.k = nn.Parameter(torch.ones(in_dim) * k_init)
        else:
            # Single k: 所有通道共享一个 k
            self.k = nn.Parameter(torch.tensor(k_init))
    
    def forward(self, X, L_norm):
        """
        Args:
            X: [N, D] 节点特征
            L_norm: [N, N] 归一化拉普拉斯矩阵
        
        Returns:
            H: [N, D] 滤波后的特征
        """
        # L @ X: [N, D]
        L_X = torch.matmul(L_norm, X)
        
        if self.channel_wise:
            # k: [D], L_X: [N, D]
            # H = X - k * L_X (广播)
            H = X - self.k.unsqueeze(0) * L_X
        else:
            H = X - self.k * L_X
        
        return H
    
    def get_k(self):
        if self.channel_wise:
            return self.k.detach().cpu().numpy()
        return self.k.item()


class MultiLayerAdaFreq(nn.Module):
    """
    多层 AdaFreq 滤波器
    
    类似 K 层 GNN，每层有独立的可学习 k 参数
    """
    
    def __init__(self, in_dim, num_layers=3, k_init=0.5, channel_wise=False):
        super().__init__()
        self.num_layers = num_layers
        self.channel_wise = channel_wise
        self.in_dim = in_dim
        
        self.filters = nn.ModuleList([
            AdaFreqFilter(in_dim, k_init=k_init, channel_wise=channel_wise)
            for _ in range(num_layers)
        ])
    
    def forward(self, X, L_norm):
        H = X
        for f in self.filters:
            H = f(H, L_norm)
        return H
    
    def get_k_list(self):
        return [f.get_k() for f in self.filters]


class OneClassHypersphere(nn.Module):
    """
    单类别超球面分类器（RHO 风格）
    
    核心思想：将正常节点映射到超球面内，异常节点映射到超球面外
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128], rep_dim=64):
        super().__init__()
        
        # 编码器
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, rep_dim))
        self.encoder = nn.Sequential(*layers)
        
        # 超球面中心（可学习）
        self.center = None
        self.rep_dim = rep_dim
    
    def forward(self, X):
        return self.encoder(X)
    
    def init_center(self, X, normal_idx):
        """用正常节点初始化中心"""
        with torch.no_grad():
            Z = self.encoder(X[normal_idx])
            self.center = Z.mean(dim=0)
    
    def compute_loss(self, Z, normal_idx, radius=None):
        """
        计算超球面损失（类似 Deep SVDD）
        
        目标：正常节点靠近中心，异常节点远离中心
        """
        if self.center is None:
            self.center = Z[normal_idx].mean(dim=0).detach()
        
        # 到中心的距离
        dist = torch.norm(Z - self.center, p=2, dim=1)
        
        # 正常节点：最小化距离
        normal_loss = dist[normal_idx].mean()
        
        return normal_loss
    
    def compute_anomaly_score(self, Z):
        """到中心的距离作为异常分数"""
        if self.center is None:
            return torch.zeros(Z.size(0), device=Z.device)
        return torch.norm(Z - self.center, p=2, dim=1)


class DualViewEncoder(nn.Module):
    """
    双视图编码器（用于对比学习）
    
    视图 1: 原始特征 + AdaFreq 滤波
    视图 2: 图结构信息（GCN 滤波）
    """
    
    def __init__(self, input_dim, hidden_dim=128, rep_dim=64):
        super().__init__()
        
        # 视图 1: AdaFreq 特征编码器
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim)
        )
        
        # 视图 2: GCN 特征编码器
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim)
        )
        
        self.center1 = None
        self.center2 = None
    
    def forward(self, H_adafreq, H_gcn):
        Z1 = self.encoder1(H_adafreq)
        Z2 = self.encoder2(H_gcn)
        return Z1, Z2
    
    def compute_loss(self, Z1, Z2, normal_idx, temp=0.5):
        """
        对比学习损失 + 单类别损失
        
        对比学习：让同一节点的两个视图表示相似
        单类别：让正常节点聚集
        """
        # 初始化中心
        if self.center1 is None:
            self.center1 = Z1[normal_idx].mean(dim=0).detach()
        if self.center2 is None:
            self.center2 = Z2[normal_idx].mean(dim=0).detach()
        
        # 单类别损失
        dist1 = torch.norm(Z1 - self.center1, p=2, dim=1)
        dist2 = torch.norm(Z2 - self.center2, p=2, dim=1)
        oc_loss = dist1[normal_idx].mean() + dist2[normal_idx].mean()
        
        # 对比损失（InfoNCE 风格）
        Z1_norm = F.normalize(Z1[normal_idx], dim=1)
        Z2_norm = F.normalize(Z2[normal_idx], dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.matmul(Z1_norm, Z2_norm.T) / temp
        
        # 正样本：对角线
        pos_sim = torch.diag(sim_matrix)
        
        # InfoNCE 损失
        labels = torch.arange(len(normal_idx), device=Z1.device)
        contra_loss = F.cross_entropy(sim_matrix, labels)
        
        return oc_loss + 0.1 * contra_loss
    
    def compute_anomaly_score(self, Z1, Z2):
        """双视图融合的异常分数"""
        if self.center1 is None or self.center2 is None:
            return torch.zeros(Z1.size(0), device=Z1.device)
        
        score1 = torch.norm(Z1 - self.center1, p=2, dim=1)
        score2 = torch.norm(Z2 - self.center2, p=2, dim=1)
        
        # 融合两个视图
        return (score1 + score2) / 2


# ============================================================================
# 辅助函数
# ============================================================================

def compute_laplacian(adj):
    """计算归一化拉普拉斯矩阵 L = I - D^{-1/2} A D^{-1/2}"""
    adj = torch.FloatTensor(adj)
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    L_norm = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    return L_norm


def compute_gcn_filter(adj, K=1):
    """计算 GCN 滤波器（A^k @ X）"""
    adj = torch.FloatTensor(adj)
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
        # 使用 F1 最优阈值
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # 选择 Youden's J statistic 最大的点
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

def experiment_single_vs_channel_k(X, L_norm, normal_idx, labels, test_idx, args):
    """
    核心实验：Single k vs Channel-wise k
    
    这是 RHO 论文的核心创新点
    """
    print("\n" + "=" * 70)
    print("实验 1: Single k vs Channel-wise k")
    print("=" * 70)
    
    input_dim = X.size(1)
    results = {}
    
    # 1. Single k
    print("\n--- Single k（所有通道共享一个 k）---")
    adafreq_single = AdaFreqFilter(input_dim, k_init=args.k_init, channel_wise=False)
    classifier_single = OneClassHypersphere(input_dim, hidden_dims=[args.hidden_dim])
    
    optimizer = torch.optim.Adam(
        list(adafreq_single.parameters()) + list(classifier_single.parameters()),
        lr=args.lr
    )
    
    best_auc = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        H = adafreq_single(X, L_norm)
        Z = classifier_single(H)
        loss = classifier_single.compute_loss(Z, normal_idx)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            classifier_single.eval()
            with torch.no_grad():
                H = adafreq_single(X, L_norm)
                Z = classifier_single(H)
                classifier_single.center = Z[normal_idx].mean(dim=0)
                scores = classifier_single.compute_anomaly_score(Z).numpy()
            
            metrics = compute_metrics(labels[test_idx], scores[test_idx])
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, k={adafreq_single.get_k():.4f}, AUC={metrics['auc']:.4f}")
            classifier_single.train()
    
    # 最终评估
    adafreq_single.eval()
    classifier_single.eval()
    with torch.no_grad():
        H = adafreq_single(X, L_norm)
        Z = classifier_single(H)
        classifier_single.center = Z[normal_idx].mean(dim=0)
        scores = classifier_single.compute_anomaly_score(Z).numpy()
    
    metrics_single = compute_metrics(labels[test_idx], scores[test_idx])
    k_single = adafreq_single.get_k()
    
    print(f"\nSingle k 结果:")
    print_metrics(metrics_single, "  ")
    print(f"  最终 k = {k_single:.4f}")
    
    results['single_k'] = {
        'metrics': metrics_single,
        'k': k_single
    }
    
    # 2. Channel-wise k
    print("\n--- Channel-wise k（每个通道有独立的 k）---")
    adafreq_channel = AdaFreqFilter(input_dim, k_init=args.k_init, channel_wise=True)
    classifier_channel = OneClassHypersphere(input_dim, hidden_dims=[args.hidden_dim])
    
    optimizer = torch.optim.Adam(
        list(adafreq_channel.parameters()) + list(classifier_channel.parameters()),
        lr=args.lr
    )
    
    best_auc = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        H = adafreq_channel(X, L_norm)
        Z = classifier_channel(H)
        loss = classifier_channel.compute_loss(Z, normal_idx)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            classifier_channel.eval()
            with torch.no_grad():
                H = adafreq_channel(X, L_norm)
                Z = classifier_channel(H)
                classifier_channel.center = Z[normal_idx].mean(dim=0)
                scores = classifier_channel.compute_anomaly_score(Z).numpy()
            
            metrics = compute_metrics(labels[test_idx], scores[test_idx])
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
            
            if (epoch + 1) % 50 == 0:
                k_channel = adafreq_channel.get_k()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, k_mean={np.mean(k_channel):.4f}, k_std={np.std(k_channel):.4f}, AUC={metrics['auc']:.4f}")
            classifier_channel.train()
    
    # 最终评估
    adafreq_channel.eval()
    classifier_channel.eval()
    with torch.no_grad():
        H = adafreq_channel(X, L_norm)
        Z = classifier_channel(H)
        classifier_channel.center = Z[normal_idx].mean(dim=0)
        scores = classifier_channel.compute_anomaly_score(Z).numpy()
    
    metrics_channel = compute_metrics(labels[test_idx], scores[test_idx])
    k_channel = adafreq_channel.get_k()
    
    print(f"\nChannel-wise k 结果:")
    print_metrics(metrics_channel, "  ")
    print(f"  k 统计: mean={np.mean(k_channel):.4f}, std={np.std(k_channel):.4f}, "
          f"min={np.min(k_channel):.4f}, max={np.max(k_channel):.4f}")
    
    results['channel_k'] = {
        'metrics': metrics_channel,
        'k_stats': {
            'mean': np.mean(k_channel),
            'std': np.std(k_channel),
            'min': np.min(k_channel),
            'max': np.max(k_channel)
        },
        'k_values': k_channel
    }
    
    # 3. 对比总结
    print("\n--- 对比总结 ---")
    improvement = (metrics_channel['auc'] - metrics_single['auc']) / metrics_single['auc'] * 100
    print(f"Channel-wise k 相比 Single k:")
    print(f"  AUC: {metrics_single['auc']:.4f} → {metrics_channel['auc']:.4f} ({improvement:+.2f}%)")
    print(f"  AP:  {metrics_single['ap']:.4f} → {metrics_channel['ap']:.4f}")
    print(f"  F1:  {metrics_single['f1']:.4f} → {metrics_channel['f1']:.4f}")
    
    if metrics_channel['auc'] > metrics_single['auc']:
        print("\n✅ 结论: Channel-wise k 优于 Single k（验证了 RHO 的核心假设）")
    else:
        print("\n⚠️ 结论: Channel-wise k 未优于 Single k（需要进一步分析）")
    
    return results


def experiment_multilayer_adafreq(X, L_norm, normal_idx, labels, test_idx, args):
    """
    实验 2: 多层 AdaFreq
    """
    print("\n" + "=" * 70)
    print("实验 2: 多层 AdaFreq（K 层，每层独立 k）")
    print("=" * 70)
    
    input_dim = X.size(1)
    results = {}
    
    for num_layers in [1, 2, 3, 4]:
        print(f"\n--- {num_layers} 层 AdaFreq ---")
        
        adafreq = MultiLayerAdaFreq(input_dim, num_layers=num_layers, 
                                    k_init=args.k_init, channel_wise=True)
        classifier = OneClassHypersphere(input_dim, hidden_dims=[args.hidden_dim])
        
        optimizer = torch.optim.Adam(
            list(adafreq.parameters()) + list(classifier.parameters()),
            lr=args.lr
        )
        
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            H = adafreq(X, L_norm)
            Z = classifier(H)
            loss = classifier.compute_loss(Z, normal_idx)
            
            loss.backward()
            optimizer.step()
        
        # 评估
        adafreq.eval()
        classifier.eval()
        with torch.no_grad():
            H = adafreq(X, L_norm)
            Z = classifier(H)
            classifier.center = Z[normal_idx].mean(dim=0)
            scores = classifier.compute_anomaly_score(Z).numpy()
        
        metrics = compute_metrics(labels[test_idx], scores[test_idx])
        k_list = adafreq.get_k_list()
        
        print(f"  AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f}")
        print(f"  各层 k 均值: {[np.mean(k) if isinstance(k, np.ndarray) else k for k in k_list]}")
        
        results[num_layers] = {
            'metrics': metrics,
            'k_list': k_list
        }
    
    # 找最佳层数
    best_layers = max(results.keys(), key=lambda x: results[x]['metrics']['auc'])
    print(f"\n最佳层数: {best_layers} (AUC: {results[best_layers]['metrics']['auc']:.4f})")
    
    return results


def experiment_fixed_k_baselines(X, L_norm, A_norm, normal_idx, labels, test_idx, args):
    """
    实验 3: 固定 k 值基线对比
    """
    print("\n" + "=" * 70)
    print("实验 3: 固定 k 值基线对比")
    print("=" * 70)
    
    input_dim = X.size(1)
    results = {}
    
    # 不同频率的滤波器
    filter_configs = [
        # (k值, 名称, 滤波类型)
        (-2.0, "k=-2.0", "高通"),
        (-1.0, "k=-1.0", "高通"),
        (-0.5, "k=-0.5", "高通"),
        (0.0, "k=0 (无滤波)", "无"),
        (0.5, "k=0.5", "低通"),
        (1.0, "k=1.0 (GCN)", "低通"),
        (1.5, "k=1.5", "低通"),
        (2.0, "k=2.0", "低通"),
    ]
    
    print(f"\n{'配置':<20} {'AUC':<10} {'AP':<10} {'F1':<10} {'滤波类型':<10}")
    print("-" * 60)
    
    for k, name, filter_type in filter_configs:
        # 创建固定 k 的滤波器
        adafreq = AdaFreqFilter(input_dim, k_init=k, channel_wise=False)
        adafreq.k.requires_grad = False
        
        classifier = OneClassHypersphere(input_dim, hidden_dims=[args.hidden_dim])
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
        
        # 训练
        adafreq.eval()
        classifier.train()
        
        with torch.no_grad():
            H = adafreq(X, L_norm)
        
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            Z = classifier(H)
            loss = classifier.compute_loss(Z, normal_idx)
            loss.backward()
            optimizer.step()
        
        # 评估
        classifier.eval()
        with torch.no_grad():
            Z = classifier(H)
            classifier.center = Z[normal_idx].mean(dim=0)
            scores = classifier.compute_anomaly_score(Z).numpy()
        
        metrics = compute_metrics(labels[test_idx], scores[test_idx])
        
        print(f"{name:<20} {metrics['auc']:<10.4f} {metrics['ap']:<10.4f} {metrics['f1']:<10.4f} {filter_type:<10}")
        
        results[name] = {
            'k': k,
            'metrics': metrics,
            'filter_type': filter_type
        }
    
    # GCN 和 SGC 基线
    print(f"\n--- 图神经网络基线 ---")
    print(f"{'方法':<20} {'AUC':<10} {'AP':<10} {'F1':<10}")
    print("-" * 50)
    
    # GCN 滤波
    H_gcn = torch.matmul(A_norm, X)
    classifier = OneClassHypersphere(input_dim, hidden_dims=[args.hidden_dim])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    classifier.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        Z = classifier(H_gcn)
        loss = classifier.compute_loss(Z, normal_idx)
        loss.backward()
        optimizer.step()
    
    classifier.eval()
    with torch.no_grad():
        Z = classifier(H_gcn)
        classifier.center = Z[normal_idx].mean(dim=0)
        scores = classifier.compute_anomaly_score(Z).numpy()
    
    metrics = compute_metrics(labels[test_idx], scores[test_idx])
    print(f"{'GCN (1层)':<20} {metrics['auc']:<10.4f} {metrics['ap']:<10.4f} {metrics['f1']:<10.4f}")
    results['GCN'] = {'metrics': metrics}
    
    # SGC (K=3)
    H_sgc = X
    for _ in range(3):
        H_sgc = torch.matmul(A_norm, H_sgc)
    
    classifier = OneClassHypersphere(input_dim, hidden_dims=[args.hidden_dim])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    classifier.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        Z = classifier(H_sgc)
        loss = classifier.compute_loss(Z, normal_idx)
        loss.backward()
        optimizer.step()
    
    classifier.eval()
    with torch.no_grad():
        Z = classifier(H_sgc)
        classifier.center = Z[normal_idx].mean(dim=0)
        scores = classifier.compute_anomaly_score(Z).numpy()
    
    metrics = compute_metrics(labels[test_idx], scores[test_idx])
    print(f"{'SGC (K=3)':<20} {metrics['auc']:<10.4f} {metrics['ap']:<10.4f} {metrics['f1']:<10.4f}")
    results['SGC'] = {'metrics': metrics}
    
    return results


def experiment_dual_view(X, L_norm, A_norm, normal_idx, labels, test_idx, args):
    """
    实验 4: 双视图对比学习
    """
    print("\n" + "=" * 70)
    print("实验 4: 双视图对比学习")
    print("=" * 70)
    
    input_dim = X.size(1)
    
    # AdaFreq 滤波
    adafreq = AdaFreqFilter(input_dim, k_init=args.k_init, channel_wise=True)
    
    # 双视图编码器
    dual_encoder = DualViewEncoder(input_dim, hidden_dim=args.hidden_dim)
    
    optimizer = torch.optim.Adam(
        list(adafreq.parameters()) + list(dual_encoder.parameters()),
        lr=args.lr
    )
    
    print("\n训练双视图模型...")
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # AdaFreq 视图
        H_adafreq = adafreq(X, L_norm)
        
        # GCN 视图
        H_gcn = torch.matmul(A_norm, X)
        
        # 双视图编码
        Z1, Z2 = dual_encoder(H_adafreq, H_gcn)
        
        # 损失
        loss = dual_encoder.compute_loss(Z1, Z2, normal_idx)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            dual_encoder.eval()
            with torch.no_grad():
                H_adafreq = adafreq(X, L_norm)
                H_gcn = torch.matmul(A_norm, X)
                Z1, Z2 = dual_encoder(H_adafreq, H_gcn)
                scores = dual_encoder.compute_anomaly_score(Z1, Z2).numpy()
            
            metrics = compute_metrics(labels[test_idx], scores[test_idx])
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, AUC={metrics['auc']:.4f}")
            dual_encoder.train()
    
    # 最终评估
    adafreq.eval()
    dual_encoder.eval()
    with torch.no_grad():
        H_adafreq = adafreq(X, L_norm)
        H_gcn = torch.matmul(A_norm, X)
        Z1, Z2 = dual_encoder(H_adafreq, H_gcn)
        scores = dual_encoder.compute_anomaly_score(Z1, Z2).numpy()
    
    metrics = compute_metrics(labels[test_idx], scores[test_idx])
    
    print(f"\n双视图模型结果:")
    print_metrics(metrics, "  ")
    
    k_channel = adafreq.get_k()
    print(f"  k 统计: mean={np.mean(k_channel):.4f}, std={np.std(k_channel):.4f}")
    
    return {
        'metrics': metrics,
        'k_stats': {
            'mean': np.mean(k_channel),
            'std': np.std(k_channel)
        }
    }


def analyze_k_distribution(k_values, feature_names=None, top_k=10):
    """
    分析 channel-wise k 的分布
    
    这有助于理解哪些特征通道更重要
    """
    print("\n" + "=" * 70)
    print("Channel-wise k 分布分析")
    print("=" * 70)
    
    k_values = np.array(k_values)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  均值: {np.mean(k_values):.4f}")
    print(f"  标准差: {np.std(k_values):.4f}")
    print(f"  最小值: {np.min(k_values):.4f}")
    print(f"  最大值: {np.max(k_values):.4f}")
    print(f"  中位数: {np.median(k_values):.4f}")
    
    # 分位数
    percentiles = [10, 25, 50, 75, 90]
    print(f"\n分位数:")
    for p in percentiles:
        print(f"  {p}%: {np.percentile(k_values, p):.4f}")
    
    # k > 0（低通）vs k < 0（高通）
    low_pass = np.sum(k_values > 0)
    high_pass = np.sum(k_values < 0)
    neutral = np.sum(np.abs(k_values) < 0.01)
    
    print(f"\n滤波类型分布:")
    print(f"  低通滤波 (k > 0): {low_pass} ({100*low_pass/len(k_values):.1f}%)")
    print(f"  高通滤波 (k < 0): {high_pass} ({100*high_pass/len(k_values):.1f}%)")
    print(f"  近中性 (|k| < 0.01): {neutral} ({100*neutral/len(k_values):.1f}%)")
    
    # 最极端的 k 值
    print(f"\n最极端的 k 值:")
    
    # 最大的 k（强低通）
    top_indices = np.argsort(k_values)[-top_k:][::-1]
    print(f"  最大的 k 值（强低通）:")
    for i, idx in enumerate(top_indices):
        print(f"    #{i+1}: 通道 {idx}, k = {k_values[idx]:.4f}")
    
    # 最小的 k（强高通）
    bottom_indices = np.argsort(k_values)[:top_k]
    print(f"  最小的 k 值（强高通）:")
    for i, idx in enumerate(bottom_indices):
        print(f"    #{i+1}: 通道 {idx}, k = {k_values[idx]:.4f}")
    
    return {
        'mean': np.mean(k_values),
        'std': np.std(k_values),
        'min': np.min(k_values),
        'max': np.max(k_values),
        'median': np.median(k_values),
        'low_pass_ratio': low_pass / len(k_values),
        'high_pass_ratio': high_pass / len(k_values)
    }


def generate_report(results, args, dataset_info):
    """生成详细分析报告"""
    print("\n" + "=" * 70)
    print("📊 详细分析报告")
    print("=" * 70)
    
    print(f"\n数据集: {args.dataset}")
    print(f"节点数: {dataset_info['num_nodes']}")
    print(f"特征维度: {dataset_info['num_features']}")
    print(f"训练正常节点: {dataset_info['num_train_normal']}")
    print(f"测试节点: {dataset_info['num_test']}")
    print(f"异常比例: {dataset_info['anomaly_ratio']:.2%}")
    
    print("\n" + "-" * 70)
    print("实验结果汇总")
    print("-" * 70)
    
    # 实验 1: Single vs Channel-wise k
    if 'exp1' in results:
        print("\n【实验 1】Single k vs Channel-wise k")
        single = results['exp1']['single_k']['metrics']
        channel = results['exp1']['channel_k']['metrics']
        
        print(f"  Single k:      AUC={single['auc']:.4f}, AP={single['ap']:.4f}, F1={single['f1']:.4f}")
        print(f"  Channel-wise k: AUC={channel['auc']:.4f}, AP={channel['ap']:.4f}, F1={channel['f1']:.4f}")
        
        improvement = (channel['auc'] - single['auc']) / single['auc'] * 100
        if improvement > 0:
            print(f"  ✅ Channel-wise k 提升: {improvement:+.2f}%")
        else:
            print(f"  ⚠️ Channel-wise k 未提升: {improvement:+.2f}%")
    
    # 实验 2: 多层 AdaFreq
    if 'exp2' in results:
        print("\n【实验 2】多层 AdaFreq")
        for layers, data in results['exp2'].items():
            m = data['metrics']
            print(f"  {layers} 层: AUC={m['auc']:.4f}, AP={m['ap']:.4f}")
    
    # 实验 3: 固定 k 基线
    if 'exp3' in results:
        print("\n【实验 3】固定 k 基线")
        # 找最佳固定 k
        best_fixed = max(
            [(k, v) for k, v in results['exp3'].items() if k not in ['GCN', 'SGC']],
            key=lambda x: x[1]['metrics']['auc']
        )
        print(f"  最佳固定 k: {best_fixed[0]}, AUC={best_fixed[1]['metrics']['auc']:.4f}")
        
        if 'GCN' in results['exp3']:
            print(f"  GCN: AUC={results['exp3']['GCN']['metrics']['auc']:.4f}")
        if 'SGC' in results['exp3']:
            print(f"  SGC: AUC={results['exp3']['SGC']['metrics']['auc']:.4f}")
    
    # 实验 4: 双视图
    if 'exp4' in results:
        print("\n【实验 4】双视图对比学习")
        m = results['exp4']['metrics']
        print(f"  双视图: AUC={m['auc']:.4f}, AP={m['ap']:.4f}, F1={m['f1']:.4f}")
    
    print("\n" + "-" * 70)
    print("核心结论")
    print("-" * 70)
    
    conclusions = []
    
    if 'exp1' in results:
        single_auc = results['exp1']['single_k']['metrics']['auc']
        channel_auc = results['exp1']['channel_k']['metrics']['auc']
        
        if channel_auc > single_auc:
            conclusions.append("✅ Channel-wise k 优于 Single k，验证了 RHO 的核心假设")
        else:
            conclusions.append("⚠️ Channel-wise k 未优于 Single k，需要进一步分析原因")
    
    if 'exp2' in results:
        best_layers = max(results['exp2'].keys(), key=lambda x: results['exp2'][x]['metrics']['auc'])
        conclusions.append(f"✅ 最佳层数为 {best_layers} 层")
    
    if 'exp3' in results and 'exp1' in results:
        channel_auc = results['exp1']['channel_k']['metrics']['auc']
        # 找最佳固定 k
        best_fixed_auc = max(
            [v['metrics']['auc'] for k, v in results['exp3'].items() if k not in ['GCN', 'SGC']]
        )
        
        if channel_auc > best_fixed_auc:
            conclusions.append(f"✅ 可学习的 channel-wise k 优于最佳固定 k ({channel_auc:.4f} > {best_fixed_auc:.4f})")
        else:
            conclusions.append(f"⚠️ 可学习的 channel-wise k 未优于最佳固定 k")
    
    for i, c in enumerate(conclusions, 1):
        print(f"  {i}. {c}")
    
    return conclusions


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='AdaFreq 自适应滤波全面验证 v2')
    parser.add_argument('--dataset', type=str, default='photo', 
                        choices=['photo', 'amazon', 'reddit', 'elliptic', 'tfinance', 'tolokers'])
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--k_init', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("AdaFreq 自适应滤波全面验证 v2")
    print("=" * 70)
    print(f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {args.dataset}")
    print(f"训练比例: {args.train_rate}")
    print(f"随机种子: {args.seed}")
    
    print("\n核心公式: g(λ) = 1 - kλ")
    print("- k > 0: 低通滤波（正常节点 = 低频）")
    print("- k < 0: 高通滤波（异常节点 = 高频）")
    print("- Channel-wise k: 每个特征通道独立学习最优滤波强度")
    
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
    L_norm = compute_laplacian(adj_dense)
    A_norm = compute_gcn_filter(adj_dense)
    
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
    
    # 存储所有实验结果
    all_results = {}
    
    # 实验 1: Single k vs Channel-wise k（核心实验）
    all_results['exp1'] = experiment_single_vs_channel_k(
        features, L_norm, normal_for_train_idx, labels, idx_test, args
    )
    
    # 分析 channel-wise k 分布
    k_analysis = analyze_k_distribution(all_results['exp1']['channel_k']['k_values'])
    all_results['k_analysis'] = k_analysis
    
    # 实验 2: 多层 AdaFreq
    all_results['exp2'] = experiment_multilayer_adafreq(
        features, L_norm, normal_for_train_idx, labels, idx_test, args
    )
    
    # 实验 3: 固定 k 基线对比
    all_results['exp3'] = experiment_fixed_k_baselines(
        features, L_norm, A_norm, normal_for_train_idx, labels, idx_test, args
    )
    
    # 实验 4: 双视图对比学习
    all_results['exp4'] = experiment_dual_view(
        features, L_norm, A_norm, normal_for_train_idx, labels, idx_test, args
    )
    
    # 生成报告
    conclusions = generate_report(all_results, args, dataset_info)
    
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
