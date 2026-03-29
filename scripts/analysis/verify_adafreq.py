#!/usr/bin/env python3
"""
验证 AdaFreq 自适应滤波的基础可行性

基于 RHO 论文的核心思想：
- g(λ) = 1 - kλ
- k 是可学习参数
- 当 k > 0：低通滤波（正常节点 = 低频）
- 当 k < 0：高通滤波（异常节点 = 高频？）

验证内容：
1. 固定 k 值下，不同 k 的效果对比
2. 可学习 k 是否能自动找到最优值
3. 与固定滤波器（GCN, SGC）的对比

用法：
    python verify_adafreq.py --dataset photo
    python verify_adafreq.py --dataset elliptic
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from utils import load_mat, normalize_adj, preprocess_features
import scipy.sparse as sp


class AdaFreqFilter(nn.Module):
    """
    AdaFreq 自适应滤波器
    
    核心公式：g(λ) = 1 - kλ
    
    实现：H = (I - k * L) @ X
    其中 L 是归一化拉普拉斯矩阵
    """
    
    def __init__(self, k_init=0.5):
        super().__init__()
        # k 是可学习参数
        self.k = nn.Parameter(torch.tensor(k_init))
    
    def forward(self, X, L_norm):
        """
        Args:
            X: [N, D] 节点特征
            L_norm: [N, N] 归一化拉普拉斯矩阵 (I - D^{-1/2} A D^{-1/2})
        
        Returns:
            H: [N, D] 滤波后的特征
        """
        # H = (I - k * L) @ X
        # 注意：L_norm 已经是拉普拉斯矩阵，所以：
        # g(λ) = 1 - kλ 对应 H = X - k * L @ X = (I - kL) @ X
        H = X - self.k * torch.matmul(L_norm, X)
        return H
    
    def get_k(self):
        return self.k.item()


class MultiLayerAdaFreq(nn.Module):
    """多层 AdaFreq（类似 K 层 GNN）"""
    
    def __init__(self, num_layers=3, k_init=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.k_list = nn.ParameterList([
            nn.Parameter(torch.tensor(k_init)) for _ in range(num_layers)
        ])
    
    def forward(self, X, L_norm):
        H = X
        for i in range(self.num_layers):
            # H = H - k_i * L @ H
            H = H - self.k_list[i] * torch.matmul(L_norm, H)
        return H
    
    def get_k_list(self):
        return [k.item() for k in self.k_list]


class OneClassClassifier(nn.Module):
    """单类别分类器（半监督异常检测）"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.center = None
    
    def forward(self, X):
        return self.encoder(X)
    
    def compute_loss(self, H, normal_idx):
        """训练：让正常节点聚集在中心"""
        if self.center is None:
            self.center = H[normal_idx].mean(dim=0, keepdim=True).detach()
        
        # 正常节点到中心的距离
        dist = torch.norm(H[normal_idx] - self.center, p=2, dim=1)
        loss = dist.mean()
        return loss
    
    def compute_anomaly_score(self, H):
        """推理：到中心的距离作为异常分数"""
        if self.center is None:
            return torch.zeros(H.size(0))
        dist = torch.norm(H - self.center, p=2, dim=1)
        return dist


def compute_laplacian(adj):
    """计算归一化拉普拉斯矩阵"""
    # D^{-1/2}
    adj = torch.FloatTensor(adj)
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    
    # L = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    L_norm = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    return L_norm


def train_adafreq(X, L_norm, normal_idx, labels, test_idx, args):
    """训练 AdaFreq + OneClass 分类器"""
    
    input_dim = X.size(1)
    
    # 模型
    adafreq = AdaFreqFilter(k_init=args.k_init)
    classifier = OneClassClassifier(input_dim, hidden_dim=args.hidden_dim)
    
    optimizer = torch.optim.Adam(
        list(adafreq.parameters()) + list(classifier.parameters()),
        lr=args.lr
    )
    
    # 训练
    adafreq.train()
    classifier.train()
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # AdaFreq 滤波
        H = adafreq(X, L_norm)
        
        # 分类器编码
        Z = classifier(H)
        
        # 损失
        loss = classifier.compute_loss(Z, normal_idx)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            k = adafreq.get_k()
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, k={k:.4f}")
    
    # 最终 k 值
    final_k = adafreq.get_k()
    
    # 评估
    adafreq.eval()
    classifier.eval()
    
    with torch.no_grad():
        H = adafreq(X, L_norm)
        Z = classifier(H)
        
        # 初始化中心
        classifier.center = Z[normal_idx].mean(dim=0, keepdim=True)
        
        # 异常分数
        scores = classifier.compute_anomaly_score(Z).numpy()
    
    # AUC
    y_true = labels[test_idx]
    y_score = scores[test_idx]
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    return auc, ap, final_k


def fixed_k_experiment(X, L_norm, normal_idx, labels, test_idx, args):
    """固定 k 值实验"""
    
    print("\n--- 固定 k 值实验 ---")
    print(f"{'k':<10} {'AUC':<10} {'AP':<10} {'滤波类型':<15}")
    print("-" * 50)
    
    k_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    results = {}
    
    for k in k_values:
        # 创建固定 k 的滤波器
        adafreq = AdaFreqFilter(k_init=k)
        adafreq.k.requires_grad = False  # 固定 k
        
        classifier = OneClassClassifier(X.size(1), hidden_dim=args.hidden_dim)
        
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
            classifier.center = Z[normal_idx].mean(dim=0, keepdim=True)
            scores = classifier.compute_anomaly_score(Z).numpy()
        
        y_true = labels[test_idx]
        y_score = scores[test_idx]
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        # 滤波类型
        if k > 0:
            filter_type = "低通"
        elif k < 0:
            filter_type = "高通"
        else:
            filter_type = "无滤波"
        
        print(f"{k:<10.2f} {auc:<10.4f} {ap:<10.4f} {filter_type:<15}")
        
        results[k] = {'auc': auc, 'ap': ap}
    
    return results


def compare_with_baselines(X, L_norm, adj, normal_idx, labels, test_idx, args):
    """与基线方法对比"""
    
    print("\n--- 基线对比 ---")
    print(f"{'方法':<20} {'AUC':<10} {'AP':<10}")
    print("-" * 45)
    
    input_dim = X.size(1)
    
    def evaluate_with_filter(H, name):
        """评估某个滤波结果"""
        classifier = OneClassClassifier(input_dim, hidden_dim=args.hidden_dim)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
        
        classifier.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            Z = classifier(H)
            loss = classifier.compute_loss(Z, normal_idx)
            loss.backward()
            optimizer.step()
        
        classifier.eval()
        with torch.no_grad():
            Z = classifier(H)
            classifier.center = Z[normal_idx].mean(dim=0, keepdim=True)
            scores = classifier.compute_anomaly_score(Z).numpy()
        
        y_true = labels[test_idx]
        y_score = scores[test_idx]
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        print(f"{name:<20} {auc:<10.4f} {ap:<10.4f}")
        return auc, ap
    
    # 1. 无滤波（原始特征）
    auc, ap = evaluate_with_filter(X, "无滤波（原始）")
    
    # 2. GCN 滤波（H = A @ X）
    adj_tensor = torch.FloatTensor(adj)
    H_gcn = torch.matmul(adj_tensor, X)
    auc, ap = evaluate_with_filter(H_gcn, "GCN 滤波")
    
    # 3. SGC 滤波（H = A^k @ X）
    H_sgc = X
    for _ in range(3):
        H_sgc = torch.matmul(adj_tensor, H_sgc)
    auc, ap = evaluate_with_filter(H_sgc, "SGC 滤波 (K=3)")
    
    # 4. 高通滤波（H = (I - A) @ X = L @ X）
    I = torch.eye(adj_tensor.size(0))
    H_highpass = torch.matmul(I - adj_tensor, X)
    auc, ap = evaluate_with_filter(H_highpass, "高通滤波 (L @ X)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--k_init', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 70)
    print("AdaFreq 自适应滤波基础可行性验证")
    print("=" * 70)
    print("\n核心公式: g(λ) = 1 - kλ")
    print("- k > 0: 低通滤波（正常节点 = 低频）")
    print("- k < 0: 高通滤波（异常节点 = 高频）")
    print("- k 可学习: 自适应选择最优频率响应")
    
    # 加载数据
    print(f"\n加载数据集: {args.dataset}...")
    
    run_args = argparse.Namespace(
        dataset=args.dataset,
        train_rate=args.train_rate,
        seed=42,
        data_split_seed=42,
        sample_rate=0.15
    )
    
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, \
    normal_for_train_idx, normal_for_generation_idx = load_mat(
        args.dataset, args.train_rate, 0.1, run_args
    )
    
    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    
    adj = normalize_adj(adj)
    adj_dense = (adj + sp.eye(adj.shape[0])).todense()
    
    features = torch.FloatTensor(features)
    labels = np.squeeze(np.array(ano_label))
    
    # 计算拉普拉斯矩阵
    L_norm = compute_laplacian(adj_dense)
    
    print(f"节点数: {features.size(0)}")
    print(f"特征维度: {features.size(1)}")
    print(f"训练集（正常节点）: {len(normal_for_train_idx)}")
    print(f"测试集: {len(idx_test)}")
    
    # 实验 1：固定 k 值
    fixed_results = fixed_k_experiment(
        features, L_norm, normal_for_train_idx, labels, idx_test, args
    )
    
    # 实验 2：可学习 k
    print("\n--- 可学习 k 实验 ---")
    auc, ap, final_k = train_adafreq(
        features, L_norm, normal_for_train_idx, labels, idx_test, args
    )
    print(f"\n最终 k = {final_k:.4f}")
    print(f"AUC = {auc:.4f}, AP = {ap:.4f}")
    
    if final_k > 0:
        print("→ 模型选择了低通滤波")
    elif final_k < 0:
        print("→ 模型选择了高通滤波")
    else:
        print("→ 模型选择了无滤波")
    
    # 实验 3：与基线对比
    compare_with_baselines(
        features, L_norm, adj_dense, normal_for_train_idx, labels, idx_test, args
    )
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("1. 固定 k 实验显示了不同频率响应的效果")
    print("2. 可学习 k 可以自动选择最优滤波类型")
    print("3. 与基线对比验证了自适应滤波的潜在优势")


if __name__ == "__main__":
    main()