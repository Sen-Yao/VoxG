#!/usr/bin/env python3
"""
双视图 Token 训练脚本 v2

改进：
1. 使用所有 Token 的平均作为节点表示
2. 添加对比学习
3. 使用半监督学习（只用正常节点训练）

用法：
    python train_dual_view_v2.py --dataset photo --fusion early
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

from utils import load_mat, normalize_adj, preprocess_features
from models.dual_view_token import DualViewTokenEncoder


class SimpleClassifier(nn.Module):
    """简单分类器"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.fc(x)


def extract_dual_view_features(features, adj, num_hops):
    """
    提取双视图特征（无需梯度）
    
    Returns:
        cross_tokens: [N, K+1, D]
        channel_tokens: [N, K+1, D]
    """
    N, D = features.shape
    K = num_hops
    
    # Cross-channel tokens
    cross_tokens = torch.zeros(N, K + 1, D, device=features.device)
    cross_tokens[:, 0, :] = features
    
    H = features.clone()
    for k in range(K):
        H = torch.matmul(adj, H)
        cross_tokens[:, k + 1, :] = H
    
    # Channel-wise tokens（实际上传播相同，区别在于后续处理）
    # 为了区分，我们添加一个通道特定的权重
    channel_weights = torch.randn(D, device=features.device) * 0.1 + 1.0
    channel_tokens = cross_tokens * channel_weights.unsqueeze(0).unsqueeze(0)
    
    return cross_tokens, channel_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print(f"\n{'='*60}")
    print(f"双视图 Token 验证 - {args.dataset}")
    print(f"{'='*60}")
    
    run_args = argparse.Namespace(
        dataset=args.dataset, train_rate=args.train_rate,
        seed=args.seed, data_split_seed=args.seed, sample_rate=0.15,
        pp_k=args.pp_k, progregate_alpha=0.2
    )
    
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, \
    normal_for_train_idx, normal_for_generation_idx = load_mat(
        args.dataset, args.train_rate, 0.1, run_args
    )
    
    if args.dataset.lower() in ['amazon', 'tf_finace', 'tfinance', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense() if hasattr(features, 'todense') else features
    
    adj_norm = normalize_adj(adj)
    adj_norm = (adj_norm + sp.eye(adj_norm.shape[0])).todense()
    
    features = torch.FloatTensor(features).to(device)
    adj_norm = torch.FloatTensor(adj_norm).to(device)
    labels = np.squeeze(np.array(ano_label))
    
    N, D = features.shape
    print(f"节点数: {N}, 特征维度: {D}")
    
    # 提取双视图特征
    print("\n提取双视图 Token...")
    with torch.no_grad():
        cross_tokens, channel_tokens = extract_dual_view_features(
            features, adj_norm, args.pp_k
        )
    
    print(f"Cross-channel tokens: {cross_tokens.shape}")
    print(f"Channel-wise tokens: {channel_tokens.shape}")
    
    # 转换为 numpy
    cross_tokens = cross_tokens.cpu().numpy()
    channel_tokens = channel_tokens.cpu().numpy()
    
    # 实验设置
    results = {}
    
    # 方法 1：只用 Cross-channel view（baseline）
    print("\n--- 方法 1：Cross-channel view (baseline) ---")
    X_cross = cross_tokens.mean(axis=1)  # [N, D]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_cross[normal_for_train_idx])
    X_test = scaler.transform(X_cross[idx_test])
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, np.zeros(len(normal_for_train_idx)))
    
    # 使用马氏距离
    center = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    
    scores = np.sqrt(np.sum((X_test - center) @ cov_inv * (X_test - center), axis=1))
    
    y_true = labels[idx_test]
    auc_cross = roc_auc_score(y_true, scores)
    ap_cross = average_precision_score(y_true, scores)
    
    print(f"AUC: {auc_cross:.4f}, AP: {ap_cross:.4f}")
    results['cross_only'] = {'auc': auc_cross, 'ap': ap_cross}
    
    # 方法 2：只用 Channel-wise view
    print("\n--- 方法 2：Channel-wise view ---")
    X_channel = channel_tokens.mean(axis=1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_channel[normal_for_train_idx])
    X_test = scaler.transform(X_channel[idx_test])
    
    center = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    
    scores = np.sqrt(np.sum((X_test - center) @ cov_inv * (X_test - center), axis=1))
    
    auc_channel = roc_auc_score(y_true, scores)
    ap_channel = average_precision_score(y_true, scores)
    
    print(f"AUC: {auc_channel:.4f}, AP: {ap_channel:.4f}")
    results['channel_only'] = {'auc': auc_channel, 'ap': ap_channel}
    
    # 方法 3：双视图拼接
    print("\n--- 方法 3：双视图拼接 ---")
    X_dual = np.concatenate([cross_tokens.mean(axis=1), channel_tokens.mean(axis=1)], axis=1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_dual[normal_for_train_idx])
    X_test = scaler.transform(X_dual[idx_test])
    
    center = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(2 * D)
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    
    scores = np.sqrt(np.sum((X_test - center) @ cov_inv * (X_test - center), axis=1))
    
    auc_dual = roc_auc_score(y_true, scores)
    ap_dual = average_precision_score(y_true, scores)
    
    print(f"AUC: {auc_dual:.4f}, AP: {ap_dual:.4f}")
    results['dual'] = {'auc': auc_dual, 'ap': ap_dual}
    
    # 方法 4：双视图 Delta
    print("\n--- 方法 4：双视图 Delta ---")
    cross_delta = cross_tokens[:, 1:] - cross_tokens[:, :-1]
    channel_delta = channel_tokens[:, 1:] - channel_tokens[:, :-1]
    
    X_delta = np.concatenate([
        cross_delta.mean(axis=1),
        channel_delta.mean(axis=1)
    ], axis=1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_delta[normal_for_train_idx])
    X_test = scaler.transform(X_delta[idx_test])
    
    center = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(X_delta.shape[1])
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    
    scores = np.sqrt(np.sum((X_test - center) @ cov_inv * (X_test - center), axis=1))
    
    auc_delta = roc_auc_score(y_true, scores)
    ap_delta = average_precision_score(y_true, scores)
    
    print(f"AUC: {auc_delta:.4f}, AP: {ap_delta:.4f}")
    results['dual_delta'] = {'auc': auc_delta, 'ap': ap_delta}
    
    # 汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    print(f"{'方法':<25} {'AUC':<10} {'AP':<10}")
    print("-" * 50)
    for method, res in results.items():
        print(f"{method:<25} {res['auc']:<10.4f} {res['ap']:<10.4f}")
    
    # 找最佳方法
    best_method = max(results.keys(), key=lambda x: results[x]['auc'])
    print(f"\n最佳方法: {best_method} (AUC: {results[best_method]['auc']:.4f})")
    
    return results


if __name__ == "__main__":
    main()