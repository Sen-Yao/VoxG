#!/usr/bin/env python3
"""
快速消融验证：ISP 是否提供额外增益

方法：使用简单分类器（LogisticRegression）验证不同输入组合
- 组合A：原始 Token
- 组合B：原始 Token + Delta
- 组合C：原始 Token + Delta + ISP

如果 C > B > A，说明 ISP 有增益。

用法：
    python ablation_isp_gain.py --dataset photo
    python ablation_isp_gain.py --dataset elliptic
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA

from utils import load_mat, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def compute_features(node_tokens):
    """
    计算不同类型的特征
    
    Returns:
        original: [N, D] - 原始 Token（平均）
        delta: [N, D] - Delta 向量（平均）
        isp: [N] - ISP（方向一致性）
        delta_norms: [N, K] - Delta 范数
    """
    N, K_plus_1, D = node_tokens.shape
    K = K_plus_1 - 1
    
    # Delta 向量
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]  # [N, K, D]
    
    # 原始 Token（平均）
    original = node_tokens.mean(dim=1)  # [N, D]
    
    # Delta 向量（平均）
    delta = delta_vectors.mean(dim=1)  # [N, D]
    
    # Delta 范数
    delta_norms = torch.norm(delta_vectors, p=2, dim=-1)  # [N, K]
    
    # ISP：方向一致性
    delta_norms_expanded = delta_norms.unsqueeze(-1) + 1e-8
    delta_directions = delta_vectors / delta_norms_expanded  # [N, K, D]
    
    consistencies = []
    for k in range(K - 1):
        dir_k = delta_directions[:, k, :]
        dir_k1 = delta_directions[:, k + 1, :]
        cos_sim = torch.sum(dir_k * dir_k1, dim=-1)
        consistencies.append(cos_sim)
    
    isp = torch.stack(consistencies, dim=1).mean(dim=1)  # [N]
    
    return original, delta, isp, delta_norms


def quick_ablation(dataset_name, original, delta, isp, delta_norms, labels, normal_idx, test_idx):
    """
    快速消融实验
    
    Args:
        normal_idx: 训练集（只有正常节点）
        test_idx: 测试集
    """
    labels = np.array(labels).flatten()
    
    original = original.numpy()
    delta = delta.numpy()
    isp = isp.numpy().reshape(-1, 1)
    delta_norms = delta_norms.numpy()
    
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"训练集（正常节点）: {len(normal_idx)}")
    print(f"测试集: {len(test_idx)}")
    
    results = {}
    
    # ========== 组合A：原始 Token ==========
    print(f"\n--- 组合A：原始 Token ---")
    X_train = original[normal_idx]
    X_test = original[test_idx]
    y_train = labels[normal_idx]
    y_test = labels[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    probas = clf.predict_proba(X_test)[:, 1]
    auc_a = roc_auc_score(y_test, probas)
    ap_a = average_precision_score(y_test, probas)
    
    print(f"AUC: {auc_a:.4f}, AP: {ap_a:.4f}")
    results['A'] = {'auc': auc_a, 'ap': ap_a}
    
    # ========== 组合B：原始 + Delta ==========
    print(f"\n--- 组合B：原始 + Delta ---")
    X_train = np.hstack([original[normal_idx], delta[normal_idx]])
    X_test = np.hstack([original[test_idx], delta[test_idx]])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = labels[normal_idx]
    
    # 如果维度太高，使用 PCA
    if X_train.shape[1] > 500:
        print(f"  使用 PCA 降维: {X_train.shape[1]} -> 256")
        pca = PCA(n_components=256)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(scaler.transform(np.hstack([original[test_idx], delta[test_idx]])))
    else:
        X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    probas = clf.predict_proba(X_test)[:, 1]
    auc_b = roc_auc_score(y_test, probas)
    ap_b = average_precision_score(y_test, probas)
    
    print(f"AUC: {auc_b:.4f}, AP: {ap_b:.4f}")
    results['B'] = {'auc': auc_b, 'ap': ap_b}
    
    # ========== 组合C：原始 + Delta + ISP ==========
    print(f"\n--- 组合C：原始 + Delta + ISP ---")
    X_train = np.hstack([original[normal_idx], delta[normal_idx], isp[normal_idx]])
    X_test = np.hstack([original[test_idx], delta[test_idx], isp[test_idx]])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    if X_train.shape[1] > 500:
        print(f"  使用 PCA 降维: {X_train.shape[1]} -> 256")
        pca = PCA(n_components=256)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(scaler.transform(np.hstack([original[test_idx], delta[test_idx], isp[test_idx]])))
    else:
        X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    probas = clf.predict_proba(X_test)[:, 1]
    auc_c = roc_auc_score(y_test, probas)
    ap_c = average_precision_score(y_test, probas)
    
    print(f"AUC: {auc_c:.4f}, AP: {ap_c:.4f}")
    results['C'] = {'auc': auc_c, 'ap': ap_c}
    
    # ========== 组合D：原始 + Delta + Delta 范数 ==========
    print(f"\n--- 组合D：原始 + Delta + Delta范数 ---")
    X_train = np.hstack([original[normal_idx], delta[normal_idx], delta_norms[normal_idx]])
    X_test = np.hstack([original[test_idx], delta[test_idx], delta_norms[test_idx]])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    if X_train.shape[1] > 500:
        print(f"  使用 PCA 降维: {X_train.shape[1]} -> 256")
        pca = PCA(n_components=256)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(scaler.transform(np.hstack([original[test_idx], delta[test_idx], delta_norms[test_idx]])))
    else:
        X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    probas = clf.predict_proba(X_test)[:, 1]
    auc_d = roc_auc_score(y_test, probas)
    ap_d = average_precision_score(y_test, probas)
    
    print(f"AUC: {auc_d:.4f}, AP: {ap_d:.4f}")
    results['D'] = {'auc': auc_d, 'ap': ap_d}
    
    # ========== 汇总 ==========
    print(f"\n{'='*70}")
    print("消融实验汇总")
    print(f"{'='*70}")
    print(f"{'组合':<35} {'AUC':<10} {'AP':<10}")
    print("-" * 55)
    print(f"{'A: 原始 Token':<35} {auc_a:<10.4f} {ap_a:<10.4f}")
    print(f"{'B: 原始 + Delta':<35} {auc_b:<10.4f} {ap_b:<10.4f}")
    print(f"{'C: 原始 + Delta + ISP':<35} {auc_c:<10.4f} {ap_c:<10.4f}")
    print(f"{'D: 原始 + Delta + Delta范数':<35} {auc_d:<10.4f} {ap_d:<10.4f}")
    
    print(f"\n--- 增益分析 ---")
    print(f"Delta 增益 (B - A): AUC {auc_b - auc_a:+.4f}, AP {ap_b - ap_a:+.4f}")
    print(f"ISP 增益 (C - B): AUC {auc_c - auc_b:+.4f}, AP {ap_c - ap_b:+.4f}")
    print(f"Delta范数 增益 (D - B): AUC {auc_d - auc_b:+.4f}, AP {ap_d - ap_b:+.4f}")
    
    if auc_c > auc_b:
        print(f"\n✅ 结论: ISP 提供了额外增益")
    else:
        print(f"\n❌ 结论: ISP 未提供额外增益")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    
    print("=" * 70)
    print("快速消融验证：ISP 是否提供额外增益？")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_args = argparse.Namespace(
        dataset=args.dataset,
        pp_k=args.pp_k,
        progregate_alpha=args.alpha,
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
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    
    print(f"\n计算 hop2token...")
    node_tokens = nagphormer_tokenization(features, adj, run_args)
    print(f"Token 形状: {node_tokens.shape}")
    
    print(f"计算特征...")
    original, delta, isp, delta_norms = compute_features(node_tokens)
    
    print(f"原始 Token: {original.shape}")
    print(f"Delta: {delta.shape}")
    print(f"ISP: {isp.shape}")
    print(f"Delta 范数: {delta_norms.shape}")
    
    labels_array = np.squeeze(np.array(ano_label))
    
    results = quick_ablation(
        args.dataset, original, delta, isp, delta_norms, 
        labels_array, normal_for_train_idx, idx_test
    )
    
    return results


if __name__ == "__main__":
    main()