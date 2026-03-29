#!/usr/bin/env python3
"""
快速消融验证：ISP 是否提供额外增益（无监督版本）

方法：使用马氏距离验证不同输入组合的效果
- 训练集：只有正常节点
- 测试：计算测试集到正常中心的马氏距离

组合：
- A：原始 Token
- B：原始 + Delta
- C：原始 + Delta + ISP
- D：原始 + Delta + Delta范数

如果 C > B > A，说明 ISP 有增益。

用法：
    python ablation_isp_gain_unsupervised.py --dataset photo
    python ablation_isp_gain_unsupervised.py --dataset elliptic
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA

from utils import load_mat, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def compute_features(node_tokens):
    """计算不同类型的特征"""
    N, K_plus_1, D = node_tokens.shape
    K = K_plus_1 - 1
    
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]
    original = node_tokens.mean(dim=1)
    delta = delta_vectors.mean(dim=1)
    delta_norms = torch.norm(delta_vectors, p=2, dim=-1)
    
    # ISP：方向一致性
    delta_norms_expanded = delta_norms.unsqueeze(-1) + 1e-8
    delta_directions = delta_vectors / delta_norms_expanded
    
    consistencies = []
    for k in range(K - 1):
        dir_k = delta_directions[:, k, :]
        dir_k1 = delta_directions[:, k + 1, :]
        cos_sim = torch.sum(dir_k * dir_k1, dim=-1)
        consistencies.append(cos_sim)
    
    isp = torch.stack(consistencies, dim=1).mean(dim=1)
    
    return original, delta, isp, delta_norms


def mahalanobis_distance(X, center, cov_inv):
    """计算马氏距离"""
    diff = X - center
    dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return dist


def quick_ablation_unsupervised(dataset_name, original, delta, isp, delta_norms, labels, normal_idx, test_idx):
    """无监督消融实验"""
    labels = np.array(labels).flatten()
    
    original = original.numpy()
    delta = delta.numpy()
    isp_np = isp.numpy().reshape(-1, 1)
    delta_norms_np = delta_norms.numpy()
    
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"训练集（正常节点）: {len(normal_idx)}")
    print(f"测试集: {len(test_idx)}, 异常: {sum(labels[test_idx]==1)}")
    
    results = {}
    
    def evaluate_feature(X_train_normal, X_test, name):
        """评估单个特征组合"""
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_normal)
        X_test_scaled = scaler.transform(X_test)
        
        # 计算正常中心
        center = X_train_scaled.mean(axis=0)
        
        # 计算协方差矩阵
        cov = np.cov(X_train_scaled.T) + 1e-6 * np.eye(X_train_scaled.shape[1])
        try:
            cov_inv = np.linalg.inv(cov)
        except:
            cov_inv = np.linalg.pinv(cov)
        
        # 计算马氏距离
        dist = mahalanobis_distance(X_test_scaled, center, cov_inv)
        
        # AUC（距离越大越异常）
        auc = roc_auc_score(labels[test_idx], dist)
        ap = average_precision_score(labels[test_idx], dist)
        
        print(f"{name}: AUC={auc:.4f}, AP={ap:.4f}")
        return auc, ap
    
    # ========== 组合A：原始 Token ==========
    print(f"\n--- 消融实验 ---")
    auc_a, ap_a = evaluate_feature(
        original[normal_idx], original[test_idx], "A: 原始 Token"
    )
    results['A'] = {'auc': auc_a, 'ap': ap_a}
    
    # ========== 组合B：原始 + Delta ==========
    X_train = np.hstack([original[normal_idx], delta[normal_idx]])
    X_test = np.hstack([original[test_idx], delta[test_idx]])
    auc_b, ap_b = evaluate_feature(X_train, X_test, "B: 原始 + Delta")
    results['B'] = {'auc': auc_b, 'ap': ap_b}
    
    # ========== 组合C：原始 + Delta + ISP ==========
    X_train = np.hstack([original[normal_idx], delta[normal_idx], isp_np[normal_idx]])
    X_test = np.hstack([original[test_idx], delta[test_idx], isp_np[test_idx]])
    auc_c, ap_c = evaluate_feature(X_train, X_test, "C: 原始 + Delta + ISP")
    results['C'] = {'auc': auc_c, 'ap': ap_c}
    
    # ========== 组合D：原始 + Delta + Delta范数 ==========
    X_train = np.hstack([original[normal_idx], delta[normal_idx], delta_norms_np[normal_idx]])
    X_test = np.hstack([original[test_idx], delta[test_idx], delta_norms_np[test_idx]])
    auc_d, ap_d = evaluate_feature(X_train, X_test, "D: 原始 + Delta + Delta范数")
    results['D'] = {'auc': auc_d, 'ap': ap_d}
    
    # ========== 组合E：原始 + ISP ==========
    X_train = np.hstack([original[normal_idx], isp_np[normal_idx]])
    X_test = np.hstack([original[test_idx], isp_np[test_idx]])
    auc_e, ap_e = evaluate_feature(X_train, X_test, "E: 原始 + ISP")
    results['E'] = {'auc': auc_e, 'ap': ap_e}
    
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
    print(f"{'E: 原始 + ISP':<35} {auc_e:<10.4f} {ap_e:<10.4f}")
    
    print(f"\n--- 增益分析 ---")
    print(f"Delta 增益 (B - A): AUC {auc_b - auc_a:+.4f}, AP {ap_b - ap_a:+.4f}")
    print(f"ISP 增益 (C - B): AUC {auc_c - auc_b:+.4f}, AP {ap_c - ap_b:+.4f}")
    print(f"Delta范数 增益 (D - B): AUC {auc_d - auc_b:+.4f}, AP {ap_d - ap_b:+.4f}")
    print(f"单独 ISP 增益 (E - A): AUC {auc_e - auc_a:+.4f}, AP {ap_e - ap_a:+.4f}")
    
    print(f"\n--- 结论 ---")
    if auc_c > auc_b and auc_c > auc_a:
        print(f"✅ ISP 提供了额外增益 (C 是最佳)")
    elif auc_c > auc_b:
        print(f"⚠️ ISP 有增益但不如 Delta")
    else:
        print(f"❌ ISP 未提供额外增益")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    
    print("=" * 70)
    print("快速消融验证：ISP 是否提供额外增益？（无监督）")
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
    
    labels_array = np.squeeze(np.array(ano_label))
    
    results = quick_ablation_unsupervised(
        args.dataset, original, delta, isp, delta_norms, 
        labels_array, normal_for_train_idx, idx_test
    )
    
    return results


if __name__ == "__main__":
    main()