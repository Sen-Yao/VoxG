#!/usr/bin/env python3
"""
Delta Vector 真正的半监督评估

修正：
- 只用 normal_for_train_idx（只有正常节点）
- 使用无监督方法：马氏距离、Isolation Forest

使用方式:
  python run_delta_oneclass.py --dataset photo --train_rate 0.05
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
import argparse
import random

from utils import load_mat, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_delta_vectors(node_tokens):
    """计算 Delta 向量"""
    return node_tokens[:, 1:] - node_tokens[:, :-1]


def compute_delta_norms(node_tokens):
    """计算 Delta 范数"""
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]
    return torch.norm(delta_vectors, p=2, dim=-1)


def evaluate_oneclass_mahalanobis(features, normal_idx, test_idx, labels):
    """
    真正的半监督评估（马氏距离）
    
    只用正常节点训练，计算测试集到正常中心的马氏距离
    """
    labels = np.array(labels)
    
    # 标准化（只用正常节点）
    scaler = StandardScaler()
    normal_features = scaler.fit_transform(features[normal_idx])
    test_features = scaler.transform(features[test_idx])
    
    # 计算马氏距离
    center = normal_features.mean(axis=0)
    cov = np.cov(normal_features.T) + 1e-6 * np.eye(normal_features.shape[1])
    
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        # 如果协方差矩阵奇异，使用伪逆
        cov_inv = np.linalg.pinv(cov)
    
    # 马氏距离
    diff = test_features - center
    dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    
    # 距离越大越异常
    auc = roc_auc_score(labels[test_idx], dist)
    ap = average_precision_score(labels[test_idx], dist)
    
    return auc, ap


def evaluate_oneclass_iforest(features, normal_idx, test_idx, labels):
    """
    Isolation Forest 方法
    """
    from sklearn.ensemble import IsolationForest
    
    labels = np.array(labels)
    
    scaler = StandardScaler()
    normal_features = scaler.fit_transform(features[normal_idx])
    test_features = scaler.transform(features[test_idx])
    
    # 训练 Isolation Forest（只用正常节点）
    clf = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    clf.fit(normal_features)
    
    # 预测（分数越低越异常）
    scores = -clf.score_samples(test_features)  # 取负，让异常分数更高
    
    auc = roc_auc_score(labels[test_idx], scores)
    ap = average_precision_score(labels[test_idx], scores)
    
    return auc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    print("="*70)
    print("Delta Vector 真正的半监督评估（修正版）")
    print(f"数据集: {args.dataset}, train_rate: {args.train_rate}")
    print("修正：只用 normal_for_train_idx（只有正常节点）")
    print("="*70)
    
    results = {
        'delta_vector': {'mahalanobis': [], 'iforest': []},
        'delta_norm': {'mahalanobis': [], 'iforest': []},
        'original_token': {'mahalanobis': [], 'iforest': []}
    }
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        run_args = argparse.Namespace(
            dataset=args.dataset,
            pp_k=args.pp_k,
            progregate_alpha=args.alpha,
            train_rate=args.train_rate,
            seed=seed,
            data_split_seed=seed,
            sample_rate=0.15
        )
        
        # 加载数据
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
        
        labels_array = np.squeeze(np.array(ano_label))
        
        # 关键修正：使用 normal_for_train_idx
        print(f"训练集（正常节点）: {len(normal_for_train_idx)}")
        print(f"测试集: {len(idx_test)}, 异常: {sum(labels_array[idx_test]==1)}")
        
        # Tokenization
        node_tokens = nagphormer_tokenization(features, adj, run_args)
        delta_vectors = compute_delta_vectors(node_tokens)
        delta_norms = compute_delta_norms(node_tokens)
        
        # 1. Delta Vector (mean) - 马氏距离
        features_mean = delta_vectors.mean(dim=1).numpy()
        auc, ap = evaluate_oneclass_mahalanobis(features_mean, normal_for_train_idx, idx_test, labels_array)
        results['delta_vector']['mahalanobis'].append(auc)
        print(f"  Delta Vector (mean) + 马氏距离: AUC={auc:.4f}")
        
        # 2. Delta Vector (mean) - Isolation Forest
        auc, ap = evaluate_oneclass_iforest(features_mean, normal_for_train_idx, idx_test, labels_array)
        results['delta_vector']['iforest'].append(auc)
        print(f"  Delta Vector (mean) + IsolationForest: AUC={auc:.4f}")
        
        # 3. Delta Norm - 马氏距离
        features_norm = delta_norms.numpy()
        auc, ap = evaluate_oneclass_mahalanobis(features_norm, normal_for_train_idx, idx_test, labels_array)
        results['delta_norm']['mahalanobis'].append(auc)
        print(f"  Delta Norm + 马氏距离: AUC={auc:.4f}")
        
        # 4. Original Token - 马氏距离
        features_orig = node_tokens.mean(dim=1).numpy()
        auc, ap = evaluate_oneclass_mahalanobis(features_orig, normal_for_train_idx, idx_test, labels_array)
        results['original_token']['mahalanobis'].append(auc)
        print(f"  Original Token + 马氏距离: AUC={auc:.4f}")
    
    # 汇总
    print("\n" + "="*70)
    print("5-seed 结果汇总（真正的半监督）")
    print("="*70)
    
    print(f"\n{'方法':<35} {'AUC (mean±std)':<20}")
    print("-"*60)
    
    for method in ['mahalanobis', 'iforest']:
        for feature in ['delta_vector', 'delta_norm', 'original_token']:
            aucs = results[feature][method]
            if aucs:
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                print(f"{feature} + {method:<15} {mean_auc:.4f} ± {std_auc:.4f}")
    
    print("\n" + "="*70)
    print("对比 SOTA")
    print("="*70)
    
    sota = {
        'photo': 0.8960,
        'Amazon': 0.9391,
        'reddit': 0.5782,
        'elliptic': 0.8509,
        't_finance': 0.8988,
        'tolokers': 0.6612
    }
    
    if args.dataset in sota:
        best_auc = max(np.mean(results['delta_vector']['mahalanobis']),
                       np.mean(results['original_token']['mahalanobis']))
        print(f"\nSOTA ({args.dataset}): {sota[args.dataset]:.4f}")
        print(f"本实验最佳: {best_auc:.4f}")
        print(f"差距: {best_auc - sota[args.dataset]:+.4f}")


if __name__ == "__main__":
    main()