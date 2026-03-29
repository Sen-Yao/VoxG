#!/usr/bin/env python3
"""
Delta Vector 半监督评估脚本

验证 Delta Vector 在真实半监督设置下的表现：
- 只用 train_rate=0.05 的标签训练
- 在测试集上评估

使用方式:
  python run_delta_semisupervised.py --dataset photo --train_rate 0.05
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
import argparse
import random
from scipy import stats

from utils import load_mat, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def compute_delta_vectors(node_tokens):
    """计算 Delta 向量"""
    return node_tokens[:, 1:] - node_tokens[:, :-1]


def compute_delta_norms(node_tokens):
    """计算 Delta 范数"""
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]
    return torch.norm(delta_vectors, p=2, dim=-1)


def evaluate_semisupervised(features, labels, train_idx, test_idx):
    """真实半监督评估"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    labels = np.array(labels)
    
    # 标准化
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_idx])
    test_features = scaler.transform(features[test_idx])
    
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    # 训练 LR
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(train_features, train_labels)
    
    # 预测
    probas = clf.predict_proba(test_features)[:, 1]
    
    auc = roc_auc_score(test_labels, probas)
    ap = average_precision_score(test_labels, probas)
    
    return auc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4', help='多个seed，逗号分隔')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    print("="*70)
    print(f"Delta Vector 半监督评估")
    print(f"数据集: {args.dataset}, train_rate: {args.train_rate}")
    print(f"Seeds: {seeds}")
    print("="*70)
    
    results = {
        'delta_vector': {'flatten': [], 'mean': [], 'last': []},
        'delta_norm': [],
        'original_token': []
    }
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 构造 args
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
        
        print(f"训练集: {len(idx_train)}, 测试集: {len(idx_test)}")
        
        # Tokenization
        node_tokens = nagphormer_tokenization(features, adj, run_args)
        delta_vectors = compute_delta_vectors(node_tokens)
        delta_norms = compute_delta_norms(node_tokens)
        
        # 1. Delta Vector - flatten
        features_flatten = delta_vectors.reshape(delta_vectors.shape[0], -1).numpy()
        auc, ap = evaluate_semisupervised(features_flatten, labels_array, idx_train, idx_test)
        results['delta_vector']['flatten'].append(auc)
        print(f"  Delta Vector (flatten): AUC={auc:.4f}, AP={ap:.4f}")
        
        # 2. Delta Vector - mean
        features_mean = delta_vectors.mean(dim=1).numpy()
        auc, ap = evaluate_semisupervised(features_mean, labels_array, idx_train, idx_test)
        results['delta_vector']['mean'].append(auc)
        print(f"  Delta Vector (mean): AUC={auc:.4f}, AP={ap:.4f}")
        
        # 3. Delta Vector - last hop
        features_last = delta_vectors[:, -1, :].numpy()
        auc, ap = evaluate_semisupervised(features_last, labels_array, idx_train, idx_test)
        results['delta_vector']['last'].append(auc)
        print(f"  Delta Vector (last): AUC={auc:.4f}, AP={ap:.4f}")
        
        # 4. Delta Norm
        features_norm = delta_norms.numpy()
        auc, ap = evaluate_semisupervised(features_norm, labels_array, idx_train, idx_test)
        results['delta_norm'].append(auc)
        print(f"  Delta Norm: AUC={auc:.4f}, AP={ap:.4f}")
        
        # 5. Original Token (对比)
        features_orig = node_tokens.mean(dim=1).numpy()
        auc, ap = evaluate_semisupervised(features_orig, labels_array, idx_train, idx_test)
        results['original_token'].append(auc)
        print(f"  Original Token (mean): AUC={auc:.4f}, AP={ap:.4f}")
    
    # 汇总结果
    print("\n" + "="*70)
    print("5-seed 结果汇总")
    print("="*70)
    
    print(f"\n{'方法':<25} {'AUC (mean±std)':<20}")
    print("-"*50)
    
    for method in ['flatten', 'mean', 'last']:
        aucs = results['delta_vector'][method]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"Delta Vector ({method})    {mean_auc:.4f} ± {std_auc:.4f}")
    
    mean_norm = np.mean(results['delta_norm'])
    std_norm = np.std(results['delta_norm'])
    print(f"Delta Norm               {mean_norm:.4f} ± {std_norm:.4f}")
    
    mean_orig = np.mean(results['original_token'])
    std_orig = np.std(results['original_token'])
    print(f"Original Token           {mean_orig:.4f} ± {std_orig:.4f}")
    
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    best_method = max(['flatten', 'mean', 'last'], 
                      key=lambda m: np.mean(results['delta_vector'][m]))
    best_auc = np.mean(results['delta_vector'][best_method])
    
    print(f"\n最佳方法: Delta Vector ({best_method})")
    print(f"  AUC: {best_auc:.4f} ± {np.std(results['delta_vector'][best_method]):.4f}")
    print(f"\nvs Delta Norm: {'+' if best_auc > mean_norm else ''}{best_auc - mean_norm:.4f}")
    print(f"vs Original Token: {'+' if best_auc > mean_orig else ''}{best_auc - mean_orig:.4f}")


if __name__ == "__main__":
    main()