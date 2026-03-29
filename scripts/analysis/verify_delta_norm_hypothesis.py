#!/usr/bin/env python3
"""
验证假设：异常节点的 Delta 范数是否更高？

用法：
    python verify_delta_norm_hypothesis.py --dataset photo
    python verify_delta_norm_hypothesis.py --dataset elliptic
    python verify_delta_norm_hypothesis.py --dataset amazon
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
import argparse
from scipy import stats
import matplotlib.pyplot as plt

from utils import load_mat, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def compute_delta_norms(node_tokens):
    """
    计算 Delta 范数
    
    Args:
        node_tokens: [N, K+1, D] - 每个 hop 的节点特征
    
    Returns:
        delta_norms: [N, K] - 每个 hop 的 Delta 范数
        delta_norm_total: [N] - 总 Delta 范数
        delta_norm_mean: [N] - 平均 Delta 范数
    """
    # Delta = X_{k+1} - X_k
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]  # [N, K, D]
    
    # Delta 范数
    delta_norms = torch.norm(delta_vectors, p=2, dim=-1)  # [N, K]
    
    # 汇总统计
    delta_norm_total = delta_norms.sum(dim=1)  # [N]
    delta_norm_mean = delta_norms.mean(dim=1)   # [N]
    
    return delta_norms, delta_norm_total, delta_norm_mean


def verify_hypothesis(dataset_name, labels, delta_norms, delta_norm_total, delta_norm_mean):
    """
    验证假设：异常节点的 Delta 范数是否更高
    """
    labels = np.array(labels).flatten()
    
    # 分离正常和异常节点
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"正常节点: {len(normal_idx)}")
    print(f"异常节点: {len(anomaly_idx)}")
    
    # 1. 总 Delta 范数对比
    print(f"\n--- 总 Delta 范数对比 ---")
    normal_total = delta_norm_total[normal_idx].numpy()
    anomaly_total = delta_norm_total[anomaly_idx].numpy()
    
    print(f"正常节点: mean={normal_total.mean():.4f}, std={normal_total.std():.4f}, median={np.median(normal_total):.4f}")
    print(f"异常节点: mean={anomaly_total.mean():.4f}, std={anomaly_total.std():.4f}, median={np.median(anomaly_total):.4f}")
    
    # 统计检验
    stat, p_value = stats.mannwhitneyu(normal_total, anomaly_total, alternative='less')
    print(f"Mann-Whitney U 检验 (正常 < 异常): stat={stat:.2f}, p={p_value:.4e}")
    
    if p_value < 0.05:
        print(f"✅ 结论: 异常节点的总 Delta 范数显著更高 (p < 0.05)")
    else:
        print(f"❌ 结论: 无显著差异 (p >= 0.05)")
    
    # 2. 平均 Delta 范数对比
    print(f"\n--- 平均 Delta 范数对比 ---")
    normal_mean = delta_norm_mean[normal_idx].numpy()
    anomaly_mean = delta_norm_mean[anomaly_idx].numpy()
    
    print(f"正常节点: mean={normal_mean.mean():.4f}, std={normal_mean.std():.4f}")
    print(f"异常节点: mean={anomaly_mean.mean():.4f}, std={anomaly_mean.std():.4f}")
    
    stat, p_value = stats.mannwhitneyu(normal_mean, anomaly_mean, alternative='less')
    print(f"Mann-Whitney U 检验: stat={stat:.2f}, p={p_value:.4e}")
    
    # 3. 逐 Hop 分析
    print(f"\n--- 逐 Hop Delta 范数对比 ---")
    print(f"{'Hop':<6} {'正常 mean':<12} {'异常 mean':<12} {'差异':<10} {'p-value':<12} {'结论':<10}")
    print("-" * 65)
    
    K = delta_norms.shape[1]
    for k in range(K):
        normal_k = delta_norms[normal_idx, k].numpy()
        anomaly_k = delta_norms[anomaly_idx, k].numpy()
        
        stat, p_value = stats.mannwhitneyu(normal_k, anomaly_k, alternative='less')
        diff = anomaly_k.mean() - normal_k.mean()
        
        conclusion = "✅ 异常更高" if p_value < 0.05 else "❌ 无显著差异"
        print(f"{k:<6} {normal_k.mean():<12.4f} {anomaly_k.mean():<12.4f} {diff:+.4f}    {p_value:<12.4e} {conclusion}")
    
    # 4. 分布可视化
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 总范数分布
        axes[0].hist(normal_total, bins=50, alpha=0.7, label=f'Normal (n={len(normal_idx)})', density=True)
        axes[0].hist(anomaly_total, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_idx)})', density=True)
        axes[0].set_xlabel('Total Delta Norm')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{dataset_name}: Total Delta Norm Distribution')
        axes[0].legend()
        
        # 平均范数分布
        axes[1].hist(normal_mean, bins=50, alpha=0.7, label='Normal', density=True)
        axes[1].hist(anomaly_mean, bins=50, alpha=0.7, label='Anomaly', density=True)
        axes[1].set_xlabel('Mean Delta Norm')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'{dataset_name}: Mean Delta Norm Distribution')
        axes[1].legend()
        
        # 逐 Hop 对比
        hop_means_normal = [delta_norms[normal_idx, k].mean().item() for k in range(K)]
        hop_means_anomaly = [delta_norms[anomaly_idx, k].mean().item() for k in range(K)]
        
        x = range(K)
        axes[2].bar([i-0.2 for i in x], hop_means_normal, width=0.4, label='Normal', alpha=0.7)
        axes[2].bar([i+0.2 for i in x], hop_means_anomaly, width=0.4, label='Anomaly', alpha=0.7)
        axes[2].set_xlabel('Hop')
        axes[2].set_ylabel('Delta Norm')
        axes[2].set_title(f'{dataset_name}: Delta Norm by Hop')
        axes[2].legend()
        axes[2].set_xticks(x)
        
        plt.tight_layout()
        plt.savefig(f'/tmp/delta_norm_verification_{dataset_name}.png', dpi=150)
        print(f"\n📊 可视化已保存: /tmp/delta_norm_verification_{dataset_name}.png")
    except Exception as e:
        print(f"\n⚠️ 可视化失败: {e}")
    
    return {
        'dataset': dataset_name,
        'normal_total_mean': normal_total.mean(),
        'anomaly_total_mean': anomaly_total.mean(),
        'p_value_total': stats.mannwhitneyu(normal_total, anomaly_total, alternative='less')[1]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    
    print("=" * 70)
    print("验证假设：异常节点的 Delta 范数是否更高？")
    print("=" * 70)
    
    # 设置
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    print(f"\n加载数据集: {args.dataset}...")
    
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
    
    labels_array = np.squeeze(np.array(ano_label))
    
    # Tokenization
    print(f"计算 hop2token (K={args.pp_k})...")
    node_tokens = nagphormer_tokenization(features, adj, run_args)
    print(f"Token 形状: {node_tokens.shape}")
    
    # 计算 Delta 范数
    print("计算 Delta 范数...")
    delta_norms, delta_norm_total, delta_norm_mean = compute_delta_norms(node_tokens)
    
    # 验证假设
    result = verify_hypothesis(
        args.dataset, labels_array, delta_norms, delta_norm_total, delta_norm_mean
    )
    
    return result


if __name__ == "__main__":
    main()