#!/usr/bin/env python3
"""
验证假设：异常节点的 ISP（方向一致性）是否更低？

ISP = Individual Smoothing Pattern（个体平滑模式）

方案A：方向一致性
- 计算 Delta 向量的方向
- 正常节点：方向一致（ISP 高）
- 异常节点：方向震荡（ISP 低）

用法：
    python verify_isp_hypothesis.py --dataset photo
    python verify_isp_hypothesis.py --dataset elliptic
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


def compute_isp_metrics(node_tokens):
    """
    计算 ISP 相关指标
    
    Args:
        node_tokens: [N, K+1, D] - 每个 hop 的节点特征
    
    Returns:
        direction_consistency: [N] - 方向一致性（越高越正常）
        delta_norms: [N, K] - Delta 范数
        cumulative_delta: [N] - 累积变化
    """
    N, K_plus_1, D = node_tokens.shape
    K = K_plus_1 - 1
    
    # Delta 向量
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]  # [N, K, D]
    
    # Delta 范数
    delta_norms = torch.norm(delta_vectors, p=2, dim=-1)  # [N, K]
    
    # ========== 方案A：方向一致性 ==========
    # 归一化 Delta 向量（只保留方向）
    delta_norms_expanded = delta_norms.unsqueeze(-1) + 1e-8  # [N, K, 1]
    delta_directions = delta_vectors / delta_norms_expanded  # [N, K, D]
    
    # 计算相邻方向的一致性
    consistencies = []
    for k in range(K - 1):
        dir_k = delta_directions[:, k, :]      # [N, D]
        dir_k1 = delta_directions[:, k + 1, :]  # [N, D]
        
        # 余弦相似度
        cos_sim = torch.sum(dir_k * dir_k1, dim=-1)  # [N]
        consistencies.append(cos_sim)
    
    # 方向一致性 = 平均余弦相似度
    direction_consistency = torch.stack(consistencies, dim=1).mean(dim=1)  # [N]
    
    # ========== 方案B：累积变化 ==========
    cumulative_delta = delta_norms.sum(dim=1)  # [N]
    
    # ========== 方案C：方向方差 ==========
    # 方向在 D 维空间中的方差（越小越一致）
    direction_mean = delta_directions.mean(dim=1)  # [N, D]
    direction_var = ((delta_directions - direction_mean.unsqueeze(1)) ** 2).sum(dim=-1).mean(dim=1)  # [N]
    
    return direction_consistency, delta_norms, cumulative_delta, direction_var


def verify_isp_hypothesis(dataset_name, labels, direction_consistency, delta_norms, cumulative_delta, direction_var):
    """
    验证假设：异常节点的 ISP 是否更低（方向一致性更低）
    """
    labels = np.array(labels).flatten()
    
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"正常节点: {len(normal_idx)}")
    print(f"异常节点: {len(anomaly_idx)}")
    
    # 1. 方向一致性对比（核心指标）
    print(f"\n{'='*50}")
    print("核心指标：方向一致性（越高 = 方向越一致 = 越正常）")
    print(f"{'='*50}")
    
    normal_dc = direction_consistency[normal_idx].numpy()
    anomaly_dc = direction_consistency[anomaly_idx].numpy()
    
    print(f"正常节点: mean={normal_dc.mean():.4f}, std={normal_dc.std():.4f}, median={np.median(normal_dc):.4f}")
    print(f"异常节点: mean={anomaly_dc.mean():.4f}, std={anomaly_dc.std():.4f}, median={np.median(anomaly_dc):.4f}")
    
    # 检验：正常节点是否方向一致性更高？
    stat, p_value = stats.mannwhitneyu(normal_dc, anomaly_dc, alternative='greater')
    print(f"Mann-Whitney U 检验 (正常 > 异常): stat={stat:.2f}, p={p_value:.4e}")
    
    if p_value < 0.05:
        print(f"✅ 结论: 正常节点方向一致性显著更高 (p < 0.05)")
        conclusion_dc = "✅ 符合预期"
    else:
        print(f"❌ 结论: 无显著差异 (p >= 0.05)")
        conclusion_dc = "❌ 不符合预期"
    
    # 2. 方向方差对比
    print(f"\n--- 方向方差（越小 = 方向越一致 = 越正常）---")
    normal_var = direction_var[normal_idx].numpy()
    anomaly_var = direction_var[anomaly_idx].numpy()
    
    print(f"正常节点: mean={normal_var.mean():.4f}, std={normal_var.std():.4f}")
    print(f"异常节点: mean={anomaly_var.mean():.4f}, std={anomaly_var.std():.4f}")
    
    stat, p_value = stats.mannwhitneyu(normal_var, anomaly_var, alternative='less')
    print(f"Mann-Whitney U 检验 (正常 < 异常): p={p_value:.4e}")
    
    if p_value < 0.05:
        print(f"✅ 结论: 正常节点方向方差显著更小")
        conclusion_var = "✅ 符合预期"
    else:
        print(f"❌ 结论: 无显著差异")
        conclusion_var = "❌ 不符合预期"
    
    # 3. 逐 Hop 方向一致性分析
    print(f"\n--- 逐 Hop 方向一致性 ---")
    print(f"{'Hop对':<8} {'正常 mean':<12} {'异常 mean':<12} {'差异':<10} {'p-value':<12} {'结论':<10}")
    print("-" * 70)
    
    N, K_plus_1, D = delta_norms.shape[0], delta_norms.shape[1] + 1, 0
    
    delta_vectors = None  # 重新计算
    for k in range(K_plus_1 - 2):
        # 这个需要重新计算，简化输出
        pass
    
    # 4. 可视化
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 方向一致性分布
        axes[0].hist(normal_dc, bins=50, alpha=0.7, label=f'Normal (n={len(normal_idx)})', density=True)
        axes[0].hist(anomaly_dc, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_idx)})', density=True)
        axes[0].set_xlabel('Direction Consistency (ISP)')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{dataset_name}: ISP Distribution')
        axes[0].axvline(normal_dc.mean(), color='blue', linestyle='--', label=f'Normal mean={normal_dc.mean():.2f}')
        axes[0].axvline(anomaly_dc.mean(), color='orange', linestyle='--', label=f'Anomaly mean={anomaly_dc.mean():.2f}')
        axes[0].legend(fontsize=8)
        
        # 方向方差分布
        axes[1].hist(normal_var, bins=50, alpha=0.7, label='Normal', density=True)
        axes[1].hist(anomaly_var, bins=50, alpha=0.7, label='Anomaly', density=True)
        axes[1].set_xlabel('Direction Variance')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'{dataset_name}: Direction Variance Distribution')
        axes[1].legend()
        
        # Delta 范数对比（参考）
        normal_cum = cumulative_delta[normal_idx].numpy()
        anomaly_cum = cumulative_delta[anomaly_idx].numpy()
        axes[2].hist(normal_cum, bins=50, alpha=0.7, label='Normal', density=True)
        axes[2].hist(anomaly_cum, bins=50, alpha=0.7, label='Anomaly', density=True)
        axes[2].set_xlabel('Cumulative Delta Norm')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'{dataset_name}: Cumulative Delta Norm')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'/tmp/isp_verification_{dataset_name}.png', dpi=150)
        print(f"\n📊 可视化已保存: /tmp/isp_verification_{dataset_name}.png")
    except Exception as e:
        print(f"\n⚠️ 可视化失败: {e}")
    
    return {
        'dataset': dataset_name,
        'normal_dc_mean': normal_dc.mean(),
        'anomaly_dc_mean': anomaly_dc.mean(),
        'p_value_dc': stats.mannwhitneyu(normal_dc, anomaly_dc, alternative='greater')[1],
        'conclusion_dc': conclusion_dc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    
    print("=" * 70)
    print("验证假设：异常节点的 ISP（方向一致性）是否更低？")
    print("=" * 70)
    print("\n理论背景：")
    print("- 正常节点：传播过程中方向一致，平滑收敛")
    print("- 异常节点：传播过程中方向震荡，难以平滑")
    print("- ISP（方向一致性）越高 = 越正常")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    
    print(f"计算 hop2token (K={args.pp_k})...")
    node_tokens = nagphormer_tokenization(features, adj, run_args)
    print(f"Token 形状: {node_tokens.shape}")
    
    print("计算 ISP 指标...")
    direction_consistency, delta_norms, cumulative_delta, direction_var = compute_isp_metrics(node_tokens)
    
    result = verify_isp_hypothesis(
        args.dataset, labels_array, direction_consistency, delta_norms, cumulative_delta, direction_var
    )
    
    return result


if __name__ == "__main__":
    main()