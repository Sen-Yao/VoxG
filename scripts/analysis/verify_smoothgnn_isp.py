#!/usr/bin/env python3
"""
验证 SmoothGNN 的真正 ISP（Individual Smoothing Pattern）

ISP 定义 (SmoothGNN, arXiv:2405.17525):
    ISP(i) = (1/K) * Σ_{t=1}^{K} ||H_i^{(t)} - H_i^{(∞)}||_2 / ||H_i^{(∞)}||_2

物理意义:
- ISP 越高 → 节点越难平滑 → 越可能是异常
- 正常节点更容易被平滑，ISP 更低

关键修正:
- 传播方式: APPNP 风格 (x^{(k+1)} = (1-α) P x^{(k)} + α x^{(0)})
- 收敛状态: x^{(∞)} = α (I - (1-α)P)^{-1} x^{(0)}
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv, spsolve
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils import load_mat, normalize_adj, preprocess_features, nagphormer_tokenization


def compute_propagation_matrix(adj_normalized):
    """
    从归一化邻接矩阵提取传播矩阵
    
    VoxG 使用: adj = normalize_adj(adj) + I
    所以 P = D^{-1/2} A D^{-1/2} = adj - I (去掉自环)
    """
    if isinstance(adj_normalized, torch.Tensor):
        P = adj_normalized.numpy() - np.eye(adj_normalized.shape[0])
    else:
        P = adj_normalized - sp.eye(adj_normalized.shape[0])
    return P


def compute_convergence_appnp(P, features, alpha):
    """
    计算 APPNP 传播的收敛状态
    
    APPNP 传播: x^{(k+1)} = (1-α) P x^{(k)} + α x^{(0)}
    收敛状态: x^{(∞)} = α (I - (1-α)P)^{-1} x^{(0)}
    
    Args:
        P: 传播矩阵 (D^{-1/2} A D^{-1/2}, 无自环)
        features: 初始特征 X
        alpha: PPR 的 teleport 概率
    
    Returns:
        H_inf: 收敛状态
    """
    print(f"  计算 APPNP 收敛状态 (alpha={alpha})...")
    
    n = P.shape[0]
    
    if isinstance(features, torch.Tensor):
        X = features.numpy()
    elif sp.issparse(features):
        X = features.toarray()
    else:
        X = np.array(features)
    
    # 构建 (I - (1-α)P)
    I_minus_P = sp.eye(n) - (1 - alpha) * sp.csr_matrix(P)
    
    # 求解 (I - (1-α)P) H = α X
    # H = α (I - (1-α)P)^{-1} X
    print(f"  求解线性系统 ({n} x {n})...")
    
    # 使用稀疏求解
    I_minus_P_csr = sp.csr_matrix(I_minus_P)
    
    # 逐列求解
    d = X.shape[1]
    H_inf = np.zeros_like(X)
    
    for i in range(d):
        if (i + 1) % 100 == 0:
            print(f"    处理第 {i+1}/{d} 维...")
        H_inf[:, i] = spsolve(I_minus_P_csr, alpha * X[:, i])
    
    print(f"  收敛状态计算完成")
    return torch.FloatTensor(H_inf)


def compute_convergence_iterative(P, features, alpha, max_iter=1000, tol=1e-10):
    """
    迭代法计算收敛状态（用于验证）
    
    Args:
        P: 传播矩阵 (无自环)
        features: 初始特征
        alpha: PPR 的 teleport 概率
    """
    print(f"  使用迭代法计算收敛状态 (alpha={alpha}, max_iter={max_iter})...")
    
    if isinstance(features, torch.Tensor):
        X = features.numpy()
    elif sp.issparse(features):
        X = features.toarray()
    else:
        X = np.array(features)
    
    P_dense = P.toarray() if sp.issparse(P) else np.array(P)
    
    H = X.copy()
    
    for i in range(max_iter):
        # x^{(k+1)} = (1-α) P x^{(k)} + α x^{(0)}
        H_new = (1 - alpha) * (P_dense @ H) + alpha * X
        diff = np.linalg.norm(H_new - H)
        
        if diff < tol:
            print(f"  收敛于第 {i+1} 次迭代, diff={diff:.2e}")
            return torch.FloatTensor(H_new), i+1
        
        H = H_new
        
        if (i + 1) % 200 == 0:
            print(f"    第 {i+1} 次迭代, diff={diff:.2e}")
    
    print(f"  ⚠️ 未收敛，达到最大迭代次数 {max_iter}, diff={diff:.2e}")
    return torch.FloatTensor(H), max_iter


def compute_smoothgnn_isp(node_tokens, H_inf, K=None):
    """
    计算 SmoothGNN 的真正 ISP
    
    Args:
        node_tokens: [N, K+1, D] - 每个 hop 的节点特征
                     其中 node_tokens[:, 0, :] = H^{(0)} = X
                     node_tokens[:, t, :] = H^{(t)} = PPR 传播 t 步
        H_inf: [N, D] - 收敛状态 H^{(∞)}
        K: 计算 ISP 的 hop 数（默认使用所有 hop）
    
    Returns:
        isp: [N] - 每个节点的 ISP 值
        hop_distances: [N, K] - 每个 hop 到收敛状态的距离
    """
    if isinstance(H_inf, torch.Tensor):
        H_inf = H_inf.numpy()
    
    N, num_hops, D = node_tokens.shape
    if K is None:
        K = num_hops - 1  # 排除第 0 hop
    
    # H_inf 的范数
    H_inf_norm = np.linalg.norm(H_inf, axis=1, keepdims=True) + 1e-8  # [N, 1]
    
    # 计算每个 hop 到收敛状态的归一化距离
    hop_distances = np.zeros((N, K))
    
    for t in range(1, K + 1):  # t = 1, 2, ..., K
        H_t = node_tokens[:, t, :].numpy()  # [N, D]
        
        # ||H_i^{(t)} - H_i^{(∞)}||_2
        diff = np.linalg.norm(H_t - H_inf, axis=1)  # [N]
        
        # 归一化距离
        hop_distances[:, t-1] = diff / H_inf_norm.flatten()
    
    # ISP = 平均归一化距离
    isp = hop_distances.mean(axis=1)  # [N]
    
    return isp, hop_distances


def compute_old_isp_metrics(node_tokens):
    """
    计算旧的 ISP 指标（方向一致性）用于对比
    """
    N, K_plus_1, D = node_tokens.shape
    K = K_plus_1 - 1
    
    # Delta 向量
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]  # [N, K, D]
    
    # Delta 范数
    delta_norms = torch.norm(delta_vectors, p=2, dim=-1)  # [N, K]
    
    # 方向一致性
    delta_norms_expanded = delta_norms.unsqueeze(-1) + 1e-8
    delta_directions = delta_vectors / delta_norms_expanded
    
    consistencies = []
    for k in range(K - 1):
        dir_k = delta_directions[:, k, :]
        dir_k1 = delta_directions[:, k + 1, :]
        cos_sim = torch.sum(dir_k * dir_k1, dim=-1)
        consistencies.append(cos_sim)
    
    direction_consistency = torch.stack(consistencies, dim=1).mean(dim=1)
    
    return direction_consistency.numpy(), delta_norms.numpy()


def verify_hypothesis(dataset_name, labels, isp, hop_distances, direction_consistency):
    """
    验证假设：异常节点的 ISP 更高
    """
    labels = np.array(labels).flatten()
    
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"正常节点: {len(normal_idx)}")
    print(f"异常节点: {len(anomaly_idx)}")
    
    # 1. SmoothGNN ISP 对比
    print(f"\n{'='*60}")
    print("核心指标：SmoothGNN ISP（到收敛状态的平均归一化距离）")
    print("ISP 越高 → 节点越难平滑 → 越可能是异常")
    print(f"{'='*60}")
    
    normal_isp = isp[normal_idx]
    anomaly_isp = isp[anomaly_idx]
    
    print(f"\n正常节点 ISP: mean={normal_isp.mean():.6f}, std={normal_isp.std():.6f}, median={np.median(normal_isp):.6f}")
    print(f"异常节点 ISP: mean={anomaly_isp.mean():.6f}, std={anomaly_isp.std():.6f}, median={np.median(anomaly_isp):.6f}")
    print(f"差异 (异常-正常): {anomaly_isp.mean() - normal_isp.mean():.6f}")
    
    # 检验：异常节点是否 ISP 更高？
    stat, p_value_greater = stats.mannwhitneyu(anomaly_isp, normal_isp, alternative='greater')
    print(f"\nMann-Whitney U 检验 (异常 > 正常): stat={stat:.2f}, p={p_value_greater:.4e}")
    
    # Cohen's d
    pooled_std = np.sqrt((normal_isp.std()**2 + anomaly_isp.std()**2) / 2)
    cohens_d = (anomaly_isp.mean() - normal_isp.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d:.4f}")
    
    if p_value_greater < 0.05:
        if anomaly_isp.mean() > normal_isp.mean():
            print(f"✅ 结论: 异常节点 ISP 显著更高 (p < 0.05)")
            conclusion_smoothgnn = "✅ 符合 SmoothGNN 假设"
        else:
            print(f"❌ 结论: 方向相反，异常节点 ISP 更低 (p < 0.05)")
            conclusion_smoothgnn = "❌ 与假设相反（异常ISP更低）"
    else:
        print(f"❌ 结论: 无显著差异 (p >= 0.05)")
        conclusion_smoothgnn = "❌ 无显著差异"
    
    # 2. 逐 hop 分析
    print(f"\n--- 逐 Hop 到收敛状态的距离 ---")
    K = hop_distances.shape[1]
    print(f"{'Hop':<6} {'正常 mean':<14} {'异常 mean':<14} {'差异':<12} {'p-value':<12} {'结论':<10}")
    print("-" * 70)
    
    hop_conclusions = []
    for t in range(K):
        normal_dist = hop_distances[normal_idx, t]
        anomaly_dist = hop_distances[anomaly_idx, t]
        
        stat, p = stats.mannwhitneyu(anomaly_dist, normal_dist, alternative='greater')
        diff = anomaly_dist.mean() - normal_dist.mean()
        
        conclusion = "✅" if p < 0.05 and diff > 0 else "❌"
        hop_conclusions.append((t+1, diff, p))
        
        print(f"t={t+1:<4} {normal_dist.mean():.6f}       {anomaly_dist.mean():.6f}       {diff:+.6f}     {p:.4e}   {conclusion}")
    
    # 3. 与旧 ISP 对比
    print(f"\n{'='*60}")
    print("对比：旧 ISP 指标（方向一致性）")
    print(f"{'='*60}")
    
    normal_dc = direction_consistency[normal_idx]
    anomaly_dc = direction_consistency[anomaly_idx]
    
    print(f"\n正常节点: mean={normal_dc.mean():.6f}, std={normal_dc.std():.6f}")
    print(f"异常节点: mean={anomaly_dc.mean():.6f}, std={anomaly_dc.std():.6f}")
    print(f"差异 (正常-异常): {normal_dc.mean() - anomaly_dc.mean():.6f}")
    
    stat, p_value_old = stats.mannwhitneyu(normal_dc, anomaly_dc, alternative='greater')
    print(f"Mann-Whitney U 检验 (正常 > 异常): p={p_value_old:.4e}")
    
    if p_value_old < 0.05:
        if normal_dc.mean() > anomaly_dc.mean():
            print(f"✅ 结论: 正常节点方向一致性更高")
            conclusion_old = "✅ 有区分度"
        else:
            print(f"❌ 结论: 方向相反")
            conclusion_old = "❌ 与预期相反"
    else:
        print(f"❌ 结论: 无显著差异")
        conclusion_old = "❌ 无显著差异"
    
    # 4. 可视化
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # SmoothGNN ISP 分布
        axes[0, 0].hist(normal_isp, bins=50, alpha=0.7, label=f'Normal (n={len(normal_idx)})', 
                       density=True, color='blue')
        axes[0, 0].hist(anomaly_isp, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_idx)})', 
                       density=True, color='red')
        axes[0, 0].set_xlabel('SmoothGNN ISP', fontsize=12)
        axes[0, 0].set_ylabel('Density', fontsize=12)
        axes[0, 0].set_title(f'{dataset_name}: SmoothGNN ISP Distribution\n(Higher = More Anomalous)', fontsize=12)
        axes[0, 0].axvline(normal_isp.mean(), color='blue', linestyle='--', 
                          label=f'Normal mean={normal_isp.mean():.4f}')
        axes[0, 0].axvline(anomaly_isp.mean(), color='red', linestyle='--', 
                          label=f'Anomaly mean={anomaly_isp.mean():.4f}')
        axes[0, 0].legend(fontsize=9)
        
        # 逐 hop 距离
        hop_t = list(range(1, K+1))
        normal_means = [hop_distances[normal_idx, t-1].mean() for t in hop_t]
        anomaly_means = [hop_distances[anomaly_idx, t-1].mean() for t in hop_t]
        normal_stds = [hop_distances[normal_idx, t-1].std() for t in hop_t]
        anomaly_stds = [hop_distances[anomaly_idx, t-1].std() for t in hop_t]
        
        x = np.arange(len(hop_t))
        width = 0.35
        axes[0, 1].bar(x - width/2, normal_means, width, yerr=normal_stds, label='Normal', 
                       color='blue', alpha=0.7, capsize=3)
        axes[0, 1].bar(x + width/2, anomaly_means, width, yerr=anomaly_stds, label='Anomaly', 
                       color='red', alpha=0.7, capsize=3)
        axes[0, 1].set_xlabel('Hop', fontsize=12)
        axes[0, 1].set_ylabel('Distance to Convergence', fontsize=12)
        axes[0, 1].set_title(f'{dataset_name}: Distance to Convergence by Hop', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f't={t}' for t in hop_t])
        axes[0, 1].legend()
        
        # 旧 ISP 对比
        axes[1, 0].hist(normal_dc, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
        axes[1, 0].hist(anomaly_dc, bins=50, alpha=0.7, label='Anomaly', density=True, color='red')
        axes[1, 0].set_xlabel('Direction Consistency (Old ISP)', fontsize=12)
        axes[1, 0].set_ylabel('Density', fontsize=12)
        axes[1, 0].set_title(f'{dataset_name}: Old ISP Distribution', fontsize=12)
        axes[1, 0].legend()
        
        # 对比汇总
        methods = ['SmoothGNN ISP', 'Old ISP']
        p_values = [p_value_greater, p_value_old]
        colors = ['green' if p < 0.05 else 'gray' for p in p_values]
        
        bars = axes[1, 1].bar(methods, -np.log10([p+1e-100 for p in p_values]), color=colors, alpha=0.7)
        axes[1, 1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1, 1].set_ylabel('-log10(p-value)', fontsize=12)
        axes[1, 1].set_title(f'{dataset_name}: Statistical Significance', fontsize=12)
        axes[1, 1].legend()
        
        for bar, p in zip(bars, p_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                          f'p={p:.2e}', ha='center', fontsize=10)
        
        plt.tight_layout()
        save_path = f'/tmp/smoothgnn_isp_{dataset_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 可视化已保存: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'dataset': dataset_name,
        'normal_isp_mean': normal_isp.mean(),
        'anomaly_isp_mean': anomaly_isp.mean(),
        'isp_diff': anomaly_isp.mean() - normal_isp.mean(),
        'p_value_smoothgnn': p_value_greater,
        'cohens_d': cohens_d,
        'conclusion_smoothgnn': conclusion_smoothgnn,
        'p_value_old': p_value_old,
        'conclusion_old': conclusion_old,
        'hop_conclusions': hop_conclusions
    }


def main():
    parser = argparse.ArgumentParser(description='验证 SmoothGNN 的真正 ISP')
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--method', type=str, default='analytical', 
                       choices=['analytical', 'iterative'],
                       help='计算收敛状态的方法')
    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.2, 
                       help='PPR 传播的 alpha (teleport probability)')
    parser.add_argument('--train_rate', type=float, default=0.05)
    args = parser.parse_args()
    
    print("=" * 70)
    print("验证 SmoothGNN ISP 假设")
    print("=" * 70)
    print("\n理论背景 (arXiv:2405.17525):")
    print("- ISP = 节点到收敛状态的平均归一化距离")
    print("- ISP 越高 → 节点越难平滑 → 越可能是异常")
    print("- 正常节点更容易被平滑，ISP 更低")
    
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
    
    # 归一化邻接矩阵
    adj_normalized = normalize_adj(adj)
    adj_normalized = (adj_normalized + sp.eye(adj_normalized.shape[0]))  # 加自环
    
    features_tensor = torch.FloatTensor(features)
    adj_tensor = torch.FloatTensor(adj_normalized.todense())
    
    labels_array = np.squeeze(np.array(ano_label))
    
    # 计算 hop features
    print(f"\n计算 hop features (K={args.pp_k})...")
    node_tokens = nagphormer_tokenization(features_tensor, adj_tensor, run_args)
    print(f"Token 形状: {node_tokens.shape}")
    
    # 计算传播矩阵（去掉自环）
    P = compute_propagation_matrix(adj_tensor)
    print(f"传播矩阵形状: {P.shape}")
    
    # 计算收敛状态
    print(f"\n计算收敛状态 H^(∞)...")
    if args.method == 'analytical':
        H_inf = compute_convergence_appnp(P, features_tensor, args.alpha)
    else:
        H_inf, iterations = compute_convergence_iterative(P, features_tensor, args.alpha)
    
    print(f"收敛状态形状: {H_inf.shape}")
    
    # 计算 SmoothGNN ISP
    print(f"\n计算 SmoothGNN ISP...")
    isp, hop_distances = compute_smoothgnn_isp(node_tokens, H_inf, K=args.pp_k)
    
    # 计算旧 ISP 对比
    print(f"计算旧 ISP（方向一致性）...")
    direction_consistency, delta_norms = compute_old_isp_metrics(node_tokens)
    
    # 验证假设
    result = verify_hypothesis(
        args.dataset, labels_array, isp, hop_distances, direction_consistency
    )
    
    # 打印总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    print(f"\nSmoothGNN ISP (到收敛的距离):")
    print(f"  正常节点: {result['normal_isp_mean']:.6f}")
    print(f"  异常节点: {result['anomaly_isp_mean']:.6f}")
    print(f"  差异: {result['isp_diff']:+.6f}")
    print(f"  p-value: {result['p_value_smoothgnn']:.4e}")
    print(f"  Cohen's d: {result['cohens_d']:.4f}")
    print(f"  结论: {result['conclusion_smoothgnn']}")
    
    print(f"\n旧 ISP (方向一致性):")
    print(f"  p-value: {result['p_value_old']:.4e}")
    print(f"  结论: {result['conclusion_old']}")
    
    return result


if __name__ == "__main__":
    main()
