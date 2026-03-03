#!/usr/bin/env python3
"""
计算 PPR Tokenization vs SPSE 的编码差异度（极简快速版）
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import json
import time

def compute_ppr_encoding(adj, max_hops=4):
    """计算 PPR 编码矩阵"""
    n = adj.shape[0]
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ adj
    
    ppr_encoding = np.zeros((n, n, max_hops))
    P_k = sp.eye(n)
    
    for k in range(max_hops):
        P_k = P_k @ P
        ppr_encoding[:, :, k] = P_k.toarray()
    
    return ppr_encoding


def compute_spse_encoding_fast(adj, max_path_length=4):
    """计算 SPSE 编码矩阵（使用矩阵乘法近似）"""
    n = adj.shape[0]
    spse_encoding = np.zeros((n, n, max_path_length))
    
    # 使用邻接矩阵的幂近似路径计数
    A = adj.toarray() if sp.issparse(adj) else adj
    A_k = np.eye(n)
    
    for k in range(max_path_length):
        A_k = A_k @ A
        spse_encoding[:, :, k] = A_k
    
    return spse_encoding


def compute_divergence_fast(ppr_encoding, spse_encoding):
    """快速计算差异度"""
    max_dim = min(ppr_encoding.shape[2], spse_encoding.shape[2])
    
    # 展平
    ppr_flat = ppr_encoding[:, :, :max_dim].flatten()
    spse_flat = spse_encoding[:, :, :max_dim].flatten()
    
    # 归一化
    ppr_norm = ppr_flat / (np.linalg.norm(ppr_flat) + 1e-10)
    spse_norm = spse_flat / (np.linalg.norm(spse_flat) + 1e-10)
    
    # 计算差异度
    diff = ppr_norm - spse_norm
    divergence = np.linalg.norm(diff) / (np.linalg.norm(spse_norm) + 1e-10)
    
    return divergence


def compute_cycle_density(G, n_nodes):
    """计算环密度"""
    cycle_counts = {}
    total_cycles = 0
    
    try:
        cycles = list(nx.simple_cycles(G.to_directed()))
        for length in range(3, 7):
            count = len([c for c in cycles if len(c) == length])
            cycle_counts[length] = count
            total_cycles += count
    except:
        for length in range(3, 7):
            cycle_counts[length] = 0
    
    max_possible = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6
    cycle_density = total_cycles / max_possible if max_possible > 0 else 0
    
    return cycle_density, cycle_counts


def generate_synthetic_graph(n_nodes, p_edge, cycle_boost):
    """生成合成图"""
    G = nx.erdos_renyi_graph(n_nodes, p_edge)
    
    nodes = list(G.nodes())
    for _ in range(int(n_nodes * cycle_boost)):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    
    return nx.adjacency_matrix(G), G


def main():
    print("="*70)
    print("PPR vs SPSE 编码差异度分析（极简版）")
    print("="*70)
    
    max_hops = 4
    max_path_length = 4
    save_dir = './figs/ppr_spse_divergence'
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # 使用更小的合成数据集
    print("\n生成合成数据集进行分析...")
    synthetic_configs = [
        ('Sparse_LowCycle', 30, 0.08, 0.1),
        ('Sparse_HighCycle', 30, 0.08, 0.5),
        ('Dense_LowCycle', 30, 0.2, 0.1),
        ('Dense_HighCycle', 30, 0.2, 0.5),
        ('Medium_Balanced', 40, 0.12, 0.3),
        ('Small_Ring', 20, 0.15, 0.6),
    ]
    
    for name, n_nodes, p_edge, cycle_boost in synthetic_configs:
        print(f"\n处理：{name}...")
        start_time = time.time()
        
        # 生成图
        adj, G = generate_synthetic_graph(n_nodes, p_edge, cycle_boost)
        n_edges = adj.nnz
        
        # 计算编码
        ppr_encoding = compute_ppr_encoding(adj, max_hops=max_hops)
        spse_encoding = compute_spse_encoding_fast(adj, max_path_length=max_path_length)
        
        # 计算差异度
        divergence = compute_divergence_fast(ppr_encoding, spse_encoding)
        
        # 计算环密度
        cycle_density, cycle_counts = compute_cycle_density(G, n_nodes)
        
        elapsed = time.time() - start_time
        
        # 保存结果
        results[name] = {
            'divergence': float(divergence),
            'cycle_density': float(cycle_density),
            'cycle_counts': cycle_counts,
            'n_nodes': n_nodes,
            'n_edges': int(n_edges),
            'elapsed_time': elapsed
        }
        
        print(f"  D(G)={divergence:.4f}, CycleDensity={cycle_density:.6f}, Time={elapsed:.2f}s")
    
    # 相关性分析
    print("\n" + "="*70)
    print("相关性分析")
    print("="*70)
    
    datasets = list(results.keys())
    divergences = [results[d]['divergence'] for d in datasets]
    cycle_densities = [results[d]['cycle_density'] for d in datasets]
    
    correlation = np.corrcoef(cycle_densities, divergences)[0, 1]
    print(f"\n差异度与环密度的相关系数：{correlation:.4f}")
    
    if correlation > 0.7:
        interpretation = "强正相关：环密度越高，PPR 和 SPSE 的差异越大"
    elif correlation > 0.4:
        interpretation = "中等正相关"
    elif correlation > -0.4:
        interpretation = "弱相关或无相关"
    else:
        interpretation = "负相关"
    
    print(f"解释：{interpretation}")
    
    # 输出报告
    print("\n" + "="*70)
    print("综合报告")
    print("="*70)
    
    print(f"\n{'数据集':25} | {'节点':4} | {'边':4} | {'D(G)':7} | {'环密度':10} | {'时间':5}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:25} | {r['n_nodes']:4} | {r['n_edges']:4} | {r['divergence']:7.4f} | {r['cycle_density']:10.6f} | {r['elapsed_time']:5.2f}s")
    
    # 保存结果
    json_path = os.path.join(save_dir, 'divergence_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    report_path = os.path.join(save_dir, 'divergence_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PPR vs SPSE Encoding Divergence Analysis Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"日期：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, r in results.items():
            f.write(f"Dataset: {name}\n")
            f.write(f"  Nodes: {r['n_nodes']}, Edges: {r['n_edges']}\n")
            f.write(f"  Divergence D(G): {r['divergence']:.4f}\n")
            f.write(f"  Cycle Density: {r['cycle_density']:.6f}\n")
            f.write(f"  Cycle Counts: {r['cycle_counts']}\n")
            f.write(f"  Time: {r['elapsed_time']:.2f}s\n\n")
        
        f.write(f"\nCorrelation(Divergence, Cycle Density): {correlation:.4f}\n")
        f.write(f"Interpretation: {interpretation}\n")
    
    print(f"\n结果已保存至：{save_dir}/")
    print("\n分析完成!")
    
    return results


if __name__ == '__main__':
    main()
