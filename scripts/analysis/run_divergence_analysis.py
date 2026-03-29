#!/usr/bin/env python3
"""PPR vs SPSE 编码差异度分析"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import json
import time

def analyze_graph(name, n_nodes, p_edge, cycle_boost):
    """分析单个图"""
    # 生成图
    G = nx.erdos_renyi_graph(n_nodes, p_edge)
    nodes = list(G.nodes())
    for _ in range(int(n_nodes * cycle_boost)):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    
    adj = nx.adjacency_matrix(G)
    n_edges = adj.nnz
    
    # PPR 编码
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ adj
    
    max_hops = 4
    ppr_encoding = np.zeros((n_nodes, n_nodes, max_hops))
    P_k = sp.eye(n_nodes)
    for k in range(max_hops):
        P_k = P_k @ P
        ppr_encoding[:, :, k] = P_k.toarray()
    
    # SPSE 编码（矩阵幂近似）
    A = adj.toarray()
    spse_encoding = np.zeros((n_nodes, n_nodes, max_hops))
    A_k = np.eye(n_nodes)
    for k in range(max_hops):
        A_k = A_k @ A
        spse_encoding[:, :, k] = A_k
    
    # 差异度
    ppr_flat = ppr_encoding.flatten()
    spse_flat = spse_encoding.flatten()
    ppr_norm = ppr_flat / (np.linalg.norm(ppr_flat) + 1e-10)
    spse_norm = spse_flat / (np.linalg.norm(spse_flat) + 1e-10)
    divergence = np.linalg.norm(ppr_norm - spse_norm) / (np.linalg.norm(spse_norm) + 1e-10)
    
    # 环密度
    cycles = list(nx.simple_cycles(G.to_directed()))
    cycle_counts = {}
    for length in range(3, 7):
        cycle_counts[length] = len([c for c in cycles if len(c) == length])
    total_cycles = sum(cycle_counts.values())
    max_possible = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6
    cycle_density = total_cycles / max_possible if max_possible > 0 else 0
    
    return {
        'name': name,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'divergence': float(divergence),
        'cycle_density': float(cycle_density),
        'cycle_counts': cycle_counts,
        'total_cycles': total_cycles
    }

# 主分析
print("="*70)
print("PPR vs SPSE 编码差异度分析")
print("="*70)

configs = [
    ('Sparse_LowCycle', 25, 0.08, 0.1),
    ('Sparse_HighCycle', 25, 0.08, 0.5),
    ('Dense_LowCycle', 25, 0.2, 0.1),
    ('Dense_HighCycle', 25, 0.2, 0.5),
    ('Medium_Balanced', 30, 0.12, 0.3),
    ('Small_Ring', 20, 0.15, 0.6),
]

results = []
for name, n_nodes, p_edge, cycle_boost in configs:
    print(f"\n分析：{name}...")
    r = analyze_graph(name, n_nodes, p_edge, cycle_boost)
    results.append(r)
    print(f"  D(G)={r['divergence']:.4f}, CycleDensity={r['cycle_density']:.6f}, Cycles={r['total_cycles']}")

# 相关性分析
divergences = [r['divergence'] for r in results]
cycle_densities = [r['cycle_density'] for r in results]
correlation = np.corrcoef(cycle_densities, divergences)[0, 1]

print("\n" + "="*70)
print("结果汇总")
print("="*70)
print(f"\n{'数据集':22} | {'节点':4} | {'边':4} | {'D(G)':7} | {'环密度':10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:22} | {r['n_nodes']:4} | {r['n_edges']:4} | {r['divergence']:7.4f} | {r['cycle_density']:10.6f}")

print(f"\n相关系数 (Divergence vs Cycle Density): {correlation:.4f}")

# 保存结果
save_dir = './figs/ppr_spse_divergence'
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, 'divergence_results.json'), 'w') as f:
    json.dump({'results': results, 'correlation': float(correlation)}, f, indent=2)

with open(os.path.join(save_dir, 'divergence_analysis_report.txt'), 'w', encoding='utf-8') as f:
    f.write("PPR vs SPSE Encoding Divergence Analysis Report\n")
    f.write("="*70 + "\n\n")
    for r in results:
        f.write(f"Dataset: {r['name']}\n")
        f.write(f"  Nodes: {r['n_nodes']}, Edges: {r['n_edges']}\n")
        f.write(f"  Divergence D(G): {r['divergence']:.4f}\n")
        f.write(f"  Cycle Density: {r['cycle_density']:.6f}\n")
        f.write(f"  Total Cycles: {r['total_cycles']}\n\n")
    f.write(f"\nCorrelation(Divergence, Cycle Density): {correlation:.4f}\n")

print(f"\n结果已保存至：{save_dir}/")
print("分析完成!")
