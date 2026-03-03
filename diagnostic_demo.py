#!/usr/bin/env python3
"""
VoxG 诊断指标演示脚本

使用模拟图数据演示诊断指标的计算方法
可用于理解 SPSE 必要性的评估逻辑
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_synthetic_graphs():
    """
    创建多种类型的合成图用于演示
    
    Returns:
        dict: 图字典 {name: (G, ano_labels)}
    """
    graphs = {}
    
    # 1. 高环状模式图 (类似社交网络/协同购买)
    print("创建高环状模式图 (类似 Photo/BlogCatalog)...")
    G_cycle = nx.barabasi_albert_graph(1000, 3)  # BA 无标度网络，富含三角形
    # 添加一些异常节点（度数异常）
    ano_labels_cycle = np.zeros(1000)
    anomaly_nodes = np.random.choice(1000, 50, replace=False)
    for node in anomaly_nodes:
        # 异常节点：连接到随机节点
        for _ in range(10):
            target = np.random.randint(0, 1000)
            if target != node:
                G_cycle.add_edge(node, target)
        ano_labels_cycle[node] = 1
    graphs['High-Cycle (Photo-like)'] = (G_cycle, ano_labels_cycle)
    
    # 2. 低环状模式图 (类似引用网络)
    print("创建低环状模式图 (类似 Cora)...")
    G_tree = nx.balanced_tree(3, 6)  # 平衡树，几乎无环
    # 添加少量边形成小环
    nodes = list(G_tree.nodes())
    for _ in range(50):
        i, j = np.random.choice(len(nodes), 2, replace=False)
        G_tree.add_edge(nodes[i], nodes[j])
    
    ano_labels_tree = np.zeros(len(G_tree.nodes()))
    anomaly_nodes = np.random.choice(len(nodes), 20, replace=False)
    for idx in anomaly_nodes:
        ano_labels_tree[idx] = 1
    graphs['Low-Cycle (Cora-like)'] = (G_tree, ano_labels_tree)
    
    # 3. 二分图 (类似 Reddit 用户 -subreddit)
    print("创建二分图 (类似 Reddit)...")
    G_bipartite = nx.bipartite.gnmk_random_graph(500, 500, 2000)
    ano_labels_bipartite = np.zeros(1000)
    anomaly_nodes = np.random.choice(1000, 30, replace=False)
    for node in anomaly_nodes:
        ano_labels_bipartite[node] = 1
    graphs['Bipartite (Reddit-like)'] = (G_bipartite, ano_labels_bipartite)
    
    # 4. 高聚类图 (类似交易网络)
    print("创建高聚类图 (类似 Amazon)...")
    G_clustered = nx.watts_strogatz_graph(1000, 10, 0.3)  # 小世界网络
    ano_labels_clustered = np.zeros(1000)
    anomaly_nodes = np.random.choice(1000, 40, replace=False)
    for node in anomaly_nodes:
        ano_labels_clustered[node] = 1
    graphs['Clustered (Amazon-like)'] = (G_clustered, ano_labels_clustered)
    
    return graphs


class DiagnosticMetricsCalculator:
    """诊断指标计算器（简化版用于演示）"""
    
    def __init__(self, G, ano_labels=None):
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.n_edges = G.number_of_edges()
        self.ano_labels = ano_labels
    
    def compute_cycle_richness(self, max_cycle_length=6):
        """计算环状模式丰富度"""
        cycle_counts = {}
        total_cycles = 0
        
        # 使用近似方法计算三角形数量（长度为 3 的环）
        triangles = sum(nx.triangles(self.G).values()) // 3
        cycle_counts[3] = triangles
        total_cycles += triangles
        
        # 对于更长的环，使用采样近似
        if self.n_nodes <= 500:
            for length in range(4, min(max_cycle_length + 1, 7)):
                try:
                    cycles = list(nx.simple_cycles(self.G.to_directed()))
                    cycles_of_length = [c for c in cycles if len(c) == length]
                    cycle_counts[length] = len(cycles_of_length) // length  # 去重
                    total_cycles += cycle_counts[length]
                except:
                    cycle_counts[length] = 0
        else:
            # 大图的近似
            for length in range(4, max_cycle_length + 1):
                cycle_counts[length] = 0  # 简化处理
        
        # 计算密度
        max_possible = self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 6
        cycle_density = total_cycles / max_possible if max_possible > 0 else 0
        
        # 综合评分
        avg_cycle_length = 3.0  # 主要是三角形
        if total_cycles > 0:
            weighted_sum = sum(l * c for l, c in cycle_counts.items())
            avg_cycle_length = weighted_sum / total_cycles
        
        richness_score = min(1.0, cycle_density * 1000 + (avg_cycle_length - 3) / 10)
        
        return {
            'total_cycles': total_cycles,
            'triangles': triangles,
            'cycle_counts_by_length': cycle_counts,
            'cycle_density': cycle_density,
            'richness_score': richness_score
        }
    
    def compute_rwse_discriminability(self, max_steps=6):
        """计算 RWSE 区分度（简化版）"""
        # 使用度数序列作为 RWSE 的近似代理
        # 真实的 RWSE 计算需要矩阵乘法，这里简化演示
        degrees = [d for _, d in self.G.degree()]
        
        # 统计度数分布
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1
        
        # 计算无法区分的节点对（度数相同）
        indistinguishable = sum(c * (c - 1) / 2 for c in degree_counts.values())
        total_pairs = self.n_nodes * (self.n_nodes - 1) / 2
        
        discriminability = 1.0 - (indistinguishable / total_pairs) if total_pairs > 0 else 0
        
        return {
            'total_node_pairs': int(total_pairs),
            'indistinguishable_pairs': int(indistinguishable),
            'discriminability': discriminability,
            'spse_necessity': 1.0 - discriminability
        }
    
    def compute_path_complexity(self, max_length=3):
        """计算路径复杂性（简化版）"""
        # 采样节点对计算路径数
        n_samples = min(100, self.n_nodes)
        sampled_nodes = np.random.choice(list(self.G.nodes()), n_samples, replace=False)
        
        path_counts = []
        for i, source in enumerate(sampled_nodes):
            for target in sampled_nodes[i+1:]:
                try:
                    paths = list(nx.all_simple_paths(self.G, source, target, cutoff=max_length))
                    if len(paths) > 0:
                        path_counts.append(len(paths))
                except:
                    pass
        
        if len(path_counts) == 0:
            return {
                'avg_paths': 0,
                'variance': 0,
                'complexity_score': 0
            }
        
        avg_paths = np.mean(path_counts)
        variance = np.var(path_counts)
        complexity_score = min(1.0, variance / (avg_paths + 1) * 0.5)
        
        return {
            'n_sampled_pairs': len(path_counts),
            'avg_paths': avg_paths,
            'variance': variance,
            'complexity_score': complexity_score
        }
    
    def compute_anomaly_structure_diff(self):
        """计算异常节点结构差异"""
        if self.ano_labels is None or np.sum(self.ano_labels) == 0:
            return {
                'error': 'No anomaly labels',
                'spse_necessity': 0.5
            }
        
        anomaly_nodes = np.where(self.ano_labels == 1)[0]
        normal_nodes = np.where(self.ano_labels == 0)[0]
        
        # 计算度数差异
        anomaly_degrees = [self.G.degree(n) for n in anomaly_nodes]
        normal_degrees = [self.G.degree(n) for n in normal_nodes]
        
        anomaly_mean = np.mean(anomaly_degrees)
        normal_mean = np.mean(normal_degrees)
        pooled_std = np.sqrt((np.std(anomaly_degrees)**2 + np.std(normal_degrees)**2) / 2 + 1e-10)
        
        cohens_d = abs(anomaly_mean - normal_mean) / pooled_std
        structure_diff_score = min(1.0, cohens_d / 3)
        
        return {
            'n_anomaly': len(anomaly_nodes),
            'n_normal': len(normal_nodes),
            'anomaly_mean_degree': anomaly_mean,
            'normal_mean_degree': normal_mean,
            'cohens_d': cohens_d,
            'structure_diff_score': structure_diff_score,
            'spse_necessity': structure_diff_score
        }
    
    def compute_all(self):
        """计算所有诊断指标"""
        print(f"\n  计算诊断指标 (节点：{self.n_nodes}, 边：{self.n_edges})...")
        
        metrics = {}
        
        print("    - 环状模式丰富度...")
        metrics['cycle_richness'] = self.compute_cycle_richness()
        
        print("    - RWSE 区分度...")
        metrics['rwse_discriminability'] = self.compute_rwse_discriminability()
        
        print("    - 路径复杂性...")
        metrics['path_complexity'] = self.compute_path_complexity()
        
        print("    - 异常结构差异...")
        metrics['anomaly_structure'] = self.compute_anomaly_structure_diff()
        
        # 综合评估
        cycle_score = metrics['cycle_richness']['richness_score']
        rwse_necessity = metrics['rwse_discriminability']['spse_necessity']
        path_score = metrics['path_complexity']['complexity_score']
        anomaly_score = metrics['anomaly_structure']['spse_necessity']
        
        overall = 0.35 * cycle_score + 0.30 * rwse_necessity + 0.20 * path_score + 0.15 * anomaly_score
        
        if overall >= 0.7:
            level = "高"
        elif overall >= 0.4:
            level = "中"
        else:
            level = "低"
        
        metrics['spse_necessity'] = {
            'overall_score': overall,
            'level': level,
            'component_scores': {
                'cycle_richness': cycle_score,
                'rwse_discriminability': 1.0 - rwse_necessity,
                'path_complexity': path_score,
                'anomaly_structure': anomaly_score
            }
        }
        
        return metrics


def print_results(results):
    """打印诊断结果"""
    print("\n" + "="*80)
    print("VoxG 诊断指标分析结果 (模拟数据)")
    print("="*80)
    
    for name, metrics in results.items():
        print(f"\n{name}")
        print("-" * 60)
        
        cycle = metrics['cycle_richness']
        rwse = metrics['rwse_discriminability']
        path = metrics['path_complexity']
        anomaly = metrics['anomaly_structure']
        spse = metrics['spse_necessity']
        
        print(f"  环状模式丰富度：{cycle['richness_score']:.3f} (三角形：{cycle['triangles']})")
        print(f"  RWSE 区分度：    {rwse['discriminability']:.3f}")
        print(f"  路径复杂性：    {path['complexity_score']:.3f}")
        print(f"  异常结构差异：  {anomaly.get('spse_necessity', 0):.3f}")
        print(f"\n  ⇒ SPSE 必要性：{spse['level']} (综合评分：{spse['overall_score']:.3f})")


def plot_comparison(results):
    """绘制诊断指标对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    datasets = list(results.keys())
    x = np.arange(len(datasets))
    
    # 1. 环状模式丰富度
    ax1 = axes[0, 0]
    cycle_scores = [results[d]['cycle_richness']['richness_score'] for d in datasets]
    bars1 = ax1.bar(x, cycle_scores, color='#3498db', alpha=0.8)
    ax1.set_ylabel('Richness Score')
    ax1.set_title('环状模式丰富度')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.split()[0] for d in datasets], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(cycle_scores):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # 2. RWSE 区分度
    ax2 = axes[0, 1]
    rwse_scores = [results[d]['rwse_discriminability']['discriminability'] for d in datasets]
    bars2 = ax2.bar(x, rwse_scores, color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Discriminability')
    ax2.set_title('RWSE 区分度')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.split()[0] for d in datasets], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(rwse_scores):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # 3. 路径复杂性
    ax3 = axes[1, 0]
    path_scores = [results[d]['path_complexity']['complexity_score'] for d in datasets]
    bars3 = ax3.bar(x, path_scores, color='#2ecc71', alpha=0.8)
    ax3.set_ylabel('Complexity Score')
    ax3.set_title('路径复杂性')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.split()[0] for d in datasets], rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    for i, v in enumerate(path_scores):
        ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # 4. SPSE 必要性综合评分
    ax4 = axes[1, 1]
    spse_scores = [results[d]['spse_necessity']['overall_score'] for d in datasets]
    colors = ['#e74c3c' if s >= 0.7 else '#f39c12' if s >= 0.4 else '#27ae60' for s in spse_scores]
    bars4 = ax4.bar(x, spse_scores, color=colors, alpha=0.8)
    ax4.set_ylabel('Overall Score')
    ax4.set_title('SPSE 必要性综合评分')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.split()[0] for d in datasets], rotation=45, ha='right')
    ax4.set_ylim(0, 1)
    ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='高必要性')
    ax4.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='中必要性')
    ax4.legend()
    for i, v in enumerate(spse_scores):
        level = '高' if v >= 0.7 else '中' if v >= 0.4 else '低'
        ax4.text(i, v + 0.02, f'{v:.2f}({level})', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = '/home/openclawvm/.openclaw/workspace/projects/VoxG/figs/diagnostic_metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存：{output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("VoxG 诊断指标演示 - 评估 SPSE 必要性")
    print("="*80)
    
    # 创建合成图
    print("\n创建合成图数据...")
    graphs = create_synthetic_graphs()
    
    # 计算诊断指标
    results = {}
    for name, (G, ano_labels) in graphs.items():
        print(f"\n分析：{name}")
        calculator = DiagnosticMetricsCalculator(G, ano_labels)
        results[name] = calculator.compute_all()
    
    # 打印结果
    print_results(results)
    
    # 绘制对比图
    print("\n生成对比图表...")
    plot_comparison(results)
    
    # 输出总结
    print("\n" + "="*80)
    print("总结与推荐")
    print("="*80)
    
    # 按 SPSE 必要性排序
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['spse_necessity']['overall_score'], 
                           reverse=True)
    
    print("\nSPSE 必要性排序:")
    for i, (name, metrics) in enumerate(sorted_results, 1):
        level = metrics['spse_necessity']['level']
        score = metrics['spse_necessity']['overall_score']
        print(f"  {i}. {name:30} | {level:3} | {score:.3f}")
    
    print("\n推荐优先级:")
    high_priority = [name for name, m in results.items() if m['spse_necessity']['overall_score'] >= 0.7]
    medium_priority = [name for name, m in results.items() if 0.4 <= m['spse_necessity']['overall_score'] < 0.7]
    low_priority = [name for name, m in results.items() if m['spse_necessity']['overall_score'] < 0.4]
    
    if high_priority:
        print(f"  高优先级：{', '.join(high_priority)}")
    if medium_priority:
        print(f"  中优先级：{', '.join(medium_priority)}")
    if low_priority:
        print(f"  低优先级：{', '.join(low_priority)}")
    
    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)


if __name__ == '__main__':
    main()
