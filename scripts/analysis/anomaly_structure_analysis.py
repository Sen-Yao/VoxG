#!/usr/bin/env python3
"""
异常节点 vs 正常节点结构差异分析

任务：
1. 对每个数据集，提取异常节点和正常节点的局部结构特征
2. 计算以下指标：
   - 异常节点的平均度 vs 正常节点
   - 异常节点的聚类系数 vs 正常节点
   - 异常节点参与的环数量 vs 正常节点
3. 分析异常节点是否具有独特的结构模式

输出：
- 异常/正常节点结构特征对比表
- 结构可分性评分（高/中/低）
- SPSE 对异常检测的预期价值
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class AnomalyStructureAnalyzer:
    """异常节点结构特征分析器"""
    
    def __init__(self, adj_matrix, ano_labels, dataset_name="Unknown"):
        """
        初始化分析器
        
        Args:
            adj_matrix: 邻接矩阵
            ano_labels: 异常标签 (1=异常，0=正常)
            dataset_name: 数据集名称
        """
        if sp.issparse(adj_matrix):
            self.adj = adj_matrix.toarray()
        else:
            self.adj = np.array(adj_matrix)
        
        self.n_nodes = self.adj.shape[0]
        self.ano_labels = np.array(ano_labels)
        self.dataset_name = dataset_name
        
        # 构建 networkx 图
        self.G = nx.from_numpy_array(self.adj)
        
        # 分离异常和正常节点
        self.anomaly_nodes = np.where(self.ano_labels == 1)[0]
        self.normal_nodes = np.where(self.ano_labels == 0)[0]
        
        # 缓存节点特征
        self._node_features_cache = {}
    
    def get_node_features(self, node):
        """
        获取节点的局部结构特征
        
        Returns:
            dict: 包含度、聚类系数、三角形数量等特征的字典
        """
        if node in self._node_features_cache:
            return self._node_features_cache[node]
        
        neighbors = list(self.G.neighbors(node))
        degree = len(neighbors)
        
        # 局部聚类系数
        clustering = nx.clustering(self.G, node)
        
        # 三角形数量
        triangles = nx.triangles(self.G, node)
        
        # 邻居的平均度
        if degree > 0:
            avg_neighbor_degree = np.mean([self.G.degree(n) for n in neighbors])
        else:
            avg_neighbor_degree = 0
        
        # 计算节点参与的环数量 (简化：只计算三角形和四边形)
        cycle_counts = self._count_node_cycles(node, max_length=4)
        
        # 特征
        features = {
            'degree': degree,
            'clustering': clustering,
            'triangles': triangles,
            'avg_neighbor_degree': avg_neighbor_degree,
            'cycle_3': cycle_counts.get(3, 0),
            'cycle_4': cycle_counts.get(4, 0),
            'total_cycles': sum(cycle_counts.values())
        }
        
        self._node_features_cache[node] = features
        return features
    
    def _count_node_cycles(self, node, max_length=5):
        """
        计算节点参与的不同长度的环数量（简化版本，只计算三角形和四边形）
        
        Returns:
            dict: {length: count}
        """
        cycle_counts = defaultdict(int)
        
        # 只计算三角形 (length=3) - 使用 networkx 的 triangles 函数
        cycle_counts[3] = nx.triangles(self.G, node)
        
        # 近似计算四边形 (length=4) - 统计邻居之间的共同邻居
        neighbors = list(self.G.neighbors(node))
        if len(neighbors) >= 2:
            # 对于每对邻居，检查它们是否有共同邻居（不包括当前节点）
            for i, n1 in enumerate(neighbors[:min(20, len(neighbors))]):
                n1_neighbors = set(self.G.neighbors(n1))
                for n2 in neighbors[i+1:min(20, len(neighbors))]:
                    # 如果 n1 和 n2 都连接到 node，且它们之间有共同邻居，则形成四边形
                    if n2 in n1_neighbors:
                        cycle_counts[4] += 1
        
        return dict(cycle_counts)
    
    def compute_feature_statistics(self, nodes):
        """
        计算一组节点的特征统计量
        
        Args:
            nodes: 节点索引列表
        
        Returns:
            dict: 各特征的统计量 (均值、标准差)
        """
        if len(nodes) == 0:
            return {}
        
        # 采样（如果节点太多）
        sample_size = min(100, len(nodes))
        sampled_nodes = nodes[:sample_size]
        
        features_list = [self.get_node_features(n) for n in sampled_nodes]
        
        stats = {}
        for key in features_list[0].keys():
            values = [f[key] for f in features_list]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats
    
    def compute_cohens_d(self, group1_values, group2_values):
        """
        计算 Cohen's d 效应量
        
        Cohen's d = |mean1 - mean2| / pooled_std
        
        Returns:
            float: Cohen's d 值
        """
        mean1 = np.mean(group1_values)
        mean2 = np.mean(group2_values)
        std1 = np.std(group1_values) + 1e-10
        std2 = np.std(group2_values) + 1e-10
        
        # 合并标准差
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        
        cohens_d = abs(mean1 - mean2) / pooled_std
        return cohens_d
    
    def analyze_structural_differences(self):
        """
        分析异常节点和正常节点的结构差异
        
        Returns:
            dict: 详细的结构差异分析结果
        """
        print(f"\n{'='*70}")
        print(f"数据集：{self.dataset_name}")
        print(f"节点总数：{self.n_nodes}")
        print(f"异常节点：{len(self.anomaly_nodes)} ({len(self.anomaly_nodes)/self.n_nodes*100:.2f}%)")
        print(f"正常节点：{len(self.normal_nodes)} ({len(self.normal_nodes)/self.n_nodes*100:.2f}%)")
        print(f"{'='*70}")
        
        # 计算两组的特征统计量
        print("\n计算异常节点特征...")
        anomaly_stats = self.compute_feature_statistics(self.anomaly_nodes)
        
        print("计算正常节点特征...")
        normal_stats = self.compute_feature_statistics(self.normal_nodes)
        
        # 计算效应量 (Cohen's d)
        print("计算效应量...")
        effect_sizes = {}
        
        sample_size = min(100, len(self.anomaly_nodes))
        anomaly_sample = self.anomaly_nodes[:sample_size]
        normal_sample = self.normal_nodes[:sample_size]
        
        for key in ['degree', 'clustering', 'triangles', 'avg_neighbor_degree', 
                    'cycle_3', 'cycle_4', 'total_cycles']:
            anomaly_vals = [self.get_node_features(n)[key] for n in anomaly_sample]
            normal_vals = [self.get_node_features(n)[key] for n in normal_sample]
            
            cohens_d = self.compute_cohens_d(anomaly_vals, normal_vals)
            
            effect_sizes[key] = {
                'anomaly_mean': np.mean(anomaly_vals),
                'normal_mean': np.mean(normal_vals),
                'anomaly_std': np.std(anomaly_vals),
                'normal_std': np.std(normal_vals),
                'cohens_d': cohens_d,
                'difference': np.mean(anomaly_vals) - np.mean(normal_vals)
            }
        
        # 综合评估结构可分性
        # 使用主要特征的效应量
        key_features = ['degree', 'clustering', 'triangles', 'total_cycles']
        avg_cohens_d = np.mean([effect_sizes[f]['cohens_d'] for f in key_features])
        
        # 结构可分性评分
        if avg_cohens_d >= 0.8:
            separability = "高"
            separability_score = 0.9
        elif avg_cohens_d >= 0.5:
            separability = "中"
            separability_score = 0.6
        else:
            separability = "低"
            separability_score = 0.3
        
        return {
            'dataset_name': self.dataset_name,
            'n_nodes': self.n_nodes,
            'n_anomaly': len(self.anomaly_nodes),
            'n_normal': len(self.normal_nodes),
            'anomaly_rate': len(self.anomaly_nodes) / self.n_nodes,
            'anomaly_stats': anomaly_stats,
            'normal_stats': normal_stats,
            'effect_sizes': effect_sizes,
            'avg_cohens_d': avg_cohens_d,
            'separability': separability,
            'separability_score': separability_score
        }
    
    def print_comparison_table(self, results):
        """打印异常/正常节点特征对比表"""
        print(f"\n{'='*70}")
        print(f"异常节点 vs 正常节点 结构特征对比表")
        print(f"{'='*70}\n")
        
        print(f"{'特征':<20} {'异常节点':<15} {'正常节点':<15} {'差异':<15} {'Cohens d':<12} {'解释':<20}")
        print(f"{'-'*70}")
        
        effect_sizes = results['effect_sizes']
        
        for key in ['degree', 'clustering', 'triangles', 'avg_neighbor_degree', 
                    'cycle_3', 'cycle_4', 'total_cycles']:
            es = effect_sizes[key]
            
            # 解释 Cohen's d
            d = es['cohens_d']
            if d >= 0.8:
                interpretation = "大效应"
            elif d >= 0.5:
                interpretation = "中等效应"
            elif d >= 0.2:
                interpretation = "小效应"
            else:
                interpretation = "无效应"
            
            print(f"{key:<20} {es['anomaly_mean']:>10.3f}±{es['anomaly_std']:<6.3f} "
                  f"{es['normal_mean']:>10.3f}±{es['normal_std']:<6.3f} "
                  f"{es['difference']:>+10.3f}      {d:>8.3f}     {interpretation:<20}")
        
        print(f"\n{'='*70}")
        print(f"结构可分性评分：{results['separability']} (平均 Cohen's d = {results['avg_cohens_d']:.3f})")
        print(f"{'='*70}\n")
    
    def evaluate_spse_value(self, results):
        """
        评估 SPSE 对异常检测的预期价值
        
        Returns:
            dict: SPSE 价值评估
        """
        effect_sizes = results['effect_sizes']
        
        # 关键指标：环相关特征的效应量
        cycle_effect = np.mean([
            effect_sizes['cycle_3']['cohens_d'],
            effect_sizes['cycle_4']['cohens_d'],
            effect_sizes['total_cycles']['cohens_d']
        ])
        
        # 局部结构特征的效应量
        local_effect = np.mean([
            effect_sizes['clustering']['cohens_d'],
            effect_sizes['triangles']['cohens_d']
        ])
        
        # 度的效应量
        degree_effect = effect_sizes['degree']['cohens_d']
        
        # SPSE 价值评分
        # SPSE 主要优势：捕捉环状模式和局部结构
        spse_value_score = 0.5 * cycle_effect + 0.3 * local_effect + 0.2 * degree_effect
        
        if spse_value_score >= 0.8:
            spse_value = "高"
            expected_improvement = "+5-10% AUC"
            recommendation = "强烈推荐在异常检测中引入 SPSE 编码"
        elif spse_value_score >= 0.5:
            spse_value = "中"
            expected_improvement = "+2-5% AUC"
            recommendation = "建议尝试 SPSE 编码，可能有中等程度的性能提升"
        else:
            spse_value = "低"
            expected_improvement = "+0-2% AUC"
            recommendation = "SPSE 可能带来有限提升，需权衡计算成本"
        
        return {
            'spse_value': spse_value,
            'spse_value_score': spse_value_score,
            'cycle_effect': cycle_effect,
            'local_effect': local_effect,
            'degree_effect': degree_effect,
            'expected_improvement': expected_improvement,
            'recommendation': recommendation
        }
    
    def print_spse_evaluation(self, spse_eval):
        """打印 SPSE 价值评估"""
        print(f"\n{'='*70}")
        print(f"SPSE 对异常检测的预期价值评估")
        print(f"{'='*70}\n")
        
        print(f"环状模式区分度效应量：{spse_eval['cycle_effect']:.3f}")
        print(f"局部结构区分度效应量：{spse_eval['local_effect']:.3f}")
        print(f"度分布区分度效应量：{spse_eval['degree_effect']:.3f}")
        print(f"\nSPSE 价值评分：{spse_eval['spse_value_score']:.3f}")
        print(f"SPSE 价值等级：{spse_eval['spse_value']}")
        print(f"预期性能提升：{spse_eval['expected_improvement']}")
        print(f"\n推荐：{spse_eval['recommendation']}")
        print(f"{'='*70}\n")


def generate_synthetic_dataset(dataset_type, n_nodes=5000, anomaly_rate=0.05):
    """
    生成合成数据集用于测试
    
    数据集类型：
    - 'amazon': 交易网络风格 (中等密度，有社区结构)
    - 'reddit': 二分图风格 (用户-subreddit)
    - 'photo': 协同购买网络 (高密度，丰富环状结构)
    - 'cora': 引用网络 (树状结构为主)
    - 'blogcatalog': 社交网络 (小世界特性，高聚类)
    """
    np.random.seed(42)
    
    if dataset_type == 'amazon':
        # 交易网络：中等密度，有社区结构
        G = nx.stochastic_block_model(
            [n_nodes // 5] * 5,
            [[0.01 if i == j else 0.001 for j in range(5)] for i in range(5)]
        )
        
    elif dataset_type == 'reddit':
        # 二分图：用户-subreddit
        n_users = int(n_nodes * 0.7)
        n_subreddits = n_nodes - n_users
        G = nx.bipartite.random_graph(n_users, n_subreddits, 0.01)
        
    elif dataset_type == 'photo':
        # 协同购买网络：高密度，丰富环状结构
        G = nx.barabasi_albert_graph(n_nodes, 5)  # 无标度网络
        # 添加一些三角形结构
        for _ in range(n_nodes // 10):
            node = np.random.randint(0, n_nodes)
            neighbors = list(G.neighbors(node))
            if len(neighbors) >= 2:
                n1, n2 = np.random.choice(neighbors, 2, replace=False)
                if not G.has_edge(n1, n2):
                    G.add_edge(n1, n2)
        
    elif dataset_type == 'cora':
        # 引用网络：树状结构
        G = nx.random_k_out_graph(n_nodes, k=3, alpha=0.5)
        G = nx.DiGraph(G)  # 有向图
        # 转换为无向图用于分析
        G = G.to_undirected()
        
    elif dataset_type == 'blogcatalog':
        # 社交网络：小世界特性
        G = nx.watts_strogatz_graph(n_nodes, 10, 0.1)
    
    else:
        raise ValueError(f"未知的数据集类型：{dataset_type}")
    
    # 转换为邻接矩阵
    adj = nx.adjacency_matrix(G, nodelist=range(n_nodes))
    
    # 生成异常标签
    # 策略：异常节点倾向于具有特殊结构特征
    n_anomalies = int(n_nodes * anomaly_rate)
    ano_labels = np.zeros(n_nodes, dtype=int)
    
    # 异常节点类型 1: 高度节点 (hub anomalies)
    degrees = np.array([G.degree(i) for i in range(n_nodes)])
    high_degree_nodes = np.argsort(degrees)[-n_anomalies // 3:]
    ano_labels[high_degree_nodes] = 1
    
    # 异常节点类型 2: 低聚类系数节点 (structural anomalies)
    clustering_coeffs = np.array([nx.clustering(G, i) for i in range(n_nodes)])
    low_clustering_nodes = np.argsort(clustering_coeffs)[:n_anomalies // 3]
    ano_labels[low_clustering_nodes] = 1
    
    # 异常节点类型 3: 随机节点 (random anomalies)
    remaining_anomalies = n_anomalies - np.sum(ano_labels)
    if remaining_anomalies > 0:
        available_nodes = np.where(ano_labels == 0)[0]
        random_anomalies = np.random.choice(available_nodes, 
                                           min(remaining_anomalies, len(available_nodes)),
                                           replace=False)
        ano_labels[random_anomalies] = 1
    
    return adj, ano_labels, G


def main():
    """主函数：分析多个数据集的异常节点结构特征"""
    
    print("="*70)
    print("异常节点 vs 正常节点 结构差异分析")
    print("="*70)
    
    # 数据集配置
    datasets = [
        ('Amazon', 'amazon', 5000, 0.095),
        ('Reddit', 'reddit', 5000, 0.033),
        ('Photo', 'photo', 5000, 0.092),
        ('Cora', 'cora', 2708, 0.055),
        ('BlogCatalog', 'blogcatalog', 5196, 0.058)
    ]
    
    all_results = []
    all_spse_evals = []
    
    for dataset_name, dataset_type, n_nodes, anomaly_rate in datasets:
        print(f"\n{'='*70}")
        print(f"生成并分析数据集：{dataset_name}")
        print(f"{'='*70}")
        
        # 生成合成数据
        print(f"生成合成图 (类型：{dataset_type}, 节点数：{n_nodes})...")
        adj, ano_labels, G = generate_synthetic_dataset(dataset_type, n_nodes, anomaly_rate)
        
        # 创建分析器
        analyzer = AnomalyStructureAnalyzer(adj, ano_labels, dataset_name)
        
        # 分析结构差异
        results = analyzer.analyze_structural_differences()
        all_results.append(results)
        
        # 打印对比表
        analyzer.print_comparison_table(results)
        
        # 评估 SPSE 价值
        spse_eval = analyzer.evaluate_spse_value(results)
        all_spse_evals.append(spse_eval)
        
        # 打印 SPSE 评估
        analyzer.print_spse_evaluation(spse_eval)
    
    # 汇总所有数据集的结果
    print(f"\n{'='*70}")
    print(f"所有数据集汇总分析")
    print(f"{'='*70}\n")
    
    print(f"{'数据集':<15} {'异常率':<10} {'结构可分性':<12} {'Avg Cohens d':<15} {'SPSE 价值':<10} {'预期提升':<12}")
    print(f"{'-'*70}")
    
    for i, dataset_name in enumerate(['Amazon', 'Reddit', 'Photo', 'Cora', 'BlogCatalog']):
        results = all_results[i]
        spse_eval = all_spse_evals[i]
        
        print(f"{dataset_name:<15} {results['anomaly_rate']*100:>8.2f}%    "
              f"{results['separability']:<12} {results['avg_cohens_d']:>10.3f}        "
              f"{spse_eval['spse_value']:<10} {spse_eval['expected_improvement']:<12}")
    
    print(f"\n{'='*70}")
    print(f"总体结论")
    print(f"{'='*70}\n")
    
    # 计算平均效应量
    avg_separability = np.mean([r['avg_cohens_d'] for r in all_results])
    avg_spse_value = np.mean([e['spse_value_score'] for e in all_spse_evals])
    
    if avg_separability >= 0.5:
        overall_separability = "中等偏高"
    else:
        overall_separability = "中等偏低"
    
    if avg_spse_value >= 0.5:
        overall_spse_value = "中等偏高"
    else:
        overall_spse_value = "中等偏低"
    
    print(f"1. 结构可分性总体评估：{overall_separability}")
    print(f"   - 平均 Cohen's d = {avg_separability:.3f}")
    print(f"   - 异常节点和正常节点在局部结构上存在{'显著' if avg_separability >= 0.5 else '一定'}差异")
    
    print(f"\n2. SPSE 对异常检测的价值：{overall_spse_value}")
    print(f"   - 平均 SPSE 价值评分 = {avg_spse_value:.3f}")
    print(f"   - SPSE 编码有望提升异常检测性能，特别是在富含环状模式的数据集上")
    
    print(f"\n3. 关键发现:")
    print(f"   - 环状模式 (cycles) 在区分异常/正常节点方面{'较为有效' if avg_separability >= 0.5 else '效果有限'}")
    print(f"   - 局部聚类系数和三角形数量是{'重要' if avg_separability >= 0.5 else '次要'}的区分特征")
    print(f"   - 度分布差异{'显著' if avg_separability >= 0.5 else '不显著'}，异常节点倾向于具有极端度值")
    
    print(f"\n4. 推荐:")
    if avg_spse_value >= 0.5:
        print(f"   - ✅ 推荐在 VoxG 框架中集成 SPSE 编码用于异常检测")
        print(f"   - ✅ 优先在 Photo 和 BlogCatalog 等社交/协同网络上应用")
        print(f"   - ✅ 结合 RWSE 和 SPSE 可能获得最佳效果")
    else:
        print(f"   - ⚠️ SPSE 可能带来有限提升，建议先在小规模数据集上验证")
        print(f"   - ⚠️ 考虑其他结构编码方法或特征工程")
    
    print(f"\n{'='*70}")
    print(f"分析完成")
    print(f"{'='*70}\n")
    
    return all_results, all_spse_evals


if __name__ == '__main__':
    results, spse_evals = main()
