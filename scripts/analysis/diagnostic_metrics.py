#!/usr/bin/env python3
"""
VoxG 诊断指标计算脚本
用于评估 VoxG 数据集是否需要 SPSE

诊断指标：
1. 环状模式丰富度 (Cycle Richness): 图中环的数量和分布
2. RWSE 区分度 (RWSE Discriminability): RWSE 无法区分的节点对比例
3. 路径计数分布 (Path Count Distribution): 不同长度简单路径的计数分布
4. 异常节点结构特征 (Anomaly Structural Features): 异常节点 vs 正常节点的局部结构差异

基于 SPSE 的 Motivation:
- RWSE 无法区分某些不同结构（如环 vs 路径）
- SPSE 使用简单路径计数，能更好捕捉环状模式
- 如果数据集富含环状模式且 RWSE 区分度低，则 SPSE 必要性高
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
from collections import defaultdict, Counter
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class VoxGDiagnosticMetrics:
    """VoxG 数据集诊断指标计算器"""
    
    def __init__(self, adj_matrix, ano_labels=None, max_cycle_length=6, max_path_length=4):
        """
        初始化诊断指标计算器
        
        Args:
            adj_matrix: 邻接矩阵 (scipy sparse 或 numpy array)
            ano_labels: 异常标签数组 (可选)
            max_cycle_length: 最大环长度
            max_path_length: 最大路径长度
        """
        if sp.issparse(adj_matrix):
            self.adj = adj_matrix.toarray()
        else:
            self.adj = np.array(adj_matrix)
        
        self.n_nodes = self.adj.shape[0]
        self.ano_labels = ano_labels
        self.max_cycle_length = max_cycle_length
        self.max_path_length = max_path_length
        
        # 构建 networkx 图
        self.G = nx.from_numpy_array(self.adj)
        
        # 缓存
        self._rwse_cache = None
        self._path_counts_cache = None
        
    def compute_rwse(self, max_steps=10):
        """
        计算随机游走结构编码 (RWSE)
        
        RWSE[i,j,k] = 从节点 i 出发，k 步后回到节点 j 的概率
        
        Returns:
            rwse: 形状为 (n_nodes, n_nodes, max_steps) 的数组
        """
        if self._rwse_cache is not None:
            return self._rwse_cache
        
        # 归一化邻接矩阵为转移概率矩阵
        row_sums = self.adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        P = self.adj / row_sums
        
        # 计算多步转移概率
        rwse = np.zeros((self.n_nodes, self.n_nodes, max_steps))
        P_k = np.eye(self.n_nodes)
        
        for k in range(max_steps):
            P_k = P_k @ P
            rwse[:, :, k] = P_k
        
        self._rwse_cache = rwse
        return rwse
    
    def compute_simple_path_counts(self, max_length=None):
        """
        计算简单路径计数 (SPSE 的核心)
        
        对于每对节点 (i,j)，计算长度为 k 的简单路径数量
        简单路径：不重复经过节点的路径
        
        Returns:
            path_counts: 字典 {(i,j): [count_len1, count_len2, ...]}
        """
        if max_length is None:
            max_length = self.max_path_length
        
        if self._path_counts_cache is not None:
            return self._path_counts_cache
        
        path_counts = defaultdict(lambda: [0] * max_length)
        
        # 使用 networkx 计算简单路径
        for source in range(self.n_nodes):
            for target in range(self.n_nodes):
                if source == target:
                    continue
                
                # 计算所有简单路径（限制长度）
                try:
                    paths = list(nx.all_simple_paths(self.G, source, target, cutoff=max_length))
                    for path in paths:
                        length = len(path) - 1  # 路径长度 = 边数
                        if 1 <= length <= max_length:
                            path_counts[(source, target)][length - 1] += 1
                except:
                    pass
        
        self._path_counts_cache = path_counts
        return path_counts
    
    def metric_cycle_richness(self):
        """
        指标 1: 环状模式丰富度
        
        计算图中不同长度的环的数量和分布
        环越多、越复杂，SPSE 的优势越明显
        
        Returns:
            dict: 包含环统计信息的字典
        """
        cycle_counts = {}
        total_cycles = 0
        
        # 计算不同长度的环
        for length in range(3, min(self.max_cycle_length + 1, self.n_nodes)):
            # 使用 networkx 查找所有长度为 length 的环
            cycles = list(nx.simple_cycles(self.G.to_directed()))
            cycles_of_length = [c for c in cycles if len(c) == length]
            cycle_counts[length] = len(cycles_of_length)
            total_cycles += len(cycles_of_length)
        
        # 计算环密度
        max_possible_cycles = self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 6  # 上界
        cycle_density = total_cycles / max_possible_cycles if max_possible_cycles > 0 else 0
        
        # 计算平均环长度
        if total_cycles > 0:
            avg_cycle_length = sum(l * c for l, c in cycle_counts.items()) / total_cycles
        else:
            avg_cycle_length = 0
        
        return {
            'total_cycles': total_cycles,
            'cycle_counts_by_length': cycle_counts,
            'cycle_density': cycle_density,
            'avg_cycle_length': avg_cycle_length,
            'cycle_richness_score': min(1.0, cycle_density * 100 + avg_cycle_length / 10)  # 综合评分
        }
    
    def metric_rwse_discriminability(self, max_steps=10, tolerance=1e-6):
        """
        指标 2: RWSE 区分度
        
        计算 RWSE 无法区分的节点对比例
        如果很多节点对有相同的 RWSE 编码，说明 RWSE 区分度低，SPSE 必要性高
        
        Returns:
            dict: 包含 RWSE 区分度统计的字典
        """
        rwse = self.compute_rwse(max_steps)
        
        # 将 RWSE 展平为节点对的特征向量
        n_pairs = self.n_nodes * (self.n_nodes - 1) // 2
        rwse_vectors = []
        
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                rwse_vectors.append(rwse[i, j, :])
        
        rwse_vectors = np.array(rwse_vectors)
        
        # 计算无法区分的节点对数量
        indistinguishable_pairs = 0
        for i in range(len(rwse_vectors)):
            for j in range(i + 1, len(rwse_vectors)):
                if np.allclose(rwse_vectors[i], rwse_vectors[j], atol=tolerance):
                    indistinguishable_pairs += 1
        
        discriminability = 1.0 - (indistinguishable_pairs / (n_pairs * (n_pairs - 1) / 2 + 1e-10))
        
        return {
            'total_node_pairs': n_pairs,
            'indistinguishable_pairs': indistinguishable_pairs,
            'rwse_discriminability': discriminability,
            'spse_necessity_from_rwse': 1.0 - discriminability  # RWSE 区分度越低，SPSE 必要性越高
        }
    
    def metric_path_count_distribution(self):
        """
        指标 3: 路径计数分布
        
        分析不同长度简单路径的计数分布
        路径计数变化越大，说明结构越复杂，SPSE 越有用
        
        Returns:
            dict: 包含路径计数分布统计的字典
        """
        path_counts = self.compute_simple_path_counts()
        
        # 统计所有路径计数的分布
        all_counts = []
        counts_by_length = defaultdict(list)
        
        for (i, j), counts in path_counts.items():
            for length_idx, count in enumerate(counts):
                if count > 0:
                    all_counts.append(count)
                    counts_by_length[length_idx + 1].append(count)
        
        if len(all_counts) == 0:
            return {
                'total_paths': 0,
                'avg_paths_per_pair': 0,
                'path_count_variance': 0,
                'path_complexity_score': 0
            }
        
        # 计算统计量
        avg_paths = np.mean(all_counts)
        variance = np.var(all_counts)
        std_dev = np.std(all_counts)
        
        # 计算路径复杂性评分
        complexity_score = min(1.0, (variance / (avg_paths + 1)) * 0.1 + len(all_counts) / (self.n_nodes * self.n_nodes))
        
        return {
            'total_paths': len(all_counts),
            'avg_paths_per_pair': avg_paths,
            'path_count_variance': variance,
            'path_count_std': std_dev,
            'paths_by_length': {l: len(counts) for l, counts in counts_by_length.items()},
            'path_complexity_score': complexity_score
        }
    
    def metric_anomaly_structural_features(self):
        """
        指标 4: 异常节点结构特征
        
        比较异常节点和正常节点的局部结构差异
        如果异常节点有显著不同的结构模式，SPSE 能更好捕捉这些差异
        
        Returns:
            dict: 包含异常节点结构特征对比的字典
        """
        if self.ano_labels is None:
            return {
                'error': 'No anomaly labels provided',
                'spse_necessity_from_anomaly': 0.5  # 默认中等必要性
            }
        
        anomaly_nodes = np.where(self.ano_labels == 1)[0]
        normal_nodes = np.where(self.ano_labels == 0)[0]
        
        if len(anomaly_nodes) == 0 or len(normal_nodes) == 0:
            return {
                'error': 'Insufficient anomaly or normal nodes',
                'spse_necessity_from_anomaly': 0.5
            }
        
        # 计算节点的局部结构特征
        def get_node_features(node):
            """获取节点的局部结构特征"""
            neighbors = list(self.G.neighbors(node))
            degree = len(neighbors)
            
            # 局部聚类系数
            clustering = nx.clustering(self.G, node)
            
            # 邻居的平均度
            if degree > 0:
                avg_neighbor_degree = np.mean([self.G.degree(n) for n in neighbors])
            else:
                avg_neighbor_degree = 0
            
            # 三角形数量
            triangles = nx.triangles(self.G, node)
            
            return {
                'degree': degree,
                'clustering': clustering,
                'avg_neighbor_degree': avg_neighbor_degree,
                'triangles': triangles
            }
        
        # 计算异常节点和正常节点的特征
        anomaly_features = [get_node_features(n) for n in anomaly_nodes[:min(100, len(anomaly_nodes))]]
        normal_features = [get_node_features(n) for n in normal_nodes[:min(100, len(normal_nodes))]]
        
        # 计算特征差异
        feature_diffs = {}
        for key in ['degree', 'clustering', 'avg_neighbor_degree', 'triangles']:
            anomaly_vals = [f[key] for f in anomaly_features]
            normal_vals = [f[key] for f in normal_features]
            
            anomaly_mean = np.mean(anomaly_vals)
            normal_mean = np.mean(normal_vals)
            anomaly_std = np.std(anomaly_vals) + 1e-10
            normal_std = np.std(normal_vals) + 1e-10
            
            # 计算效应量 (Cohen's d)
            pooled_std = np.sqrt((anomaly_std**2 + normal_std**2) / 2)
            cohens_d = abs(anomaly_mean - normal_mean) / pooled_std
            
            feature_diffs[key] = {
                'anomaly_mean': anomaly_mean,
                'normal_mean': normal_mean,
                'cohens_d': cohens_d
            }
        
        # 计算总体结构差异评分
        avg_cohens_d = np.mean([d['cohens_d'] for d in feature_diffs.values()])
        structural_difference_score = min(1.0, avg_cohens_d / 2)  # 归一化到 [0,1]
        
        return {
            'n_anomaly_nodes': len(anomaly_nodes),
            'n_normal_nodes': len(normal_nodes),
            'feature_differences': feature_diffs,
            'avg_cohens_d': avg_cohens_d,
            'structural_difference_score': structural_difference_score,
            'spse_necessity_from_anomaly': structural_difference_score  # 结构差异越大，SPSE 越有用
        }
    
    def compute_all_metrics(self):
        """
        计算所有诊断指标
        
        Returns:
            dict: 包含所有诊断指标的字典
        """
        print(f"计算 VoxG 诊断指标 (节点数：{self.n_nodes})...")
        
        metrics = {}
        
        # 1. 环状模式丰富度
        print("  [1/4] 计算环状模式丰富度...")
        metrics['cycle_richness'] = self.metric_cycle_richness()
        
        # 2. RWSE 区分度
        print("  [2/4] 计算 RWSE 区分度...")
        metrics['rwse_discriminability'] = self.metric_rwse_discriminability()
        
        # 3. 路径计数分布
        print("  [3/4] 计算路径计数分布...")
        metrics['path_count_distribution'] = self.metric_path_count_distribution()
        
        # 4. 异常节点结构特征
        print("  [4/4] 计算异常节点结构特征...")
        metrics['anomaly_structural_features'] = self.metric_anomaly_structural_features()
        
        # 综合评估 SPSE 必要性
        spse_necessity = self._compute_spse_necessity(metrics)
        metrics['spse_necessity'] = spse_necessity
        
        return metrics
    
    def _compute_spse_necessity(self, metrics):
        """
        综合评估 SPSE 必要性
        
        基于以下因素：
        1. 环状模式丰富度 (越高越需要 SPSE)
        2. RWSE 区分度 (越低越需要 SPSE)
        3. 路径复杂性 (越高越需要 SPSE)
        4. 异常节点结构差异 (越大越需要 SPSE)
        
        Returns:
            dict: SPSE 必要性评估结果
        """
        # 提取各指标评分
        cycle_score = metrics['cycle_richness']['cycle_richness_score']
        rwse_necessity = metrics['rwse_discriminability']['spse_necessity_from_rwse']
        path_score = metrics['path_count_distribution']['path_complexity_score']
        anomaly_score = metrics['anomaly_structural_features']['spse_necessity_from_anomaly']
        
        # 加权平均 (可根据实际情况调整权重)
        weights = {
            'cycle': 0.35,      # 环状模式最重要
            'rwse': 0.30,       # RWSE 区分度
            'path': 0.20,       # 路径复杂性
            'anomaly': 0.15     # 异常结构差异
        }
        
        overall_score = (
            weights['cycle'] * cycle_score +
            weights['rwse'] * rwse_necessity +
            weights['path'] * path_score +
            weights['anomaly'] * anomaly_score
        )
        
        # 判定等级
        if overall_score >= 0.7:
            level = "高"
            recommendation = "强烈推荐引入 SPSE。数据集富含环状模式，RWSE 区分度低，SPSE 能显著提升性能。"
        elif overall_score >= 0.4:
            level = "中"
            recommendation = "建议考虑 SPSE。数据集有一定复杂性，SPSE 可能带来中等程度的性能提升。"
        else:
            level = "低"
            recommendation = "SPSE 必要性较低。数据集结构相对简单，RWSE 可能已足够。"
        
        return {
            'overall_score': overall_score,
            'level': level,
            'recommendation': recommendation,
            'component_scores': {
                'cycle_richness': cycle_score,
                'rwse_discriminability': 1.0 - rwse_necessity,  # 转换为区分度
                'path_complexity': path_score,
                'anomaly_structure': anomaly_score
            },
            'weights': weights
        }


def load_voxg_dataset(dataset_name, data_dir='./dataset'):
    """
    加载 VoxG 数据集
    
    Args:
        dataset_name: 数据集名称 (Amazon, Reddit, Photo, 等)
        data_dir: 数据目录
    
    Returns:
        adj: 邻接矩阵
        ano_labels: 异常标签
    """
    import os
    
    # 尝试 .mat 格式
    mat_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if os.path.exists(mat_path):
        data = sio.loadmat(mat_path)
        
        # 提取邻接矩阵
        adj = data['Network'] if 'Network' in data else data['A']
        
        # 提取异常标签
        if 'Label' in data:
            ano_labels = np.squeeze(np.array(data['Label']))
        elif 'gnd' in data:
            ano_labels = np.squeeze(np.array(data['gnd']))
        else:
            ano_labels = None
        
        return adj, ano_labels
    
    # 尝试其他格式
    raise FileNotFoundError(f"数据集 {dataset_name} 未找到，请检查路径：{data_dir}")


def main():
    """主函数：在多个 VoxG 数据集上计算诊断指标"""
    
    # 定义要分析的数据集
    datasets = ['Amazon', 'Reddit', 'Photo', 'Cora', 'BlogCatalog']
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"分析数据集：{dataset_name}")
        print('='*60)
        
        try:
            # 加载数据集
            adj, ano_labels = load_voxg_dataset(dataset_name)
            print(f"加载成功：{adj.shape[0]} 节点，{np.sum(ano_labels) if ano_labels is not None else 0} 异常节点")
            
            # 创建诊断器
            diagnostic = VoxGDiagnosticMetrics(adj, ano_labels)
            
            # 计算所有指标
            metrics = diagnostic.compute_all_metrics()
            
            # 保存结果
            results[dataset_name] = metrics
            
            # 打印摘要
            print(f"\n--- {dataset_name} 诊断结果 ---")
            print(f"环状模式丰富度：{metrics['cycle_richness']['cycle_richness_score']:.3f}")
            print(f"RWSE 区分度：{1.0 - metrics['rwse_discriminability']['spse_necessity_from_rwse']:.3f}")
            print(f"路径复杂性：{metrics['path_count_distribution']['path_complexity_score']:.3f}")
            print(f"异常结构差异：{metrics['anomaly_structural_features']['spse_necessity_from_anomaly']:.3f}")
            print(f"\nSPSE 必要性：{metrics['spse_necessity']['level']} ({metrics['spse_necessity']['overall_score']:.3f})")
            print(f"建议：{metrics['spse_necessity']['recommendation']}")
            
        except FileNotFoundError as e:
            print(f"跳过 {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
        except Exception as e:
            print(f"分析 {dataset_name} 失败：{e}")
            results[dataset_name] = {'error': str(e)}
    
    # 输出综合报告
    print(f"\n{'='*60}")
    print("VoxG 数据集 SPSE 必要性综合报告")
    print('='*60)
    
    for dataset_name, metrics in results.items():
        if 'error' not in metrics:
            level = metrics['spse_necessity']['level']
            score = metrics['spse_necessity']['overall_score']
            print(f"{dataset_name:15} | SPSE 必要性：{level:3} | 综合评分：{score:.3f}")
    
    return results


if __name__ == '__main__':
    main()
