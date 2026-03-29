import numpy as np
import scipy.io as sio
import networkx as nx
from collections import defaultdict

print('='*60)
print('VoxG 数据集环密度诊断')
print('='*60)

datasets = ['Amazon', 'reddit', 'photo', 'elliptic', 't_finance', 'tolokers']

for name in datasets:
    print(f'\n加载 {name}...')
    try:
        data = sio.loadmat(f'dataset/{name}.mat')
        adj = data['Network'] if 'Network' in data else data['A']
        labels = data.get('Label', data.get('gnd', None))
        
        # 转换为 networkx (兼容旧版本)
        G = nx.from_scipy_sparse_matrix(adj)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # 计算三角形数量（环的代理指标）
        triangles = sum(nx.triangles(G).values()) // 3
        
        # 计算平均聚类系数
        avg_clustering = nx.average_clustering(G)
        
        # 计算环密度代理指标
        max_possible = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6
        cycle_density = triangles / max_possible if max_possible > 0 else 0
        
        # SPSE 必要性评分
        spse_score = min(1.0, cycle_density * 1000 + avg_clustering)
        
        if spse_score >= 0.7:
            level = '高'
        elif spse_score >= 0.4:
            level = '中'
        else:
            level = '低'
        
        print(f'  节点：{n_nodes:,}, 边：{n_edges:,}')
        print(f'  三角形：{triangles:,}')
        print(f'  平均聚类系数：{avg_clustering:.4f}')
        print(f'  环密度：{cycle_density:.6f}')
        print(f'  ⇒ SPSE 必要性：{level} (评分：{spse_score:.3f})')
        
    except Exception as e:
        print(f'  ❌ 失败：{e}')

print('\n' + '='*60)
print('诊断完成')
print('='*60)
