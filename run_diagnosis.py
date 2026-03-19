"""
运行诊断实验：分析特征重构 vs 结构重构
"""
import torch
import scipy.io as scio
import argparse
import numpy as np
from VoxGFormer import VoxGFormer
from utils import load_mat_VoxG
from diagnostics import diagnose_structure_reconstruction, analyze_dataset_characteristics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--pp_k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"诊断实验: {args.dataset}")
    print(f"{'='*60}")
    
    # 加载数据
    data_path = f'/root/gpufree-data/linziyao/VoxG/data/{args.dataset}.mat'
    data = scio.loadmat(data_path)
    
    features = torch.FloatTensor(data['X'])
    adj = torch.FloatTensor(data['A'])
    labels = data['Y'].flatten()
    
    print(f"\n数据加载完成: {features.shape[0]} 节点, {features.shape[1]} 维特征")
    
    # 分析数据集特性
    analyze_dataset_characteristics(features, adj, labels)
    
    # 创建模型（随机初始化）
    torch.manual_seed(args.seed)
    model = VoxGFormer(features.shape[1], args.embedding_dim, 'prelu', args)
    
    print(f"\n运行诊断...")
    results = diagnose_structure_reconstruction(model, features, adj, labels, args)
    
    print(f"\n诊断完成！")
    print(f"\n建议:")
    if results['correlation'] < 0.3:
        print(f"  特征和结构重构误差低相关 → 结构重构有潜力提升性能")
    else:
        print(f"  特征和结构重构误差高相关 → 可能需要更精细的设计")
    
    if results['struct_auc'] > results['feat_auc']:
        print(f"  结构重构优于特征重构 → 结构异常主导")
    else:
        print(f"  特征重构优于结构重构 → 特征异常主导")
    
    improvement = results['best_auc'] - results['feat_auc']
    print(f"\n潜在提升: {improvement*100:.2f}% (组合 vs 仅特征)")

if __name__ == '__main__':
    main()
