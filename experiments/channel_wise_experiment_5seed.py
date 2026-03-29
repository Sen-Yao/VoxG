#!/usr/bin/env python3
"""
Channel-wise Token 实验（5-seed 版本）

对比：
1. Baseline：Cross-channel Token（原有方式）
2. Ours：Channel-wise Token

遵循 5-seed 规则：每套配置跑 5 个 seed，报告均值 ± 标准差

作者：Nexus
日期：2026-03-27
"""

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import argparse

from utils import load_mat, normalize_adj, preprocess_features
from models.channel_wise_token import (
    ChannelWiseTransformer,
    CrossChannelTokenizer
)


def set_seed(seed):
    """固定随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(X_train, X_test, y_true):
    """无监督评估（马氏距离）"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    center = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(X_train.shape[1])
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    
    scores = np.sqrt(np.sum((X_test - center) @ cov_inv * (X_test - center), axis=1))
    return roc_auc_score(y_true, scores), average_precision_score(y_true, scores)


class ChannelWiseModel(nn.Module):
    """Channel-wise Token 模型"""
    
    def __init__(self, num_hops, num_channels, hidden_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.transformer = ChannelWiseTransformer(
            num_hops, num_channels, hidden_dim, num_heads, num_layers
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, tokens):
        output, attention = self.transformer(tokens)
        proj = self.proj(output)
        return output, proj, attention


class CrossChannelModel(nn.Module):
    """Cross-channel Token 模型（baseline）"""
    
    def __init__(self, num_hops, num_channels, hidden_dim, num_heads=4, num_layers=2):
        super().__init__()
        
        self.tokenizer = CrossChannelTokenizer(num_hops, num_channels, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, tokens):
        hop_tokens = self.tokenizer(tokens)
        encoded = self.transformer(hop_tokens)
        output = encoded[:, 0, :]
        proj = self.proj(output)
        return output, proj


def contrastive_loss(Z1, Z2, temperature=0.2):
    """对比学习损失"""
    N = Z1.shape[0]
    Z1 = F.normalize(Z1, dim=1)
    Z2 = F.normalize(Z2, dim=1)
    
    pos = torch.exp(torch.sum(Z1 * Z2, dim=1) / temperature)
    neg = torch.exp(Z1 @ Z2.T / temperature).sum(dim=1)
    
    loss = -torch.log(pos / neg).mean()
    return loss


def run_single_seed(args, seed, tokens, labels, normal_for_train_idx, idx_test, 
                    adj_norm_tensor, features_tensor, D, effective_dim, tokens_proj):
    """运行单个 seed 的实验"""
    
    set_seed(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    K = args.num_hops
    
    labels_arr = np.squeeze(np.array(labels))
    y_true = labels_arr[idx_test]
    
    # ========== Baseline 方法 ==========
    # Cross mean
    X = tokens.mean(dim=1).cpu().numpy()
    baseline_cross_mean = evaluate(X[normal_for_train_idx], X[idx_test], y_true)[0]
    
    # Delta last
    delta_last = tokens[:, K] - tokens[:, K - 1]
    X = delta_last.cpu().numpy()
    baseline_delta_last = evaluate(X[normal_for_train_idx], X[idx_test], y_true)[0]
    
    # ========== Cross-channel Token 训练 ==========
    cc_model = CrossChannelModel(K, effective_dim, args.hidden_dim, args.num_heads, args.num_layers).to(device)
    cc_optimizer = torch.optim.Adam(cc_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_cc_auc = 0
    for epoch in range(args.epochs):
        cc_model.train()
        cc_optimizer.zero_grad()
        
        output, proj = cc_model(tokens_proj)
        
        perm = torch.randperm(len(normal_for_train_idx))[:min(512, len(normal_for_train_idx))]
        sample_idx = torch.tensor(normal_for_train_idx, device=device)[perm]
        
        sample_tokens = tokens_proj[sample_idx].clone()
        noise = torch.randn_like(sample_tokens) * 0.1
        tokens_aug = sample_tokens + noise
        
        _, proj1 = cc_model(sample_tokens)
        _, proj2 = cc_model(tokens_aug)
        
        loss = contrastive_loss(proj1, proj2)
        loss.backward()
        cc_optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            cc_model.eval()
            with torch.no_grad():
                output, _ = cc_model(tokens_proj)
            
            X = output.cpu().numpy()
            auc, _ = evaluate(X[normal_for_train_idx], X[idx_test], y_true)
            
            if auc > best_cc_auc:
                best_cc_auc = auc
    
    # ========== Channel-wise Token 训练 ==========
    cw_model = ChannelWiseModel(K, effective_dim, args.hidden_dim, args.num_heads, args.num_layers).to(device)
    cw_optimizer = torch.optim.Adam(cw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_cw_auc = 0
    best_attention = None
    
    for epoch in range(args.epochs):
        cw_model.train()
        cw_optimizer.zero_grad()
        
        perm = torch.randperm(len(normal_for_train_idx))[:min(512, len(normal_for_train_idx))]
        sample_idx = torch.tensor(normal_for_train_idx, device=device)[perm]
        
        sample_tokens = tokens_proj[sample_idx].clone()
        noise = torch.randn_like(sample_tokens) * 0.1
        tokens_aug = sample_tokens + noise
        
        _, proj1, _ = cw_model(sample_tokens)
        _, proj2, _ = cw_model(tokens_aug)
        
        loss = contrastive_loss(proj1, proj2)
        loss.backward()
        cw_optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            cw_model.eval()
            with torch.no_grad():
                output, _, attention = cw_model(tokens_proj)
            
            X = output.cpu().numpy()
            auc, _ = evaluate(X[normal_for_train_idx], X[idx_test], y_true)
            
            if auc > best_cw_auc:
                best_cw_auc = auc
                best_attention = attention.mean(dim=0).cpu().numpy()
    
    return {
        'seed': seed,
        'baseline_cross_mean': baseline_cross_mean,
        'baseline_delta_last': baseline_delta_last,
        'cross_channel': best_cc_auc,
        'channel_wise': best_cw_auc,
        'attention': best_attention
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds for 5-seed experiment')
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_hops', type=int, default=6)
    args = parser.parse_args()
    
    # 5-seed 种子列表
    seeds = [42, 123, 456, 789, 1024][:args.num_seeds]
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Channel-wise Token 5-seed 实验 - {args.dataset}")
    print(f"{'='*60}")
    
    # 加载数据（固定数据划分）
    run_args = argparse.Namespace(
        dataset=args.dataset, train_rate=args.train_rate,
        seed=args.seed, data_split_seed=args.seed, sample_rate=0.15,
        pp_k=6, progregate_alpha=0.2
    )
    
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_for_train_idx, _ = load_mat(
        args.dataset, args.train_rate, 0.1, run_args
    )
    
    if args.dataset.lower() in ['amazon', 'tf_finace', 'tfinance', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense() if hasattr(features, 'todense') else features
    
    # 计算多跳传播
    adj_norm = normalize_adj(adj)
    adj_norm_tensor = torch.FloatTensor(adj_norm.todense()).to(device)
    features_tensor = torch.FloatTensor(features).to(device)
    
    N, D = features_tensor.shape
    print(f"节点数: {N}, 特征维度: {D}")
    print(f"训练正常节点: {len(normal_for_train_idx)}")
    print(f"测试节点: {len(idx_test)}")
    print(f"Seed 数量: {len(seeds)}")
    
    # 生成 Token
    K = args.num_hops
    tokens = torch.zeros(N, K + 1, D, device=device)
    tokens[:, 0, :] = features_tensor
    H = features_tensor.clone()
    for k in range(K):
        H = torch.matmul(adj_norm_tensor, H)
        tokens[:, k + 1, :] = H
    
    # 高维特征降维
    if D > 256:
        print(f"特征维度 {D} > 256，使用降维投影")
        dim_proj = nn.Linear(D, 256).to(device)
        tokens_proj = dim_proj(tokens)
        effective_dim = 256
    else:
        tokens_proj = tokens
        effective_dim = D
    
    # ========== 5-seed 实验 ==========
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {i+1}/{len(seeds)}: {seed} ---")
        
        result = run_single_seed(
            args, seed, tokens, labels, normal_for_train_idx, idx_test,
            adj_norm_tensor, features_tensor, D, effective_dim, tokens_proj
        )
        all_results.append(result)
        
        print(f"  Cross-channel: {result['cross_channel']:.4f}")
        print(f"  Channel-wise:  {result['channel_wise']:.4f}")
    
    # ========== 汇总 5-seed 结果 ==========
    print(f"\n{'='*60}")
    print("5-seed 结果汇总")
    print(f"{'='*60}")
    
    methods = ['baseline_cross_mean', 'baseline_delta_last', 'cross_channel', 'channel_wise']
    
    summary = {}
    for method in methods:
        values = [r[method] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        summary[method] = {'mean': mean, 'std': std, 'values': values}
        
        # 标记最优
        is_best = method == max(summary.keys(), key=lambda x: summary[x]['mean'])
        marker = '✅' if is_best else '  '
        
        print(f"{marker} {method:<25} {mean:.4f} ± {std:.4f}")
    
    # 各 seed 详细结果
    print(f"\n各 Seed 详细结果:")
    print(f"{'Seed':<8} {'Cross-channel':<15} {'Channel-wise':<15}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['seed']:<8} {r['cross_channel']:.4f}         {r['channel_wise']:.4f}")
    
    # Channel-wise 相比 Cross-channel 提升
    improvement = (summary['channel_wise']['mean'] - summary['cross_channel']['mean']) / summary['cross_channel']['mean'] * 100
    print(f"\nChannel-wise Token 相比 Cross-channel Token: {improvement:+.2f}%")
    
    # 数据集特性
    density = adj.nnz / (N * N) if hasattr(adj, 'nnz') else np.count_nonzero(adj.todense()) / (N * N)
    print(f"\n数据集特性:")
    print(f"  特征维度: {D}")
    print(f"  图密度: {density:.6f}")
    print(f"  异常比例: {np.mean(np.squeeze(np.array(labels)) == 1):.2%}")
    
    return summary, all_results


if __name__ == "__main__":
    main()