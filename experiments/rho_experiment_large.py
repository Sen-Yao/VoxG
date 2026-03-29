#!/usr/bin/env python3
"""
RHO 模块实验脚本 - Elliptic 大图版本

使用批量对比损失，支持大图训练
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


def evaluate(X_train, X_test, y_true):
    """无监督评估"""
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


def batch_contrastive_loss(Z1, Z2, temperature=0.2, batch_size=4096):
    """批量对比损失"""
    N = Z1.shape[0]
    Z1 = F.normalize(Z1, dim=1)
    Z2 = F.normalize(Z2, dim=1)
    
    total_loss = 0.0
    num_batches = (N + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)
        
        z1 = Z1[start:end]
        z2 = Z2[start:end]
        
        # 正样本：对角线
        pos = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
        
        # 负样本：z1 @ Z2.T
        neg = torch.exp(z1 @ Z2.T / temperature).sum(dim=1)
        
        loss = -torch.log(pos / neg).mean()
        total_loss += loss.item() * (end - start)
    
    return total_loss / N


class RHOModelLarge(nn.Module):
    """支持大图的 RHO 模型"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Channel-wise: 每个维度独立的 k
        self.k_cw = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim))
            for _ in range(num_layers)
        ])
        
        # Cross-channel: 共享 k
        self.k_cc = nn.Parameter(torch.ones(num_layers))
        
        self.fc_cw = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.fc_cc = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.proj_cw = nn.Linear(hidden_dim, hidden_dim)
        self.proj_cc = nn.Linear(hidden_dim, hidden_dim)
        
        self.num_layers = num_layers
    
    def forward(self, L, X):
        # 编码
        H = self.encoder(X)
        
        # Channel-wise 滤波
        H_cw = H.clone()
        for i in range(self.num_layers):
            LH = torch.matmul(L, H_cw)
            H_cw = H_cw - self.k_cw[i].unsqueeze(0) * LH
            H_cw = F.relu(self.fc_cw[i](H_cw))
        
        # Cross-channel 滤波
        H_cc = H.clone()
        for i in range(self.num_layers):
            LH = torch.matmul(L, H_cc)
            H_cc = H_cc - self.k_cc[i] * LH
            H_cc = F.relu(self.fc_cc[i](H_cc))
        
        # 投影
        Z_cw = self.proj_cw(H_cw)
        Z_cc = self.proj_cc(H_cc)
        
        return H_cw, H_cc, Z_cw, Z_cc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='elliptic')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"RHO 模块实验（大图版本） - {args.dataset}")
    print(f"{'='*60}")
    
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
    
    # 计算拉普拉斯矩阵
    adj_norm = normalize_adj(adj)
    L = sp.eye(adj.shape[0]) - adj_norm
    L = torch.FloatTensor(L.todense()).to(device)
    adj_norm_tensor = torch.FloatTensor(adj_norm.todense()).to(device)
    
    features = torch.FloatTensor(features).to(device)
    labels = np.squeeze(np.array(ano_label))
    
    N, D = features.shape
    print(f"节点数: {N}, 特征维度: {D}")
    print(f"训练正常节点: {len(normal_for_train_idx)}")
    print(f"测试节点: {len(idx_test)}")
    
    y_true = labels[idx_test]
    
    # ========== Baseline ==========
    print(f"\n--- Baseline 方法 ---")
    
    num_hops = 6
    tokens = torch.zeros(N, num_hops + 1, D, device=device)
    tokens[:, 0, :] = features
    H = features.clone()
    for k in range(num_hops):
        H = torch.matmul(adj_norm_tensor, H)
        tokens[:, k + 1, :] = H
    
    baseline_results = {}
    
    # Cross mean
    X = tokens.mean(dim=1).cpu().numpy()
    baseline_results['cross_mean'] = evaluate(X[normal_for_train_idx], X[idx_test], y_true)[0]
    
    # Delta last
    delta_last = tokens[:, num_hops] - tokens[:, num_hops - 1]
    X = delta_last.cpu().numpy()
    baseline_results['delta_last'] = evaluate(X[normal_for_train_idx], X[idx_test], y_true)[0]
    
    # Delta mean
    deltas = tokens[:, 1:] - tokens[:, :-1]
    X = deltas.mean(dim=1).cpu().numpy()
    baseline_results['delta_mean'] = evaluate(X[normal_for_train_idx], X[idx_test], y_true)[0]
    
    for method, auc in baseline_results.items():
        print(f"  {method}: AUC={auc:.4f}")
    
    # ========== RHO 模型 ==========
    print(f"\n--- RHO 模型训练 ---")
    
    model = RHOModelLarge(D, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_results = {'channel_wise': 0, 'cross_channel': 0, 'both': 0}
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        H_cw, H_cc, Z_cw, Z_cc = model(L, features)
        
        # 批量对比损失
        loss = batch_contrastive_loss(
            Z_cw[normal_for_train_idx], 
            Z_cc[normal_for_train_idx],
            batch_size=args.batch_size
        )
        loss = torch.tensor(loss, requires_grad=True, device=device)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                H_cw, H_cc, _, _ = model(L, features)
            
            # 评估
            X = H_cw.cpu().numpy()
            auc_cw, _ = evaluate(X[normal_for_train_idx], X[idx_test], y_true)
            
            X = H_cc.cpu().numpy()
            auc_cc, _ = evaluate(X[normal_for_train_idx], X[idx_test], y_true)
            
            X = torch.cat([H_cw, H_cc], dim=1).cpu().numpy()
            auc_both, _ = evaluate(X[normal_for_train_idx], X[idx_test], y_true)
            
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, CW={auc_cw:.4f}, CC={auc_cc:.4f}, Both={auc_both:.4f}")
            
            if auc_cw > best_results['channel_wise']:
                best_results['channel_wise'] = auc_cw
            if auc_cc > best_results['cross_channel']:
                best_results['cross_channel'] = auc_cc
            if auc_both > best_results['both']:
                best_results['both'] = auc_both
            
            model.train()
    
    # ========== 分析 k 值 ==========
    print(f"\n--- 学习到的 k 值 ---")
    print("\nChannel-wise k:")
    for i, k in enumerate(model.k_cw):
        k = k.detach().cpu().numpy()
        print(f"  Layer {i+1}: mean={k.mean():.4f}, std={k.std():.4f}")
    
    print("\nCross-channel k:")
    for i, k in enumerate(model.k_cc):
        print(f"  Layer {i+1}: k={k.item():.4f}")
    
    # ========== 汇总 ==========
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    
    all_results = {
        **{f"baseline_{k}": v for k, v in baseline_results.items()},
        'RHO Channel-wise': best_results['channel_wise'],
        'RHO Cross-channel': best_results['cross_channel'],
        'RHO Both': best_results['both']
    }
    
    for method, auc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        marker = '✅' if auc == max(all_results.values()) else '  '
        print(f"{marker} {method:<25} {auc:.4f}")
    
    # 数据集特性
    density = adj.nnz / (N * N) if hasattr(adj, 'nnz') else np.count_nonzero(adj.todense()) / (N * N)
    print(f"\n数据集特性:")
    print(f"  特征维度: {D}")
    print(f"  图密度: {density:.6f}")
    print(f"  异常比例: {np.mean(labels == 1):.2%}")
    
    return all_results


if __name__ == "__main__":
    main()