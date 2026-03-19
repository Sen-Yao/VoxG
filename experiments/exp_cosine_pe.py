#!/usr/bin/env python3
"""
实验：添加 Cosine 位置编码到 GGADFormer
基于 TransGAD 论文的方法

创新点：
1. 为每个 hop 添加可学习的位置编码
2. 使用 Cosine 位置编码捕获图的拓扑信息
3. 增强模型对不同 hop 信息的区分能力

运行方式：
python experiments/exp_cosine_pe.py --dataset photo --device 1
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import argparse
import copy
import numpy as np

from utils import load_mat, nagphormer_tokenization
from GGADFormer import GGADFormer
from run import parse_args

class CosinePositionalEncoding(nn.Module):
    """
    Cosine 位置编码，适用于图 Transformer
    
    与 NLP 的 Sinusoidal 不同，我们使用纯 Cosine 函数
    因为图的 hop 具有对称性（距离）
    """
    def __init__(self, d_model, max_len=10):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 只使用 cosine（TransGAD 的做法）
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # 偶数和奇数都用 cosine
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    每个 hop 有独立的可学习 embedding
    """
    def __init__(self, d_model, max_len=10):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class GGADFormer_WithPE(nn.Module):
    """
    带位置编码的 GGADFormer
    """
    def __init__(self, base_model, pe_type='cosine', d_model=256):
        super().__init__()
        self.base_model = base_model
        
        # 添加位置编码
        if pe_type == 'cosine':
            self.pos_encoder = CosinePositionalEncoding(d_model, max_len=10)
        else:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=10)
            
        self.pe_type = pe_type
        
    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):
        """
        在 token 进入 Transformer 之前添加位置编码
        """
        # 注意：input_tokens 是原始特征维度，需要先投影
        # 这里我们在 base_model 的 TransformerEncoder 中处理
        
        # 修改 base_model 的 token_projection 后添加位置编码
        # 由于架构限制，我们需要修改 base_model 内部
        
        return self.base_model(input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse)


def add_pe_to_model(model, args, pe_type='cosine'):
    """
    直接修改模型，在 token_projection 后添加位置编码
    
    这是侵入式修改，直接改变 GGADFormer 的 forward 逻辑
    """
    original_transformer_encoder = model.TransformerEncoder
    
    # 创建位置编码层
    if pe_type == 'cosine':
        pos_encoder = CosinePositionalEncoding(args.embedding_dim, max_len=10)
    else:
        pos_encoder = LearnablePositionalEncoding(args.embedding_dim, max_len=10)
    
    pos_encoder = pos_encoder.to(model.device)
    
    def new_transformer_encoder(tokens):
        # 延迟初始化 token_projection
        if model.token_projection is None:
            input_dim = tokens.shape[-1]
            model.token_projection = nn.Linear(input_dim, args.embedding_dim).to(tokens.device)
        
        # 投影到 embedding 空间
        emb = model.token_projection(tokens)
        
        # 添加位置编码
        emb = pos_encoder(emb)
        
        # 继续原有的 Transformer 层
        for i, l in enumerate(model.layers):
            emb, current_attention_weights = model.layers[i](emb)
            if i == len(model.layers) - 1:
                attention_weights = current_attention_weights
                agg_attention_weights = torch.mean(attention_weights, dim=1)
        
        emb = model.final_ln(emb)
        attention_scores = agg_attention_weights[:, 0, :]
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)
        
        return emb
    
    # 替换 TransformerEncoder 方法
    model.TransformerEncoder = new_transformer_encoder
    
    return model


def run_experiment(dataset='photo', device=1, pe_type='cosine', num_epochs=100):
    """
    运行单个实验
    """
    print(f"\n{'='*60}")
    print(f"实验: {pe_type} 位置编码 | 数据集: {dataset} | GPU: {device}")
    print(f"{'='*60}\n")
    
    # 解析参数
    args = parse_args()
    args.dataset = dataset
    args.device = device
    args.num_epoch = num_epochs
    args.use_pe = True
    args.pe_type = pe_type
    
    # 加载数据
    print("加载数据...")
    adj, features, ano_labels, str_ano_labels, attr_ano_labels, idx_train, idx_val, idx_test, normal_for_train_idx = load_mat(
        args.dataset, args.train_rate, args.val_rate, args
    )
    
    # Tokenization
    print("Tokenization...")
    input_features = nagphormer_tokenization(features, adj, args)
    
    # 创建模型
    print("创建模型...")
    n_in = input_features.shape[2]
    n_h = args.embedding_dim
    
    model = GGADFormer(n_in, n_h, 'prelu', args)
    
    # 添加位置编码
    if args.use_pe:
        print(f"添加 {pe_type} 位置编码...")
        model = add_pe_to_model(model, args, pe_type=pe_type)
    
    model = model.to(f'cuda:{device}')
    
    # 训练（简化版，实际训练逻辑在 run.py 中）
    print("模型已创建，请使用 run.py 进行完整训练")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--pe_type', type=str, default='cosine', choices=['cosine', 'learnable'])
    parser.add_argument('--num_epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    run_experiment(
        dataset=args.dataset,
        device=args.device,
        pe_type=args.pe_type,
        num_epochs=args.num_epochs
    )