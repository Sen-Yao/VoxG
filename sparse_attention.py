"""
Exphormer-style Sparse Attention Implementation for VoxG
Implements Local, Global, and Expander attention mechanisms

Note: In VoxGFormer, attention is computed on the token sequence dimension (pp_k+1),
not the graph node dimension. The adjacency matrix can be used to create attention patterns
within each node's token sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class LocalAttention(nn.Module):
    """
    Local attention based on graph adjacency - only computes attention within k-hop neighbors
    Uses sparse matrix optimization for efficiency
    
    For VoxG: Applies local window attention over token sequence
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, k_hop=1):
        super(LocalAttention, self).__init__()
        
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.k_hop = k_hop
        
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)
        
    def forward(self, q, k, v, adj=None, attn_bias=None):
        """
        Args:
            q, k, v: query, key, value tensors [B, N, D] where N = pp_k+1 (token sequence length)
            adj: adjacency matrix (not used for local token attention, but kept for API)
            attn_bias: optional attention bias
        """
        orig_q_size = q.size()
        batch_size = q.size(0)
        seq_len = q.size(1)
        d_k = self.att_size
        d_v = self.att_size
        
        # Linear projections
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        
        q = q.transpose(1, 2)  # [B, H, N, d_k]
        k = k.transpose(1, 2).transpose(2, 3)  # [B, H, d_k, N]
        v = v.transpose(1, 2)  # [B, H, N, d_v]
        
        # Compute attention scores
        q = q * self.scale
        attn_scores = torch.matmul(q, k)  # [B, H, N, N]
        
        # Create local attention mask (window around each position)
        # For token sequence, local means positions within k_hop distance
        local_mask = torch.zeros(1, 1, seq_len, seq_len, device=q.device)
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) <= self.k_hop:
                    local_mask[0, 0, i, j] = 1.0
        
        # Mask out non-local positions
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(local_mask == 0, neg_inf)
        
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        # Softmax and dropout
        attention_weights = torch.softmax(attn_scores, dim=-1)
        attention_weights = self.att_dropout(attention_weights)
        
        # Apply attention to values
        x = torch.matmul(attention_weights, v)  # [B, H, N, d_v]
        
        # Reshape output
        x = x.transpose(1, 2).contiguous()  # [B, N, H, d_v]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x, attention_weights


class GlobalAttention(nn.Module):
    """
    Selective global attention for important nodes/tokens
    Important positions participate in global attention
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, 
                 num_global_nodes=16, selection_strategy='first'):
        super(GlobalAttention, self).__init__()
        
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.num_global_nodes = num_global_nodes
        self.selection_strategy = selection_strategy
        
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)
    
    def forward(self, q, k, v, adj=None, attn_bias=None):
        """
        Args:
            q, k, v: query, key, value tensors [B, N, D]
            adj: adjacency matrix (not used, kept for API)
            attn_bias: optional attention bias
        """
        orig_q_size = q.size()
        batch_size = q.size(0)
        seq_len = q.size(1)
        d_k = self.att_size
        d_v = self.att_size
        
        # Linear projections
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        
        q = q.transpose(1, 2)  # [B, H, N, d_k]
        k = k.transpose(1, 2).transpose(2, 3)  # [B, H, d_k, N]
        v = v.transpose(1, 2)  # [B, H, N, d_v]
        
        # Compute attention scores
        q = q * self.scale
        attn_scores = torch.matmul(q, k)  # [B, H, N, N]
        
        # Create global attention mask
        # Select important positions (first position is the anchor in VoxG)
        global_mask = torch.zeros(1, 1, seq_len, seq_len, device=q.device)
        
        if self.selection_strategy == 'first':
            # First position (self token) can attend globally, all can attend to first
            global_mask[0, 0, 0, :] = 1.0  # First position attends to all
            global_mask[0, 0, :, 0] = 1.0  # All attend to first
            global_mask[0, 0, 1:, 1:] = 1.0  # Local for others (can adjust)
        else:
            # All positions can attend to first num_global_nodes
            k_global = min(self.num_global_nodes, seq_len)
            global_mask[0, 0, :k_global, :] = 1.0
            global_mask[0, 0, :, :k_global] = 1.0
        
        # Apply mask
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(global_mask == 0, neg_inf)
        
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        # Softmax and dropout
        attention_weights = torch.softmax(attn_scores, dim=-1)
        attention_weights = self.att_dropout(attention_weights)
        
        # Apply attention to values
        x = torch.matmul(attention_weights, v)  # [B, H, N, d_v]
        
        # Reshape output
        x = x.transpose(1, 2).contiguous()  # [B, N, H, d_v]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x, attention_weights


class ExpanderAttention(nn.Module):
    """
    Expander graph attention - random connections for long-range dependencies
    Maintains spectral gap properties for good mixing
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, 
                 expansion_degree=3, seed=None):
        super(ExpanderAttention, self).__init__()
        
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.expansion_degree = expansion_degree
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)
        
        # Cache for expander mask
        self.cached_expander_mask = None
        self.cached_seq_len = None
        
    def generate_expander_mask(self, seq_len, device):
        """
        Generate expander graph mask - random connections for spectral expansion
        Args:
            seq_len: sequence length
            device: torch device
        Returns:
            expander_mask: [1, 1, N, N] binary mask
        """
        # Check cache
        if self.cached_seq_len == seq_len and self.cached_expander_mask is not None:
            return self.cached_expander_mask.to(device)
        
        # Create random expander graph
        expander_mask = torch.zeros(1, 1, seq_len, seq_len, device=device)
        
        # Add self-loops
        expander_mask[0, 0].fill_diagonal_(1)
        
        # Add random connections
        degree = min(self.expansion_degree, seq_len - 1)
        for i in range(seq_len):
            # Randomly select expansion_degree positions
            if seq_len > 1:
                indices = torch.randperm(seq_len - 1, device=device)[:degree]
                # Adjust indices to avoid self-connection
                indices = indices + (indices >= i).long()
                expander_mask[0, 0, i, indices] = 1
                expander_mask[0, 0, indices, i] = 1  # Symmetric
        
        # Cache the mask
        self.cached_seq_len = seq_len
        self.cached_expander_mask = expander_mask
        
        return expander_mask
    
    def forward(self, q, k, v, adj=None, attn_bias=None):
        """
        Args:
            q, k, v: query, key, value tensors [B, N, D]
            adj: adjacency matrix (not used, kept for API)
            attn_bias: optional attention bias
        """
        orig_q_size = q.size()
        batch_size = q.size(0)
        seq_len = q.size(1)
        d_k = self.att_size
        d_v = self.att_size
        
        # Linear projections
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        
        q = q.transpose(1, 2)  # [B, H, N, d_k]
        k = k.transpose(1, 2).transpose(2, 3)  # [B, H, d_k, N]
        v = v.transpose(1, 2)  # [B, H, N, d_v]
        
        # Compute attention scores
        q = q * self.scale
        attn_scores = torch.matmul(q, k)  # [B, H, N, N]
        
        # Apply expander mask
        expander_mask = self.generate_expander_mask(seq_len, q.device)
        
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(expander_mask == 0, neg_inf)
        
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        # Softmax and dropout
        attention_weights = torch.softmax(attn_scores, dim=-1)
        attention_weights = self.att_dropout(attention_weights)
        
        # Apply attention to values
        x = torch.matmul(attention_weights, v)  # [B, H, N, d_v]
        
        # Reshape output
        x = x.transpose(1, 2).contiguous()  # [B, N, H, d_v]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x, attention_weights


class SparseMultiHeadAttention(nn.Module):
    """
    Combined sparse attention combining Local, Global, and Expander attention
    Based on Exphormer architecture
    
    For VoxG: Applies sparse attention patterns over the token sequence (pp_k+1)
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads,
                 use_local=True, use_global=True, use_expander=True,
                 k_hop=1, num_global_nodes=16, expansion_degree=3,
                 local_weight=0.4, global_weight=0.3, expander_weight=0.3):
        super(SparseMultiHeadAttention, self).__init__()
        
        self.use_local = use_local
        self.use_global = use_global
        self.use_expander = use_expander
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        
        # Initialize attention modules
        if use_local:
            self.local_attention = LocalAttention(
                hidden_size, attention_dropout_rate, num_heads, k_hop=k_hop
            )
        
        if use_global:
            self.global_attention = GlobalAttention(
                hidden_size, attention_dropout_rate, num_heads, 
                num_global_nodes=num_global_nodes
            )
        
        if use_expander:
            self.expander_attention = ExpanderAttention(
                hidden_size, attention_dropout_rate, num_heads,
                expansion_degree=expansion_degree
            )
        
        # Learnable combination weights
        self.num_components = sum([use_local, use_global, use_expander])
        if self.num_components > 1:
            # Learnable combination weights
            self.combine_weights = nn.Parameter(
                torch.ones(self.num_components) / self.num_components
            )
        
        # Output projection
        self.output_dropout = nn.Dropout(attention_dropout_rate)
        
    def forward(self, q, k, v, adj=None, attn_bias=None):
        """
        Combined sparse attention forward
        Args:
            q, k, v: query, key, value [B, N, D] where N = pp_k+1 (token sequence)
            adj: adjacency matrix (kept for API compatibility)
            attn_bias: optional attention bias
        Returns:
            output: [B, N, D]
            attention_weights: aggregated attention weights
        """
        outputs = []
        attention_weights_dict = {}
        
        # Local attention
        if self.use_local:
            local_out, local_attn = self.local_attention(q, k, v, adj, attn_bias)
            outputs.append(local_out)
            attention_weights_dict['local'] = local_attn
        
        # Global attention
        if self.use_global:
            global_out, global_attn = self.global_attention(q, k, v, adj, attn_bias)
            outputs.append(global_out)
            attention_weights_dict['global'] = global_attn
        
        # Expander attention
        if self.use_expander:
            expander_out, expander_attn = self.expander_attention(q, k, v, adj, attn_bias)
            outputs.append(expander_out)
            attention_weights_dict['expander'] = expander_attn
        
        # Combine outputs
        if self.num_components == 1:
            output = outputs[0]
            attention_weights = list(attention_weights_dict.values())[0]
        else:
            # Weighted combination
            if self.training:
                # Softmax over weights for learnable combination
                weights = F.softmax(self.combine_weights, dim=0)
                output = sum(w * o for w, o in zip(weights, outputs))
            else:
                # Simple average during inference
                output = sum(outputs) / len(outputs)
            
            # Average attention weights for visualization
            attention_weights = torch.stack(list(attention_weights_dict.values())).mean(dim=0)
        
        output = self.output_dropout(output)
        
        return output, attention_weights
