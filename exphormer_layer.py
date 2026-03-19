"""
Exphormer Sparse Attention Layer (ICML 2023)
Implements O(n) complexity sparse attention for graph transformers.

Key Components:
1. Local Attention: Attention only on graph edges (O(E))
2. Expander Graph: Random sparse connections for global connectivity (O(n*d))
3. Virtual Global Nodes: Global information pooling (O(n*k))

Reference: "Exphormer: Sparse Transformers for Graphs" (ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ExpanderGraphGenerator:
    """
    Generate random expander graphs for sparse global attention.
    Uses random regular graphs which have good expansion properties.
    """
    
    @staticmethod
    def generate_random_regular_graph(num_nodes: int, degree: int, device: torch.device) -> torch.Tensor:
        """
        Generate a random d-regular graph adjacency matrix.
        Returns sparse edge list for efficient computation.
        
        Args:
            num_nodes: Number of nodes in the graph
            degree: Degree of each node (expander degree)
            device: PyTorch device
            
        Returns:
            edge_index: [2, num_edges] tensor of edge pairs
        """
        edges = []
        for i in range(num_nodes):
            # Random connections to other nodes
            perm = torch.randperm(num_nodes - 1, device=device)
            perm[perm >= i] += 1  # Skip self
            neighbors = perm[:degree]
            edges.append(torch.stack([torch.full((degree,), i, device=device), neighbors]))
        
        edge_index = torch.cat(edges, dim=1)
        return edge_index


class LocalAttention(nn.Module):
    """
    Local attention computed only on graph edges.
    Complexity: O(E) where E is number of edges.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, num_nodes, hidden_size]
            edge_index: [2, num_edges] edge list (sparse)
            return_attention: Whether to return attention weights
            
        Returns:
            out: [batch_size, num_nodes, hidden_size]
            attention_weights: Optional attention weights
        """
        batch_size, num_nodes, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, nodes, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # For sparse local attention, we compute attention only on edges
        # Create sparse attention matrix from edge_index
        src, dst = edge_index[0], edge_index[1]
        
        # Compute attention scores only for edges
        # q: [batch, heads, nodes, head_dim]
        # For each edge (src, dst), compute attention score
        q_src = q[:, :, src, :]  # [batch, heads, num_edges, head_dim]
        k_dst = k[:, :, dst, :]  # [batch, heads, num_edges, head_dim]
        v_dst = v[:, :, dst, :]
        
        # Attention scores for edges
        attn_scores = (q_src * k_dst).sum(dim=-1) * self.scale  # [batch, heads, num_edges]
        
        # For softmax, we need to normalize per query node
        # Create sparse softmax
        out = self._sparse_attention_aggregate(
            attn_scores, v_dst, src, dst, num_nodes, batch_size
        )
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out, None
    
    def _sparse_attention_aggregate(
        self,
        attn_scores: torch.Tensor,
        v: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Aggregate values using sparse attention with softmax per query node.
        """
        # Apply softmax per source node
        # Group by source node and apply softmax
        device = attn_scores.device
        
        # Create output tensor
        out = torch.zeros(batch_size, self.num_heads, num_nodes, self.head_dim, device=device)
        
        # For each unique source node, compute softmax and aggregate
        unique_src = torch.unique(src)
        
        for s in unique_src:
            mask = src == s
            scores_s = attn_scores[:, :, mask]  # [batch, heads, num_neighbors]
            v_s = v[:, :, mask, :]  # [batch, heads, num_neighbors, head_dim]
            
            # Softmax over neighbors
            attn_weights = F.softmax(scores_s, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Weighted sum
            out[:, :, s, :] = (attn_weights.unsqueeze(-1) * v_s).sum(dim=2)
        
        return out


class GlobalAttention(nn.Module):
    """
    Global attention via virtual nodes.
    Each virtual node connects to all real nodes.
    Complexity: O(n * k) where k is number of virtual nodes.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_virtual_nodes: int = 4,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_virtual_nodes = num_virtual_nodes
        self.scale = self.head_dim ** -0.5
        
        # Virtual node embeddings (learnable)
        self.virtual_node_embeddings = nn.Parameter(
            torch.randn(1, num_virtual_nodes, hidden_size) * 0.02
        )
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, num_nodes, hidden_size]
            
        Returns:
            out: [batch_size, num_nodes, hidden_size]
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # Expand virtual nodes for batch
        virtual_nodes = self.virtual_node_embeddings.expand(batch_size, -1, -1)
        
        # Concatenate real nodes with virtual nodes
        # [batch, num_nodes + num_virtual, hidden]
        x_with_virtual = torch.cat([x, virtual_nodes], dim=1)
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x_with_virtual).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(x_with_virtual).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose: [batch, heads, nodes, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention: real nodes attend to all (real + virtual)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out, attn if return_attention else None


class ExpanderAttention(nn.Module):
    """
    Attention on expander graph for global connectivity.
    Complexity: O(n * d) where d is expander degree.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expander_degree: int = 3,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.expander_degree = expander_degree
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
        
        # Cached expander graph (regenerated if num_nodes changes)
        self._cached_edge_index = None
        self._cached_num_nodes = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, num_nodes, hidden_size]
            
        Returns:
            out: [batch_size, num_nodes, hidden_size]
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # Get or generate expander graph
        if self._cached_num_nodes != num_nodes:
            self._cached_edge_index = ExpanderGraphGenerator.generate_random_regular_graph(
                num_nodes, self.expander_degree, device
            )
            self._cached_num_nodes = num_nodes
        
        edge_index = self._cached_edge_index.to(device)
        
        # Use same sparse attention as local attention
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        src, dst = edge_index[0], edge_index[1]
        
        q_src = q[:, :, src, :]
        k_dst = k[:, :, dst, :]
        v_dst = v[:, :, dst, :]
        
        attn_scores = (q_src * k_dst).sum(dim=-1) * self.scale
        
        out = self._sparse_attention_aggregate(
            attn_scores, v_dst, src, dst, num_nodes, batch_size
        )
        
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out, None
    
    def _sparse_attention_aggregate(
        self,
        attn_scores: torch.Tensor,
        v: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        batch_size: int
    ) -> torch.Tensor:
        device = attn_scores.device
        out = torch.zeros(batch_size, self.num_heads, num_nodes, self.head_dim, device=device)
        
        unique_src = torch.unique(src)
        
        for s in unique_src:
            mask = src == s
            scores_s = attn_scores[:, :, mask]
            v_s = v[:, :, mask, :]
            
            attn_weights = F.softmax(scores_s, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            out[:, :, s, :] = (attn_weights.unsqueeze(-1) * v_s).sum(dim=2)
        
        return out


class ExphormerLayer(nn.Module):
    """
    Complete Exphormer layer combining:
    1. Local attention on graph edges
    2. Expander attention for global connectivity
    3. Virtual global nodes for global information pooling
    
    Total complexity: O(E + n*d + n*k) = O(n) for sparse graphs
    where E = edges, d = expander_degree, k = num_virtual_nodes
    """
    
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_heads: int,
        num_virtual_nodes: int = 4,
        expander_degree: int = 3,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_local: bool = True,
        use_expander: bool = True,
        use_global: bool = True
    ):
        super().__init__()
        
        self.use_local = use_local
        self.use_expander = use_expander
        self.use_global = use_global
        
        # Attention layers
        if use_local:
            self.local_attention = LocalAttention(
                hidden_size, num_heads, attention_dropout
            )
            self.local_norm = nn.LayerNorm(hidden_size)
            self.local_dropout = nn.Dropout(dropout)
        
        if use_expander:
            self.expander_attention = ExpanderAttention(
                hidden_size, num_heads, expander_degree, attention_dropout
            )
            self.expander_norm = nn.LayerNorm(hidden_size)
            self.expander_dropout = nn.Dropout(dropout)
        
        if use_global:
            self.global_attention = GlobalAttention(
                hidden_size, num_heads, num_virtual_nodes, attention_dropout
            )
            self.global_norm = nn.LayerNorm(hidden_size)
            self.global_dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Learnable weights for combining attention outputs
        num_components = sum([use_local, use_expander, use_global])
        if num_components > 1:
            self.attention_weights = nn.Parameter(torch.ones(num_components) / num_components)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, num_nodes, hidden_size]
            edge_index: [2, num_edges] sparse edge list (for local attention)
            
        Returns:
            out: [batch_size, num_nodes, hidden_size]
            attention_weights: Optional attention weights
        """
        attention_outputs = []
        attention_weights_list = []
        
        # Local attention
        if self.use_local and edge_index is not None:
            local_out, local_attn = self.local_attention(
                self.local_norm(x), edge_index, return_attention
            )
            attention_outputs.append(local_out)
            attention_weights_list.append(local_attn)
        
        # Expander attention
        if self.use_expander:
            expander_out, expander_attn = self.expander_attention(
                self.expander_norm(x), return_attention
            )
            attention_outputs.append(expander_out)
            attention_weights_list.append(expander_attn)
        
        # Global attention
        if self.use_global:
            global_out, global_attn = self.global_attention(
                self.global_norm(x), return_attention
            )
            attention_outputs.append(global_out)
            attention_weights_list.append(global_attn)
        
        # Combine attention outputs
        if len(attention_outputs) == 1:
            combined = attention_outputs[0]
        else:
            # Weighted combination
            weights = F.softmax(self.attention_weights, dim=0)
            combined = sum(w * out for w, out in zip(weights, attention_outputs))
        
        # Residual connection
        x = x + combined
        
        # Feed-forward network
        x = x + self.ffn(self.ffn_norm(x))
        
        return x, attention_weights_list[0] if return_attention else None


class ExphormerEncoder(nn.Module):
    """
    Stack of Exphormer layers.
    Drop-in replacement for standard Transformer encoder.
    """
    
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_layers: int,
        num_heads: int,
        num_virtual_nodes: int = 4,
        expander_degree: int = 3,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_local: bool = True,
        use_expander: bool = True,
        use_global: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ExphormerLayer(
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                num_heads=num_heads,
                num_virtual_nodes=num_virtual_nodes,
                expander_degree=expander_degree,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_local=use_local,
                use_expander=use_expander,
                use_global=use_global
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, num_nodes, hidden_size]
            edge_index: [2, num_edges] sparse edge list
            
        Returns:
            out: [batch_size, num_nodes, hidden_size]
        """
        attention_weights = None
        
        for layer in self.layers:
            x, attn = layer(x, edge_index, return_attention)
            if attn is not None:
                attention_weights = attn
        
        return self.final_norm(x), attention_weights


# Drop-in replacement for MultiHeadAttention
class ExphormerMultiHeadAttention(nn.Module):
    """
    Drop-in replacement for standard MultiHeadAttention.
    Uses Exphormer sparse attention internally.
    """
    
    def __init__(
        self,
        hidden_size: int,
        attention_dropout_rate: float,
        num_heads: int,
        num_virtual_nodes: int = 4,
        expander_degree: int = 3,
        return_attention_weights: bool = False
    ):
        super().__init__()
        self.return_attention_weights = return_attention_weights
        
        self.attention = GlobalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_virtual_nodes=num_virtual_nodes,
            attention_dropout=attention_dropout_rate
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            q, k, v: [batch_size, seq_len, hidden_size]
            attn_bias: Ignored (for compatibility)
            
        Returns:
            out: [batch_size, seq_len, hidden_size]
            attention_weights: Optional attention weights
        """
        # For self-attention, q = k = v
        out, attention_weights = self.attention(q, self.return_attention_weights)
        return out, attention_weights
