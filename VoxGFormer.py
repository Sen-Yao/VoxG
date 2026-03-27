import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math

from losses.contrastive_loss import GraphContrastiveLoss
from check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory
from models.prompt_tuning import PromptTuning, PromptTuningConfig, freeze_non_prompt_parameters, get_prompt_parameters
from exphormer_layer import ExphormerLayer, ExphormerEncoder


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for linear layers.
    Wraps an existing linear layer and adds low-rank adaptation.
    
    h = W_0 x + BA x
    where B is (out_features, r) and A is (r, in_features)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        original_weight: torch.Tensor = None,
        original_bias: torch.Tensor = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Main linear layer (frozen when using LoRA)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if original_weight is not None:
            self.weight.data.copy_(original_weight)
        if original_bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.data.copy_(original_bias)
        else:
            self.register_parameter('bias', None)
        
        # LoRA low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        # A is initialized with Kaiming uniform
        # B is initialized to zero so initial output equals original
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Track whether weights are merged
        self.merged = False
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or not self.enabled:
            # Use merged weights or original only
            return F.linear(x, self.weight, self.bias)
        
        # Original output
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA addition: (x @ A^T) @ B^T * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into the base weights for inference."""
        if not self.merged:
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.weight.data.add_(delta_w)
            self.merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from the base weights."""
        if self.merged:
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.weight.data.sub_(delta_w)
            self.merged = False
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, return_attention_weights=False, 
                 use_lora=False, lora_rank=8, lora_alpha=16.0, lora_dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.return_attention_weights = return_attention_weights
        self.use_lora = use_lora

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        # Create linear projections
        if use_lora:
            # Use LoRA-wrapped linear layers for Q and V
            self.linear_q = LoRALinear(
                in_features=hidden_size,
                out_features=num_heads * att_size,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            self.linear_k = nn.Linear(hidden_size, num_heads * att_size)  # K remains standard
            self.linear_v = LoRALinear(
                in_features=hidden_size,
                out_features=num_heads * att_size,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
        else:
            self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
            self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
            self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        k = k.transpose(1, 2)                  # [b, h, k_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]

        if not self.return_attention_weights:
            # Use Flash Attention via PyTorch SDPA for speed and memory efficiency
            attn_mask = attn_bias
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.att_dropout.p if self.training else 0.0,
                scale=self.scale
            )
            attention_weights = None
        else:
            # Compute attention explicitly when we need to return weights
            k_t = k.transpose(2, 3)  # [b, h, d_k, k_len]
            q_scaled = q * self.scale
            attn_scores = torch.matmul(q_scaled, k_t)  # [b, h, q_len, k_len]
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attention_weights = torch.softmax(attn_scores, dim=3)
            x = self.att_dropout(attention_weights)
            x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, 
                 return_attention_weights=False, use_lora=False, lora_rank=8, lora_alpha=16.0, lora_dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)

        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads,
            return_attention_weights=return_attention_weights,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None):


        y = self.self_attention_norm(x)
        y, attention_weights = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        # transformer and FFN LayerNorm and related operations
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x, attention_weights




class ExphormerEncoderLayer(nn.Module):
    """
    Exphormer-based encoder layer with O(n) sparse attention.
    Drop-in replacement for EncoderLayer when use_exphormer=True.
    """
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads,
                 return_attention_weights=False, num_virtual_nodes=4, expander_degree=3):
        super(ExphormerEncoderLayer, self).__init__()
        
        self.return_attention_weights = return_attention_weights
        
        # Exphormer sparse attention layer
        self.exphormer = ExphormerLayer(
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            num_heads=num_heads,
            num_virtual_nodes=num_virtual_nodes,
            expander_degree=expander_degree,
            dropout=dropout_rate,
            attention_dropout=attention_dropout_rate,
            use_local=False,  # No local attention for sequence data (no graph structure)
            use_expander=True,
            use_global=True
        )
    
    def forward(self, x, attn_bias=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attn_bias: Optional attention bias (ignored for Exphormer)
            
        Returns:
            x: [batch_size, seq_len, hidden_size]
            attention_weights: None (Exphormer doesn't return full attention matrix)
        """
        x, attention_weights = self.exphormer(x, edge_index=None, return_attention=self.return_attention_weights)
        return x, attention_weights

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits



class VoxGFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super(VoxGFormer, self).__init__()

        # Set device
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.args = args

        # LoRA configuration
        self.use_lora = getattr(args, 'use_lora', False)
        self.lora_rank = getattr(args, 'lora_rank', 8)
        self.lora_alpha = getattr(args, 'lora_alpha', 16.0)
        self.lora_dropout = getattr(args, 'lora_dropout', 0.0)
        
        if self.use_lora:
            print(f"\n=== LoRA Configuration ===")
            print(f"  rank: {self.lora_rank}")
            print(f"  alpha: {self.lora_alpha}")
            print(f"  dropout: {self.lora_dropout}")
            print(f"  target: Q and V projections")
            print(f"===========================\n")

        # Prompt Tuning configuration
        self.use_prompt_tuning = getattr(args, "use_prompt_tuning", False)
        self.num_prompt_tokens = getattr(args, "num_prompt_tokens", 10)
        self.prompt_init_method = getattr(args, "prompt_init", "random")
        self.prompt_dropout = getattr(args, "prompt_dropout", 0.0)
        
        if self.use_prompt_tuning:
            print(f"\n=== Prompt Tuning Configuration ===")
            print(f"  num_tokens: {self.num_prompt_tokens}")
            print(f"  init_method: {self.prompt_init_method}")
            print(f"  dropout: {self.prompt_dropout}")
            print(f"====================================\n")
            
            self.prompt_tuning = PromptTuning(
                num_tokens=self.num_prompt_tokens,
                hidden_dim=args.embedding_dim,
                init_method=self.prompt_init_method,
                dropout=self.prompt_dropout
            )
        else:
            self.prompt_tuning = None

        # Set batch size
        self.batchsize = getattr(args, 'batchsize', None)
        
        self.gcn1 = GCN(args.embedding_dim, args.embedding_dim, activation)
        self.gcn2 = GCN(args.embedding_dim, args.embedding_dim, activation)

        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.act = nn.ReLU()

        self.n_in = n_in

        # Graph Transformer with optional LoRA or Exphormer
        self.use_exphormer = getattr(args, 'use_exphormer', False)
        self.num_virtual_nodes = getattr(args, 'exphormer_virtual_nodes', 4)
        self.expander_degree = getattr(args, 'exphormer_degree', 3)
        
        if self.use_exphormer:
            print("=== Exphormer Configuration ===")
            print(f"  virtual_nodes: {self.num_virtual_nodes}")
            print(f"  expander_degree: {self.expander_degree}")
            print("  complexity: O(n) sparse attention")
            print("===============================")
        
        encoders = []
        for i in range(args.GT_num_layers):
            return_attn = (i == args.GT_num_layers - 1)
            if self.use_exphormer:
                # Use Exphormer sparse attention (O(n) complexity)
                encoders.append(ExphormerEncoderLayer(
                    args.embedding_dim, args.GT_ffn_dim, args.GT_dropout,
                    args.GT_attention_dropout, args.GT_num_heads,
                    return_attention_weights=return_attn,
                    num_virtual_nodes=self.num_virtual_nodes,
                    expander_degree=self.expander_degree
                ))
            else:
                # Use standard attention or LoRA
                encoders.append(EncoderLayer(
                    args.embedding_dim, args.GT_ffn_dim, args.GT_dropout,
                    args.GT_attention_dropout, args.GT_num_heads,
                    return_attention_weights=return_attn,
                    use_lora=self.use_lora,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout
                ))
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.embedding_dim)
        self.read_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.token_projection = nn.Linear(self.n_in, args.embedding_dim)

        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, (args.pp_k+1) * self.n_in)
        )

        # Reconstruction loss
        self.recon_loss_fn = nn.MSELoss()

        # Projection layer for reconstruction error
        self.reconstruction_proj = nn.Sequential(
            nn.Linear((args.pp_k+1) * n_in, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, args.embedding_dim)
        )

        # Move model to device
        self.to(self.device)

        # Freeze non-LoRA/Prompt Tuning parameters if using LoRA or Prompt Tuning
        if self.use_lora or self.use_prompt_tuning:
            self._freeze_non_peft_parameters()
            self._print_parameter_stats()

        # Contrastive learning module
        self.use_contrastive = getattr(args, 'use_contrastive', False)
        if self.use_contrastive:
            self.contrastive_loss_fn = GraphContrastiveLoss(
                hidden_dim=args.embedding_dim,
                temperature=getattr(args, 'contrastive_temp', 0.1),
                aug_ratio=getattr(args, 'contrastive_aug_ratio', 0.2)
            ).to(self.device)
        else:
            self.contrastive_loss_fn = None
    
    def _freeze_non_lora_parameters(self):
        """Freeze all parameters except LoRA parameters."""
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        print(f"LoRA Mode: Frozen {frozen_count:,} params, trainable {trainable_count:,} params")
    
    def _freeze_non_peft_parameters(self):
        """Freeze all parameters except LoRA and Prompt Tuning parameters."""
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'soft_prompts' in name or 'prompt_tuning' in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        mode = []
        if self.use_lora:
            mode.append("LoRA")
        if self.use_prompt_tuning:
            mode.append("Prompt Tuning")
        mode_str = " + ".join(mode)
        print(f"{mode_str} Mode: Frozen {frozen_count:,} params, trainable {trainable_count:,} params")
    
    def _print_parameter_stats(self):
        """Print detailed parameter statistics."""
        total_params = 0
        trainable_params = 0
        lora_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
        
        print(f"\n=== Model Parameter Statistics ===")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  LoRA parameters: {lora_params:,}")
        if total_params > 0:
            print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
        print(f"=================================\n")
    
    def get_lora_parameters(self):
        """Get only LoRA parameters for optimizer."""
        return [p for n, p in self.named_parameters() if 'lora_A' in n or 'lora_B' in n]
    
    def get_prompt_parameters(self):
        """Get only Prompt Tuning parameters for optimizer."""
        return [p for n, p in self.named_parameters() if 'soft_prompts' in n or 'prompt_tuning' in n]
    
    def get_peft_parameters(self):
        """Get all PEFT parameters (LoRA + Prompt Tuning) for optimizer."""
        return [p for n, p in self.named_parameters() 
                if 'lora_A' in n or 'lora_B' in n or 'soft_prompts' in n or 'prompt_tuning' in n]
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base weights for inference."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
    
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from base weights."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()

    def _get_cosine_pe(self, seq_len, embed_dim, device):
        """
        Generate Cosine positional encoding (TransGAD style)
        """
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * 
                            (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def TransformerEncoder(self, tokens):
        """
        Inputs:
            - tokens: Input node token sequence [batch_size, pp_k+1, feature_dim]
        Outputs:
            - emb: Encoded output [1, batch_size, embedding_dim]
        """
        emb = self.token_projection(tokens)
        
        # Apply Prompt Tuning: prepend soft prompts to the sequence
        num_prompt_tokens = 0
        if self.use_prompt_tuning and self.prompt_tuning is not None:
            emb = self.prompt_tuning(emb)
            num_prompt_tokens = self.prompt_tuning.num_tokens
        
        # Add Cosine positional encoding
        if getattr(self.args, 'use_cosine_pe', True):
            seq_len = emb.size(1)
            pe = self._get_cosine_pe(seq_len, emb.size(-1), emb.device)
            emb = emb + pe.unsqueeze(0)  # broadcast to batch dimension
        
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # Get attention from last layer
                attention_weights = current_attention_weights
                # Aggregate multi-head attention
                agg_attention_weights = torch.mean(attention_weights, dim=1)
        
        emb = self.final_ln(emb)

        # Remove prompt outputs before attention-based pooling
        # attention_weights shape: [batch, num_heads, seq_len, seq_len]
        # agg_attention_weights: [batch, seq_len, seq_len]
        
        if num_prompt_tokens > 0:
            # Remove prompt tokens from the sequence dimension
            emb = emb[:, num_prompt_tokens:, :]
            # Adjust attention weights to remove prompt tokens
            # We want attention from real tokens, so slice the query dimension
            agg_attention_weights = agg_attention_weights[:, num_prompt_tokens:, :]
            # Also slice the key dimension to only include real tokens
            agg_attention_weights = agg_attention_weights[:, :, num_prompt_tokens:]

        # attention_scores: [N, args.pp_k+1]
        attention_scores = agg_attention_weights[:, 0, :]

        # Pooling based on attention_scores
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)

        return emb

    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):

        # input_tokens: (N, args.pp_k+1, d)
        emb = self.TransformerEncoder(input_tokens)

        # Generate global center point
        h_mean = torch.mean(emb, dim=1, keepdim=True)

        outlier_emb = None
        emb_combine = None
        noised_normal_for_generation_emb = None

        gna_loss = torch.tensor(0.0, device=emb.device)
        proj_loss = torch.tensor(0.0, device=emb.device)
        uniformity_loss = torch.tensor(0.0, device=emb.device)
        loss_ring = torch.tensor(0.0, device=emb.device)
        con_loss = torch.tensor(0.0, device=emb.device)
        loss_rec = torch.tensor(0.0, device=emb.device)
        if train_flag:
            # Efficient reshuffling
            perm = torch.randperm(normal_for_train_idx.size(0), device=normal_for_train_idx.device)
            normal_for_train_idx = normal_for_train_idx[perm]
            normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]            
            normal_for_generation_emb = emb[:, normal_for_generation_idx, :]
            # Noise
            noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
            noised_normal_for_generation_emb = normal_for_generation_emb + noise

            # Reconstruction learning
            reconstructed_tokens = self.token_decoder(emb).squeeze(0)  # [num_nodes, (args.pp_k+1)*n_in]
            reconstruction_error = reconstructed_tokens - input_tokens.view(-1, (args.pp_k+1) * self.n_in)
            # Project reconstruction error to embedding dimension
            reconstruction_error_proj = self.reconstruction_proj(reconstruction_error[normal_for_generation_idx, :])

            # Ablation study:
            if args.ablation_random_dir:
                norms = torch.norm(reconstruction_error_proj, p=2, dim=1, keepdim=True)
                random_vec = torch.randn_like(reconstruction_error_proj)
                random_dir = torch.nn.functional.normalize(random_vec, p=2, dim=1)
                reconstruction_error_proj = norms * random_dir

            outlier_emb = normal_for_generation_emb + args.outlier_beta * reconstruction_error_proj
            outlier_emb = outlier_emb.squeeze(0)

            # Ring alignment loss
            outlier_to_center_dist = torch.norm(outlier_emb - h_mean.squeeze(0), p=2, dim=1)
            ring_out_range_loss = torch.relu(args.ring_R_min - outlier_to_center_dist)
            ring_in_range_loss = torch.relu(outlier_to_center_dist - args.ring_R_max)

            loss_ring = torch.mean(ring_out_range_loss + ring_in_range_loss)

            # Contrastive learning loss
            loss_contrastive = torch.tensor(0.0, device=emb.device)
            if self.use_contrastive and self.contrastive_loss_fn is not None:
                if normal_for_train_idx is not None and len(normal_for_train_idx) > 1:
                    loss_contrastive = self.contrastive_loss_fn(emb[0, normal_for_train_idx, :])
            # Re-encode reconstructed tokens
            reconstructed_tokens_vector = torch.reshape(reconstructed_tokens, (-1, args.pp_k+1, self.n_in))
            reencoded_emb = self.TransformerEncoder(reconstructed_tokens_vector)[:, normal_for_generation_idx, :].detach().squeeze(0)
            loss_rec = self.compute_rec_loss(input_tokens, reconstructed_tokens, normal_for_generation_emb, reencoded_emb, normal_for_generation_idx)

            emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)

            f_1 = self.fc1(emb_combine)
        else:
            loss_contrastive = torch.tensor(0.0, device=emb.device)
            f_1 = self.fc1(emb)
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        logits = self.fc3(f_2)
        emb = emb.clone()

        return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring, loss_contrastive

    def compute_rec_loss(self, input_tokens, reconstructed_tokens, normal_for_generation_emb, reencoded_emb, normal_for_generation_idx):
        """
        Compute reconstruction loss in token and embedding spaces
        """
        token_rec_loss = self.recon_loss_fn(reconstructed_tokens, input_tokens.view(-1, (self.args.pp_k+1) * self.n_in))
        emb_rec_loss = torch.mean(torch.norm(normal_for_generation_emb.squeeze(0) - reencoded_emb, dim=-1))
        loss_rec = self.args.lambda_rec_tok * token_rec_loss + self.args.lambda_rec_emb * emb_rec_loss
        return loss_rec


    # InfoNCE uniformity loss
    def compute_infoNCE_uniformity_loss(self, emb, normal_for_train_idx, args):
        """
        Compute InfoNCE uniformity loss to push apart different normal nodes
        """
        normal_emb = emb[0, normal_for_train_idx, :]
        num_normal = normal_emb.size(0)
        
        if num_normal < 2:
            return torch.tensor(0.0, device=emb.device)
        
        normal_emb_norm = F.normalize(normal_emb, p=2, dim=1)
        similarity_matrix = torch.mm(normal_emb_norm, normal_emb_norm.t())
        similarity_matrix = similarity_matrix / args.GNA_temp
        
        mask = torch.eye(num_normal, device=emb.device, dtype=torch.bool)
        similarity_matrix_masked = similarity_matrix.masked_fill(mask, float('-inf'))
        log_sum_exp_values = torch.logsumexp(similarity_matrix_masked, dim=1)
        uniformity_loss = log_sum_exp_values.mean()
        
        return uniformity_loss