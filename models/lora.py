"""
LoRA (Low-Rank Adaptation) implementation for VoxGFormer
Based on: https://arxiv.org/abs/2106.09685

This module provides parameter-efficient fine-tuning by adding low-rank 
decomposition matrices to existing linear layers, specifically for 
attention Q/V projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


class LoRALinear(nn.Module):
    """
    LoRA wrapper for linear layers.
    Wraps an existing linear layer and adds low-rank adaptation.
    
    h = W_0 x + BA x
    where B is (out_features, r) and A is (r, in_features)
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = True
    ):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        
        # Store the original weights (frozen)
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # LoRA low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        # A is initialized with Kaiming uniform (like standard linear)
        # B is initialized to zero so initial output equals original
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Track whether weights are merged
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Use merged weights
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
            # W_new = W + BA * scaling
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


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ['linear_q', 'linear_v'],
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, int]]:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA
        target_modules: List of module names to apply LoRA to
        verbose: Whether to print information
    
    Returns:
        Modified model with LoRA layers
        Dictionary with parameter counts
    """
    param_counts = {
        'total_original': 0,
        'total_lora_trainable': 0,
        'lora_layers': 0
    }
    
    # Count original parameters
    for name, param in model.named_parameters():
        param_counts['total_original'] += param.numel()
    
    # Track LoRA parameters
    lora_params = []
    
    def replace_linear_with_lora(name: str, module: nn.Module, parent: nn.Module, attr_name: str):
        """Replace a linear layer with LoRA-wrapped version."""
        nonlocal lora_params
        
        # Check if this module name matches any target
        for target in target_modules:
            if target in name:
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(parent, attr_name, lora_layer)
                    lora_params.append((name, lora_layer))
                    if verbose:
                        print(f"  Applied LoRA to: {name} (in={module.in_features}, out={module.out_features})")
                    return True
        return False
    
    # Walk through model and apply LoRA
    if verbose:
        print(f"Applying LoRA with rank={rank}, alpha={alpha}")
        print(f"Target modules: {target_modules}")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules()).get(parent_name)
            else:
                parent = model
                attr_name = name
            
            if parent is not None:
                replace_linear_with_lora(name, module, parent, attr_name)
    
    # Count LoRA trainable parameters
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param_counts['total_lora_trainable'] += param.numel()
    
    param_counts['lora_layers'] = len(lora_params)
    
    if verbose:
        print(f"\nLoRA Statistics:")
        print(f"  Total original parameters: {param_counts['total_original']:,}")
        print(f"  LoRA trainable parameters: {param_counts['total_lora_trainable']:,}")
        print(f"  LoRA layers: {param_counts['lora_layers']}")
        reduction = (1 - param_counts['total_lora_trainable'] / param_counts['total_original']) * 100
        print(f"  Parameter reduction: {reduction:.2f}%")
    
    return model, param_counts


def freeze_non_lora_parameters(model: nn.Module, verbose: bool = True):
    """
    Freeze all parameters except LoRA parameters.
    This ensures only LoRA weights are updated during training.
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    if verbose:
        print(f"Frozen {frozen_count:,} parameters, keeping {trainable_count:,} trainable (LoRA only)")


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only LoRA parameters for optimizer.
    Use this when you want to train only LoRA weights.
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.
    Useful for saving/loading LoRA weights separately.
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
    """
    Load LoRA parameters from state dict.
    """
    for name, param in model.named_parameters():
        if name in lora_state:
            param.data.copy_(lora_state[name])


class LoRAConfig:
    """Configuration for LoRA."""
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: List[str] = ['linear_q', 'linear_v'],
        freeze_base: bool = True
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.freeze_base = freeze_base
    
    def __repr__(self):
        return (f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, dropout={self.dropout}, "
                f"target_modules={self.target_modules}, freeze_base={self.freeze_base})")
