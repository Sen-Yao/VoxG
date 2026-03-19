"""
Prompt Tuning implementation for VoxGFormer
Based on: "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)

This module provides parameter-efficient fine-tuning by adding learnable soft prompt tokens
at the beginning of the input sequence, specifically designed for graph anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal


class PromptTuning(nn.Module):
    """
    Soft Prompt Tuning for Graph Transformer.
    
    Prepends learnable soft prompt tokens to the input sequence before the Transformer encoder.
    For graph data, these prompts act as task-specific context that guides the model attention.
    """
    
    def __init__(
        self,
        num_tokens: int = 10,
        hidden_dim: int = 256,
        init_method: str = "random",
        init_range: float = 0.5,
        num_classes: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.init_method = init_method
        self.num_classes = num_classes
        
        # Main soft prompts: [num_tokens, hidden_dim]
        self.soft_prompts = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        
        # Dropout for prompts (optional)
        self.prompt_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize prompts based on method
        self._init_prompts(init_range)
        
        # For class_specific initialization, we also learn class embeddings
        if init_method == "class_specific":
            self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
            nn.init.normal_(self.class_embeddings, mean=0.0, std=0.02)
        else:
            self.class_embeddings = None
        
        print(f"[Prompt Tuning] Initialized with {num_tokens} tokens, hidden_dim={hidden_dim}, init={init_method}")
    
    def _init_prompts(self, init_range: float = 0.5):
        """Initialize soft prompts based on the specified method."""
        if self.init_method == "random":
            nn.init.normal_(self.soft_prompts, mean=0.0, std=0.02)
        elif self.init_method == "uniform":
            nn.init.uniform_(self.soft_prompts, a=-init_range, b=init_range)
        elif self.init_method == "class_specific":
            nn.init.normal_(self.soft_prompts, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.soft_prompts, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, class_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Prepend soft prompts to input sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            class_idx: Optional class indices for class_specific prompts [batch_size]
            
        Returns:
            Tensor of shape [batch_size, num_tokens + seq_len, hidden_dim]
        """
        batch_size = x.size(0)
        
        # Expand prompts to batch dimension
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply dropout to prompts
        prompts = self.prompt_dropout(prompts)
        
        # For class_specific initialization, add class-specific bias
        if self.init_method == "class_specific" and self.class_embeddings is not None and class_idx is not None:
            class_emb = self.class_embeddings[class_idx]
            prompts = prompts + class_emb.unsqueeze(1) * 0.1
        
        # Concatenate prompts with input
        return torch.cat([prompts, x], dim=1)
    
    def get_num_tokens(self) -> int:
        return self.num_tokens
    
    def extra_repr(self) -> str:
        return f"num_tokens={self.num_tokens}, hidden_dim={self.hidden_dim}, init_method={self.init_method}"


class PromptTuningConfig:
    """Configuration for Prompt Tuning."""
    
    def __init__(
        self,
        num_tokens: int = 10,
        hidden_dim: int = 256,
        init_method: str = "random",
        init_range: float = 0.5,
        dropout: float = 0.0,
        freeze_base: bool = True,
        deep_prompt: bool = False,
        num_layers: int = 3
    ):
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.init_method = init_method
        self.init_range = init_range
        self.dropout = dropout
        self.freeze_base = freeze_base
        self.deep_prompt = deep_prompt
        self.num_layers = num_layers
    
    def __repr__(self):
        return f"PromptTuningConfig(num_tokens={self.num_tokens}, hidden_dim={self.hidden_dim}, init_method={self.init_method})"


class DeepPromptTuning(nn.Module):
    """
    Deep Prompt Tuning: adds learnable prompts at each Transformer layer.
    """
    
    def __init__(
        self,
        num_tokens: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 3,
        init_method: str = "random",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create prompts for each layer
        self.layer_prompts = nn.ModuleList([
            PromptTuning(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                init_method=init_method,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        print(f"[Deep Prompt Tuning] Initialized {num_layers} layers with {num_tokens} tokens each")
    
    def forward(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        if layer_idx >= self.num_layers:
            layer_idx = self.num_layers - 1
        return self.layer_prompts[layer_idx](x)
    
    def get_num_tokens(self) -> int:
        return self.num_tokens


def apply_prompt_tuning_to_model(
    model: nn.Module,
    config: PromptTuningConfig,
    verbose: bool = True
) -> nn.Module:
    """
    Apply prompt tuning to a model.
    """
    if verbose:
        print(f"\n=== Applying Prompt Tuning ===")
        print(f"  num_tokens: {config.num_tokens}")
        print(f"  hidden_dim: {config.hidden_dim}")
        print(f"  init_method: {config.init_method}")
        print(f"  freeze_base: {config.freeze_base}")
    
    # Add prompt tuning module to model
    if config.deep_prompt:
        model.prompt_tuning = DeepPromptTuning(
            num_tokens=config.num_tokens,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            init_method=config.init_method,
            dropout=config.dropout
        )
    else:
        model.prompt_tuning = PromptTuning(
            num_tokens=config.num_tokens,
            hidden_dim=config.hidden_dim,
            init_method=config.init_method,
            dropout=config.dropout
        )
    
    model.prompt_tuning_config = config
    
    # Freeze base model parameters if requested
    if config.freeze_base:
        freeze_non_prompt_parameters(model, verbose=verbose)
    
    return model


def freeze_non_prompt_parameters(model: nn.Module, verbose: bool = True):
    """Freeze all parameters except prompt tuning parameters."""
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if "prompt_tuning" in name or "soft_prompts" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    if verbose:
        print(f"  Frozen {frozen_count:,} parameters, keeping {trainable_count:,} trainable (prompt tuning)")


def get_prompt_parameters(model: nn.Module) -> list:
    """Get only prompt tuning parameters for optimizer."""
    prompt_params = []
    for name, param in model.named_parameters():
        if "prompt_tuning" in name or "soft_prompts" in name:
            prompt_params.append(param)
    return prompt_params


def get_prompt_tuning_state_dict(model: nn.Module) -> dict:
    """Get state dict containing only prompt tuning parameters."""
    prompt_state = {}
    for name, param in model.named_parameters():
        if "prompt_tuning" in name or "soft_prompts" in name:
            prompt_state[name] = param.data.clone()
    return prompt_state


def load_prompt_tuning_state_dict(model: nn.Module, prompt_state: dict):
    """Load prompt tuning parameters from state dict."""
    for name, param in model.named_parameters():
        if name in prompt_state:
            param.data.copy_(prompt_state[name])


def remove_prompt_outputs(output: torch.Tensor, num_prompt_tokens: int) -> torch.Tensor:
    """Remove prompt token outputs from the sequence."""
    return output[:, num_prompt_tokens:, :]


def create_graph_prompt_init(
    node_features: torch.Tensor,
    num_prompts: int,
    method: str = "mean"
) -> torch.Tensor:
    """
    Create initialization for soft prompts based on graph node features.
    """
    if method == "mean":
        mean_feature = node_features.mean(dim=0, keepdim=True)
        prompts = mean_feature + torch.randn(num_prompts, node_features.size(1)) * 0.02
    elif method == "random_sample":
        num_nodes = node_features.size(0)
        indices = torch.randperm(num_nodes)[:num_prompts]
        prompts = node_features[indices]
    else:
        prompts = torch.randn(num_prompts, node_features.size(1)) * 0.02
    
    return prompts
