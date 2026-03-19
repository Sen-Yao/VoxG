"""
Curriculum Learning Module for Graph Anomaly Detection

This module implements difficulty-based curriculum learning to improve
anomaly detection performance by training on easier samples first.

Difficulty metrics:
1. Node degree: Lower degree = easier (fewer neighbors to aggregate)
2. Neighbor consistency: Higher consistency = easier (stable neighborhood)  
3. Feature entropy: Lower entropy = easier (more predictable features)
"""

import torch
import numpy as np
from scipy import sparse
from typing import Optional, Tuple, List, Dict, Literal
from dataclasses import dataclass
import warnings


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    strategy: Literal['linear', 'step', 'exp'] = 'linear'
    start_ratio: float = 0.3  # Start with 30% easiest samples
    end_ratio: float = 1.0    # End with all samples
    step_epochs: int = 50     # For step strategy, epochs per step
    exp_gamma: float = 0.02   # For exp strategy, growth rate
    
    # Difficulty scoring weights
    degree_weight: float = 0.4
    consistency_weight: float = 0.4
    entropy_weight: float = 0.2


class DifficultyScorer:
    """
    Calculate sample difficulty scores based on graph structure and features.
    
    Lower score = easier sample (should be trained first)
    Higher score = harder sample (should be trained later)
    """
    
    def __init__(
        self,
        adj: sparse.csr_matrix,
        features: np.ndarray,
        config: Optional[CurriculumConfig] = None,
        normal_train_idx: Optional[np.ndarray] = None
    ):
        """
        Args:
            adj: Adjacency matrix (scipy sparse)
            features: Node features (numpy array)
            config: Curriculum configuration
            normal_train_idx: Indices of normal training samples
        """
        self.adj = adj
        self.features = features
        self.config = config or CurriculumConfig()
        self.normal_train_idx = normal_train_idx
        self.num_nodes = adj.shape[0]
        
        # Cache for computed scores
        self._degree_scores: Optional[np.ndarray] = None
        self._consistency_scores: Optional[np.ndarray] = None
        self._entropy_scores: Optional[np.ndarray] = None
        self._combined_scores: Optional[np.ndarray] = None
    
    def compute_degree_scores(self) -> np.ndarray:
        """
        Compute difficulty based on node degree.
        Lower degree = easier (less information to aggregate)
        
        Returns:
            Normalized degree scores [0, 1], higher = harder
        """
        degrees = np.array(self.adj.sum(axis=1)).flatten()
        
        # Normalize to [0, 1]
        max_degree = degrees.max()
        min_degree = degrees.min()
        
        if max_degree == min_degree:
            self._degree_scores = np.zeros(self.num_nodes)
        else:
            self._degree_scores = (degrees - min_degree) / (max_degree - min_degree)
        
        return self._degree_scores
    
    def compute_consistency_scores(self) -> np.ndarray:
        """
        Compute difficulty based on neighbor feature consistency.
        Higher consistency = easier (stable neighborhood)
        
        Uses cosine similarity between node and its neighbors.
        
        Returns:
            Normalized consistency scores [0, 1], higher = harder
        """
        consistency = np.zeros(self.num_nodes)
        degrees = np.array(self.adj.sum(axis=1)).flatten()
        
        # Convert features to float for computation
        features = self.features.astype(np.float32)
        
        # Normalize features for cosine similarity
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features_normalized = features / feat_norms
        
        for i in range(self.num_nodes):
            if degrees[i] == 0:
                # Isolated nodes: set to mean consistency
                consistency[i] = 0.5
                continue
            
            # Get neighbors
            neighbors = self.adj[i].nonzero()[1]
            
            if len(neighbors) == 0:
                consistency[i] = 0.5
                continue
            
            # Compute average cosine similarity with neighbors
            node_feat = features_normalized[i]
            neighbor_feats = features_normalized[neighbors]
            
            similarities = np.dot(neighbor_feats, node_feat)
            avg_similarity = np.mean(similarities)
            
            # Higher similarity = easier, so invert for difficulty
            consistency[i] = 1.0 - avg_similarity  # Now higher = harder
        
        self._consistency_scores = consistency
        return self._consistency_scores
    
    def compute_entropy_scores(self) -> np.ndarray:
        """
        Compute difficulty based on feature entropy.
        Lower entropy = easier (more predictable/regular features)
        
        Returns:
            Normalized entropy scores [0, 1], higher = harder
        """
        features = self.features.astype(np.float32)
        
        # Compute entropy for each node across its feature dimensions
        entropy_per_node = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            feat = features[i]
            
            # Skip constant features
            feat_range = feat.max() - feat.min()
            if feat_range < 1e-8:
                entropy_per_node[i] = 0.0
                continue
            
            # Normalize to [0, 1] for probability-like interpretation
            feat_normalized = (feat - feat.min()) / (feat_range + 1e-8)
            
            # Add small epsilon for numerical stability
            feat_normalized = feat_normalized + 1e-8
            feat_normalized = feat_normalized / feat_normalized.sum()
            
            # Compute entropy
            entropy = -np.sum(feat_normalized * np.log2(feat_normalized + 1e-10))
            entropy_per_node[i] = entropy
        
        # Normalize to [0, 1]
        max_entropy = entropy_per_node.max()
        min_entropy = entropy_per_node.min()
        
        if max_entropy == min_entropy:
            self._entropy_scores = np.zeros(self.num_nodes)
        else:
            self._entropy_scores = (entropy_per_node - min_entropy) / (max_entropy - min_entropy)
        
        return self._entropy_scores
    
    def compute_all_scores(self) -> np.ndarray:
        """
        Compute all difficulty scores and combine them.
        
        Returns:
            Combined difficulty scores [0, 1], higher = harder
        """
        # Compute individual scores
        degree_scores = self.compute_degree_scores()
        consistency_scores = self.compute_consistency_scores()
        entropy_scores = self.compute_entropy_scores()
        
        # Weighted combination
        combined = (
            self.config.degree_weight * degree_scores +
            self.config.consistency_weight * consistency_scores +
            self.config.entropy_weight * entropy_scores
        )
        
        # Normalize to [0, 1]
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        
        self._combined_scores = combined
        return self._combined_scores
    
    def get_difficulty_ranking(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get ranking of samples by difficulty (easiest first).
        
        Args:
            indices: Subset of indices to rank. If None, use all nodes.
            
        Returns:
            Array of indices sorted by difficulty (easiest first)
        """
        if self._combined_scores is None:
            self.compute_all_scores()
        
        if indices is not None:
            scores_subset = self._combined_scores[indices]
            # Sort by score (ascending: easiest first)
            sorted_order = np.argsort(scores_subset)
            return indices[sorted_order]
        else:
            return np.argsort(self._combined_scores)


class CurriculumScheduler:
    """
    Manage curriculum learning schedule.
    
    Determines which samples to use at each epoch based on difficulty.
    """
    
    def __init__(
        self,
        scorer: DifficultyScorer,
        config: Optional[CurriculumConfig] = None,
        train_indices: Optional[np.ndarray] = None
    ):
        """
        Args:
            scorer: DifficultyScorer instance with computed scores
            config: Curriculum configuration
            train_indices: Indices of training samples (None = all nodes)
        """
        self.scorer = scorer
        self.config = config or CurriculumConfig()
        self.train_indices = train_indices
        
        # Get difficulty ranking for training samples
        self.ranked_indices = scorer.get_difficulty_ranking(train_indices)
        self.num_samples = len(self.ranked_indices)
        
        # Current ratio
        self._current_ratio = self.config.start_ratio
    
    def get_current_ratio(self, epoch: int, total_epochs: int) -> float:
        """
        Calculate the current sampling ratio based on epoch and strategy.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Current ratio of samples to use [start_ratio, end_ratio]
        """
        if self.config.strategy == 'linear':
            # Linear interpolation from start_ratio to end_ratio
            progress = min(epoch / total_epochs, 1.0)
            ratio = self.config.start_ratio + progress * (
                self.config.end_ratio - self.config.start_ratio
            )
        
        elif self.config.strategy == 'step':
            # Step-wise increase every step_epochs
            num_steps = total_epochs // self.config.step_epochs
            current_step = epoch // self.config.step_epochs
            progress = min(current_step / max(num_steps, 1), 1.0)
            ratio = self.config.start_ratio + progress * (
                self.config.end_ratio - self.config.start_ratio
            )
        
        elif self.config.strategy == 'exp':
            # Exponential growth
            # ratio(t) = end_ratio - (end_ratio - start_ratio) * exp(-gamma * t)
            ratio = self.config.end_ratio - (
                self.config.end_ratio - self.config.start_ratio
            ) * np.exp(-self.config.exp_gamma * epoch)
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        self._current_ratio = ratio
        return ratio
    
    def get_epoch_samples(self, epoch: int, total_epochs: int) -> np.ndarray:
        """
        Get sample indices for the current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Array of sample indices to train on this epoch
        """
        ratio = self.get_current_ratio(epoch, total_epochs)
        num_samples = max(1, int(self.num_samples * ratio))
        
        # Return easiest samples up to current ratio
        return self.ranked_indices[:num_samples]
    
    def get_epoch_weights(
        self, 
        epoch: int, 
        total_epochs: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get sample weights for weighted sampling.
        
        Easier samples get higher weights early, then equal weights later.
        
        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            device: Device to place tensor on
            
        Returns:
            Tensor of weights for each sample
        """
        ratio = self.get_current_ratio(epoch, total_epochs)
        num_hard = int(self.num_samples * (1 - ratio))
        num_easy = self.num_samples - num_hard
        
        # Easy samples get weight 1.0, hard samples get weight based on progress
        weights = np.ones(self.num_samples)
        hard_weight = ratio  # Gradually increase weight of hard samples
        
        if num_hard > 0:
            weights[-num_hard:] = hard_weight
        
        # Normalize
        weights = weights / weights.sum() * self.num_samples
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            weights_tensor = weights_tensor.to(device)
        
        return weights_tensor
    
    def log_progress(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Get progress metrics for logging.
        
        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            Dict with progress metrics
        """
        ratio = self.get_current_ratio(epoch, total_epochs)
        num_samples = int(self.num_samples * ratio)
        
        return {
            'curriculum_ratio': ratio,
            'curriculum_samples': num_samples,
            'curriculum_total_samples': self.num_samples,
            'curriculum_percent': ratio * 100
        }


def create_curriculum_scheduler(
    adj: sparse.csr_matrix,
    features: np.ndarray,
    normal_train_idx: np.ndarray,
    strategy: str = 'linear',
    start_ratio: float = 0.3,
    end_ratio: float = 1.0,
    **kwargs
) -> CurriculumScheduler:
    """
    Convenience function to create a curriculum scheduler.
    
    Args:
        adj: Adjacency matrix
        features: Node features
        normal_train_idx: Indices of normal training samples
        strategy: Curriculum strategy ('linear', 'step', 'exp')
        start_ratio: Starting ratio of samples
        end_ratio: Ending ratio of samples
        **kwargs: Additional config options
        
    Returns:
        CurriculumScheduler instance
    """
    config = CurriculumConfig(
        strategy=strategy,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        **kwargs
    )
    
    scorer = DifficultyScorer(
        adj=adj,
        features=features,
        config=config,
        normal_train_idx=normal_train_idx
    )
    
    scheduler = CurriculumScheduler(
        scorer=scorer,
        config=config,
        train_indices=normal_train_idx
    )
    
    return scheduler
