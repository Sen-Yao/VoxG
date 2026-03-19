import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GraphContrastiveLoss(nn.Module):
    """
    CVGAD style contrastive loss for graph anomaly detection.
    """
    
    def __init__(self, hidden_dim, temperature=0.1, aug_ratio=0.2):
        super(GraphContrastiveLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.aug_ratio = aug_ratio
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def augment_features(self, x, drop_rate=None):
        if drop_rate is None:
            drop_rate = self.aug_ratio
            
        if x.dim() == 2:
            drop_mask = torch.bernoulli(torch.ones(x.size(1), device=x.device) * (1 - drop_rate))
            return x * drop_mask.unsqueeze(0)
        else:
            drop_mask = torch.bernoulli(torch.ones(x.size(-1), device=x.device) * (1 - drop_rate))
            return x * drop_mask.unsqueeze(0).unsqueeze(0)
    
    def augment_edges(self, edge_index, num_nodes, drop_rate=0.1):
        if edge_index is None:
            return None
            
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return edge_index
            
        keep_mask = torch.bernoulli(torch.ones(num_edges, device=edge_index.device) * (1 - drop_rate)).bool()
        
        if keep_mask.sum() == 0:
            keep_mask[0] = True
            
        new_edge_index = edge_index[:, keep_mask]
        return new_edge_index
    
    def augment_tokens(self, tokens, drop_rate=None):
        if drop_rate is None:
            drop_rate = self.aug_ratio
        return self.augment_features(tokens, drop_rate)
    
    def forward(self, h_original, h_aug=None, node_ids=None, return_aug=False):
        if h_original.dim() == 3:
            h_original = h_original.squeeze(0)
        
        if h_aug is None:
            h_aug = self.augment_features(h_original)
            h_aug = h_aug + torch.randn_like(h_aug) * 0.01
        
        if h_aug.dim() == 3:
            h_aug = h_aug.squeeze(0)
        
        if node_ids is not None:
            h_original = h_original[node_ids]
            h_aug = h_aug[node_ids]
        
        z_original = self.projection_head(h_original)
        z_aug = self.projection_head(h_aug)
        
        z_original = F.normalize(z_original, p=2, dim=1)
        z_aug = F.normalize(z_aug, p=2, dim=1)
        
        num_nodes = z_original.size(0)
        if num_nodes < 2:
            return torch.tensor(0.0, device=h_original.device)
        
        sim_matrix = torch.mm(z_original, z_aug.t()) / self.temperature
        labels = torch.arange(num_nodes, device=h_original.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        if return_aug:
            return loss, h_aug
        return loss


class CVGADContrastiveLoss(nn.Module):
    """
    Full CVGAD style contrastive loss with progressive purification.
    """
    
    def __init__(self, hidden_dim, temperature=0.1, 
                 feature_drop_rate=0.2, edge_drop_rate=0.1,
                 use_projection=True):
        super(CVGADContrastiveLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.feature_drop_rate = feature_drop_rate
        self.edge_drop_rate = edge_drop_rate
        
        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def project(self, h):
        if self.use_projection:
            return self.projection(h)
        return h
    
    def create_clean_view(self, tokens, edge_index=None):
        noise = torch.randn_like(tokens) * 0.05
        return tokens + noise
    
    def create_augmented_view(self, tokens, edge_index=None):
        drop_mask = torch.bernoulli(
            torch.ones(tokens.size(-1), device=tokens.device) * (1 - self.feature_drop_rate)
        )
        augmented = tokens * drop_mask.unsqueeze(0).unsqueeze(0)
        augmented = augmented + torch.randn_like(augmented) * 0.1
        return augmented
    
    def forward(self, h_original, h_clean=None, h_aug=None, 
                node_mask=None, return_views=False):
        if h_original.dim() == 3:
            h_original = h_original.squeeze(0)
        
        if node_mask is not None:
            h_original = h_original[node_mask]
            if h_clean is not None:
                h_clean = h_clean[node_mask] if h_clean.dim() == 2 else h_clean[node_mask]
            if h_aug is not None:
                h_aug = h_aug[node_mask] if h_aug.dim() == 2 else h_aug[node_mask]
        
        num_nodes = h_original.size(0)
        if num_nodes < 2:
            loss = torch.tensor(0.0, device=h_original.device)
            if return_views:
                return loss, None, None
            return loss
        
        z_original = self.project(h_original)
        
        if h_clean is None:
            z_clean = z_original
        else:
            z_clean = self.project(h_clean.squeeze(0) if h_clean.dim() == 3 else h_clean)
        
        if h_aug is None:
            z_aug = z_clean
        else:
            z_aug = self.project(h_aug.squeeze(0) if h_aug.dim() == 3 else h_aug)
        
        z_original = F.normalize(z_original, p=2, dim=1)
        z_clean = F.normalize(z_clean, p=2, dim=1)
        z_aug = F.normalize(z_aug, p=2, dim=1)
        
        sim_clean = torch.mm(z_original, z_clean.t()) / self.temperature
        sim_aug = torch.mm(z_original, z_aug.t()) / self.temperature
        
        labels = torch.arange(num_nodes, device=h_original.device)
        
        loss_clean = F.cross_entropy(sim_clean, labels)
        loss_aug = F.cross_entropy(sim_aug, labels)
        
        loss = (loss_clean + loss_aug) / 2
        
        if return_views:
            return loss, z_clean, z_aug
        return loss


def compute_contrastive_loss(emb, normal_idx, temperature=0.1, device=None):
    """
    Simplified contrastive loss function.
    """
    if device is None:
        device = emb.device
    
    if emb.dim() == 3:
        emb = emb.squeeze(0)
    
    normal_emb = emb[normal_idx]
    num_nodes = normal_emb.size(0)
    
    if num_nodes < 2:
        return torch.tensor(0.0, device=device)
    
    normal_emb_norm = F.normalize(normal_emb, p=2, dim=1)
    
    sim_matrix = torch.mm(normal_emb_norm, normal_emb_norm.t()) / temperature
    
    mask = torch.eye(num_nodes, device=device, dtype=torch.bool)
    sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))
    
    noise = torch.randn_like(normal_emb) * 0.01
    augmented = F.normalize(normal_emb + noise, p=2, dim=1)
    sim_positive = torch.sum(normal_emb_norm * augmented, dim=1) / temperature
    
    log_sum_exp = torch.logsumexp(sim_matrix_masked, dim=1)
    loss = -sim_positive + log_sum_exp
    
    return loss.mean()
