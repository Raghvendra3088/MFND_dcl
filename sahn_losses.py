"""
sahn_losses.py
==============
Semantic-Ambiguity-Aware Hard-Negative Dual Contrastive Learning (SAHN-DCL).

Key Features:
1. Valid Negative Masking (Diagonal exclusion, high-overlap duplicate exclusion).
2. Semantic Ambiguity Proxy (A_ij) to prevent over-penalizing confusing/uncertain pairs.
3. Bounded Hard-Negative Weighting (strictly prevents NaN, inf, or gradient explosion).
4. Full backward-compatibility: reproduces baseline loss exactly when use_sahn=False.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sample_negative_mask(labels):
    """
    Constructs sample-level negative mask:
    M[i, j] = 1 if labels[i] != labels[j] and i != j
    """
    labels = labels.view(-1, 1)
    batch_size = labels.shape[0]
    identity = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    same_label = torch.eq(labels, labels.T)
    neg_mask = (~same_label) & (~identity)
    return neg_mask.float()


def build_modality_negative_mask(text_features, overlap_threshold=0.85):
    """
    Constructs modality-level valid negative mask:
    - Diagonal (i == j) is positive (excluded from negatives).
    - Off-diagonal pairs with extremely high text similarity (near duplicates/same event)
      are masked out to prevent false-negative penalties.
    """
    batch_size = text_features.shape[0]
    identity = torch.eye(batch_size, dtype=torch.bool, device=text_features.device)
    
    # Check text-text overlap
    t_norm = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
    t_sim = torch.matmul(t_norm, t_norm.T)
    high_overlap = t_sim >= overlap_threshold
    
    valid_neg = (~identity) & (~high_overlap)
    return valid_neg.float()


def compute_ambiguity_proxy(sim_matrix):
    """
    Computes semantic ambiguity score A_ij in [0, 1].
    Ambiguity is highest near the decision threshold (sim ~ mean_sim)
    and lowest at extreme high or extreme low similarity.
    """
    mean_sim = sim_matrix.mean()
    std_sim = sim_matrix.std().clamp_min(1e-4)
    normalized_dist = (sim_matrix - mean_sim) / std_sim
    # Bell-shaped curve centered at mean_sim: 1.0 at mean, decays towards 0 at extremes
    ambiguity = torch.exp(-0.5 * (normalized_dist ** 2))
    return ambiguity.clamp(0.0, 1.0)


class SAHNSupConLoss(nn.Module):
    """
    Semantic-Ambiguity-Aware Hard-Negative Supervised Contrastive Loss.
    """
    def __init__(
        self,
        temperature=0.2,
        alpha=1.0,           # Hard-negative strength
        beta=0.5,            # Ambiguity suppression factor
        use_sahn=True,
        learnable_temp=True,
        min_temperature=0.05,
        max_temperature=1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_sahn = use_sahn
        self.learnable_temp = learnable_temp
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        if learnable_temp:
            init_val = (temperature - min_temperature) / (max_temperature - min_temperature)
            init_val = min(max(init_val, 0.01), 0.99)
            init_logit = math.log(init_val / (1.0 - init_val))
            self.temperature_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(temperature, dtype=torch.float32))

    def get_temperature(self):
        if not self.learnable_temp:
            return self.fixed_temperature
        logit = torch.clamp(self.temperature_logit, min=-10.0, max=10.0)
        return self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(logit)

    def forward(self, features, labels):
        batch_size = features.shape[0]
        if batch_size <= 1:
            return features.sum() * 0.0

        features = F.normalize(features, p=2, dim=-1, eps=1e-8)
        labels_col = labels.view(-1, 1)
        identity = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        pos_mask = torch.eq(labels_col, labels_col.T) & (~identity)

        temp = self.get_temperature()
        sim = torch.matmul(features, features.T) / temp
        sim = torch.clamp(sim, min=-50.0, max=50.0)
        sim = sim.masked_fill(identity, -1e4)

        if self.use_sahn:
            neg_mask = build_sample_negative_mask(labels)
            raw_sim = torch.matmul(features, features.T)
            ambiguity = compute_ambiguity_proxy(raw_sim)
            hardness = (raw_sim > 0.0).float() * raw_sim
            
            # Bounded weight: 1.0 <= W <= 1.0 + alpha
            w_ij = 1.0 + self.alpha * hardness * (1.0 - self.beta * ambiguity)
            w_ij = torch.clamp(w_ij, min=1.0, max=1.0 + self.alpha)
            
            log_w = torch.log(w_ij) * neg_mask
            sim = sim + log_w

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_count = pos_mask.sum(dim=1)
        valid_anchors = pos_count > 0

        if not valid_anchors.any():
            return features.sum() * 0.0

        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1)[valid_anchors] / pos_count[valid_anchors]
        return -mean_log_prob.mean()


class SAHNModalityConsistencyLoss(nn.Module):
    """
    Semantic-Ambiguity-Aware Hard-Negative Modality Consistency Loss.
    """
    def __init__(
        self,
        temperature=0.1,
        alpha=0.8,
        beta=0.5,
        overlap_threshold=0.85,
        use_sahn=True,
        learnable_temp=True,
        min_temperature=0.05,
        max_temperature=1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.overlap_threshold = overlap_threshold
        self.use_sahn = use_sahn
        self.learnable_temp = learnable_temp
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.ce_loss = nn.CrossEntropyLoss()

        if learnable_temp:
            init_val = (temperature - min_temperature) / (max_temperature - min_temperature)
            init_val = min(max(init_val, 0.01), 0.99)
            init_logit = math.log(init_val / (1.0 - init_val))
            self.temperature_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(temperature, dtype=torch.float32))

    def get_temperature(self):
        if not self.learnable_temp:
            return self.fixed_temperature
        logit = torch.clamp(self.temperature_logit, min=-10.0, max=10.0)
        return self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(logit)

    def _apply_sahn(self, logits, text_features, image_features):
        batch_size = logits.shape[0]
        neg_mask = build_modality_negative_mask(text_features, self.overlap_threshold)
        
        raw_sim = torch.matmul(
            F.normalize(text_features, p=2, dim=-1, eps=1e-8),
            F.normalize(image_features, p=2, dim=-1, eps=1e-8).T
        )
        ambiguity = compute_ambiguity_proxy(raw_sim)
        hardness = (raw_sim > 0.0).float() * raw_sim

        w_ij = 1.0 + self.alpha * hardness * (1.0 - self.beta * ambiguity)
        w_ij = torch.clamp(w_ij, min=1.0, max=1.0 + self.alpha)
        
        log_w = torch.log(w_ij) * neg_mask
        return logits + log_w

    def forward(self, text_features, image_features):
        batch_size = text_features.shape[0]
        if batch_size <= 1:
            return (text_features.sum() + image_features.sum()) * 0.0

        t_norm = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
        i_norm = F.normalize(image_features, p=2, dim=-1, eps=1e-8)
        temp = self.get_temperature()

        logits_ti = torch.matmul(t_norm, i_norm.T) / temp
        logits_it = torch.matmul(i_norm, t_norm.T) / temp

        logits_ti = torch.clamp(logits_ti, min=-50.0, max=50.0)
        logits_it = torch.clamp(logits_it, min=-50.0, max=50.0)

        if self.use_sahn:
            logits_ti = self._apply_sahn(logits_ti, text_features, image_features)
            logits_it = self._apply_sahn(logits_it, image_features, text_features)

        targets = torch.arange(batch_size, device=text_features.device)
        l_ti = self.ce_loss(logits_ti, targets)
        l_it = self.ce_loss(logits_it, targets)
        return 0.5 * (l_ti + l_it)
