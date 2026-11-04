import torch
import torch.nn as nn
import math

class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature=0.2,
        hard_negative_k=5,
        hard_negative_weight=1.5,
        min_temperature=0.05,
        max_temperature=1.0,
        learnable_temp=True
    ):
        super().__init__()
        self.hard_negative_k = hard_negative_k
        self.hard_negative_weight = hard_negative_weight
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.learnable_temp = learnable_temp
        
        if learnable_temp:
            initial_value = (temperature - min_temperature) / (max_temperature - min_temperature)
            initial_value = min(max(initial_value, 0.01), 0.99)
            initial_logit = math.log(initial_value / (1 - initial_value))
            self.temperature_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(temperature, dtype=torch.float32))

    def get_temperature(self):
        if not self.learnable_temp:
            return self.fixed_temperature
        # Clamp logit to prevent extreme temperature values and nan gradients
        logit = torch.clamp(self.temperature_logit, min=-10.0, max=10.0)
        temperature = self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(logit)
        return temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        # Epsilon to prevent div by zero in norm
        features = nn.functional.normalize(features, p=2, dim=1, eps=1e-8)
        labels = labels.view(-1, 1)
        
        positive_mask = torch.eq(labels, labels.T).float()
        identity_mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask - identity_mask
        negative_mask = (1.0 - positive_mask - identity_mask).clamp(min=0.0)
        
        temperature = self.get_temperature()
        similarity = torch.matmul(features, features.T) / temperature
        # Clamp similarity to prevent extreme values and NaN in backward pass
        similarity = torch.clamp(similarity, min=-50.0, max=50.0)
        similarity = similarity.masked_fill(identity_mask.bool(), -1e4)

        # Mismatch-Aware Hard-negative selection (Vectorized)
        masked_similarity = similarity.masked_fill(negative_mask == 0, -1e4)
        
        valid_k = min(self.hard_negative_k, batch_size - 1)
        if valid_k > 0:
            top_scores, top_indices = torch.topk(masked_similarity, k=valid_k, dim=1)
            hard_negative_mask = torch.zeros_like(negative_mask)
            hard_negative_mask.scatter_(1, top_indices, 1.0)
            
            # NOVELTY FIX: Only apply hard negative weighting if confusing (similarity > 0)
            confusing_mask = (similarity > 0.0).float()
            hard_negative_mask = hard_negative_mask * confusing_mask
        else:
            hard_negative_mask = torch.zeros_like(negative_mask)

        # Increase hard-negative importance safely
        log_weight = math.log(self.hard_negative_weight)
        weighted_similarity = similarity + (hard_negative_mask * log_weight)

        log_prob = (
            weighted_similarity - torch.logsumexp(weighted_similarity, dim=1, keepdim=True)
        )

        positive_count = positive_mask.sum(dim=1)
        valid_samples = positive_count > 0

        if not valid_samples.any():
            return features.sum() * 0.0

        mean_log_prob = (positive_mask * log_prob).sum(dim=1)[valid_samples] / (positive_count[valid_samples] + 1e-8)
        loss = -mean_log_prob.mean()
        return loss

class MultimodalConsistencyLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        hard_negative_k=5,
        hard_negative_weight=1.5,
        min_temperature=0.05,
        max_temperature=1.0,
        learnable_temp=True
    ):
        super().__init__()
        self.hard_negative_k = hard_negative_k
        self.hard_negative_weight = hard_negative_weight
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.learnable_temp = learnable_temp
        
        if learnable_temp:
            initial_value = (temperature - min_temperature) / (max_temperature - min_temperature)
            initial_value = min(max(initial_value, 0.01), 0.99)
            initial_logit = math.log(initial_value / (1 - initial_value))
            self.temperature_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(temperature, dtype=torch.float32))
        self.loss_fn = nn.CrossEntropyLoss()

    def get_temperature(self):
        if not self.learnable_temp:
            return self.fixed_temperature
        logit = torch.clamp(self.temperature_logit, min=-10.0, max=10.0)
        temperature = self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(logit)
        return temperature

    def apply_hard_negative_weight(self, logits):
        batch_size = logits.shape[0]
        identity_mask = torch.eye(batch_size, device=logits.device)
        negative_mask = 1.0 - identity_mask
        masked_logits = logits.masked_fill(negative_mask == 0, -1e4)
        
        valid_k = min(self.hard_negative_k, batch_size - 1)
        if valid_k > 0:
            top_scores, top_indices = torch.topk(masked_logits, k=valid_k, dim=1)
            hard_negative_mask = torch.zeros_like(negative_mask)
            hard_negative_mask.scatter_(1, top_indices, 1.0)
            
            # NOVELTY FIX: Only penalize if confusing (cross-modal similarity > 0)
            confusing_mask = (logits > 0.0).float()
            hard_negative_mask = hard_negative_mask * confusing_mask
        else:
            hard_negative_mask = torch.zeros_like(negative_mask)
            
        log_weight = math.log(self.hard_negative_weight)
        logits = logits + (hard_negative_mask * log_weight)
        return logits

    def forward(self, text_features, image_features):
        text_features = nn.functional.normalize(text_features, p=2, dim=1, eps=1e-8)
        image_features = nn.functional.normalize(image_features, p=2, dim=1, eps=1e-8)
        
        temperature = self.get_temperature()
        logits_ti = torch.matmul(text_features, image_features.T) / temperature
        logits_it = torch.matmul(image_features, text_features.T) / temperature
        
        logits_ti = torch.clamp(logits_ti, min=-50.0, max=50.0)
        logits_it = torch.clamp(logits_it, min=-50.0, max=50.0)
        
        logits_ti = self.apply_hard_negative_weight(logits_ti)
        logits_it = self.apply_hard_negative_weight(logits_it)
        
        batch_size = text_features.shape[0]
        labels = torch.arange(batch_size, device=text_features.device)
        
        loss_t_i = self.loss_fn(logits_ti, labels)
        loss_i_t = self.loss_fn(logits_it, labels)
        
        return (loss_t_i + loss_i_t) / 2
