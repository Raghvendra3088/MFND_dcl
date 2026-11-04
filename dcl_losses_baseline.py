import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        features = nn.functional.normalize(features, p=2, dim=1)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        identity_mask = torch.eye(batch_size, device=features.device)
        mask = mask - identity_mask  # Remove self-comparison

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits = anchor_dot_contrast - torch.log(torch.exp(anchor_dot_contrast).sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        positive_samples_exist = mask_sum > 0
        
        if not positive_samples_exist.any():
            return torch.tensor(0.0, device=features.device)
            
        log_prob = (mask * logits).sum(1)[positive_samples_exist] / mask_sum[positive_samples_exist]
        
        loss = -log_prob.mean()
        return loss

class MultimodalConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MultimodalConsistencyLoss, self).__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text_features, image_features):
        text_features = nn.functional.normalize(text_features, p=2, dim=1)
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        
        logits_ti = torch.matmul(text_features, image_features.T) / self.temperature
        logits_it = torch.matmul(image_features, text_features.T) / self.temperature
        
        batch_size = text_features.shape[0]
        labels = torch.arange(batch_size, device=text_features.device)
        
        loss_t_i = self.loss_fn(logits_ti, labels)
        loss_i_t = self.loss_fn(logits_it, labels)
        
        return (loss_t_i + loss_i_t) / 2
