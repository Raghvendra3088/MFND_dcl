"""
dcl_losses.py
=============
Dual Contrastive Learning losses, Section 3.5.

1. Sample-level contrastive loss (Sec 3.5.1, Eq. 6-8) -- a standard
   supervised contrastive (SupCon) loss operating on the fused
   sample-level representation z_s, pulling same-label samples together
   and pushing different-label samples apart.

2. Modal-level contrastive loss (Sec 3.5.2, Eq. 9-11) -- symmetric InfoNCE
   between the text and image projections of the SAME post (positive pair
   = matching text/image from the same post; negatives = all other
   text/image pairs in the batch), i.e. a CLIP-style contrastive loss
   applied post-fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleContrastiveLoss(nn.Module):
    """Supervised contrastive loss, Eq. 6-8.

    L_sample = -(1/|B|) * sum_{i in B} (1/|P(i)|) *
                    sum_{p in P(i)} log( exp(s_ip) / sum_{k!=i} exp(s_ik) )

    where B = {i : |P(i)| > 0} ensures every anchor has >=1 positive in the
    batch, s_ij = (z_i . z_j) / tau is the temperature-scaled cosine
    similarity of L2-normalized sample representations, and P(i) is the set
    of same-label samples (excluding i itself).
    """

    def __init__(self, temperature=0.2):
        super().__init__()
        self.tau = temperature

    def forward(self, z_sample, labels):
        # z_sample: [N, dp] (unnormalized projection-head output)
        # labels:   [N] integer class labels
        device = z_sample.device
        N = z_sample.size(0)

        z = F.normalize(z_sample, dim=-1)               # Eq. 6, L2 normalization
        sim = torch.matmul(z, z.T) / self.tau            # Eq. 7, s_ij

        # mask out self-similarity (k != i) in the denominator
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim_masked = sim.masked_fill(self_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_masked, dim=1)   # log sum_{k!=i} exp(s_ik)

        labels = labels.view(-1, 1)
        same_label = (labels == labels.T)                # [N, N] boolean
        positive_mask = same_label & (~self_mask)         # P(i): same label, excl. self

        has_positive = positive_mask.sum(dim=1) > 0        # B = {i : |P(i)| > 0}
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # log p_ip = s_ip - log_denom_i  for every positive p of anchor i
        log_prob = sim - log_denom.unsqueeze(1)            # [N, N]
        pos_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
        num_positives = positive_mask.sum(dim=1).clamp(min=1).float()
        mean_log_prob_pos = pos_log_prob / num_positives    # (1/|P(i)|) sum_p log p_ip

        loss_per_anchor = -mean_log_prob_pos[has_positive]
        loss = loss_per_anchor.mean()                       # average over B
        return loss


class ModalContrastiveLoss(nn.Module):
    """Symmetric cross-modal InfoNCE loss, Eq. 9-11.

    Builds text->image similarity matrix S_t2v[m,n] = (z_t^m . z_v^n) / tau_m
    and computes two InfoNCE losses treating the diagonal as positives:
        L_t2v: text m must match its own image m among all images in batch
        L_v2t: image n must match its own text n among all texts in batch
    L_modal = (L_t2v + L_v2t) / 2
    """

    def __init__(self, temperature=0.2):
        super().__init__()
        self.tau = temperature

    def forward(self, z_text, z_image):
        # z_text, z_image: [N, dp], paired by row index (same post)
        device = z_text.device
        N = z_text.size(0)

        z_t = F.normalize(z_text, dim=-1)
        z_v = F.normalize(z_image, dim=-1)

        logits_t2v = torch.matmul(z_t, z_v.T) / self.tau   # [N, N], S_t2v
        targets = torch.arange(N, device=device)

        loss_t2v = F.cross_entropy(logits_t2v, targets)     # Eq. 9
        loss_v2t = F.cross_entropy(logits_t2v.T, targets)    # Eq. 10 (symmetric)

        loss = 0.5 * (loss_t2v + loss_v2t)                   # Eq. 11
        return loss
