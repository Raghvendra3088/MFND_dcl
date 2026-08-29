"""
model.py
========
Full MFND-DCL architecture (Fig. 2), assembled from pre-extracted, frozen
CLIP features:

  CLIP text/image features
        -> Gated Cross-Attention (gca.py)
        -> fused sample representation z_s (concat of enhanced text+image)
        -> Projection head  -> z_sample (for L_sample), and
           the pre-fusion enhanced_text/enhanced_image are separately
           projected -> z_text, z_image (for L_modal)
        -> Classification head -> fake/real logits (for L_CE)
"""

import torch
import torch.nn as nn

from gca import GatedCrossAttention


class MFND_DCL(nn.Module):
    def __init__(self, d_model=512, d_k=512, proj_dim=128,
                 classifier_dropout=0.3, num_classes=2):
        super().__init__()
        self.gca = GatedCrossAttention(d_model=d_model, d_k=d_k)

        # Modal-level projection heads (map enhanced per-modality features
        # into the space where L_modal / Eq. 9-11 is computed)
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim)
        )

        # Sample-level projection head (Fig. 2: "Projection head" branch,
        # feeds L_sample / Eq. 6-8)
        fused_dim = d_model * 2  # concat(enhanced_text, enhanced_image)
        self.sample_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Linear(fused_dim, proj_dim)
        )

        # Classification head (Fig. 2: "Classification head" branch, feeds
        # L_CE / "Fake News Detection")
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, text_feat, image_feat):
        """
        text_feat, image_feat: [B, d_model]  (frozen CLIP embeddings)
        Returns a dict with everything needed to compute L_total (Eq. 12).
        """
        enhanced_text, enhanced_image = self.gca(text_feat, image_feat)

        fused = torch.cat([enhanced_text, enhanced_image], dim=-1)  # sample repr z_s
        z_sample = self.sample_proj(fused)          # for L_sample
        z_text = self.text_proj(enhanced_text)       # for L_modal
        z_image = self.image_proj(enhanced_image)     # for L_modal
        logits = self.classifier(fused)                # for L_CE

        return {
            "logits": logits,
            "z_sample": z_sample,
            "z_text": z_text,
            "z_image": z_image,
            "fused": fused,
        }
