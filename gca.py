"""
gca.py
======
Gated Cross-Attention (GCA), Section 3.4 / Fig. 3, Eq. 1-5.

Given text feature sequence T and image feature sequence I (both projected
to R^d), the module:
  1. Projects each modality into Q/K/V (Eq. 1-2).
  2. Computes text->image cross-attention (text as query) and, symmetrically,
     image->text cross-attention (Eq. 3, applied in both directions).
  3. Computes a learnable gate g in [0,1] from the concatenation of the
     original query feature and the attention output (Eq. 4).
  4. Uses the gate to blend the attended (cross-modal) signal with the
     original modality feature (Eq. 5), so the model can suppress
     misleading/irrelevant cross-modal information on a per-sample basis.
  5. Applies residual connection + LayerNorm to produce the enhanced
     modality features, matching "Finally, the enhanced modality features
     are obtained through residual connection and layer normalization."

Fig. 2 additionally shows a "1D average pooling" step after GCA that
collapses the (possibly sequence-valued) enhanced features into a single
fixed-size vector per modality before they are handed to the DCL heads.
Because we use CLIP's pooled (non-token-level) text/image embeddings as
input (Sec 3.3), each modality is naturally a length-1 "sequence"; the
implementation below still supports longer sequences (e.g. if you later
swap in token-level CLIP features) and applies the 1D average pooling in
that general case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """One-directional scaled dot-product cross-attention: query modality
    attends over key/value modality. Implements Eq. 1-3."""

    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)

    def forward(self, query_src, kv_src):
        # query_src: [B, Lq, d_model]   kv_src: [B, Lk, d_model]
        Q = self.Wq(query_src)   # [B, Lq, d_k]
        K = self.Wk(kv_src)      # [B, Lk, d_k]
        V = self.Wv(kv_src)      # [B, Lk, d_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)  # [B, Lq, Lk]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, Lq, d_k]
        return out


class GatedCrossAttention(nn.Module):
    """Bidirectional Gated Cross-Attention: enhances both text and image
    features using each other, with a learnable gate controlling how much
    cross-modal information is fused in (Eq. 4-5)."""

    def __init__(self, d_model=512, d_k=512, dropout=0.1):
        super().__init__()
        self.text_to_image_attn = CrossAttentionBlock(d_model, d_k)  # text queries image
        self.image_to_text_attn = CrossAttentionBlock(d_model, d_k)  # image queries text

        # Gate: sigmoid(W_g . concat(original, attended) + b_g), Eq. 4
        self.gate_text = nn.Linear(d_model + d_k, d_model)
        self.gate_image = nn.Linear(d_model + d_k, d_model)

        self.ln_text = nn.LayerNorm(d_model)
        self.ln_image = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, image_feat):
        """
        text_feat, image_feat: [B, d_model]  (pooled CLIP features)
        Internally unsqueezed to [B, 1, d_model] to reuse a general
        sequence-based attention implementation; the "1D average pooling"
        of Fig. 2 is then just squeeze(1) in this pooled-feature setting.
        """
        T = text_feat.unsqueeze(1)   # [B, 1, d]
        I = image_feat.unsqueeze(1)  # [B, 1, d]

        # text->image cross attention (text is the query, Eq. 3)
        attn_t2i = self.text_to_image_attn(query_src=T, kv_src=I)  # [B, 1, d_k]
        # image->text cross attention (image is the query, symmetric)
        attn_i2t = self.image_to_text_attn(query_src=I, kv_src=T)  # [B, 1, d_k]

        # gate for text branch: g = sigmoid(Wg . concat(T, attn_t2i) + bg)  (Eq. 4)
        g_text = torch.sigmoid(self.gate_text(torch.cat([T, attn_t2i], dim=-1)))
        # gated fusion (Eq. 5): blend attended cross-modal info with the
        # original modality feature
        fused_text = g_text * attn_t2i + (1 - g_text) * T

        g_image = torch.sigmoid(self.gate_image(torch.cat([I, attn_i2t], dim=-1)))
        fused_image = g_image * attn_i2t + (1 - g_image) * I

        # residual connection + LayerNorm ("enhanced modality features")
        enhanced_text = self.ln_text(T + self.dropout(fused_text))
        enhanced_image = self.ln_image(I + self.dropout(fused_image))

        # "1D average pooling" (Fig. 2) -> fixed-size vector per modality
        enhanced_text = enhanced_text.mean(dim=1)    # [B, d]
        enhanced_image = enhanced_image.mean(dim=1)  # [B, d]

        return enhanced_text, enhanced_image
