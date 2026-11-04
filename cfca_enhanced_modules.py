import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Importance(nn.Module):
    """a_i = softmax_i( w^T tanh(W h_i + b) ), padding -> 0 weight."""
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x, mask):
        s = self.net(x).squeeze(-1).masked_fill(~mask, -1e4)
        return F.softmax(s, dim=-1)

class MHGatedCrossAttn(nn.Module):
    """x attends y. Per-token gate g_i. LayerNorm before (norm1) and after (norm2)."""
    def __init__(self, d=128, heads=4, drop=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.gate = nn.Linear(2 * d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, y, y_mask, y_prior=None):
        B, Lx, d = x.shape
        Ly = y.size(1)
        xn = self.n1(x)
        q = self.q(xn).view(B, Lx, self.h, self.dh).transpose(1, 2)
        k = self.k(y).view(B, Ly, self.h, self.dh).transpose(1, 2)
        v = self.v(y).view(B, Ly, self.h, self.dh).transpose(1, 2)
        logit = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if y_prior is not None:
            logit = logit + torch.log(y_prior.clamp_min(1e-6))[:, None, None, :]
        logit = logit.masked_fill(~y_mask[:, None, None, :], -1e4)
        A = logit.softmax(-1)
        C = self.o((A @ v).transpose(1, 2).reshape(B, Lx, d))
        g = torch.sigmoid(self.gate(torch.cat([xn, C], -1)))
        out = self.n2(x + self.drop(g * C + (1 - g) * xn))
        return out, g, A.mean(1)

def align_matrix(t, a, tmask, v, b, mode="max"):
    t, v = F.normalize(t, dim=-1), F.normalize(v, dim=-1)
    if mode == "sum":
        return (a.unsqueeze(-1) * t).sum(1) @ (b.unsqueeze(-1) * v).sum(1).T
    S = torch.einsum("ild,jmd->ijlm", t, v)
    t2v = (S.max(-1).values * a.unsqueeze(1)).sum(-1)
    S_m = S.masked_fill(~tmask[:, None, :, None], -1e4)
    v2t = (S_m.max(-2).values * b.unsqueeze(0)).sum(-1)
    return 0.5 * (t2v + v2t)

class LearnableTau(nn.Module):
    """tau strictly clamped in [0.05, 0.50] as required by Change 5."""
    def __init__(self, tau=0.2, tmin=0.05, tmax=0.50):
        super().__init__()
        self.tmin, self.tmax = tmin, tmax
        x = min(max((tau - tmin) / (tmax - tmin), 0.01), 0.99)
        self.theta = nn.Parameter(torch.tensor(math.log(x / (1 - x))))

    def forward(self):
        return self.tmin + (self.tmax - self.tmin) * torch.sigmoid(self.theta)

class FGLoss(nn.Module):
    def __init__(self, tau=0.2, tmin=0.05, tmax=0.50):
        super().__init__()
        self.tau = LearnableTau(tau, tmin=tmin, tmax=tmax)

    def forward(self, S):
        logits = S / self.tau()
        y = torch.arange(S.size(0), device=S.device)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))

class CFCAIEnhanced(nn.Module):
    def __init__(
        self,
        d_text=1024,
        d_img=1024,
        d_clip_cls=1024,
        d_roberta=768,
        d_vit=768,
        d=128,
        heads=4,
        drop=0.3,
        use_multiscale=True,      # Change 1
        use_ensemble=True,        # Change 2
        use_importance=True,
        use_fg=True,
        align_mode="max",
        word_mask_p=0.10,         # Change 4
        patch_mask_k=7,           # Change 4 (7 out of 49)
    ):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_ensemble = use_ensemble
        self.use_importance = use_importance
        self.use_fg = use_fg
        self.align_mode = align_mode
        self.word_mask_p = word_mask_p
        self.patch_mask_k = patch_mask_k

        # Bottlenecks for token sequences
        br = lambda din: nn.Sequential(nn.Linear(din, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, d))
        self.proj_t = br(d_text)
        self.proj_i = br(d_img)
        self.imp_t, self.imp_i = Importance(d), Importance(d)
        self.ca_t2i = MHGatedCrossAttn(d, heads, drop=drop)
        self.ca_i2t = MHGatedCrossAttn(d, heads, drop=drop)

        # Change 1: Multi-scale global CLS projections
        if use_multiscale:
            self.proj_t_cls = nn.Sequential(nn.LayerNorm(d_clip_cls), nn.Linear(d_clip_cls, d), nn.ReLU(), nn.Dropout(drop))
            self.proj_i_cls = nn.Sequential(nn.LayerNorm(d_clip_cls), nn.Linear(d_clip_cls, d), nn.ReLU(), nn.Dropout(drop))
            dim_t = 2 * d
            dim_v = 2 * d
        else:
            dim_t = d
            dim_v = d

        # Change 2: Deep Feature Ensembling with RoBERTa & ViT
        if use_ensemble:
            self.proj_roberta = nn.Sequential(nn.LayerNorm(d_roberta), nn.Linear(d_roberta, d), nn.ReLU(), nn.Dropout(drop))
            self.proj_vit = nn.Sequential(nn.LayerNorm(d_vit), nn.Linear(d_vit, d), nn.ReLU(), nn.Dropout(drop))
            dim_ens = 2 * d
        else:
            dim_ens = 0

        # Contrastive sample projection head
        self.sample_proj = nn.Sequential(nn.Linear(dim_t + dim_v, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

        # Final Classifier
        in_dim = dim_t + dim_v + dim_ens + (1 if use_fg else 0)
        self.clf = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 2)
        )

    def forward(self, tt, tm, it, clip_t_cls=None, clip_i_cls=None, roberta_cls=None, vit_cls=None):
        B = tt.size(0)

        # Change 4: Multimodal Augmentation during training
        if self.training:
            if self.word_mask_p > 0:
                drop_mask = (torch.rand(tt.shape[:2], device=tt.device) >= self.word_mask_p).unsqueeze(-1)
                tt = tt * drop_mask
            if self.patch_mask_k > 0:
                # Clone it to avoid modifying the in-memory tensor in place
                it = it.clone()
                for b in range(B):
                    perm = torch.randperm(it.size(1), device=it.device)[:self.patch_mask_k]
                    it[b, perm] = 0.0

        Ht, Hi = self.proj_t(tt), self.proj_i(it)
        im = torch.ones(Hi.shape[:2], dtype=torch.bool, device=Hi.device)

        if self.use_importance:
            a, b = self.imp_t(Ht, tm), self.imp_i(Hi, im)
        else:
            a = tm.float() / tm.sum(1, keepdim=True).clamp_min(1.0)
            b = torch.full(Hi.shape[:2], 1.0 / Hi.size(1), device=Hi.device)

        Tt, gT, _ = self.ca_t2i(Ht, Hi, im, b)
        Ii, gI, _ = self.ca_i2t(Hi, Ht, tm, a)

        T_att = (a.unsqueeze(-1) * Tt).sum(1)
        I_att = (b.unsqueeze(-1) * Ii).sum(1)

        # Change 1: Multi-scale pooling
        if self.use_multiscale and clip_t_cls is not None and clip_i_cls is not None:
            t_cls_p = self.proj_t_cls(clip_t_cls)
            i_cls_p = self.proj_i_cls(clip_i_cls)
            t_final = torch.cat([t_cls_p, T_att], dim=-1)
            v_final = torch.cat([i_cls_p, I_att], dim=-1)
        else:
            t_final = T_att
            v_final = I_att

        S = align_matrix(Ht, a, tm, Hi, b, self.align_mode)

        feats = [t_final, v_final]

        # Change 2: Deep Feature Ensembling
        if self.use_ensemble and roberta_cls is not None and vit_cls is not None:
            rob_p = self.proj_roberta(roberta_cls)
            vit_p = self.proj_vit(vit_cls)
            feats.extend([rob_p, vit_p])

        if self.use_fg:
            feats.append(S.diag().unsqueeze(1))

        fused = torch.cat(feats, dim=-1)
        logits = self.clf(fused)
        z_sample = self.sample_proj(torch.cat([t_final, v_final], dim=-1))

        return dict(
            logits=logits,
            z_sample=z_sample,
            z_text=t_final,
            z_image=v_final,
            S=S,
            a=a,
            b=b,
            gT=gT,
            gI=gI
        )

