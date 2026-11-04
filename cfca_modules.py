"""
cfca_modules.py  --  CFCA-I (Claim-aware Fine-grained Cross-modal Alignment + Inconsistency)
Built to match the V2 (GitHub-style) MFND-DCL: d_proj=128, 4 heads, dual LayerNorm,
256-d fused classifier input.  Existing SupConLoss / MultimodalConsistencyLoss from
dcl_losses.py are injected as callables (no hard dependency here).

Inputs per sample:
  text_tok  [B,L,Dt]  token features   text_mask [B,L] bool (True = real token)
  img_tok   [B,M,Di]  patch features   meta [B,2] optional (temporal dims of the baseline)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- importance
class Importance(nn.Module):
    """a_i = softmax_i( w^T tanh(W h_i + b) ), padding -> 0 weight."""
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x, mask):
        s = self.net(x).squeeze(-1).masked_fill(~mask, -1e4)
        return F.softmax(s, dim=-1)


# ------------------------------------------- multi-head gated cross-attention
class MHGatedCrossAttn(nn.Module):
    """x attends y.  Per-token gate g_i.  LayerNorm before (norm1) and after (norm2)."""
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
        logit = q @ k.transpose(-1, -2) / math.sqrt(self.dh)          # [B,h,Lx,Ly]
        if y_prior is not None:                                        # importance as prior
            logit = logit + torch.log(y_prior.clamp_min(1e-6))[:, None, None, :]
        logit = logit.masked_fill(~y_mask[:, None, None, :], -1e4)
        A = logit.softmax(-1)
        C = self.o((A @ v).transpose(1, 2).reshape(B, Lx, d))
        g = torch.sigmoid(self.gate(torch.cat([xn, C], -1)))
        out = self.n2(x + self.drop(g * C + (1 - g) * xn))
        return out, g, A.mean(1)


# ------------------------------------------------------------ alignment score
def align_matrix(t, a, tmask, v, b, mode="max"):
    """
    S[i,j] = alignment of text i with image j.
    mode="max": 0.5*( sum_l a_il * max_m cos(t_il, v_jm) + sum_m b_jm * max_l cos(t_il, v_jm) )
                -> genuinely word-to-region (each word picks its best patch).
    mode="sum": sum_l sum_m a_l b_m cos  ==  (sum a_l t_l) . (sum b_m v_m)  (bilinear!)
                -> collapses to pooled dot product; use ONLY as a control experiment.
    """
    t, v = F.normalize(t, dim=-1), F.normalize(v, dim=-1)
    if mode == "sum":
        return (a.unsqueeze(-1) * t).sum(1) @ (b.unsqueeze(-1) * v).sum(1).T
    S = torch.einsum("ild,jmd->ijlm", t, v)                            # [N,N,L,M]
    t2v = (S.max(-1).values * a.unsqueeze(1)).sum(-1)
    S_m = S.masked_fill(~tmask[:, None, :, None], -1e4)               # ignore padded words
    v2t = (S_m.max(-2).values * b.unsqueeze(0)).sum(-1)
    return 0.5 * (t2v + v2t)


class LearnableTau(nn.Module):
    """tau = tmin + (tmax-tmin)*sigmoid(theta)  (same form as dcl_losses.py)."""
    def __init__(self, tau=0.2, tmin=0.05, tmax=1.0):
        super().__init__()
        self.tmin, self.tmax = tmin, tmax
        x = min(max((tau - tmin) / (tmax - tmin), 0.01), 0.99)
        self.theta = nn.Parameter(torch.tensor(math.log(x / (1 - x))))

    def forward(self):
        return self.tmin + (self.tmax - self.tmin) * torch.sigmoid(self.theta)


class FGLoss(nn.Module):
    """Symmetric InfoNCE over the fine-grained alignment matrix S (Eq. L_FG)."""
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = LearnableTau(tau)

    def forward(self, S):
        logits = S / self.tau()
        y = torch.arange(S.size(0), device=S.device)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


# --------------------------------------------------------- inconsistency head
class InconsistencyHead(nn.Module):
    def __init__(self, d=128, p=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, p))
        self.score = nn.Linear(p, 1)

    def forward(self, t, v):
        f = self.mlp(torch.cat([(t - v).abs(), t * v], -1))
        return f, self.score(f).squeeze(-1)


# ----------------------------------------------------------------- full model
class CFCAI(nn.Module):
    def __init__(self, d_text, d_img, d=128, heads=4, drop=0.3, extra_dim=2,
                 use_importance=True, use_fg=True, use_inc=True, align_mode="max"):
        super().__init__()
        br = lambda din: nn.Sequential(nn.Linear(din, 512), nn.ReLU(), nn.Dropout(drop),
                                       nn.Linear(512, d))
        self.proj_t, self.proj_i = br(d_text), br(d_img)           # same bottleneck as V2
        self.imp_t, self.imp_i = Importance(d), Importance(d)
        self.ca_t2i, self.ca_i2t = MHGatedCrossAttn(d, heads), MHGatedCrossAttn(d, heads)
        self.inc = InconsistencyHead(d)
        self.sample_proj = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.use_importance, self.use_fg, self.use_inc = use_importance, use_fg, use_inc
        self.align_mode, self.extra_dim = align_mode, extra_dim
        in_dim = 2 * d + (32 if use_inc else 0) + (1 if use_fg else 0) + extra_dim
        self.clf = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(256, 2))

    def _pair(self, tt, tm, it):
        Ht, Hi = self.proj_t(tt), self.proj_i(it)
        im = torch.ones(Hi.shape[:2], dtype=torch.bool, device=Hi.device)
        if self.use_importance:
            a, b = self.imp_t(Ht, tm), self.imp_i(Hi, im)
        else:
            a = tm.float() / tm.sum(1, keepdim=True)
            b = torch.full(Hi.shape[:2], 1.0 / Hi.size(1), device=Hi.device)
        Tt, gT, _ = self.ca_t2i(Ht, Hi, im, b)
        Ii, gI, _ = self.ca_i2t(Hi, Ht, tm, a)
        T = (a.unsqueeze(-1) * Tt).sum(1)
        I = (b.unsqueeze(-1) * Ii).sum(1)
        return dict(Ht=Ht, Hi=Hi, a=a, b=b, T=T, I=I, gT=gT, gI=gI)

    def forward(self, tt, tm, it, meta=None):
        p = self._pair(tt, tm, it)
        feats = [p["T"], p["I"]]
        S = align_matrix(p["Ht"], p["a"], tm, p["Hi"], p["b"], self.align_mode)
        if self.use_inc:
            f_inc, s_inc = self.inc(p["T"], p["I"])
            feats.append(f_inc)
        if self.use_fg:
            feats.append(S.diag().unsqueeze(1))
        if self.extra_dim:
            feats.append(meta if meta is not None else p["T"].new_zeros(p["T"].size(0), self.extra_dim))
        return dict(logits=self.clf(torch.cat(feats, -1)),
                    z_sample=self.sample_proj(torch.cat([p["T"], p["I"]], -1)),
                    z_text=p["T"], z_image=p["I"], S=S, a=p["a"], b=p["b"],
                    gT=p["gT"], gI=p["gI"])

    def inc_loss(self, tt, tm, it, labels, real_label=0):
        """Consistent target=0 for REAL posts' own pairs; inconsistent target=1 for the same
        texts with another real post's image. Fake posts are never labelled 'consistent'."""
        real = (labels == real_label).nonzero().squeeze(-1)
        if real.numel() < 2:
            return tt.new_zeros(())
        neg = real.roll(int(torch.randint(1, real.numel(), (1,))))
        pos_p = self._pair(tt[real], tm[real], it[real])
        neg_p = self._pair(tt[real], tm[real], it[neg])
        s = torch.cat([self.inc(pos_p["T"], pos_p["I"])[1], self.inc(neg_p["T"], neg_p["I"])[1]])
        y = torch.cat([torch.zeros(real.numel()), torch.ones(real.numel())]).to(s.device)
        return F.binary_cross_entropy_with_logits(s, y)


# ------------------------------------------------------------- total loss
def cfcai_total_loss(model, out, batch, labels, supcon_fn, modal_fn, fg_loss,
                     lam_s=0.5, lam_m=0.3, lam_fg=0.3, lam_inc=0.2):
    """supcon_fn(z, labels) and modal_fn(z_t, z_v) = your dcl_losses.py criteria."""
    tt, tm, it = batch
    loss = F.cross_entropy(out["logits"], labels)
    loss = loss + lam_s * supcon_fn(out["z_sample"], labels)
    loss = loss + lam_m * modal_fn(out["z_text"], out["z_image"])
    if model.use_fg:
        loss = loss + lam_fg * fg_loss(out["S"])
    if model.use_inc:
        loss = loss + lam_inc * model.inc_loss(tt, tm, it, labels)
    return loss
