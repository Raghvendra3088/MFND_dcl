"""
ablation.py
===========
Reproduces the three variants in Table 5 (Section 5.2):

  1. w/o GCA & DCL : CLIP features -> simple pooling/concatenation -> classifier
                      (no GCA module, no contrastive losses; lambda_sample =
                      lambda_modal = 0)
  2. w/o DCL       : adds the GCA module on top of variant 1, but still
                      trains with lambda_sample = lambda_modal = 0
                      (classification loss only)
  3. Ours (full)    : GCA + both DCL losses (this is exactly train.py)

Usage:
    python ablation.py --dataset weibo \
        --train_dir data/weibo/features/train \
        --val_dir   data/weibo/features/val \
        --test_dir  data/weibo/features/test
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config, set_seed
from dataset import MFNDFeatureDataset, collate_fn
from model import MFND_DCL
from dcl_losses import SampleContrastiveLoss, ModalContrastiveLoss
from train import evaluate


class SimpleConcatBaseline(nn.Module):
    """Variant 1: 'w/o GCA DCL'. No cross-attention, no gating -- just
    concatenate the raw pooled CLIP text/image features and classify."""

    def __init__(self, d_model=512, classifier_dropout=0.3, num_classes=2):
        super().__init__()
        fused_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, text_feat, image_feat):
        fused = torch.cat([text_feat, image_feat], dim=-1)
        logits = self.classifier(fused)
        return {"logits": logits, "z_sample": None, "z_text": None,
                "z_image": None, "fused": fused}


def run_variant(variant, cfg, train_loader, val_loader, test_loader, device):
    set_seed()

    if variant == "wo_gca_dcl":
        model = SimpleConcatBaseline(
            d_model=cfg.d_model, classifier_dropout=cfg.classifier_dropout).to(device)
        use_gca, use_dcl = False, False
    elif variant == "wo_dcl":
        model = MFND_DCL(
            d_model=cfg.d_model, d_k=cfg.d_k, proj_dim=cfg.proj_dim,
            classifier_dropout=cfg.classifier_dropout).to(device)
        use_gca, use_dcl = True, False
    elif variant == "full":
        model = MFND_DCL(
            d_model=cfg.d_model, d_k=cfg.d_k, proj_dim=cfg.proj_dim,
            classifier_dropout=cfg.classifier_dropout).to(device)
        use_gca, use_dcl = True, True
    else:
        raise ValueError(variant)

    weight = (torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
              if cfg.class_weights is not None else None)
    ce_loss_fn = nn.CrossEntropyLoss(weight=weight)
    sample_loss_fn = SampleContrastiveLoss(temperature=cfg.tau)
    modal_loss_fn = ModalContrastiveLoss(temperature=cfg.tau)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for text_feat, image_feat, labels in train_loader:
            text_feat, image_feat, labels = (
                text_feat.to(device), image_feat.to(device), labels.to(device))
            out = model(text_feat, image_feat)
            loss = ce_loss_fn(out["logits"], labels)
            if use_dcl:
                loss = loss + cfg.lambda_sample * sample_loss_fn(out["z_sample"], labels)
                loss = loss + cfg.lambda_modal * modal_loss_fn(out["z_text"], out["z_image"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_state, patience = val_metrics["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n=== Variant: {variant} ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    return test_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["weibo", "pheme"])
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    args = ap.parse_args()

    cfg = get_config(args.dataset)
    device = cfg.device

    train_ds = MFNDFeatureDataset(args.train_dir)
    val_ds = MFNDFeatureDataset(args.val_dir)
    test_ds = MFNDFeatureDataset(args.test_dir)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    results = {}
    for variant in ["wo_gca_dcl", "wo_dcl", "full"]:
        results[variant] = run_variant(variant, cfg, train_loader, val_loader, test_loader, device)

    print("\n=== Summary (compare against paper's Table 5) ===")
    for variant, m in results.items():
        print(f"{variant:12s}  acc={m['accuracy']:.4f}  "
              f"fake_f1={m['f1_fake']:.4f}  real_f1={m['f1_real']:.4f}")
