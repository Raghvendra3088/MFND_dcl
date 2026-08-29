"""
train.py
========
Trains MFND-DCL with the total loss of Eq. 12:

    L_total = L_CE + lambda_sample * L_sample + lambda_modal * L_modal

using the exact recipe from Section 4.4:
  - Adam optimizer, gradient clipping
  - max 100 epochs, early stopping on validation F1
  - Weibo: batch_size=128, lr=1e-4, dropout=0.3
  - PHEME: batch_size=256, lr=6e-5, class-weighted CE with weights [1.0, 2.4]
  - global seed = 129

Usage:
    python train.py --dataset weibo \
        --train_dir data/weibo/features/train \
        --val_dir   data/weibo/features/val \
        --test_dir  data/weibo/features/test

Note: the paper's Weibo split (Sec 4.1.1) is a single 80/20 train/test
split with no explicit validation set; if you don't have a separate
val split, carve one out of the training set (e.g. 10%) purely for the
early-stopping criterion, since the paper trains "until the validation
F1-score ceases to improve" - do NOT use the test set for this.
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import get_config, set_seed
from dataset import MFNDFeatureDataset, collate_fn
from model import MFND_DCL
from dcl_losses import SampleContrastiveLoss, ModalContrastiveLoss


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_feat, image_feat, labels in loader:
            text_feat, image_feat = text_feat.to(device), image_feat.to(device)
            out = model(text_feat, image_feat)
            preds = out["logits"].argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)
    macro_f1 = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)[2]
    return {
        "accuracy": acc,
        "precision_real": precision[0], "recall_real": recall[0], "f1_real": f1[0],
        "precision_fake": precision[1], "recall_fake": recall[1], "f1_fake": f1[1],
        "macro_f1": macro_f1,
    }


def train(args):
    cfg = get_config(args.dataset)
    set_seed()
    device = cfg.device

    train_ds = MFNDFeatureDataset(args.train_dir)
    val_ds = MFNDFeatureDataset(args.val_dir)
    test_ds = MFNDFeatureDataset(args.test_dir)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn)

    model = MFND_DCL(
        d_model=cfg.d_model, d_k=cfg.d_k, proj_dim=cfg.proj_dim,
        classifier_dropout=cfg.classifier_dropout,
    ).to(device)

    if cfg.class_weights is not None:
        weight = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    else:
        weight = None
    ce_loss_fn = nn.CrossEntropyLoss(weight=weight)
    sample_loss_fn = SampleContrastiveLoss(temperature=cfg.tau)
    modal_loss_fn = ModalContrastiveLoss(temperature=cfg.tau)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for text_feat, image_feat, labels in train_loader:
            text_feat = text_feat.to(device)
            image_feat = image_feat.to(device)
            labels = labels.to(device)

            out = model(text_feat, image_feat)

            l_ce = ce_loss_fn(out["logits"], labels)
            l_sample = sample_loss_fn(out["z_sample"], labels)
            l_modal = modal_loss_fn(out["z_text"], out["z_image"])

            # Eq. 12
            loss = l_ce + cfg.lambda_sample * l_sample + cfg.lambda_modal * l_modal

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item() * text_feat.size(0)

        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macroF1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val F1 improvement for {cfg.early_stopping_patience} epochs).")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Final Test Results ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"\nSaved best model to {args.save_path}")
    return test_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["weibo", "pheme"])
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--save_path", default="checkpoints/mfnd_dcl_best.pt")
    args = ap.parse_args()
    train(args)
