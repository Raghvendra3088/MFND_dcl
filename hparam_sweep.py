"""
hparam_sweep.py
================
Reproduces Table 6 (temperature sweep, Sec 5.3) and Table 7 (learning-rate
sweep, Sec 5.4), both run on the Weibo dataset in the paper.

Usage:
    python hparam_sweep.py --sweep temperature --train_dir ... --val_dir ... --test_dir ...
    python hparam_sweep.py --sweep lr           --train_dir ... --val_dir ... --test_dir ...
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

TEMPERATURE_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # Table 6
LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]                              # Table 7


def run_once(cfg, tau, lr, train_loader, val_loader, test_loader, device):
    set_seed()
    model = MFND_DCL(d_model=cfg.d_model, d_k=cfg.d_k, proj_dim=cfg.proj_dim,
                      classifier_dropout=cfg.classifier_dropout).to(device)
    ce_loss_fn = nn.CrossEntropyLoss()
    sample_loss_fn = SampleContrastiveLoss(temperature=tau)
    modal_loss_fn = ModalContrastiveLoss(temperature=tau)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for text_feat, image_feat, labels in train_loader:
            text_feat, image_feat, labels = (
                text_feat.to(device), image_feat.to(device), labels.to(device))
            out = model(text_feat, image_feat)
            loss = (ce_loss_fn(out["logits"], labels)
                    + cfg.lambda_sample * sample_loss_fn(out["z_sample"], labels)
                    + cfg.lambda_modal * modal_loss_fn(out["z_text"], out["z_image"]))
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
    return evaluate(model, test_loader, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["temperature", "lr"], required=True)
    ap.add_argument("--dataset", default="weibo")  # paper runs this sweep on Weibo
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

    print(f"{'value':>10s} {'accuracy':>10s} {'precision':>10s} {'recall':>10s} {'f1(macro)':>10s}")
    if args.sweep == "temperature":
        for tau in TEMPERATURE_GRID:
            m = run_once(cfg, tau=tau, lr=cfg.lr,
                         train_loader=train_loader, val_loader=val_loader,
                         test_loader=test_loader, device=device)
            print(f"{tau:>10.2f} {m['accuracy']:>10.4f} "
                  f"{(m['precision_fake']+m['precision_real'])/2:>10.4f} "
                  f"{(m['recall_fake']+m['recall_real'])/2:>10.4f} {m['macro_f1']:>10.4f}")
    else:
        for lr in LR_GRID:
            m = run_once(cfg, tau=cfg.tau, lr=lr,
                         train_loader=train_loader, val_loader=val_loader,
                         test_loader=test_loader, device=device)
            print(f"{lr:>10.0e} {m['accuracy']:>10.4f} "
                  f"{(m['precision_fake']+m['precision_real'])/2:>10.4f} "
                  f"{(m['recall_fake']+m['recall_real'])/2:>10.4f} {m['macro_f1']:>10.4f}")
