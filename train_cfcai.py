"""
train_cfcai.py
==============
Rigorous Training & Evaluation Pipeline for CFCA-I (Claim-aware Fine-grained Cross-modal Alignment + Inconsistency).
- Ultra-fast vectorized in-memory data loader (~1s per epoch loading overhead).
- Line-buffered real-time stdout logging.
- Rigorous seed-controlled 90/10 train/val split (zero test leakage).
- Early stopping on validation Macro F1 (patience=10).
- Dynamic temperature logging (tau_s, tau_m, tau_fg).
- Saves test predictions per seed (preds_{model}_seed{seed}.npy).
- Ablation ladder models: M2, M3, M4, M5, M6, M5a.
"""

import os
import sys
import copy
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure line-buffered output to log files
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

torch.set_float32_matmul_precision('high')

from cfca_modules import CFCAI, FGLoss, cfcai_total_loss
from dcl_losses import SupConLoss, MultimodalConsistencyLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFCATokenDataset:
    def __init__(self, data_dir, drop_temporal=False):
        data_path = Path(data_dir)
        print(f"Loading token dataset from {data_path} into memory...")
        self.text = torch.from_numpy(np.array(np.lib.format.open_memmap(data_path / "text.npy", mode="r"))).float()
        self.text_mask = torch.from_numpy(np.array(np.lib.format.open_memmap(data_path / "text_mask.npy", mode="r")))
        self.patch = torch.from_numpy(np.array(np.lib.format.open_memmap(data_path / "patch.npy", mode="r"))).float()
        self.labels = torch.from_numpy(np.load(data_path / "labels.npy")).long()
        self.drop_temporal = drop_temporal
        if drop_temporal:
            self.meta = None
        else:
            self.meta = torch.from_numpy(np.array(np.lib.format.open_memmap(data_path / "meta.npy", mode="r"))).float()
        print(f"Loaded {len(self.labels)} samples into memory successfully.")

    def __len__(self):
        return len(self.labels)


class FastDataLoader:
    def __init__(self, text, text_mask, patch, meta, labels, batch_size=64, shuffle=True, generator=None):
        self.text = text
        self.text_mask = text_mask
        self.patch = patch
        self.meta = meta
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.n = len(labels)

    def __iter__(self):
        if self.shuffle:
            if self.generator is not None:
                indices = torch.randperm(self.n, generator=self.generator)
            else:
                indices = torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)
            
        for i in range(0, self.n, self.batch_size):
            b_idx = indices[i:i + self.batch_size]
            tt = self.text[b_idx]
            tm = self.text_mask[b_idx]
            it = self.patch[b_idx]
            meta = self.meta[b_idx] if self.meta is not None else None
            lbl = self.labels[b_idx]
            yield tt, tm, it, meta, lbl

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tt, tm, it, meta, labels in loader:
            tt, tm, it = tt.to(device), tm.to(device), it.to(device)
            if meta is not None:
                meta = meta.to(device)
            out = model(tt, tm, it, meta)
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
        "preds": all_preds,
        "labels": all_labels
    }


def get_model_config(model_name):
    configs = {
        "M2": dict(use_importance=False, use_fg=False, use_inc=False, align_mode="max", drop_temporal=False, extra_dim=2),
        "M3": dict(use_importance=True, use_fg=False, use_inc=False, align_mode="max", drop_temporal=False, extra_dim=2),
        "M4": dict(use_importance=True, use_fg=True, use_inc=False, align_mode="max", drop_temporal=False, extra_dim=2),
        "M5": dict(use_importance=True, use_fg=True, use_inc=True, align_mode="max", drop_temporal=False, extra_dim=2),
        "M6": dict(use_importance=True, use_fg=True, use_inc=True, align_mode="sum", drop_temporal=False, extra_dim=2),
        "M5a": dict(use_importance=True, use_fg=True, use_inc=True, align_mode="max", drop_temporal=True, extra_dim=0),
    }
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Supported: {list(configs.keys())}")
    return configs[model_name]


def train_single_run(args, seed, full_train_ds, test_ds):
    set_seed(seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    cfg = get_model_config(args.model)

    val_size = int(len(full_train_ds) * 0.1)
    train_size = len(full_train_ds) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train_ds), generator=split_gen)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_text = full_train_ds.text[train_idx]
    train_tm = full_train_ds.text_mask[train_idx]
    train_patch = full_train_ds.patch[train_idx]
    train_meta = full_train_ds.meta[train_idx] if full_train_ds.meta is not None else None
    train_labels = full_train_ds.labels[train_idx]

    val_text = full_train_ds.text[val_idx]
    val_tm = full_train_ds.text_mask[val_idx]
    val_patch = full_train_ds.patch[val_idx]
    val_meta = full_train_ds.meta[val_idx] if full_train_ds.meta is not None else None
    val_labels = full_train_ds.labels[val_idx]

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = FastDataLoader(train_text, train_tm, train_patch, train_meta, train_labels,
                                  batch_size=args.bs, shuffle=True, generator=train_gen)
    val_loader = FastDataLoader(val_text, val_tm, val_patch, val_meta, val_labels,
                                batch_size=args.bs, shuffle=False)
    test_loader = FastDataLoader(test_ds.text, test_ds.text_mask, test_ds.patch, test_ds.meta, test_ds.labels,
                                 batch_size=args.bs, shuffle=False)

    model = CFCAI(
        d_text=1024, d_img=1024, d=128, heads=4, drop=args.dropout,
        extra_dim=cfg["extra_dim"],
        use_importance=cfg["use_importance"],
        use_fg=cfg["use_fg"],
        use_inc=cfg["use_inc"],
        align_mode=cfg["align_mode"]
    ).to(device)

    supcon_fn = SupConLoss(temperature=0.2, learnable_temp=True).to(device)
    modal_fn = MultimodalConsistencyLoss(temperature=0.1, learnable_temp=True).to(device)
    fg_loss = FGLoss(tau=0.2).to(device)

    params_to_opt = list(model.parameters()) + list(supcon_fn.parameters()) + list(modal_fn.parameters())
    if cfg["use_fg"]:
        params_to_opt += list(fg_loss.parameters())

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    print(f"\n--- Starting {args.model} | Seed {seed} on {device} ---", flush=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0

        for tt, tm, it, meta, labels in train_loader:
            tt, tm, it = tt.to(device), tm.to(device), it.to(device)
            if meta is not None:
                meta = meta.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(tt, tm, it, meta)

            loss = cfcai_total_loss(
                model, out, (tt, tm, it), labels,
                supcon_fn, modal_fn, fg_loss,
                lam_s=args.lam_s, lam_m=args.lam_m, lam_fg=args.lam_fg, lam_inc=args.lam_inc
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * tt.size(0)

        avg_loss = total_loss / train_size
        val_metrics = evaluate(model, val_loader, device)

        tau_s = supcon_fn.get_temperature().item()
        tau_m = modal_fn.get_temperature().item()
        tau_fg = fg_loss.tau().item() if cfg["use_fg"] else 0.0

        if epoch % 5 == 0 or val_metrics["macro_f1"] > best_val_f1:
            print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | tau_s: {tau_s:.3f} | tau_m: {tau_m:.3f} | tau_fg: {tau_fg:.3f}", flush=True)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (Best: Epoch {best_epoch:02d}, Val F1: {best_val_f1:.4f})", flush=True)
                break

    # Load best validation checkpoint
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["tau_s"] = supcon_fn.get_temperature().item()
    test_metrics["tau_m"] = modal_fn.get_temperature().item()
    test_metrics["tau_fg"] = fg_loss.tau().item() if cfg["use_fg"] else 0.0

    print(f"[{args.model} | Seed {seed}] TEST ACC: {test_metrics['accuracy'] * 100:.2f}% | MACRO F1: {test_metrics['macro_f1'] * 100:.2f}% | Best Epoch: {best_epoch}", flush=True)

    # Save predictions
    out_dir = Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"preds_{args.model}_seed{seed}.npy"
    np.save(pred_path, test_metrics["preds"])
    print(f"Saved predictions to {pred_path}", flush=True)

    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["M2", "M3", "M4", "M5", "M6", "M5a"])
    ap.add_argument("--train_dir", type=str, default="../data/weibo/tokens/train")
    ap.add_argument("--test_dir", type=str, default="../data/weibo/tokens/test")
    ap.add_argument("--pred_dir", type=str, default="predictions")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 129, 2024, 3407])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lam_s", type=float, default=0.5)
    ap.add_argument("--lam_m", type=float, default=0.3)
    ap.add_argument("--lam_fg", type=float, default=0.3)
    ap.add_argument("--lam_inc", type=float, default=0.2)
    args = ap.parse_args()

    cfg = get_model_config(args.model)
    full_train_ds = CFCATokenDataset(args.train_dir, drop_temporal=cfg["drop_temporal"])
    test_ds = CFCATokenDataset(args.test_dir, drop_temporal=cfg["drop_temporal"])

    results = []
    for s in args.seeds:
        res = train_single_run(args, s, full_train_ds, test_ds)
        results.append(res)

    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["macro_f1"] * 100 for r in results]
    tau_s_list = [r["tau_s"] for r in results]
    tau_m_list = [r["tau_m"] for r in results]
    tau_fg_list = [r["tau_fg"] for r in results]

    print("\n" + "=" * 60, flush=True)
    print(f"FINAL 5-SEED RESULTS FOR MODEL {args.model}:", flush=True)
    print(f"  Accuracy:  {np.mean(accs):.2f}% ± {np.std(accs, ddof=1):.2f}%  (Seeds: {[round(x, 2) for x in accs]})", flush=True)
    print(f"  Macro F1:  {np.mean(f1s):.2f}% ± {np.std(f1s, ddof=1):.2f}%", flush=True)
    print(f"  Learned Temperatures: tau_s={np.mean(tau_s_list):.3f}, tau_m={np.mean(tau_m_list):.3f}, tau_fg={np.mean(tau_fg_list):.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
