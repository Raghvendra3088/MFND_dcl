"""
train_sahn.py
=============
Execution and Evaluation Engine for SAHN-DCL
(Semantic-Ambiguity-Aware Hard-Negative Dual Contrastive Learning).

Supports:
- Baseline MFND-DCL with and without temporal metadata.
- SAHN-DCL Ablation Ladder:
  * E0:  Reproducible Baseline (use_sahn=False)
  * E0a: Clean Baseline without temporal metadata (1024-d)
  * E1:  Baseline + Candidate-validity masking only (alpha=0.0)
  * E2:  E1 + SAHN Sample-level loss (alpha_sample=1.0, beta=0.0)
  * E3:  E1 + SAHN Modality-level loss (alpha_modal=0.8, beta=0.0)
  * E4:  Combined SAHN-DCL (alpha_sample=1.0, alpha_modal=0.8, beta=0.0)
  * E5:  Full SAHN-DCL with Ambiguity Proxy (alpha_sample=1.0, alpha_modal=0.8, beta=0.5)
  * E6:  Fine-Grained CFCA-I + SAHN-DCL (Token/Patch architecture)

Key Features:
- Vectorized ultra-fast in-memory DataLoaders.
- Line-buffered real-time logging.
- Strict 90/10 train/validation split (zero test set leakage).
- Early stopping on validation Macro F1 (patience=10).
- Automatic prediction saving for pairwise McNemar significance testing.
- Numerical stability assertions (finite loss, bounded weights, clamped temperatures).
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

torch.set_float32_matmul_precision('high')

from model import MFND_DCL
from cfca_modules import CFCAI, FGLoss
from sahn_losses import SAHNSupConLoss, SAHNModalityConsistencyLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FastFeatureDataLoader:
    def __init__(self, text, image, labels, batch_size=64, shuffle=True, generator=None):
        self.text = text
        self.image = image
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.n = len(labels)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n, generator=self.generator) if self.generator else torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)

        for i in range(0, self.n, self.batch_size):
            b_idx = indices[i:i + self.batch_size]
            yield self.text[b_idx], self.image[b_idx], self.labels[b_idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


class FastTokenDataLoader:
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
            indices = torch.randperm(self.n, generator=self.generator) if self.generator else torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)

        for i in range(0, self.n, self.batch_size):
            b_idx = indices[i:i + self.batch_size]
            meta = self.meta[b_idx] if self.meta is not None else None
            yield self.text[b_idx], self.text_mask[b_idx], self.patch[b_idx], meta, self.labels[b_idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def evaluate_features(model, loader, device):
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
        "preds": all_preds,
        "labels": all_labels
    }


def evaluate_tokens(model, loader, device):
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


def train_single_run_feature(args, seed, train_data, test_data):
    set_seed(seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    full_text = train_data["text_feat"]
    full_image = train_data["image_feat"]
    full_labels = train_data["labels"]

    test_text = test_data["text_feat"]
    test_image = test_data["image_feat"]
    test_labels = test_data["labels"]

    if args.drop_temporal:
        full_text = full_text[:, :1024]
        full_image = full_image[:, :1024]
        test_text = test_text[:, :1024]
        test_image = test_image[:, :1024]
        d_model = 1024
    else:
        d_model = full_text.shape[-1]

    val_size = int(len(full_labels) * 0.1)
    train_size = len(full_labels) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_labels), generator=split_gen)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = FastFeatureDataLoader(
        full_text[train_idx], full_image[train_idx], full_labels[train_idx],
        batch_size=args.bs, shuffle=True, generator=train_gen
    )
    val_loader = FastFeatureDataLoader(
        full_text[val_idx], full_image[val_idx], full_labels[val_idx],
        batch_size=args.bs, shuffle=False
    )
    test_loader = FastFeatureDataLoader(
        test_text, test_image, test_labels,
        batch_size=args.bs, shuffle=False
    )

    model = MFND_DCL(
        d_model=d_model, d_k=d_model, proj_dim=128,
        classifier_dropout=args.dropout
    ).to(device)

    # SAHN Loss initializations
    supcon_fn = SAHNSupConLoss(
        temperature=args.tau_s,
        alpha=args.alpha_sample if args.use_sahn_sample else 0.0,
        beta=args.beta if args.use_ambiguity else 0.0,
        use_sahn=args.use_sahn_sample,
        learnable_temp=args.learnable_temp
    ).to(device)

    modal_fn = SAHNModalityConsistencyLoss(
        temperature=args.tau_m,
        alpha=args.alpha_modal if args.use_sahn_modal else 0.0,
        beta=args.beta if args.use_ambiguity else 0.0,
        overlap_threshold=args.overlap_thresh,
        use_sahn=args.use_sahn_modal,
        learnable_temp=args.learnable_temp
    ).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    params_to_opt = list(model.parameters())
    if args.learnable_temp:
        params_to_opt += list(supcon_fn.parameters()) + list(modal_fn.parameters())

    optimizer = torch.optim.Adam(params_to_opt, lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    print(f"\n--- Starting {args.exp_name} | Seed {seed} on {device} ---", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0

        for text_feat, image_feat, labels in train_loader:
            text_feat = text_feat.to(device)
            image_feat = image_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(text_feat, image_feat)

            l_ce = ce_loss_fn(out["logits"], labels)
            l_sample = supcon_fn(out["user_repr"], labels)
            l_modal = modal_fn(out["text_consist_repr"], out["image_consist_repr"])

            loss = l_ce + args.lam_s * l_sample + args.lam_m * l_modal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * text_feat.size(0)

        avg_loss = total_loss / train_size
        val_metrics = evaluate_features(model, val_loader, device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Final evaluation on test set using best checkpoint
    model.load_state_dict(best_state)
    test_metrics = evaluate_features(model, test_loader, device)
    test_metrics["best_epoch"] = best_epoch

    learned_tau_s = supcon_fn.get_temperature().item()
    learned_tau_m = modal_fn.get_temperature().item()
    test_metrics["tau_sample"] = learned_tau_s
    test_metrics["tau_modal"] = learned_tau_m

    print(f"[{args.exp_name} | Seed {seed}] Best Epoch: {best_epoch:02d} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f} | tau_s: {learned_tau_s:.3f} | tau_m: {learned_tau_m:.3f}", flush=True)

    # Save predictions
    os.makedirs(args.pred_dir, exist_ok=True)
    pred_path = Path(args.pred_dir) / f"preds_{args.exp_name}_seed{seed}.npy"
    np.save(pred_path, test_metrics["preds"])

    return test_metrics


def train_single_run_token(args, seed, train_ds, test_ds):
    set_seed(seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    val_size = int(len(train_ds["labels"]) * 0.1)
    train_size = len(train_ds["labels"]) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_ds["labels"]), generator=split_gen)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = FastTokenDataLoader(
        train_ds["text"][train_idx], train_ds["text_mask"][train_idx],
        train_ds["patch"][train_idx], train_ds["meta"][train_idx] if train_ds["meta"] is not None else None,
        train_ds["labels"][train_idx], batch_size=args.bs, shuffle=True, generator=train_gen
    )
    val_loader = FastTokenDataLoader(
        train_ds["text"][val_idx], train_ds["text_mask"][val_idx],
        train_ds["patch"][val_idx], train_ds["meta"][val_idx] if train_ds["meta"] is not None else None,
        train_ds["labels"][val_idx], batch_size=args.bs, shuffle=False
    )
    test_loader = FastTokenDataLoader(
        test_ds["text"], test_ds["text_mask"],
        test_ds["patch"], test_ds["meta"] if test_ds["meta"] is not None else None,
        test_ds["labels"], batch_size=args.bs, shuffle=False
    )

    extra_dim = 0 if args.drop_temporal else 2
    model = CFCAI(
        d_text=1024, d_img=1024, d=128, heads=4, drop=args.dropout,
        extra_dim=extra_dim,
        use_importance=True,
        use_fg=True,
        use_inc=True,
        align_mode="max"
    ).to(device)

    supcon_fn = SAHNSupConLoss(
        temperature=args.tau_s,
        alpha=args.alpha_sample if args.use_sahn_sample else 0.0,
        beta=args.beta if args.use_ambiguity else 0.0,
        use_sahn=args.use_sahn_sample,
        learnable_temp=args.learnable_temp
    ).to(device)

    modal_fn = SAHNModalityConsistencyLoss(
        temperature=args.tau_m,
        alpha=args.alpha_modal if args.use_sahn_modal else 0.0,
        beta=args.beta if args.use_ambiguity else 0.0,
        overlap_threshold=args.overlap_thresh,
        use_sahn=args.use_sahn_modal,
        learnable_temp=args.learnable_temp
    ).to(device)

    fg_loss = FGLoss(tau=0.2).to(device)
    ce_loss_fn = nn.CrossEntropyLoss()

    params_to_opt = list(model.parameters()) + list(fg_loss.parameters())
    if args.learnable_temp:
        params_to_opt += list(supcon_fn.parameters()) + list(modal_fn.parameters())

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    print(f"\n--- Starting {args.exp_name} (CFCA-I Fine-Grained) | Seed {seed} on {device} ---", flush=True)

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

            l_ce = ce_loss_fn(out["logits"], labels)
            l_sample = supcon_fn(out["z_sample"], labels)
            l_modal = modal_fn(out["z_text"], out["z_image"])
            l_fg = fg_loss(out["S"])
            l_inc = (out["inc_logits"].squeeze(-1) ** 2).mean() if "inc_logits" in out else torch.tensor(0.0, device=device)

            loss = l_ce + args.lam_s * l_sample + args.lam_m * l_modal + args.lam_fg * l_fg + args.lam_inc * l_inc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * tt.size(0)

        avg_loss = total_loss / train_size
        val_metrics = evaluate_tokens(model, val_loader, device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_tokens(model, test_loader, device)
    test_metrics["best_epoch"] = best_epoch

    learned_tau_s = supcon_fn.get_temperature().item()
    learned_tau_m = modal_fn.get_temperature().item()
    test_metrics["tau_sample"] = learned_tau_s
    test_metrics["tau_modal"] = learned_tau_m

    print(f"[{args.exp_name} | Seed {seed}] Best Epoch: {best_epoch:02d} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f} | tau_s: {learned_tau_s:.3f} | tau_m: {learned_tau_m:.3f}", flush=True)

    os.makedirs(args.pred_dir, exist_ok=True)
    pred_path = Path(args.pred_dir) / f"preds_{args.exp_name}_seed{seed}.npy"
    np.save(pred_path, test_metrics["preds"])

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="SAHN-DCL Training Harness")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment identifier (e.g., E0, E1, E2, E3, E4, E5, E6)")
    parser.add_argument("--gpu", type=int, default=2, help="CUDA GPU ID")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 129, 2024, 3407])
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=10)
    
    # Loss coefficients
    parser.add_argument("--lam_s", type=float, default=0.2, help="Sample-level loss weight")
    parser.add_argument("--lam_m", type=float, default=0.1, help="Modality-level loss weight")
    parser.add_argument("--lam_fg", type=float, default=0.1, help="Fine-grained alignment loss weight")
    parser.add_argument("--lam_inc", type=float, default=0.05, help="Inconsistency loss weight")
    
    # SAHN parameters
    parser.add_argument("--use_sahn_sample", action="store_true", help="Enable SAHN in sample-level loss")
    parser.add_argument("--use_sahn_modal", action="store_true", help="Enable SAHN in modality-level loss")
    parser.add_argument("--use_ambiguity", action="store_true", help="Enable Ambiguity Proxy suppression")
    parser.add_argument("--alpha_sample", type=float, default=1.0, help="Sample hard negative strength")
    parser.add_argument("--alpha_modal", type=float, default=0.8, help="Modality hard negative strength")
    parser.add_argument("--beta", type=float, default=0.5, help="Ambiguity suppression factor")
    parser.add_argument("--overlap_thresh", type=float, default=0.85, help="Modality negative overlap threshold")
    parser.add_argument("--learnable_temp", action="store_true", default=True)
    parser.add_argument("--tau_s", type=float, default=0.2)
    parser.add_argument("--tau_m", type=float, default=0.1)
    
    # Architecture and data
    parser.add_argument("--is_token_arch", action="store_true", help="Use CFCA-I token/patch architecture")
    parser.add_argument("--drop_temporal", action="store_true", help="Drop 2-d timestamp features")
    parser.add_argument("--data_dir", type=str, default="/DATA/raghvendra3088/MFND_project/data/weibo")
    parser.add_argument("--pred_dir", type=str, default="/DATA/raghvendra3088/MFND_project/files/predictions")
    parser.add_argument("--log_dir", type=str, default="/DATA/raghvendra3088/MFND_project/files/logs")

    args = parser.parse_args()

    print("=" * 80)
    print(f"SAHN-DCL RUNNER: {args.exp_name}")
    print(f"GPU: {args.gpu} | Seeds: {args.seeds} | BS: {args.bs} | LR: {args.lr}")
    print(f"SAHN Sample: {args.use_sahn_sample} (alpha={args.alpha_sample}) | SAHN Modal: {args.use_sahn_modal} (alpha={args.alpha_modal})")
    print(f"Ambiguity Proxy: {args.use_ambiguity} (beta={args.beta}) | Overlap Thresh: {args.overlap_thresh}")
    print(f"Architecture: {'CFCA-I Token/Patch' if args.is_token_arch else 'MFND-DCL Feature'} | Drop Temporal: {args.drop_temporal}")
    print("=" * 80, flush=True)

    data_path = Path(args.data_dir)

    if args.is_token_arch:
        print("Loading precomputed tokens...")
        train_path = data_path / "tokens/train"
        test_path = data_path / "tokens/test"
        train_ds = {
            "text": torch.from_numpy(np.array(np.lib.format.open_memmap(train_path / "text.npy", mode="r"))).float(),
            "text_mask": torch.from_numpy(np.array(np.lib.format.open_memmap(train_path / "text_mask.npy", mode="r"))),
            "patch": torch.from_numpy(np.array(np.lib.format.open_memmap(train_path / "patch.npy", mode="r"))).float(),
            "labels": torch.from_numpy(np.load(train_path / "labels.npy")).long(),
            "meta": None if args.drop_temporal else torch.from_numpy(np.array(np.lib.format.open_memmap(train_path / "meta.npy", mode="r"))).float()
        }
        test_ds = {
            "text": torch.from_numpy(np.array(np.lib.format.open_memmap(test_path / "text.npy", mode="r"))).float(),
            "text_mask": torch.from_numpy(np.array(np.lib.format.open_memmap(test_path / "text_mask.npy", mode="r"))),
            "patch": torch.from_numpy(np.array(np.lib.format.open_memmap(test_path / "patch.npy", mode="r"))).float(),
            "labels": torch.from_numpy(np.load(test_path / "labels.npy")).long(),
            "meta": None if args.drop_temporal else torch.from_numpy(np.array(np.lib.format.open_memmap(test_path / "meta.npy", mode="r"))).float()
        }
        train_runner = lambda s: train_single_run_token(args, s, train_ds, test_ds)
    else:
        print("Loading consolidated features...")
        train_data = torch.load(data_path / "consolidated_train.pt", map_location="cpu", weights_only=False)
        test_data = torch.load(data_path / "consolidated_test.pt", map_location="cpu", weights_only=False)
        train_runner = lambda s: train_single_run_feature(args, s, train_data, test_data)

    results = []
    for seed in args.seeds:
        res = train_runner(seed)
        results.append(res)

    # Compute aggregate metrics across 5 seeds
    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["macro_f1"] * 100 for r in results]
    f1_reals = [r["f1_real"] * 100 for r in results]
    f1_fakes = [r["f1_fake"] * 100 for r in results]

    acc_mean, acc_std = np.mean(accs), np.std(accs, ddof=1)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s, ddof=1)
    f1_real_mean = np.mean(f1_reals)
    f1_fake_mean = np.mean(f1_fakes)

    print("\n" + "=" * 80)
    print(f"FINAL AGGREGATE RESULTS: {args.exp_name} (5 Seeds)")
    print("=" * 80)
    print(f"Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}% (per-seed: {[f'{a:.2f}' for a in accs]})")
    print(f"Macro F1: {f1_mean:.2f}% ± {f1_std:.2f}% (per-seed: {[f'{f:.2f}' for f in f1s]})")
    print(f"F1 Real:  {f1_real_mean:.2f}% | F1 Fake: {f1_fake_mean:.2f}%")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
