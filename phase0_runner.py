"""
phase0_runner.py
================
Rigorous Phase 0 Evaluation Harness for MFND-DCL:
1. Zero-variance debugging (seed verification, initial param checksum, batch hash).
2. Per-seed prediction saving (preds_{model}_seed{s}.npy).
3. Ablation ladder across 5 seeds [42, 123, 129, 2024, 3407]:
   - M0:  Baseline V2 (1026-d, fixed tau=0.2, no hard-negative)
   - M0a: Baseline V2 without temporal metadata (1024-d pure Chinese-CLIP)
   - M1a: Learnable tau only (1026-d, learnable tau, no hard-negative)
   - M1b: Hard-negative only (1026-d, fixed tau=0.2, hard-neg k=5, w=1.5)
   - M_novelty: Full Novelty (1026-d, learnable tau + hard-negative)
4. Pairwise McNemar tests, true mean +/- sample std, and inter-seed variance metrics.
"""

import os
import copy
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Optimize matmul
torch.set_float32_matmul_precision('high')

from config import get_config
from model import MFND_DCL
from dcl_losses import SupConLoss, MultimodalConsistencyLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MFNDFeatureDataset(Dataset):
    def __init__(self, feature_dir, drop_temporal=False, in_memory=True):
        self.files = sorted(Path(feature_dir).glob("*.pt"))
        self.drop_temporal = drop_temporal
        self.in_memory = in_memory
        if len(self.files) == 0:
            raise RuntimeError(f"No cached feature files found in {feature_dir}.")

        if in_memory:
            self.cached = []
            for f in self.files:
                rec = torch.load(f, map_location="cpu")
                t = rec["text_feat"]
                i = rec["image_feat"]
                if drop_temporal:
                    t = t[:1024]
                    i = i[:1024]
                self.cached.append((t, i, int(rec["label"])))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.cached[idx]
        rec = torch.load(self.files[idx], map_location="cpu")
        t_feat = rec["text_feat"]
        i_feat = rec["image_feat"]
        if self.drop_temporal:
            t_feat = t_feat[:1024]
            i_feat = i_feat[:1024]
        return t_feat, i_feat, rec["label"]


def collate_fn(batch):
    text_feats = torch.stack([b[0] for b in batch], dim=0)
    image_feats = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return text_feats, image_feats, labels


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
        "preds": all_preds,
        "labels": all_labels
    }


def mcnemar_test(preds_a, preds_b, targets):
    """
    McNemar's test with continuity correction.
    b: A correct, B wrong
    c: A wrong, B correct
    """
    correct_a = (preds_a == targets)
    correct_b = (preds_b == targets)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_val = 1.0 - chi2_dist.cdf(chi2, df=1)
    return chi2, p_val, b, c


def train_single_run(args, seed, exp_name):
    set_seed(seed)
    cfg = get_config(args.dataset)
    device = cfg.device

    full_train_ds = MFNDFeatureDataset(args.train_dir, drop_temporal=args.drop_temporal)
    test_ds = MFNDFeatureDataset(args.test_dir, drop_temporal=args.drop_temporal)

    val_size = int(len(full_train_ds) * 0.1)
    train_size = len(full_train_ds) - val_size

    # Rigorous seed-controlled split
    split_gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full_train_ds, [train_size, val_size], generator=split_gen)

    # Rigorous seed-controlled DataLoader
    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=False, generator=train_gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn)

    d_model = 1024 if args.drop_temporal else cfg.d_model
    model = MFND_DCL(
        d_model=d_model, d_k=d_model, proj_dim=cfg.proj_dim,
        classifier_dropout=cfg.classifier_dropout,
    ).to(device)

    # Initial parameter checksum & first batch inspection
    param_checksum = sum(p.sum().item() for p in model.parameters())
    first_batch = next(iter(train_loader))
    first_batch_label_sum = first_batch[2].sum().item()
    first_batch_feat_norm = first_batch[0].norm().item()
    print(f"[{exp_name} | Seed {seed}] Param Checksum: {param_checksum:.4f} | First Batch LabelSum: {first_batch_label_sum} | FeatNorm: {first_batch_feat_norm:.2f}")

    ce_loss_fn = nn.CrossEntropyLoss()

    # Configure DCL loss variants
    learnable_temp = (args.mode in ["learnable_tau", "novelty_both"])
    use_hard_neg = (args.mode in ["hard_neg", "novelty_both"])

    sample_loss_fn = SupConLoss(
        temperature=cfg.tau,
        hard_negative_k=cfg.hard_negative_k if use_hard_neg else 0,
        hard_negative_weight=cfg.hard_negative_weight if use_hard_neg else 1.0,
        min_temperature=cfg.min_temperature,
        max_temperature=cfg.max_temperature,
        learnable_temp=learnable_temp
    ).to(device)

    modal_loss_fn = MultimodalConsistencyLoss(
        temperature=cfg.tau / 2.0 if cfg.tau > 0.1 else 0.1,
        hard_negative_k=cfg.hard_negative_k if use_hard_neg else 0,
        hard_negative_weight=cfg.hard_negative_weight if use_hard_neg else 1.0,
        min_temperature=cfg.min_temperature,
        max_temperature=cfg.max_temperature,
        learnable_temp=learnable_temp
    ).to(device)

    params_to_opt = list(model.parameters())
    if learnable_temp:
        params_to_opt += list(sample_loss_fn.parameters()) + list(modal_loss_fn.parameters())

    optimizer = torch.optim.Adam(params_to_opt, lr=cfg.lr, weight_decay=1e-5)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for text_feat, image_feat, labels in train_loader:
            text_feat = text_feat.to(device)
            image_feat = image_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(text_feat, image_feat)

            l_ce = ce_loss_fn(out["logits"], labels)
            l_sample = sample_loss_fn(out["user_repr"], labels)
            l_modal = modal_loss_fn(out["text_consist_repr"], out["image_consist_repr"])

            loss = l_ce + cfg.lambda_sample * l_sample + cfg.lambda_modal * l_modal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item() * text_feat.size(0)

        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                break

    # Evaluate best checkpoint on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    test_metrics["best_epoch"] = best_epoch

    learned_tau_sample = sample_loss_fn.get_temperature().item() if hasattr(sample_loss_fn, 'get_temperature') else cfg.tau
    learned_tau_modal = modal_loss_fn.get_temperature().item() if hasattr(modal_loss_fn, 'get_temperature') else cfg.tau / 2.0
    test_metrics["tau_sample"] = learned_tau_sample
    test_metrics["tau_modal"] = learned_tau_modal

    print(f"[{exp_name} | Seed {seed}] Best Epoch: {best_epoch:02d} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f} | tau_s: {learned_tau_sample:.3f}")

    return test_metrics


def run_experiment_suite(args):
    SEEDS = [42, 123, 129, 2024, 3407]
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    exp_configs = [
        ("M0", "baseline", False),          # Baseline V2 (1026-d, fixed tau=0.2, no hard neg)
        ("M0a", "baseline", True),          # Baseline V2 WITHOUT temporal metadata (1024-d)
        ("M1a", "learnable_tau", False),    # Novelty 2 only: Learnable tau (1026-d)
        ("M1b", "hard_neg", False),         # Novelty 1 only: Hard-neg only (1026-d)
        ("M_novelty", "novelty_both", False)# Novelties 1 + 2 combined (1026-d)
    ]

    if args.single_exp:
        exp_configs = [c for c in exp_configs if c[0] == args.single_exp]

    for exp_id, mode, drop_temporal in exp_configs:
        print(f"\n{'='*70}\nSTARTING EXPERIMENT: {exp_id} (mode={mode}, drop_temporal={drop_temporal})\n{'='*70}")
        args.mode = mode
        args.drop_temporal = drop_temporal

        exp_acc, exp_f1, exp_fake_f1, exp_real_f1 = [], [], [], []
        preds_list = []
        targets = None

        for s in SEEDS:
            m = train_single_run(args, s, exp_id)
            exp_acc.append(m["accuracy"])
            exp_f1.append(m["macro_f1"])
            exp_fake_f1.append(m["f1_fake"])
            exp_real_f1.append(m["f1_real"])
            preds_list.append(m["preds"])
            targets = m["labels"]

            # Save per-seed prediction array
            pred_file = pred_dir / f"preds_{exp_id}_seed{s}.npy"
            np.save(pred_file, m["preds"])

        # Check prediction discordance across seeds
        hamming_matrix = np.zeros((len(SEEDS), len(SEEDS)))
        for i in range(len(SEEDS)):
            for j in range(len(SEEDS)):
                hamming_matrix[i, j] = np.mean(preds_list[i] != preds_list[j])
        mean_interseed_disagreement = np.mean(hamming_matrix[np.triu_indices(len(SEEDS), k=1)])

        results[exp_id] = {
            "acc_mean": np.mean(exp_acc),
            "acc_std": np.std(exp_acc, ddof=1),
            "f1_mean": np.mean(exp_f1),
            "f1_std": np.std(exp_f1, ddof=1),
            "fake_f1_mean": np.mean(exp_fake_f1),
            "real_f1_mean": np.mean(exp_real_f1),
            "interseed_disagreement": mean_interseed_disagreement,
            "preds_seeds": preds_list,
            "raw_acc": exp_acc,
            "raw_f1": exp_f1
        }

        print(f"\n--- {exp_id} Summary across 5 Seeds ---")
        print(f"Accuracy: {results[exp_id]['acc_mean']*100:.2f}% +/- {results[exp_id]['acc_std']*100:.3f}%")
        print(f"Macro F1: {results[exp_id]['f1_mean']*100:.2f}% +/- {results[exp_id]['f1_std']*100:.3f}%")
        print(f"Inter-seed Prediction Disagreement: {mean_interseed_disagreement*100:.2f}% of samples")

    # Save ground truth targets
    if targets is not None:
        np.save(pred_dir / "test_targets.npy", targets)

    # Statistical Significance (McNemar vs M0)
    if "M0" in results:
        print(f"\n{'='*70}\nPAIRWISE STATISTICAL COMPARISONS AGAINST M0 (Baseline V2 1026-d)\n{'='*70}")
        m0_preds = results["M0"]["preds_seeds"][0] # compare seed 129 / seed 42
        for comp_id in ["M0a", "M1a", "M1b", "M_novelty"]:
            if comp_id in results:
                comp_preds = results[comp_id]["preds_seeds"][0]
                chi2, p_val, b, c = mcnemar_test(m0_preds, comp_preds, targets)
                mean_diff = (results[comp_id]["acc_mean"] - results["M0"]["acc_mean"]) * 100
                print(f"{comp_id} vs M0: Delta = {mean_diff:+.2f}% | McNemar chi2 = {chi2:.3f}, p-val = {p_val:.4f} (b={b}, c={c})")

    # Print Full Comparative Table
    print(f"\n{'='*80}\nFINAL PUBLICATION-GRADE RESULTS SUMMARY\n{'='*80}")
    print(f"{'Exp ID':<10} | {'Description':<32} | {'Accuracy (%)':<16} | {'Macro F1 (%)':<16} | {'Fake F1 (%)':<12}")
    print("-" * 88)
    desc_map = {
        "M0": "Baseline V2 (1026-d)",
        "M0a": "Baseline V2 (1024-d, No Temporal)",
        "M1a": "Novelty 2 (Learnable tau)",
        "M1b": "Novelty 1 (Hard-Negative)",
        "M_novelty": "Novelty 1+2 (Combined)"
    }
    for k, v in results.items():
        desc = desc_map.get(k, k)
        print(f"{k:<10} | {desc:<32} | {v['acc_mean']*100:.2f} +/- {v['acc_std']*100:.2f}  | {v['f1_mean']*100:.2f} +/- {v['f1_std']*100:.2f}  | {v['fake_f1_mean']*100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weibo")
    parser.add_argument("--train_dir", default="../data/weibo/features_cn_clip/train")
    parser.add_argument("--test_dir", default="../data/weibo/features_cn_clip/test")
    parser.add_argument("--pred_dir", default="./predictions")
    parser.add_argument("--single_exp", default=None, choices=["M0", "M0a", "M1a", "M1b", "M_novelty"])
    args = parser.parse_args()

    # Expand paths
    args.train_dir = str(Path(args.train_dir).expanduser())
    args.test_dir = str(Path(args.test_dir).expanduser())

    run_experiment_suite(args)
