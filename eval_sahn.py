"""
eval_sahn.py
============
Computes statistical significance, McNemar chi2, exact p-values, and 
generates final comparison and ablation tables for SAHN-DCL in plain text (.txt).
"""

import sys
import glob
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import chi2 as chi2_dist


def mcnemar_test(preds_a, preds_b, targets):
    correct_a = (preds_a == targets)
    correct_b = (preds_b == targets)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_val = 1.0 - chi2_dist.cdf(chi2, df=1)
    return chi2, p_val, b, c


def evaluate_all():
    base_dir = Path("/DATA/raghvendra3088/MFND_project")
    pred_dir = base_dir / "files/predictions"
    docs_dir = base_dir / "files/docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth labels
    test_data = torch.load(base_dir / "data/weibo/consolidated_test.pt", map_location="cpu", weights_only=False)
    targets = test_data["labels"].numpy()

    # Discover experiments
    experiments = [
        ("E0", "MFND-DCL Baseline (Paper Arch + Meta)"),
        ("E0a", "MFND-DCL Clean Baseline (No Meta)"),
        ("E1", "E1: Baseline + Negative Masking Only (alpha=0)"),
        ("E2", "E2: E1 + SAHN Sample-Level Loss (alpha_s=1.0)"),
        ("E3", "E3: E1 + SAHN Modality-Level Loss (alpha_m=0.8)"),
        ("E4", "E4: Combined SAHN-DCL (Sample + Modal, beta=0)"),
        ("E5", "E5: Full SAHN-DCL + Ambiguity Proxy (beta=0.5)"),
        ("E6", "E6: Fine-Grained CFCA-I + Full SAHN-DCL"),
    ]

    seeds = [42, 123, 129, 2024, 3407]
    stats = {}

    for exp_id, desc in experiments:
        exp_preds = []
        for s in seeds:
            f = pred_dir / f"preds_{exp_id}_seed{s}.npy"
            if f.exists():
                exp_preds.append(np.load(f))
        
        if len(exp_preds) == 0:
            continue

        accs, f1s, r_f1s, f_f1s = [], [], [], []
        for p in exp_preds:
            accs.append(accuracy_score(targets, p) * 100)
            prec, rec, f1, _ = precision_recall_fscore_support(targets, p, average=None, labels=[0, 1], zero_division=0)
            r_f1s.append(f1[0] * 100)
            f_f1s.append(f1[1] * 100)
            macro_f1 = precision_recall_fscore_support(targets, p, average="macro", zero_division=0)[2]
            f1s.append(macro_f1 * 100)

        # Ensemble / concatenated predictions across seeds for McNemar
        all_exp_preds = np.concatenate(exp_preds)
        all_targets = np.tile(targets, len(exp_preds))

        stats[exp_id] = {
            "desc": desc,
            "n_seeds": len(exp_preds),
            "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0,
            "r_f1_mean": np.mean(r_f1s), "f_f1_mean": np.mean(f_f1s),
            "preds_concat": all_exp_preds,
            "targets_concat": all_targets
        }

    # Generate Plain Text Output
    out_lines = []
    def log(msg=""):
        print(msg)
        out_lines.append(msg)

    log("=" * 100)
    log("       SAHN-DCL RIGOROUS EXPERIMENTAL EVALUATION RESULTS (WEIBO DATASET)")
    log("=" * 100)
    log(f"Evaluation Protocol: 5 Random Seeds [42, 123, 129, 2024, 3407] | 90/10 Val Split | Test N=1996")
    log()

    log("TABLE 1: MAIN RESULTS & ARCHITECTURE COMPARISON (MEAN ± SAMPLE STD)")
    log("-" * 100)
    header = f"{'Model / Experiment':<42} | {'Accuracy (%)':<16} | {'Macro F1 (%)':<16} | {'McNemar vs E0':<18}"
    log(header)
    log("-" * 100)

    base_concat = stats["E0"]["preds_concat"] if "E0" in stats else None
    targets_concat = stats["E0"]["targets_concat"] if "E0" in stats else None

    for exp_id in ["E0", "E0a", "E4", "E5", "E6"]:
        if exp_id not in stats:
            continue
        s = stats[exp_id]
        acc_str = f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}"
        f1_str = f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f}"

        if exp_id == "E0" or base_concat is None:
            mcn_str = "Baseline (Ref)"
        else:
            chi2, p_val, b, c = mcnemar_test(s["preds_concat"], base_concat, targets_concat)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            mcn_str = f"p={p_val:.2e} {sig}"

        log(f"{s['desc']:<42} | {acc_str:<16} | {f1_str:<16} | {mcn_str:<18}")

    log("-" * 100)
    log("Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. not significant")
    log()

    log("TABLE 2: SAHN-DCL COMPONENT ABLATION LADDER")
    log("-" * 100)
    header_abl = f"{'Configuration':<42} | {'Acc (%)':<14} | {'Macro F1 (%)':<14} | {'Δ vs Base':<12} | {'McNemar p':<12}"
    log(header_abl)
    log("-" * 100)

    for exp_id, _ in experiments:
        if exp_id not in stats:
            continue
        s = stats[exp_id]
        acc_str = f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}"
        f1_str = f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f}"
        delta = s["acc_mean"] - stats["E0"]["acc_mean"] if "E0" in stats else 0.0
        delta_str = f"{delta:+.2f}%"

        if exp_id == "E0" or base_concat is None:
            p_str = "-"
        else:
            chi2, p_val, b, c = mcnemar_test(s["preds_concat"], base_concat, targets_concat)
            p_str = f"{p_val:.2e}"

        log(f"{s['desc']:<42} | {acc_str:<14} | {f1_str:<14} | {delta_str:<12} | {p_str:<12}")

    log("-" * 100)
    log("=" * 100)

    out_file = docs_dir / "SAHN_DCL_RESULTS.txt"
    with open(out_file, "w") as fp:
        fp.write("\n".join(out_lines) + "\n")
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    evaluate_all()
