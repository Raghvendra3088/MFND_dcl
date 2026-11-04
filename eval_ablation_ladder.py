"""
eval_ablation_ladder.py
=======================
Calculates true sample mean +/- std, McNemar significance tests, and generates
the final publication-grade comparison table between Base Paper, Baselines (M0, M0a),
and all CFCA-I models (M2-M6, M5a).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
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


def load_all_predictions(pred_dir, targets, pt_to_csv_idx, seeds=[42, 123, 129, 2024, 3407]):
    pred_path = Path(pred_dir)
    models = ["M0", "M0a", "M1a", "M1b", "M_novelty", "M2", "M3", "M4", "M5", "M6", "M5a"]
    results = {}

    for m in models:
        model_preds = []
        metrics_list = []
        available = True
        for s in seeds:
            p_file = pred_path / f"preds_{m}_seed{s}.npy"
            if not p_file.exists():
                available = False
                break
            preds = np.load(p_file)
            # Check if this model used the old pt loader (M0, M0a, M1a, M1b, M_novelty)
            if m in ["M0", "M0a", "M1a", "M1b", "M_novelty"]:
                preds = preds[pt_to_csv_idx]
            model_preds.append(preds)

            acc = accuracy_score(targets, preds) * 100
            p, r, f1, _ = precision_recall_fscore_support(targets, preds, average=None, labels=[0, 1], zero_division=0)
            mf1 = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)[2] * 100
            metrics_list.append({
                "seed": s, "acc": acc, "mf1": mf1,
                "p0": p[0]*100, "r0": r[0]*100, "f1_0": f1[0]*100,
                "p1": p[1]*100, "r1": r[1]*100, "f1_1": f1[1]*100
            })

        if available:
            concat_preds = np.concatenate(model_preds)
            results[m] = {
                "per_seed": metrics_list,
                "preds_per_seed": model_preds,
                "concat_preds": concat_preds,
                "mean_acc": np.mean([x["acc"] for x in metrics_list]),
                "std_acc": np.std([x["acc"] for x in metrics_list], ddof=1),
                "mean_f1": np.mean([x["mf1"] for x in metrics_list]),
                "std_f1": np.std([x["mf1"] for x in metrics_list], ddof=1),
                "mean_p0": np.mean([x["p0"] for x in metrics_list]),
                "mean_r0": np.mean([x["r0"] for x in metrics_list]),
                "mean_p1": np.mean([x["p1"] for x in metrics_list]),
                "mean_r1": np.mean([x["r1"] for x in metrics_list]),
            }
    return results


def main():
    test_csv = "../data/weibo/weibo_test.csv"
    pred_dir = "predictions"
    pt_dir = "../data/weibo/features_cn_clip/test"

    print("Loading test metadata and targets...")
    df = pd.read_csv(test_csv)
    targets = np.load("../data/weibo/tokens/test/labels.npy")

    # Map pt order to csv order
    sorted_files = sorted(Path(pt_dir).glob("*.pt"))
    pt_post_ids = [int(f.stem) for f in sorted_files]
    csv_post_ids = df["post_id"].tolist()
    pt_to_csv_idx = [pt_post_ids.index(pid) for pid in csv_post_ids]

    results = load_all_predictions(pred_dir, targets, pt_to_csv_idx)
    concat_targets = np.tile(targets, 5)

    print("\n" + "=" * 90)
    print("                     CFCA-I ABLATION LADDER & STATISTICAL EVALUATION")
    print("=" * 90)
    print(f"{'Model':<12} | {'Accuracy':<16} | {'Macro F1':<16} | {'Prec (R/F)':<15} | {'Rec (R/F)':<15}")
    print("-" * 90)

    # Base paper reference
    print(f"{'Paper (Qian)':<12} | 95.50%          | 95.80%          | 95.6 / 95.4     | 95.3 / 95.7")

    model_display_names = {
        "M0": "M0 (Base V2 +Meta)",
        "M0a": "M0a (Base V2 Clean)",
        "M1a": "M1a (Learn Tau)",
        "M1b": "M1b (Hard Neg)",
        "M_novelty": "M_nov (Tau+HardN)",
        "M2": "M2 (CA Uniform)",
        "M3": "M3 (CA+Importance)",
        "M4": "M4 (+L_FG Max)",
        "M5": "M5 (Full CFCA-I)",
        "M6": "M6 (Sum Collapsed)",
        "M5a": "M5a (CFCA-I Clean)",
    }

    for m, info in results.items():
        name = model_display_names.get(m, m)
        acc_str = f"{info['mean_acc']:.2f}% ± {info['std_acc']:.2f}%"
        f1_str = f"{info['mean_f1']:.2f}% ± {info['std_f1']:.2f}%"
        p_str = f"{info['mean_p0']:.1f} / {info['mean_p1']:.1f}"
        r_str = f"{info['mean_r0']:.1f} / {info['mean_r1']:.1f}"
        print(f"{name:<12} | {acc_str:<16} | {f1_str:<16} | {p_str:<15} | {r_str:<15}")

    print("=" * 90)

    # Pairwise McNemar tests
    print("\n" + "=" * 90)
    print("                      PAIRWISE MCNEMAR SIGNIFICANCE TESTS (5 Seeds)")
    print("=" * 90)
    comparisons = [
        ("M0a", "M5a", "Clean Baseline vs Clean CFCA-I (No temporal meta)"),
        ("M0", "M5", "Baseline V2 vs Full CFCA-I (+Meta)"),
        ("M4", "M5", "CFCA-I without vs with Inconsistency Head"),
        ("M6", "M5", "Sum/Bilinear Collapse vs Dual-Max Alignment"),
        ("M2", "M3", "Uniform vs Claim/Region Importance"),
        ("M3", "M4", "Importance vs Fine-grained Alignment Loss"),
    ]

    for m_a, m_b, desc in comparisons:
        if m_a in results and m_b in results:
            preds_a = results[m_a]["concat_preds"]
            preds_b = results[m_b]["concat_preds"]
            chi2, p_val, b, c = mcnemar_test(preds_a, preds_b, concat_targets)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            print(f"[{m_a} vs {m_b}] {desc}")
            print(f"   b ({m_a} win) = {b}, c ({m_b} win) = {c} | Chi2 = {chi2:.3f}, p = {p_val:.4e} ({sig})\n")


if __name__ == "__main__":
    main()
