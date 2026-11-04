"""
launch_enhanced_v2.py
=====================
Multi-GPU parallel launcher for Enhanced CFCA-I V2 experiments.
Runs E0v2–E4v2, 5 seeds each = 25 total runs.
Distributes across GPUs 0, 2, 4 (avoids busy GPUs 1, 3).

Usage:
  nohup python launch_enhanced_v2.py > enhanced_v2_master.log 2>&1 &
  tail -f enhanced_v2_master.log

Estimated time: ~35-50 min total (30-epoch runs, 3 parallel GPUs)
Results saved to: /DATA/raghvendra3088/MFND_project/files/docs/ENHANCED_CFCAI_V2_RESULTS.txt
"""

import subprocess
import sys
import time
import os
import numpy as np
from pathlib import Path
from itertools import cycle

PYTHON = "/DATA/raghvendra3088/MFND_env/bin/python"
SCRIPT = "/DATA/raghvendra3088/MFND_project/files/train_cfca_enhanced_v2.py"
DATA_ROOT = "/DATA/raghvendra3088/MFND_project/data/weibo"
PRED_DIR = "/DATA/raghvendra3088/MFND_project/files/predictions"
OUT_DIR = "/DATA/raghvendra3088/MFND_project/files/docs"
LOG_DIR = "/DATA/raghvendra3088/MFND_project/files/logs_v2"

CONFIGS = ["E0v2", "E1v2", "E2v2", "E3v2", "E4v2"]
SEEDS = [42, 123, 129, 2024, 3407]
GPUS = [0, 2, 4]  # Only use GPUs 0, 2, 4

# V2 training args
TRAIN_ARGS = [
    "--lr", "2e-4",
    "--lr_min", "1e-5",
    "--max_epochs", "30",
    "--patience", "15",
    "--bs", "64",
    "--dropout", "0.3",
    "--lam_fg", "0.15",
]

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Build all jobs
jobs = [(cfg, seed) for cfg in CONFIGS for seed in SEEDS]
print(f"Total jobs: {len(jobs)} | Configs: {CONFIGS} | Seeds: {SEEDS}", flush=True)
print(f"Distributing across GPUs: {GPUS}", flush=True)

gpu_cycle = cycle(GPUS)
processes = []
gpu_assignments = {}

for (cfg, seed) in jobs:
    gpu = next(gpu_cycle)
    log_file = f"{LOG_DIR}/v2_{cfg}_seed{seed}.log"
    cmd = [
        PYTHON, SCRIPT,
        "--config", cfg,
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--data_root", DATA_ROOT,
        "--pred_dir", PRED_DIR,
    ] + TRAIN_ARGS

    print(f"Launching: {cfg} seed={seed} gpu={gpu}", flush=True)
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    processes.append((proc, cfg, seed, gpu, log_file))
    gpu_assignments[(cfg, seed)] = gpu
    time.sleep(0.3)  # small stagger to avoid simultaneous data loading spikes

print(f"\nAll {len(processes)} processes launched. Waiting for completion...\n", flush=True)

# Wait for all to finish
finished = 0
while processes:
    for item in list(processes):
        proc, cfg, seed, gpu, log_file = item
        ret = proc.poll()
        if ret is not None:
            status = "OK" if ret == 0 else f"FAILED(rc={ret})"
            print(f"[{status}] {cfg} seed={seed} gpu={gpu}", flush=True)
            processes.remove(item)
            finished += 1
    if processes:
        time.sleep(10)

print(f"\nAll {finished} runs completed. Collecting results...\n", flush=True)

# Collect and summarize results
results = {}
for cfg in CONFIGS:
    accs, f1s = [], []
    for seed in SEEDS:
        pred_path = Path(PRED_DIR) / f"preds_v2_{cfg}_seed{seed}.npy"
        test_label_path = Path(DATA_ROOT) / "tokens" / "test" / "labels.npy"
        if pred_path.exists():
            preds = np.load(pred_path)
            labels = np.load(test_label_path)
            acc = (preds == labels).mean() * 100
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, preds, average="macro") * 100
            accs.append(acc)
            f1s.append(f1)
        else:
            print(f"  [MISSING] {pred_path}", flush=True)
    if accs:
        results[cfg] = {
            "accs": accs,
            "f1s": f1s,
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs, ddof=0),
            "mean_f1": np.mean(f1s),
            "std_f1": np.std(f1s, ddof=0),
        }

# Write results file
out_file = Path(OUT_DIR) / "ENHANCED_CFCAI_V2_RESULTS.txt"
with open(out_file, "w") as f:
    f.write("=" * 90 + "\n")
    f.write("      ENHANCED CFCA-I V2: RESULTS (patch_mask_k=3, LayerNorm, CosineAnnealing)\n")
    f.write("=" * 90 + "\n\n")
    f.write("V2 Changes vs V1:\n")
    f.write("  Fix 1: patch_mask_k 7 -> 3  (6% masking, was 14%)\n")
    f.write("  Fix 2: BatchNorm1d -> LayerNorm in classifier\n")
    f.write("  Fix 3: E0v2 uses lam_fg=0.0 (true no-FG baseline, was 0.15)\n")
    f.write("  Fix 4: CosineAnnealingLR (2e-4 -> 1e-5 over 30 epochs)\n")
    f.write("  Fix 5: max_epochs=30, patience=15 (was 20/10)\n\n")
    f.write("Seeds: [42, 123, 129, 2024, 3407]\n\n")

    for cfg in CONFIGS:
        if cfg in results:
            r = results[cfg]
            seed_str = ", ".join(f"{a:.2f}" for a in r["accs"])
            f.write(f"Config {cfg}:\n")
            f.write(f"  Accuracy: {r['mean_acc']:.2f}% +/- {r['std_acc']:.2f}%  (Seeds: [{seed_str}])\n")
            f.write(f"  Macro-F1: {r['mean_f1']:.2f}% +/- {r['std_f1']:.2f}%\n\n")
        else:
            f.write(f"Config {cfg}: NO RESULTS\n\n")

    # Comparison table
    f.write("-" * 90 + "\n")
    f.write(f"{'Config':<10} | {'Acc V2':>12} | {'F1 V2':>12} | {'Acc V1 (ref)':>14} | {'Δ':>8}\n")
    f.write("-" * 90 + "\n")
    # V1 reference values
    v1_ref = {
        "E0v2": 90.53, "E1v2": 92.21, "E2v2": 92.51, "E3v2": 92.71, "E4v2": 93.13
    }
    v1_labels = {
        "E0v2": "E0 (90.53±1.13)", "E1v2": "E1 (92.21±1.06)",
        "E2v2": "E2 (92.51±0.41)", "E3v2": "E3 (92.71±0.61)", "E4v2": "E4 (93.13±1.41)"
    }
    for cfg in CONFIGS:
        if cfg in results:
            r = results[cfg]
            v1 = v1_ref.get(cfg, 0)
            delta = r["mean_acc"] - v1
            f.write(
                f"{cfg:<10} | {r['mean_acc']:>8.2f}±{r['std_acc']:.2f} | "
                f"{r['mean_f1']:>8.2f}±{r['std_f1']:.2f} | "
                f"{v1_labels.get(cfg, 'N/A'):>14} | {delta:>+8.2f}%\n"
            )
    f.write("-" * 90 + "\n")
    f.write("\nBase Paper (Qian et al.) reported: 95.50% (WITH timestamp shortcut)\n")
    f.write("Our best clean (M4, V1): 93.33% +/- 0.63%\n")
    if "E4v2" in results:
        r4 = results["E4v2"]
        f.write(f"Our best clean (E4v2, V2): {r4['mean_acc']:.2f}% +/- {r4['std_acc']:.2f}%\n")

print(f"\nResults saved to: {out_file}", flush=True)
print("Done!", flush=True)

