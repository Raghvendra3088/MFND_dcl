import subprocess
import time
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import binomtest

configs = ["E0", "E1", "E2", "E3", "E4"]
seeds = [42, 123, 129, 2024, 3407]
gpus = [0, 2, 4]  # High memory free GPUs

py_bin = "/DATA/raghvendra3088/MFND_env/bin/python"
train_script = "/DATA/raghvendra3088/MFND_project/files/train_cfca_enhanced.py"
pred_dir = Path("/DATA/raghvendra3088/MFND_project/files/predictions")
pred_dir.mkdir(parents=True, exist_ok=True)

# Generate list of tasks
tasks = []
for cfg in configs:
    for s in seeds:
        pred_file = pred_dir / f"preds_enhanced_{cfg}_seed{s}.npy"
        if not pred_file.exists():
            tasks.append((cfg, s))
        else:
            print(f"Skipping {cfg} seed {s} (already finished: {pred_file})")

print(f"Total remaining tasks to run: {len(tasks)}")

running_processes = {}
task_idx = 0

while task_idx < len(tasks) or len(running_processes) > 0:
    # Check finished processes
    finished_gpus = []
    for gpu, (p, (cfg, s)) in list(running_processes.items()):
        ret = p.poll()
        if ret is not None:
            if ret == 0:
                print(f"[SUCCESS] GPU {gpu} finished {cfg} seed {s}")
            else:
                print(f"[ERROR] GPU {gpu} failed {cfg} seed {s} with return code {ret}")
            finished_gpus.append(gpu)

    for g in finished_gpus:
        del running_processes[g]

    # Assign new tasks to free GPUs
    for gpu in gpus:
        if gpu not in running_processes and task_idx < len(tasks):
            cfg, s = tasks[task_idx]
            task_idx += 1
            cmd = [
                py_bin, train_script,
                "--config", cfg,
                "--seed", str(s),
                "--gpu", str(gpu),
                "--bs", "64",
                "--lr", "2e-4",
                "--max_epochs", "20",
                "--patience", "10"
            ]
            print(f"Launching {cfg} seed {s} on GPU {gpu}...")
            p = subprocess.Popen(cmd)
            running_processes[gpu] = (p, (cfg, s))
            time.sleep(2)  # small gap

    time.sleep(2)

print("\nAll training runs complete! Aggregating results across all 5 seeds...")

labels_true = np.load("/DATA/raghvendra3088/MFND_project/data/weibo/tokens/test/labels.npy")

results_summary = []
results_summary.append("=========================================================================================")
results_summary.append("      ENHANCED CFCA-I: MULTI-SCALE POOLING + DEEP ENSEMBLE + AUGMENTATION RESULTS")
results_summary.append("=========================================================================================\n")

res_dict = {}

for cfg in configs:
    accs, f1s = [], []
    for s in seeds:
        p_path = pred_dir / f"preds_enhanced_{cfg}_seed{s}.npy"
        if p_path.exists():
            preds = np.load(p_path)
            acc = accuracy_score(labels_true, preds) * 100.0
            f1 = f1_score(labels_true, preds, average="macro") * 100.0
            accs.append(acc)
            f1s.append(f1)
        else:
            print(f"Missing prediction: {p_path}")

    if len(accs) > 0:
        mean_acc, std_acc = np.mean(accs), np.std(accs, ddof=1)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)
        res_dict[cfg] = (mean_acc, std_acc, mean_f1, std_f1, accs)
        results_summary.append(f"Config {cfg}:")
        results_summary.append(f"  Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%  (Individual seeds: {[round(x, 2) for x in accs]})")
        results_summary.append(f"  Macro-F1: {mean_f1:.2f}% +/- {std_f1:.2f}%\n")

# Statistical significance vs Baseline E0
if "E0" in res_dict and "E4" in res_dict:
    e0_all = np.concatenate([np.load(pred_dir / f"preds_enhanced_E0_seed{s}.npy") for s in seeds])
    e4_all = np.concatenate([np.load(pred_dir / f"preds_enhanced_E4_seed{s}.npy") for s in seeds])
    y_all = np.tile(labels_true, len(seeds))

    e0_corr = (e0_all == y_all)
    e4_corr = (e4_all == y_all)

    n_e0_only = np.sum(e0_corr & ~e4_corr)
    n_e4_only = np.sum(~e0_corr & e4_corr)
    p_val = binomtest(min(n_e0_only, n_e4_only), n_e0_only + n_e4_only, 0.5).pvalue

    results_summary.append("Paired Statistical Test (E0 vs Full Enhanced E4 across 9,980 predictions):")
    results_summary.append(f"  E0 Correct / E4 Wrong: {n_e0_only}")
    results_summary.append(f"  E0 Wrong / E4 Correct: {n_e4_only}")
    results_summary.append(f"  Exact Binomial p-value: {p_val:.4e}")
    results_summary.append("=========================================================================================")

report_text = "\n".join(results_summary)
print(report_text)

out_file = Path("/DATA/raghvendra3088/MFND_project/files/docs/ENHANCED_CFCAI_RESULTS.txt")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    f.write(report_text)

print(f"\nSaved final results to {out_file}")

