import subprocess
import json
import numpy as np
from pathlib import Path
from scipy.stats import binomtest

seeds = [42, 123, 129, 2024, 3407]
modes = ['baseline', 'hard_negative']

results = {m: [] for m in modes}

for mode in modes:
    print(f'=== Running RoViT {mode} across 5 seeds ===')
    for s in seeds:
        cmd = [
            '/DATA/raghvendra3088/MFND_env/bin/python',
            '/DATA/raghvendra3088/MFND_project/files/train_rovit.py',
            '--mode', mode,
            '--seed', str(s),
            '--gpu', '4',
            '--epochs', '15',
            '--lr', '1e-4',
            '--lambda_con', '0.1'
        ]
        out = subprocess.check_output(cmd, text=True)
        print(out.strip())
        
        # Load accuracy and f1
        preds = np.load(f'/DATA/raghvendra3088/MFND_project/files/predictions/preds_rovit_{mode}_seed{s}.npy')
        trues = np.load('/DATA/raghvendra3088/MFND_project/data/weibo/rovit/test_labels.npy')
        acc = (preds == trues).mean()
        
        from sklearn.metrics import f1_score
        f1 = f1_score(trues, preds, average='macro')
        results[mode].append({'seed': s, 'acc': acc, 'f1': f1})

summary_lines = []
summary_lines.append('================================================================================')
summary_lines.append('           ROVIT-DETECT & HN-ROVIT EXPERIMENTAL RESULTS (WEIBO CLEAN)')
summary_lines.append('================================================================================\n')

for mode in modes:
    accs = [r['acc'] * 100 for r in results[mode]]
    f1s = [r['f1'] * 100 for r in results[mode]]
    mean_acc, std_acc = np.mean(accs), np.std(accs, ddof=1)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)
    summary_lines.append(f'Model: {mode.upper()}')
    summary_lines.append(f'  Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%  (Seeds: {[round(x, 2) for x in accs]})')
    summary_lines.append(f'  Macro-F1: {mean_f1:.2f}% +/- {std_f1:.2f}%  (Seeds: {[round(x, 2) for x in f1s]})\n')

# Paired McNemar Test
b_preds = np.concatenate([np.load(f'/DATA/raghvendra3088/MFND_project/files/predictions/preds_rovit_baseline_seed{s}.npy') for s in seeds])
h_preds = np.concatenate([np.load(f'/DATA/raghvendra3088/MFND_project/files/predictions/preds_rovit_hard_negative_seed{s}.npy') for s in seeds])
trues_all = np.tile(np.load('/DATA/raghvendra3088/MFND_project/data/weibo/rovit/test_labels.npy'), len(seeds))

b_corr = (b_preds == trues_all)
h_corr = (h_preds == trues_all)

b_only = np.sum(b_corr & ~h_corr)
h_only = np.sum(~b_corr & h_corr)
both = np.sum(b_corr & h_corr)
neither = np.sum(~b_corr & ~h_corr)

# Exact Binomial test
p_val = binomtest(min(b_only, h_only), b_only + h_only, 0.5).pvalue if (b_only + h_only) > 0 else 1.0

summary_lines.append('Paired Statistical Significance (Baseline vs HN-RoViT across 5 seeds):')
summary_lines.append(f'  RoViT Correct / HN-RoViT Wrong: {b_only}')
summary_lines.append(f'  RoViT Wrong / HN-RoViT Correct: {h_only}')
summary_lines.append(f'  Exact Binomial p-value: {p_val:.4e}')
summary_lines.append('================================================================================')

report_text = '\n'.join(summary_lines)
print('\n' + report_text)

with open('/DATA/raghvendra3088/MFND_project/files/docs/ROVIT_RESULTS.txt', 'w') as f:
    f.write(report_text)
print('Results saved to docs/ROVIT_RESULTS.txt')
