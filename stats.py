"""
stats.py
========
Rigorous statistical analysis toolkit for fake news detection:
1. Per-seed exact McNemar test (binomial distribution, 2-tailed).
2. Holm-Bonferroni correction across families of comparisons.
3. Paired bootstrap (10,000 resamples) for confidence intervals of differences.
4. Sample standard deviation (ddof=1) and mean.
"""

import numpy as np
from scipy.stats import binomtest, ttest_rel, wilcoxon


def exact_mcnemar_test(preds_a, preds_b, targets):
    """
    Computes exact two-tailed binomial McNemar test.
    b: Model A correct, Model B wrong
    c: Model A wrong, Model B correct
    """
    correct_a = (preds_a == targets)
    correct_b = (preds_b == targets)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return 1.0, b, c
    res = binomtest(b, n, p=0.5, alternative='two-sided')
    return res.pvalue, b, c


def holm_bonferroni_correction(p_values):
    """
    Applies step-down Holm-Bonferroni adjustment to an array/list of p-values.
    Returns adjusted p-values.
    """
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = m - rank
        adj = p * multiplier
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(cum_max, 1.0)
    return adjusted


def paired_bootstrap_diff(preds_a, preds_b, targets, metric_fn, n_resamples=10000, seed=42):
    """
    Computes paired bootstrap confidence interval (95%) for (Metric_A - Metric_B).
    """
    rng = np.random.RandomState(seed)
    N = len(targets)
    diffs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        boot_idx = rng.randint(0, N, size=N)
        m_a = metric_fn(targets[boot_idx], preds_a[boot_idx])
        m_b = metric_fn(targets[boot_idx], preds_b[boot_idx])
        diffs[i] = m_a - m_b

    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    p_val = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return mean_diff, (ci_low, ci_high), min(p_val, 1.0)
