"""
test_sahn.py
============
Unit test suite for SAHN-DCL (Phase 2, 4, 8).
Verifies:
1. Mask correctness (diagonal exclusion, high overlap exclusion).
2. Ambiguity proxy range [0, 1].
3. Weight bounds and numerical stability (no NaN/inf).
4. Gradient flow.
5. Baseline equivalence when use_sahn=False.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sahn_losses import (
    build_sample_negative_mask,
    build_modality_negative_mask,
    compute_ambiguity_proxy,
    SAHNSupConLoss,
    SAHNModalityConsistencyLoss,
)


def run_unit_tests():
    print("=== Running SAHN-DCL Unit Tests ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, D = 16, 128

    # 1. Mask Correctness
    print("\n--- Test 1: Mask Correctness & Diagonal Exclusion ---")
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1] * 2, device=device)
    m_sample = build_sample_negative_mask(labels)
    assert m_sample.shape == (B, B), f"Expected shape ({B}, {B}), got {m_sample.shape}"
    diag = torch.diag(m_sample)
    assert (diag == 0).all(), "Diagonal must be 0 in sample negative mask!"

    # Test same-label pairs are 0
    for i in range(B):
        for j in range(B):
            if labels[i] == labels[j]:
                assert m_sample[i, j] == 0, f"Same class pair ({i}, {j}) marked as negative!"
            else:
                assert m_sample[i, j] == 1, f"Different class pair ({i}, {j}) missed from negatives!"
    print("✓ Sample negative mask strictly verified!")

    # 2. Modality Negative Mask & Overlap Exclusion
    print("\n--- Test 2: Modality Negative Mask & Overlap Exclusion ---")
    t_feat = torch.randn(B, D, device=device)
    # Force sample 0 and 1 to be identical (near duplicates)
    t_feat[1] = t_feat[0]
    m_modal = build_modality_negative_mask(t_feat, overlap_threshold=0.85)
    assert m_modal.shape == (B, B)
    assert (torch.diag(m_modal) == 0).all(), "Diagonal must be 0 in modality negative mask!"
    assert m_modal[0, 1] == 0, "Near-duplicate sample 1 was not excluded from negative mask of sample 0!"
    print("✓ Modality negative mask and overlap exclusion verified!")

    # 3. Ambiguity Proxy Range
    print("\n--- Test 3: Ambiguity Proxy Bounds ---")
    sim = torch.linspace(-1.0, 1.0, 100, device=device).view(10, 10)
    amb = compute_ambiguity_proxy(sim)
    assert (amb >= 0.0).all() and (amb <= 1.0).all(), "Ambiguity proxy out of bounds [0, 1]!"
    assert not torch.isnan(amb).any(), "Ambiguity proxy contains NaN!"
    print(f"✓ Ambiguity proxy bounded in [{amb.min():.4f}, {amb.max():.4f}] without NaNs!")

    # 4. Numerical Stability & Finite Loss/Gradients
    print("\n--- Test 4: Loss & Finite Gradients ---")
    feat = torch.randn(B, D, device=device, requires_grad=True)
    loss_sup = SAHNSupConLoss(alpha=1.0, beta=0.5, use_sahn=True).to(device)
    l_val = loss_sup(feat, labels)
    assert torch.isfinite(l_val), f"Loss is not finite: {l_val}"
    l_val.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all(), "Gradients contain NaN or Inf!"
    print(f"✓ SAHN SupCon loss is finite ({l_val.item():.4f}) with clean gradients!")

    img_feat = torch.randn(B, D, device=device, requires_grad=True)
    text_feat = torch.randn(B, D, device=device, requires_grad=True)
    loss_modal = SAHNModalityConsistencyLoss(alpha=0.8, beta=0.5, use_sahn=True).to(device)
    l_m_val = loss_modal(text_feat, img_feat)
    assert torch.isfinite(l_m_val), f"Modality loss is not finite: {l_m_val}"
    l_m_val.backward()
    assert torch.isfinite(text_feat.grad).all() and torch.isfinite(img_feat.grad).all()
    print(f"✓ SAHN Modality loss is finite ({l_m_val.item():.4f}) with clean gradients!")

    # 5. Baseline Equivalence (When use_sahn=False)
    print("\n--- Test 5: Baseline Equivalence when use_sahn=False ---")
    loss_sup_disabled = SAHNSupConLoss(temperature=0.2, use_sahn=False, learnable_temp=False).to(device)
    # Compute baseline reference manually
    f_norm = F.normalize(feat.detach(), p=2, dim=-1)
    sim_ref = torch.matmul(f_norm, f_norm.T) / 0.2
    sim_ref.masked_fill_(torch.eye(B, dtype=torch.bool, device=device), -1e4)
    log_p = sim_ref - torch.logsumexp(sim_ref, dim=1, keepdim=True)
    labels_col = labels.view(-1, 1)
    pos_mask = (labels_col == labels_col.T) & (~torch.eye(B, dtype=torch.bool, device=device))
    ref_loss = -(pos_mask.float() * log_p).sum(dim=1)[pos_mask.sum(1) > 0] / pos_mask.sum(1)[pos_mask.sum(1) > 0]
    expected_val = ref_loss.mean()

    actual_val = loss_sup_disabled(feat.detach(), labels)
    diff = (expected_val - actual_val).abs().item()
    assert diff < 1e-5, f"Disabled SAHN loss did not match baseline! Diff: {diff}"
    print(f"✓ Baseline equivalence strictly verified! (Diff: {diff:.2e})")

    # 6. Empty Negative Row Handling
    print("\n--- Test 6: Empty Negative Handling ---")
    # Batch where all samples have the exact same label
    all_same_labels = torch.zeros(B, dtype=torch.long, device=device)
    loss_empty = loss_sup(feat.detach(), all_same_labels)
    assert torch.isfinite(loss_empty), "Loss became NaN on all-same label batch!"
    print("✓ Empty negative row handled safely with zero NaN!")

    print("\n==========================================")
    print("🎉 ALL SAHN-DCL UNIT TESTS PASSED (6/6)!")
    print("==========================================")


if __name__ == "__main__":
    run_unit_tests()
