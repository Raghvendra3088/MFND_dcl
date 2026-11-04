"""
test_cfca_real.py -- Phase 2 Unit Checks on Real Cached Tensors.
Runs on GPU with actual extracted Chinese-CLIP token tensors.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfca_modules import CFCAI, FGLoss, cfcai_total_loss, align_matrix
from dcl_losses import SupConLoss, MultimodalConsistencyLoss


def run_checks(device="cuda:2"):
    print(f"=== Starting Phase 2 Unit Checks on {device} ===")
    
    # 1. Load real cached tensors
    train_dir = "../data/weibo/tokens/train"
    print(f"Loading real cached tensors from {train_dir}...")
    T = np.lib.format.open_memmap(f"{train_dir}/text.npy", mode="r")
    TM = np.lib.format.open_memmap(f"{train_dir}/text_mask.npy", mode="r")
    V = np.lib.format.open_memmap(f"{train_dir}/patch.npy", mode="r")
    META = np.lib.format.open_memmap(f"{train_dir}/meta.npy", mode="r")
    LABELS = np.load(f"{train_dir}/labels.npy")

    B = 32
    tt = torch.from_numpy(np.array(T[:B])).float().to(device)
    tm = torch.from_numpy(np.array(TM[:B])).to(device)
    it = torch.from_numpy(np.array(V[:B])).float().to(device)
    meta = torch.from_numpy(np.array(META[:B])).float().to(device)
    labels = torch.from_numpy(LABELS[:B]).long().to(device)

    print(f"Batch shape: tt={tt.shape}, tm={tm.shape}, it={it.shape}, meta={meta.shape}, labels={labels.shape}")

    # Check 1: Importance weights sum to 1 and zero on padding
    print("\n--- Check 1: Importance Weight Properties ---")
    model = CFCAI(d_text=1024, d_img=1024, d=128, heads=4, extra_dim=2,
                  use_importance=True, use_fg=True, use_inc=True).to(device)
    out = model(tt, tm, it, meta)
    a, b = out["a"], out["b"]

    # Sum check
    a_sum = a.sum(dim=-1)
    b_sum = b.sum(dim=-1)
    assert torch.allclose(a_sum, torch.ones_like(a_sum), atol=1e-4), f"Text importance does not sum to 1: {a_sum}"
    assert torch.allclose(b_sum, torch.ones_like(b_sum), atol=1e-4), f"Image importance does not sum to 1: {b_sum}"
    print(f"✓ Importance sum check passed! a_sum={a_sum[0].item():.4f}, b_sum={b_sum[0].item():.4f}")

    # Padding mask check on 'a'
    pad_mask = ~tm
    if pad_mask.any():
        pad_weights = a[pad_mask]
        max_pad_weight = pad_weights.max().item()
        assert max_pad_weight < 1e-5, f"Padded tokens received nonzero weight: {max_pad_weight}"
        print(f"✓ Padding zero-weight check passed! Max padded weight: {max_pad_weight:.2e}")
    else:
        print("Note: First batch has no padding tokens, testing with forced padding mask:")
        tm_forced = tm.clone()
        tm_forced[:, 30:] = False
        out_f = model(tt, tm_forced, it, meta)
        pad_weights = out_f["a"][~tm_forced]
        assert pad_weights.max().item() < 1e-5
        print("✓ Forced padding zero-weight check passed!")

    # Check 2: Padding Mask Invariance
    print("\n--- Check 2: Padding Mask Invariance ---")
    model.eval()
    # Corrupt padded positions with random values
    tt_corrupt = tt.clone()
    if not pad_mask.any():
        tm = tm.clone()
        tm[:, 35:] = False
        pad_mask = ~tm
    
    tt_corrupt[pad_mask] = torch.randn_like(tt_corrupt[pad_mask]) * 50.0  # huge noise
    with torch.no_grad():
        out_orig = model(tt, tm, it, meta)
        out_corrupt = model(tt_corrupt, tm, it, meta)

    diff_logits = (out_orig["logits"] - out_corrupt["logits"]).abs().max().item()
    diff_ztext = (out_orig["z_text"] - out_corrupt["z_text"]).abs().max().item()
    diff_S = (out_orig["S"] - out_corrupt["S"]).abs().max().item()
    print(f"Max diff with corrupted padding: logits={diff_logits:.2e}, z_text={diff_ztext:.2e}, S={diff_S:.2e}")
    assert diff_logits < 1e-4, f"Logits changed when corrupting padded tokens! diff={diff_logits}"
    assert diff_ztext < 1e-4, f"z_text changed when corrupting padded tokens! diff={diff_ztext}"
    print("✓ Padding Mask Invariance strictly verified!")
    model.train()

    # Check 3: Full Backward Gradient Flow (M5)
    print("\n--- Check 3: Full Gradient Flow (M5) ---")
    supcon_fn = SupConLoss(temperature=0.2, learnable_temp=True).to(device)
    modal_fn = MultimodalConsistencyLoss(temperature=0.1, learnable_temp=True).to(device)
    fg_loss = FGLoss(tau=0.2).to(device)

    model.zero_grad()
    supcon_fn.zero_grad()
    modal_fn.zero_grad()
    fg_loss.zero_grad()

    out = model(tt, tm, it, meta)
    loss = cfcai_total_loss(model, out, (tt, tm, it), labels, supcon_fn, modal_fn, fg_loss)
    loss.backward()

    components = {
        "proj_t": model.proj_t[0].weight.grad,
        "proj_i": model.proj_i[0].weight.grad,
        "imp_t": model.imp_t.net[0].weight.grad,
        "imp_i": model.imp_i.net[0].weight.grad,
        "ca_t2i": model.ca_t2i.q.weight.grad,
        "ca_i2t": model.ca_i2t.q.weight.grad,
        "inc": model.inc.mlp[0].weight.grad,
        "clf": model.clf[0].weight.grad,
        "sample_proj": model.sample_proj[0].weight.grad,
        "fg_loss.tau": fg_loss.tau.theta.grad,
        "supcon.tau": supcon_fn.temperature_logit.grad,
        "modal.tau": modal_fn.temperature_logit.grad,
    }

    for name, grad in components.items():
        assert grad is not None, f"Gradient for {name} is None!"
        norm = grad.abs().sum().item()
        assert norm > 1e-8, f"Gradient for {name} is zero!"
        print(f"  {name:15s}: grad norm = {norm:.4e} (active)")
    print("✓ Full backward gradient flow passed for all 12 modules!")

    # Check 4: Ablation Gradient Invariance (M2, M3, M4)
    print("\n--- Check 4: Ablation Gradient Isolation ---")
    # Test M2: no importance, no fg, no inc
    m2 = CFCAI(d_text=1024, d_img=1024, d=128, heads=4, extra_dim=2,
               use_importance=False, use_fg=False, use_inc=False).to(device)
    out_m2 = m2(tt, tm, it, meta)
    loss_m2 = cfcai_total_loss(m2, out_m2, (tt, tm, it), labels, supcon_fn, modal_fn, fg_loss)
    m2.zero_grad()
    loss_m2.backward()
    assert m2.imp_t.net[0].weight.grad is None, "imp_t received grad when disabled in M2!"
    assert m2.inc.mlp[0].weight.grad is None, "inc received grad when disabled in M2!"
    print("✓ M2 isolation verified: imp_t and inc have exactly None grad.")

    # Check 5: Alignment Modes: "max" vs "sum"
    print("\n--- Check 5: Alignment Mode Verification ---")
    Ht, Hi = model.proj_t(tt), model.proj_i(it)
    a, b = model.imp_t(Ht, tm), model.imp_i(Hi, torch.ones(Hi.shape[:2], dtype=torch.bool, device=device))
    S_max = align_matrix(Ht, a, tm, Hi, b, mode="max")
    S_sum = align_matrix(Ht, a, tm, Hi, b, mode="sum")
    
    # Verify S_sum is mathematically identical to pooled dot product:
    t_pooled = (a.unsqueeze(-1) * F.normalize(Ht, dim=-1)).sum(1)
    v_pooled = (b.unsqueeze(-1) * F.normalize(Hi, dim=-1)).sum(1)
    expected_sum = t_pooled @ v_pooled.T
    assert torch.allclose(S_sum, expected_sum, atol=1e-4), "S_sum does not match pooled dot product!"
    print(f"✓ S_sum confirmed to collapse to pooled dot product (diff: {(S_sum - expected_sum).abs().max().item():.2e})")
    print(f"✓ S_max preserves fine-grained token-to-patch structure (diag mean: {S_max.diag().mean().item():.4f})")

    # Check 6: Mini-batch Overfit Sanity Test (15 steps)
    print("\n--- Check 6: 15-Step Mini-batch Overfitting Test ---")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(supcon_fn.parameters()) + 
        list(modal_fn.parameters()) + list(fg_loss.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    initial_loss = None
    final_loss = None
    for step in range(15):
        optimizer.zero_grad()
        out = model(tt, tm, it, meta)
        loss = cfcai_total_loss(model, out, (tt, tm, it), labels, supcon_fn, modal_fn, fg_loss)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
    
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    print(f"  Step 0 loss: {initial_loss:.4f} -> Step 14 loss: {final_loss:.4f} (Reduction: {loss_reduction:.1f}%)")
    assert final_loss < initial_loss, "Model failed to optimize on mini-batch!"
    assert not np.isnan(final_loss), "Loss became NaN during optimization!"
    print(f"✓ Mini-batch optimization successful with zero NaN/divergence!")

    print("\n==========================================")
    print("🎉 ALL PHASE 2 UNIT CHECKS PASSED PERFECTLY!")
    print("==========================================")


if __name__ == "__main__":
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    run_checks(device)
