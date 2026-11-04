import os
import sys
import copy
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

torch.set_float32_matmul_precision('high')

from cfca_enhanced_modules import CFCAIEnhanced, FGLoss
from dcl_losses import SupConLoss, MultimodalConsistencyLoss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EnhancedDataset:
    def __init__(self, data_root, split='train'):
        data_path = Path(data_root)
        print(f"Loading {split} dataset from {data_path}...")
        
        # 1. Tokens
        tok_dir = data_path / "tokens" / split
        self.text = torch.from_numpy(np.array(np.lib.format.open_memmap(tok_dir / "text.npy", mode="r"))).float()
        self.text_mask = torch.from_numpy(np.array(np.lib.format.open_memmap(tok_dir / "text_mask.npy", mode="r")))
        self.patch = torch.from_numpy(np.array(np.lib.format.open_memmap(tok_dir / "patch.npy", mode="r"))).float()
        self.labels = torch.from_numpy(np.load(tok_dir / "labels.npy")).long()

        # 2. Chinese-CLIP CLS
        clip_dir = data_path / "cnclip_cls"
        self.clip_t = torch.from_numpy(np.load(clip_dir / f"{split}_text_cls.npy")).float()
        self.clip_i = torch.from_numpy(np.load(clip_dir / f"{split}_img_cls.npy")).float()

        # 3. RoViT CLS (RoBERTa & ViT)
        rovit_dir = data_path / "rovit"
        self.roberta = torch.from_numpy(np.load(rovit_dir / f"{split}_text_cls.npy")).float()
        self.vit = torch.from_numpy(np.load(rovit_dir / f"{split}_img_cls.npy")).float()

        print(f"Loaded {len(self.labels)} {split} samples into memory successfully.")

    def __len__(self):
        return len(self.labels)

class FastEnhancedLoader:
    def __init__(self, text, text_mask, patch, clip_t, clip_i, roberta, vit, labels,
                 batch_size=64, shuffle=True, generator=None):
        self.text = text
        self.text_mask = text_mask
        self.patch = patch
        self.clip_t = clip_t
        self.clip_i = clip_i
        self.roberta = roberta
        self.vit = vit
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.n = len(labels)

    def __iter__(self):
        if self.shuffle:
            if self.generator is not None:
                indices = torch.randperm(self.n, generator=self.generator)
            else:
                indices = torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)

        for i in range(0, self.n, self.batch_size):
            b_idx = indices[i:i + self.batch_size]
            yield (
                self.text[b_idx],
                self.text_mask[b_idx],
                self.patch[b_idx],
                self.clip_t[b_idx],
                self.clip_i[b_idx],
                self.roberta[b_idx],
                self.vit[b_idx],
                self.labels[b_idx]
            )

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            tt, tm, it, ct, ci, rob, vit, labels = [x.to(device) for x in batch]
            out = model(tt, tm, it, ct, ci, rob, vit)
            probs = F.softmax(out["logits"], dim=-1).cpu()
            preds = probs.argmax(dim=-1)
            all_preds.append(preds)
            all_labels.append(labels.cpu())
            all_probs.append(probs)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)[2]
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "probs": all_probs,
        "labels": all_labels
    }

def get_ablation_config(config_name):
    # E0: Baseline Clean CFCA-I (M5a reproduction)
    # E1: + Multi-Scale Pooling (Change 1)
    # E2: + Multi-Scale + Deep Ensemble (Change 1 + 2)
    # E3: + L_FG Warm-Up Curriculum (Change 1 + 2 + 3)
    # E4: Full Enhanced (+ Augmentation & Temp Clamp) (All 5 Changes)
    configs = {
        "E0": dict(use_multiscale=False, use_ensemble=False, warmup=False, word_mask_p=0.0, patch_mask_k=0),
        "E1": dict(use_multiscale=True,  use_ensemble=False, warmup=False, word_mask_p=0.0, patch_mask_k=0),
        "E2": dict(use_multiscale=True,  use_ensemble=True,  warmup=False, word_mask_p=0.0, patch_mask_k=0),
        "E3": dict(use_multiscale=True,  use_ensemble=True,  warmup=True,  word_mask_p=0.0, patch_mask_k=0),
        "E4": dict(use_multiscale=True,  use_ensemble=True,  warmup=True,  word_mask_p=0.10, patch_mask_k=7),
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Supported: {list(configs.keys())}")
    return configs[config_name]

def train_single_run(args, seed, full_train_ds, test_ds):
    set_seed(seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    cfg = get_ablation_config(args.config)

    val_size = int(len(full_train_ds) * 0.10)
    train_size = len(full_train_ds) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train_ds), generator=split_gen)
    tr_idx, va_idx = perm[:train_size], perm[train_size:]

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = FastEnhancedLoader(
        full_train_ds.text[tr_idx], full_train_ds.text_mask[tr_idx], full_train_ds.patch[tr_idx],
        full_train_ds.clip_t[tr_idx], full_train_ds.clip_i[tr_idx],
        full_train_ds.roberta[tr_idx], full_train_ds.vit[tr_idx], full_train_ds.labels[tr_idx],
        batch_size=args.bs, shuffle=True, generator=train_gen
    )
    val_loader = FastEnhancedLoader(
        full_train_ds.text[va_idx], full_train_ds.text_mask[va_idx], full_train_ds.patch[va_idx],
        full_train_ds.clip_t[va_idx], full_train_ds.clip_i[va_idx],
        full_train_ds.roberta[va_idx], full_train_ds.vit[va_idx], full_train_ds.labels[va_idx],
        batch_size=args.bs, shuffle=False
    )
    test_loader = FastEnhancedLoader(
        test_ds.text, test_ds.text_mask, test_ds.patch,
        test_ds.clip_t, test_ds.clip_i,
        test_ds.roberta, test_ds.vit, test_ds.labels,
        batch_size=args.bs, shuffle=False
    )

    model = CFCAIEnhanced(
        d_text=1024, d_img=1024, d_clip_cls=1024, d_roberta=768, d_vit=768,
        d=128, heads=4, drop=args.dropout,
        use_multiscale=cfg["use_multiscale"],
        use_ensemble=cfg["use_ensemble"],
        use_importance=True,
        use_fg=True,
        align_mode="max",
        word_mask_p=cfg["word_mask_p"],
        patch_mask_k=cfg["patch_mask_k"]
    ).to(device)

    # Change 5: Temperature strictly clamped in [0.05, 0.50]
    supcon_fn = SupConLoss(temperature=0.15, min_temperature=0.05, max_temperature=0.50, learnable_temp=True).to(device)
    modal_fn = MultimodalConsistencyLoss(temperature=0.10, min_temperature=0.05, max_temperature=0.50, learnable_temp=True).to(device)
    fg_loss = FGLoss(tau=0.15, tmin=0.05, tmax=0.50).to(device)

    params = list(model.parameters()) + list(supcon_fn.parameters()) + list(modal_fn.parameters()) + list(fg_loss.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    print(f"\n--- Running Enhanced CFCA-I [{args.config}] | Seed {seed} on {device} ---", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0

        # Change 3: Linear Warm-Up Curriculum for Fine-Grained Loss (L_FG)
        if cfg["warmup"]:
            if epoch <= 5:
                cur_lam_fg = 0.0
            else:
                cur_lam_fg = args.lam_fg * (epoch - 5) / (args.max_epochs - 5)
        else:
            cur_lam_fg = args.lam_fg

        for batch in train_loader:
            tt, tm, it, ct, ci, rob, vit, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()

            out = model(tt, tm, it, ct, ci, rob, vit)

            loss_cls = F.cross_entropy(out["logits"], labels)
            loss_sup = supcon_fn(out["z_sample"], labels)
            loss_mod = modal_fn(out["z_text"], out["z_image"])
            loss_fg = fg_loss(out["S"]) if cur_lam_fg > 0 else 0.0

            loss = loss_cls + args.lam_s * loss_sup + args.lam_m * loss_mod + cur_lam_fg * loss_fg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * tt.size(0)

        avg_loss = total_loss / train_size
        val_metrics = evaluate(model, val_loader, device)

        tau_s = supcon_fn.get_temperature().item()
        tau_m = modal_fn.get_temperature().item()
        tau_fg = fg_loss.tau().item()

        if epoch % 5 == 0 or val_metrics["macro_f1"] > best_val_f1:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | lam_fg: {cur_lam_fg:.3f} | tau_s: {tau_s:.3f} | tau_m: {tau_m:.3f} | tau_fg: {tau_fg:.3f}", flush=True)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (Best: Epoch {best_epoch:02d}, Val F1: {best_val_f1:.4f})", flush=True)
                break

    # Evaluate on Test Set using best validation checkpoint
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    test_metrics["best_epoch"] = best_epoch
    print(f"[{args.config} Seed {seed} FINISHED] Test Acc: {test_metrics['accuracy']*100:.2f}% | Test F1: {test_metrics['macro_f1']*100:.2f}% (Best Val F1: {best_val_f1*100:.2f}%)", flush=True)

    out_dir = Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"preds_enhanced_{args.config}_seed{seed}.npy", test_metrics["preds"])
    np.save(out_dir / f"probs_enhanced_{args.config}_seed{seed}.npy", test_metrics["probs"])

    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Enhanced CFCA-I Training Runner")
    parser.add_argument("--config", type=str, default="E4", choices=["E0", "E1", "E2", "E3", "E4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lam_s", type=float, default=0.2)
    parser.add_argument("--lam_m", type=float, default=0.1)
    parser.add_argument("--lam_fg", type=float, default=0.15)
    parser.add_argument("--data_root", type=str, default="/DATA/raghvendra3088/MFND_project/data/weibo")
    parser.add_argument("--pred_dir", type=str, default="/DATA/raghvendra3088/MFND_project/files/predictions")
    args = parser.parse_args()

    full_train_ds = EnhancedDataset(args.data_root, split="train")
    test_ds = EnhancedDataset(args.data_root, split="test")

    train_single_run(args, args.seed, full_train_ds, test_ds)

if __name__ == "__main__":
    main()

