"""
extract_tokens_cnclip.py -- One-time offline extraction of TOKEN and PATCH features using cn_clip.
Runs on GPU. Compatible with OFA-Sys cn_clip ViT-H-14 local model.

Usage:
  python extract_tokens_cnclip.py --csv ../data/weibo/weibo_train.csv --out ../data/weibo/tokens/train --L 64 --pool 7 --bs 32
  python extract_tokens_cnclip.py --csv ../data/weibo/weibo_test.csv --out ../data/weibo/tokens/test --L 64 --pool 7 --bs 32
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import load_from_name


@torch.no_grad()
def extract_split(a):
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() and a.gpu >= 0 else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Chinese-CLIP ViT-H-14 on {dev}...")
    model, preprocess = load_from_name("ViT-H-14", device=dev, download_root=a.model_dir)
    model.eval()

    df = pd.read_csv(a.csv)
    N = len(df)
    D = 1024  # ViT-H-14 joint projection dim

    os.makedirs(a.out, exist_ok=True)
    T = np.lib.format.open_memmap(f"{a.out}/text.npy", "w+", np.float16, (N, a.L, D))
    TM = np.lib.format.open_memmap(f"{a.out}/text_mask.npy", "w+", np.bool_, (N, a.L))
    V = np.lib.format.open_memmap(f"{a.out}/patch.npy", "w+", np.float16, (N, a.pool ** 2, D))
    META = np.lib.format.open_memmap(f"{a.out}/meta.npy", "w+", np.float32, (N, 2))
    np.save(f"{a.out}/labels.npy", df["label"].values.astype(np.int64))

    csv_dir = Path(a.csv).parent
    img_fallback = Image.new("RGB", (224, 224), (0, 0, 0))

    for s in tqdm(range(0, N, a.bs), desc=f"Extracting {Path(a.csv).name}"):
        rows = df.iloc[s:s + a.bs]
        e = s + len(rows)

        # 1. Process Text Tokens
        texts = rows["text"].fillna("").tolist()
        ti = clip.tokenize(texts, context_length=a.L).to(dev)
        t_mask = (ti != 0).cpu().numpy()

        text_out = model.bert(ti)
        th = text_out[0]  # [B, L, 1024]
        if isinstance(model.text_projection, torch.Tensor):
            tp = th @ model.text_projection
        else:
            tp = model.text_projection(th)
        tp = F.normalize(tp, dim=-1)

        # 2. Process Image Patches
        imgs_batch = []
        for _, row in rows.iterrows():
            img_paths = str(row.get("image_path", "")).split(";")
            loaded_img = None
            for p in img_paths:
                p_full = p if os.path.isabs(p) else os.path.join(csv_dir, p)
                if not os.path.exists(p_full):
                    # Also try relative to current dir
                    p_full = p
                try:
                    loaded_img = Image.open(p_full).convert("RGB")
                    break
                except Exception:
                    continue
            if loaded_img is None:
                loaded_img = img_fallback
            imgs_batch.append(preprocess(loaded_img))

        v = model.visual
        px = torch.stack(imgs_batch, dim=0).to(dev).to(v.conv1.weight.dtype)

        # Vision transformer forward for patch tokens
        x = v.conv1(px)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = v.transformer(x)
        x = v.ln_post(x)
        patch_tokens = x[:, 1:]  # drop CLS token [B, 256, 1280]

        if isinstance(v.proj, torch.Tensor):
            vp = patch_tokens @ v.proj
        else:
            vp = v.proj(patch_tokens)
        vp = F.normalize(vp, dim=-1)

        # Adaptive pool 16x16 -> pool x pool (e.g. 7x7 = 49)
        B_curr, P, _ = vp.shape
        g = int(P ** 0.5)
        vp_grid = vp.transpose(1, 2).reshape(B_curr, D, g, g)
        vp_pooled = F.adaptive_avg_pool2d(vp_grid.float(), (a.pool, a.pool)).flatten(2).transpose(1, 2)
        vp_pooled = F.normalize(vp_pooled, dim=-1)

        # 3. Process Temporal Metadata (kept for ablation comparisons)
        meta_batch = []
        for _, row in rows.iterrows():
            timestamp_str = str(row.get("timestamp", ""))
            try:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                hour_norm = 2 * np.pi * dt.hour / 24
                meta_batch.append([np.sin(hour_norm), np.cos(hour_norm)])
            except Exception:
                hour_norm = 2 * np.pi * 14 / 24
                meta_batch.append([np.sin(hour_norm), np.cos(hour_norm)])

        # Write to memmap
        T[s:e] = tp.float().cpu().numpy().astype(np.float16)
        TM[s:e] = t_mask
        V[s:e] = vp_pooled.cpu().numpy().astype(np.float16)
        META[s:e] = np.array(meta_batch, dtype=np.float32)

    for m in (T, TM, V, META):
        m.flush()
    print(f"Extraction complete for {a.out}! Shapes: Text {T.shape}, Patch {V.shape}, Meta {META.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--model_dir", default="./")
    extract_split(ap.parse_args())
