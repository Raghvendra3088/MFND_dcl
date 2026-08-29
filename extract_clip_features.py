"""
extract_clip_features.py
==========================
Implements Section 3.3 "CLIP representation":

    "we use CLIP's visual encoder (ViT-B/16) and text encoder to extract the
    image and text features, respectively. Let the extracted text feature be
    f_t in R^{d_t}, and the image feature be f_v in R^{d_v} (for posts
    containing multiple images, we take the average of their features)."

Because MFND-DCL FREEZES CLIP and only trains the GCA + DCL heads on top
(Sec 5.1.2: "a frozen CLIP encoder used only for feature extraction in a
separate preprocessing stage ... the pre-extracted features are loaded
directly, so the CLIP forward pass does not contribute to the reported
training time"), features are extracted ONCE up front and cached to disk.
This matches the paper's design exactly and is also what makes their
reported per-epoch times (2.7s/epoch Weibo, 0.9s/epoch PHEME) achievable.

Usage:
    python extract_clip_features.py --csv data/weibo/weibo_train.csv \
        --out_dir data/weibo/features/train --lang zh
"""

import os
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-base-patch16"  # CLIP ViT-B/16, as specified in Sec 3.3
MAX_TOKENS = 77  # CLIP's native context length


@torch.no_grad()
def extract(csv_path, out_dir, device="cuda", batch_size=32):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    df = pd.read_csv(csv_path)
    # a post may reference multiple images separated by ';' in the image_path
    # column (adjust if your preprocessing script emits a different format)
    for start in tqdm(range(0, len(df), batch_size), desc=f"CLIP features: {csv_path}"):
        batch = df.iloc[start:start + batch_size]

        texts = batch["text"].fillna("").tolist()
        text_inputs = processor(text=texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=MAX_TOKENS).to(device)
        text_feats = model.get_text_features(**text_inputs)          # [B, 512]
        text_feats = text_feats.float().cpu()

        image_feats_batch = []
        for _, row in batch.iterrows():
            image_paths = str(row["image_path"]).split(";")
            imgs = []
            for p in image_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue
            if not imgs:
                image_feats_batch.append(torch.zeros(512))
                continue
            img_inputs = processor(images=imgs, return_tensors="pt").to(device)
            feats = model.get_image_features(**img_inputs).float().cpu()  # [k, 512]
            # "for posts containing multiple images, we take the average of
            # their features" -- Sec 3.3
            image_feats_batch.append(feats.mean(dim=0))
        image_feats = torch.stack(image_feats_batch, dim=0)

        for i, (_, row) in enumerate(batch.iterrows()):
            pid = row.get("post_id", row.get("tweet_id"))
            torch.save(
                {"text_feat": text_feats[i], "image_feat": image_feats[i],
                 "label": int(row["label"])},
                out_dir / f"{pid}.pt",
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    extract(args.csv, args.out_dir, args.device, args.batch_size)
