"""
preprocess_weibo.py
====================
Prepares the Weibo multimodal rumor dataset (Jin et al., 2017) to match the
statistics reported in the paper (Table 1):

    Split   Fake   Real   Total
    Train   3748   3783   7531
    Test    1000    996   1996
    TOTAL   4748   4779   9527

Assumes you already have the raw Jin et al. Weibo dataset downloaded, with
the standard layout:
    weibo/
        tweets/train_rumor.txt
        tweets/train_nonrumor.txt
        tweets/test_rumor.txt
        tweets/test_nonrumor.txt
        rumor_images/*.jpg
        nonrumor_images/*.jpg

Each txt file is formatted as (Jin et al.'s original release):
    line 1: post_id | image_id(s) | ...
    line 2: post text
    line 3: post_id | image_id(s) | ...
    line 4: post text
    ...

This script:
  1. Parses text + associated image ids for every post.
  2. Keeps only posts that have >=1 loadable, non-corrupted image
     (paper: "unimodal samples filtered out to ensure multimodal integrity").
  3. Removes near-duplicate images via perceptual hashing (an accessible
     stand-in for the paper's Locality Sensitive Hashing / LSH step).
  4. Filters out images that are too small or too large.
  5. Writes a clean, unified CSV per split: post_id, text, image_path, label
     (label: 1 = fake, 0 = real), preserving the event-based split that
     ships with the raw data (train/test files are already event-disjoint
     in the original release, matching Sec 4.1.1's claim that "samples from
     the same event do not appear across different splits").
"""

import os
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
import imagehash

MIN_IMG_SIDE = 100     # "very small ... images were filtered" -- paper does not give
MAX_IMG_SIDE = 4000    # exact thresholds; these are conservative, commonly-used defaults.
                        # Adjust if you know the exact thresholds used by Jin et al. / the authors.
HASH_DIST_THRESHOLD = 5  # perceptual-hash Hamming distance below which two images
                          # are treated as near-duplicates (LSH-style de-dup)


def parse_split_file(path):
    """Parses one of Jin et al.'s train_rumor.txt / train_nonrumor.txt style files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    records = []
    for i in range(0, len(lines) - 1, 2):
        meta_line, text_line = lines[i], lines[i + 1]
        parts = meta_line.split("|")
        post_id = parts[0]
        image_ids = [p for p in parts[1:-2] if p]  # image id list (varies by release)
        records.append({"post_id": post_id, "image_ids": image_ids, "text": text_line})
    return records


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            if w < MIN_IMG_SIDE or h < MIN_IMG_SIDE:
                return False
            if w > MAX_IMG_SIDE or h > MAX_IMG_SIDE:
                return False
        return True
    except Exception:
        return False


def dedup_by_phash(records_with_images, image_root):
    """Removes near-duplicate images across the corpus using perceptual hashing,
    an accessible stand-in for the paper's LSH-based de-duplication step."""
    seen_hashes = []
    kept = []
    for rec in records_with_images:
        img_path = os.path.join(image_root, rec["image_file"])
        try:
            h = imagehash.phash(Image.open(img_path))
        except Exception:
            continue
        is_dup = any((h - sh) <= HASH_DIST_THRESHOLD for sh in seen_hashes)
        if not is_dup:
            seen_hashes.append(h)
            kept.append(rec)
    return kept


def build_split(txt_path, image_dir, label, image_ext_candidates=(".jpg", ".jpeg", ".png")):
    parsed = parse_split_file(txt_path)
    out = []
    for rec in parsed:
        chosen_img = None
        for img_id in rec["image_ids"]:
            for ext in image_ext_candidates:
                candidate = f"{img_id}{ext}"
                full = os.path.join(image_dir, candidate)
                if os.path.exists(full) and is_valid_image(full):
                    chosen_img = candidate
                    break
            if chosen_img:
                break
        if chosen_img is None:
            continue  # drop unimodal (text-only / no valid image) samples
        out.append({
            "post_id": rec["post_id"],
            "text": rec["text"],
            "image_file": chosen_img,
            "label": label,
        })
    return out


def main(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [
            (raw_dir / "tweets/train_rumor.txt", raw_dir / "rumor_images", 1),
            (raw_dir / "tweets/train_nonrumor.txt", raw_dir / "nonrumor_images", 0),
        ],
        "test": [
            (raw_dir / "tweets/test_rumor.txt", raw_dir / "rumor_images", 1),
            (raw_dir / "tweets/test_nonrumor.txt", raw_dir / "nonrumor_images", 0),
        ],
    }

    for split_name, sources in splits.items():
        all_records = []
        for txt_path, image_dir, label in sources:
            recs = build_split(txt_path, image_dir, label)
            recs = dedup_by_phash(recs, image_dir)
            for r in recs:
                r["image_path"] = str(image_dir / r["image_file"])
            all_records.extend(recs)

        df = pd.DataFrame(all_records)[["post_id", "text", "image_path", "label"]]
        df.to_csv(out_dir / f"weibo_{split_name}.csv", index=False)
        n_fake = (df.label == 1).sum()
        n_real = (df.label == 0).sum()
        print(f"[{split_name}] fake={n_fake} real={n_real} total={len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Path to raw Jin et al. Weibo release")
    ap.add_argument("--out_dir", default="./data/weibo", help="Where to write cleaned CSVs")
    args = ap.parse_args()
    main(args.raw_dir, args.out_dir)
