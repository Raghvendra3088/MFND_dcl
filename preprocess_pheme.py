"""
preprocess_pheme.py
=====================
Prepares the PHEME rumor dataset (Zubiaga et al., 2017) to match Table 2 of
the paper:

    Event               Rumours   Non-rumours   Total
    Charlie Hebdo         458        1621        2079
    Ferguson               284         859        1143
    Germanwings Crash      238         231         469
    Ottawa Shooting        470         420         890
    Sydney Siege            522         699        1221
    TOTAL                  1972        3830        5802

Assumes the standard PHEME release layout (threaded conversations), e.g.:
    pheme-rnr-dataset/
        charliehebdo/
            rumours/<tweet_id>/source-tweet/<tweet_id>.json
            non-rumours/<tweet_id>/source-tweet/<tweet_id>.json
        ferguson/
        germanwings-crash/
        ottawashooting/
        sydneysiege/

Only SOURCE tweets are used (one row per story/thread), matching the
"5,802 tweets" total reported in the paper.

This script:
  1. Walks each event directory, reading every source-tweet JSON.
  2. Applies the retweet-count threshold (paper: "sampled based on a
     retweet threshold (>=100 retweets)") to keep only high-impact posts.
  3. Extracts: tweet id, text, any attached media URL/local image (if the
     PHEME release you downloaded includes cached images -- PHEME's default
     public release is text/JSON-only, so if you don't have images, see the
     NOTE below), event name, label (1 = rumour, 0 = non-rumour).
  4. Performs the paper's 70/10/20 train/val/test split (Sec 4.1.2).

NOTE ON IMAGES: The publicly released PHEME corpus (Zubiaga et al., 2017)
ships tweet JSON/text only; it does not bundle image files. To reproduce the
image modality for PHEME you must separately fetch media referenced by each
tweet's `entities.media[].media_url` field (subject to Twitter's terms and
current API access) before extracting CLIP image features. Any source tweet
whose image cannot be retrieved should be dropped so that, like Weibo, only
genuinely multimodal samples remain (unimodal samples are filtered per the
paper's stated methodology of requiring paired text+image evidence).
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RETWEET_THRESHOLD = 100
EVENT_DIRS = {
    "charliehebdo": "Charlie Hebdo",
    "ferguson": "Ferguson",
    "germanwings-crash": "Germanwings Crash",
    "ottawashooting": "Ottawa Shooting",
    "sydneysiege": "Sydney Siege",
}


def load_source_tweets(pheme_root, event_dir):
    rows = []
    for label_name, label in (("rumours", 1), ("non-rumours", 0)):
        base = Path(pheme_root) / event_dir / label_name
        if not base.exists():
            continue
        for thread_dir in base.iterdir():
            src_dir = thread_dir / "source-tweet"
            if not src_dir.exists():
                continue
            json_files = list(src_dir.glob("*.json"))
            if not json_files:
                continue
            with open(json_files[0], "r", encoding="utf-8") as f:
                tw = json.load(f)

            retweet_count = tw.get("retweet_count", 0)
            if retweet_count < RETWEET_THRESHOLD:
                continue  # paper: retweet threshold >= 100

            media_url = None
            media = tw.get("entities", {}).get("media", [])
            if media:
                media_url = media[0].get("media_url")

            rows.append({
                "tweet_id": tw.get("id_str", thread_dir.name),
                "text": tw.get("text", ""),
                "media_url": media_url,
                "event": EVENT_DIRS.get(event_dir, event_dir),
                "label": label,
            })
    return rows


def main(pheme_root, out_dir, require_image=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for event_dir in EVENT_DIRS:
        all_rows.extend(load_source_tweets(pheme_root, event_dir))

    df = pd.DataFrame(all_rows)
    print("Per-event counts before image filtering:")
    print(df.groupby(["event", "label"]).size())

    if require_image:
        df = df[df.media_url.notna()].reset_index(drop=True)
        print(f"\nAfter requiring an attached image: {len(df)} multimodal samples")
        print("NOTE: run a separate image-download step (see module docstring) "
              "before this filter has any effect -- `media_url` must be resolved "
              "to a local file path and validated as loadable.")

    # paper: 70% train / 10% val / 20% test, stratified by label to preserve
    # the ~34% rumour / 66% non-rumour balance in every split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df.label, random_state=129)
    val_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, stratify=temp_df.label, random_state=129)
    # 0.3 * (2/3) = 0.20 of the full set -> test; 0.3 * (1/3) = 0.10 -> val

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df.to_csv(out_dir / f"pheme_{name}.csv", index=False)
        print(f"[{name}] rumour={sum(split_df.label==1)} "
              f"non-rumour={sum(split_df.label==0)} total={len(split_df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheme_root", required=True,
                     help="Path to pheme-rnr-dataset root (one folder per event)")
    ap.add_argument("--out_dir", default="./data/pheme")
    ap.add_argument("--require_image", action="store_true", default=True)
    args = ap.parse_args()
    main(args.pheme_root, args.out_dir, args.require_image)
