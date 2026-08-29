# Reproducing MFND-DCL (Qian et al., 2026, *Information Processing & Management*)

This is a complete, faithful re-implementation of **"MFND-DCL: Multimodal
fake news detection based on dual contrastive learning"**, built directly
from the equations, figures, and stated hyperparameters in the paper.

**Important caveat:** the paper states its code is released at
`https://github.com/chenyujia-max/MFND-DCL`, but as of writing that
repository is not publicly resolvable/discoverable. Everything here is
therefore reconstructed from the paper's text (Sections 3-5), not copied
from an official codebase. Architectural choices that the paper doesn't
fully pin down numerically (e.g. exact image-size filtering thresholds,
early-stopping patience) are flagged explicitly below and in code comments
— you should treat those as reasonable defaults, not guaranteed-identical
settings.

---

## 0. Requirements

**Hardware** (paper's own setup, Sec 4.4):
- NVIDIA A100-PCIe 40GB GPU (paper). Any modern CUDA GPU with ≥8GB VRAM
  will work — the trainable part of the model is <1.5M parameters; CLIP
  feature extraction is the only GPU-memory-heavy step, and it's frozen
  and run only once.

**Software:**
```
Python 3.10+
torch >= 2.1.0
transformers >= 4.36.0   (HuggingFace CLIP ViT-B/16)
scikit-learn, pandas, numpy, Pillow, tqdm
jieba                    (Chinese text handling for Weibo)
imagehash, opencv-python (near-duplicate image filtering for Weibo)
```
Install with:
```bash
pip install -r requirements.txt
```

**Data** (you said you already have both):
- **Weibo**: Jin et al. (2017) multimodal rumor dataset — raw release with
  `tweets/*.txt` + `rumor_images/`, `nonrumor_images/` folders.
- **PHEME**: Zubiaga et al. (2017) — `pheme-rnr-dataset/<event>/{rumours,non-rumours}/<thread_id>/source-tweet/*.json`.
  Note: the public PHEME release is text/JSON only — it does **not** bundle
  images. To reproduce the image modality you must separately resolve each
  tweet's `entities.media[].media_url` to a downloaded file (see
  `preprocess_pheme.py` docstring for details) before CLIP feature
  extraction. Any tweet whose image can't be resolved is dropped, exactly
  as unimodal samples are dropped from Weibo.

---

## 1. Project layout

```
mfnd_dcl/
├── config.py                 # exact hyperparameters, per-dataset
├── preprocess_weibo.py        # Sec 4.1.1 data cleaning -> CSV
├── preprocess_pheme.py        # Sec 4.1.2 data cleaning -> CSV
├── extract_clip_features.py   # Sec 3.3 CLIP ViT-B/16 feature extraction
├── dataset.py                 # loads cached CLIP features
├── gca.py                     # Sec 3.4 Gated Cross-Attention (Eq. 1-5)
├── dcl_losses.py               # Sec 3.5 sample-level + modal-level losses (Eq. 6-11)
├── model.py                    # full MFND-DCL architecture (Fig. 2)
├── train.py                    # main training loop (Eq. 12, Sec 4.4)
├── ablation.py                 # reproduces Table 5 (3 variants)
├── hparam_sweep.py             # reproduces Tables 6 & 7
└── requirements.txt
```

---

## 2. Step-by-step pipeline

### Step 1 — Clean and split the raw data (Sections 4.1.1 / 4.1.2)

```bash
# Weibo: keeps only multimodal samples, de-duplicates near-identical
# images, filters degenerate image sizes, preserves the event-disjoint
# train/test split shipped with the raw dataset.
python preprocess_weibo.py --raw_dir /path/to/raw_weibo --out_dir data/weibo

# Expected output counts (Table 1):
#   train: fake=3748 real=3783 total=7531
#   test:  fake=1000 real=996  total=1996
```

```bash
# PHEME: keeps only source tweets with >=100 retweets, requires a
# resolvable image, splits 70/10/20 stratified by label (Sec 4.1.2).
python preprocess_pheme.py --pheme_root /path/to/pheme-rnr-dataset --out_dir data/pheme

# Expected raw event counts before the retweet/image filters (Table 2):
#   Charlie Hebdo: 458 rumours / 1621 non-rumours
#   Ferguson: 284 / 859
#   Germanwings Crash: 238 / 231
#   Ottawa Shooting: 470 / 420
#   Sydney Siege: 522 / 699
#   TOTAL: 1972 / 3830 = 5802
```

> If your counts don't match exactly, check dataset version/release date —
> both Weibo and PHEME have circulated in slightly different cleaned
> variants over the years, and the paper doesn't specify a checksum/version
> tag for either.

### Step 2 — Extract frozen CLIP (ViT-B/16) features (Section 3.3)

The paper explicitly freezes CLIP and treats feature extraction as an
offline preprocessing stage (Sec 5.1.2) — this is what makes training so
fast (2.7s/epoch Weibo, 0.9s/epoch PHEME). Run this once per split:

```bash
python extract_clip_features.py --csv data/weibo/weibo_train.csv \
    --out_dir data/weibo/features/train

python extract_clip_features.py --csv data/weibo/weibo_test.csv \
    --out_dir data/weibo/features/test

# carve a validation split out of train for early stopping (paper's Weibo
# split has no separate val set — see note in train.py)
```
Repeat analogously for `data/pheme/pheme_{train,val,test}.csv`.

Each `.pt` file cached contains: `text_feat` (512-d), `image_feat` (512-d,
averaged across images if multiple), and `label`.

### Step 3 — Train MFND-DCL (Section 4.4, Eq. 12)

```bash
python train.py --dataset weibo \
    --train_dir data/weibo/features/train \
    --val_dir   data/weibo/features/val \
    --test_dir  data/weibo/features/test \
    --save_path checkpoints/mfnd_dcl_weibo.pt
```

```bash
python train.py --dataset pheme \
    --train_dir data/pheme/features/train \
    --val_dir   data/pheme/features/val \
    --test_dir  data/pheme/features/test \
    --save_path checkpoints/mfnd_dcl_pheme.pt
```

This runs the exact recipe from Section 4.4:
| Setting | Weibo | PHEME |
|---|---|---|
| Batch size | 128 | 256 |
| Learning rate | 1e-4 | 6e-5 |
| Dropout | 0.3 | 0.3 (paper doesn't restate this for PHEME) |
| Class weights | none | [1.0, 2.4] |
| λ_sample / λ_modal | 0.5 / 0.3 | 0.5 / 0.3 |
| τ (temperature) | 0.2 | 0.2 |
| Optimizer | Adam + grad clip | Adam + grad clip |
| Max epochs | 100 (early stop on val F1) | 100 (early stop on val F1) |
| Seed | 129 | 129 |

**Target results to match (Tables 3-4):**
- Weibo: Accuracy **0.955**, Fake F1 **0.957**, Real F1 **0.953**
- PHEME: Accuracy **0.901**, Precision **0.846**, Recall **0.867**, F1 **0.856**

### Step 4 — Ablation study (Section 5.2 / Table 5)

```bash
python ablation.py --dataset weibo \
    --train_dir data/weibo/features/train \
    --val_dir   data/weibo/features/val \
    --test_dir  data/weibo/features/test
```
Reproduces all three rows of Table 5: `w/o GCA DCL`, `w/o DCL`, and the
full model, so you can confirm each module's individual contribution.

### Step 5 — Hyperparameter sensitivity (Sections 5.3 / 5.4, Tables 6-7)

```bash
python hparam_sweep.py --sweep temperature --dataset weibo \
    --train_dir data/weibo/features/train --val_dir data/weibo/features/val --test_dir data/weibo/features/test

python hparam_sweep.py --sweep lr --dataset weibo \
    --train_dir data/weibo/features/train --val_dir data/weibo/features/val --test_dir data/weibo/features/test
```
Expected peak: **τ = 0.2** and **lr = 1e-4**, both giving F1 ≈ 0.9565.

### Step 6 — Feature-space visualization (Section 5.5 / Fig. 8, optional)

Not scripted above (it's a diagnostic, not part of the training pipeline),
but easy to add: after training, pull `out["fused"]` for a batch of test
samples at different checkpoints (or every N epochs during training),
reduce to 3D with `sklearn.decomposition.PCA(n_components=3)`, and compute
`sklearn.metrics.silhouette_score(features, labels)`. Paper reports
0.0225 (no DCL) → 0.6338 (epoch 15) → 0.6476 (epoch 25) on Weibo.

---

## 3. Critical implementation notes (read before you run large sweeps)

1. **Temperature discrepancy in the paper itself.** Section 4.4 states
   τ=0.1 as the standard setting, but the headline Table 3/4/5 numbers
   (F1=0.9565 on Weibo) exactly match the τ=0.2 row of Table 6. `config.py`
   defaults to **τ=0.2** to match the reported results — see the long
   comment at the top of `config.py` if you want τ=0.1 instead (expect
   roughly 1 point lower accuracy/F1).

2. **CLIP is frozen.** Do not fine-tune CLIP — the paper is explicit that
   only the GCA module and the two projection/classification heads are
   trainable (1.46M params on Weibo, 0.94M on PHEME per Sec 5.1.2). If your
   trainable parameter count differs a lot from these, you've likely made
   CLIP (or part of it) trainable by mistake.

3. **PHEME has no images in the public release.** This is the single
   biggest practical obstacle to an exact reproduction — you must source
   images yourself (subject to platform terms/availability at the time you
   attempt this), and unresolved-image tweets should be dropped, matching
   how Weibo drops unimodal samples.

4. **The GCA equations describe a single query/key/value token per
   modality** (pooled CLIP embeddings, not patch/token sequences), so the
   "cross-attention" in this setup is effectively a single-token gated
   fusion rather than a multi-token attention pattern. `gca.py` is written
   generically (accepts `[B, L, d]`) so you can swap in CLIP's token-level
   outputs later if you want to explore beyond what's literally described.

5. **Reproducing exact digits.** Even with identical hyperparameters,
   expect results within roughly ±0.5-1.0 points of the paper's numbers
   rather than bit-for-bit identical values — the paper doesn't specify
   PyTorch/CUDA/transformers versions, exact weight initialization scheme,
   or non-determinism settings beyond the single global seed, all of which
   can shift results slightly even with fixed seeds on different hardware.

6. **Validation split for Weibo.** The paper's Weibo protocol (Sec 4.1.1)
   is a plain 80/20 train/test split with no stated validation set, yet
   Section 4.4 requires a validation F1 for early stopping. Carve a small
   held-out slice (e.g. 10%) out of the *training* set for this purpose —
   never use the test set for early-stopping decisions.

7. **Image-size filtering thresholds (Weibo).** The paper says only that
   "very small or excessively large images were filtered" without giving
   numeric thresholds. `preprocess_weibo.py` uses conservative defaults
   (`MIN_IMG_SIDE=100`, `MAX_IMG_SIDE=4000`) — adjust if your final image
   count doesn't match Table 1's totals.

---

## 4. Evaluation metrics (Section 4.3, Eq. 13-16)

All four scripts above report exactly the metrics defined in the paper:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **F1** = 2 · Precision · Recall / (Precision + Recall)

computed per-class (fake / real) as in Tables 3 and 5, and macro-averaged
where the paper reports a single number (Table 4, PHEME).
