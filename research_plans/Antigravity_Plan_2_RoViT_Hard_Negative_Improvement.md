# Antigravity Implementation Plan 2 — RoViT-Inspired Hard-Negative Learning for Multimodal Fake News Detection

> **Goal:** Build a research extension that keeps the strong RoBERTa + ViT encoders from RoViT-Detect but explicitly trains the model to distinguish difficult real/fake examples and misleading text-image pairings.
>
> **Working name:** `HN-RoViT` (Hard-Negative-aware RoBERTa–ViT).
>
> **Research claim to test, not assume:** hard-negative-aware training improves discrimination on confusing examples and reduces false predictions compared with the same RoViT late-fusion baseline. It may or may not improve overall accuracy; evaluate macro-F1, per-class recall, calibration, and hard-subset metrics too.

---

## 0. Critical distinction

Plan 1 reproduces the paper-style late-fusion architecture. This plan is a **new proposed method**, not the exact RoViT-Detect model and not a claim about what the source paper already implements.

The source paper uses RoBERTa + ViT, global CLS features, late fusion, and an MLP with cross-entropy. This extension adds explicit hard-example selection and a supervised contrastive objective. Start with global embeddings and a separate hard-negative training objective; only add token-to-patch cross-attention as a separate ablation if compute and timeline permit.

Do not assume that every visually similar or semantically similar sample is a valid negative. In fake-news datasets, samples can share an event, image, or claim and still share the same label.

---

## 1. Proposed architecture

```text
Text T ─> Chinese RoBERTa ─> token states Ht [B, L, Dt]
                    └─────> CLS text vector zt [B, D]
Image I ─> ViT-Base/16 ───> patch states Hv [B, P+1, Dv]
                    └─────> CLS image vector zv [B, D]
                                      |
                         projection heads (shared dim d)
                          pt = normalize(Wt zt)
                          pv = normalize(Wv zv)
                                      |
              ┌───────────────────────┴───────────────────────┐
              |                                               |
      late-fusion classifier                         hard-pair scorer
   concat(zt, zv) -> MLP -> logits              score(text, image) / similarity
              |                                               |
     cross-entropy L_cls                       hard-negative pair mining
              └───────────────────────┬───────────────────────┘
                                      |
          supervised contrastive / pairwise hard-negative loss
                                      |
        L_total = L_cls + λ_con L_supcon + λ_pair L_pair
```

### Optional extension (do not enable in the first implementation)

```text
Ht token states × Hv image patch states
  -> cross-modal attention / token-patch alignment
  -> optional cross-modal fused vector
  -> classifier
```

The optional local interaction module addresses a different issue—fine-grained text-image interaction. It should be evaluated separately from hard-negative training to identify which component contributes improvement.

---

## 2. Implementation strategy: safest first version

### Version A — Hard-negative learning without changing the encoder/fusion architecture

Keep the same RoBERTa + ViT CLS concatenation classifier as Plan 1. Add:
1. normalized projection heads for text and image features;
2. supervised contrastive loss over correctly labeled examples;
3. a label-aware hard-negative sampler for the contrastive branch;
4. optionally, a pairwise ranking loss for verified mismatched text-image pairs;
5. a weighted total loss.

This isolates the effect of hard-negative learning. It is the recommended first experiment.

### Version B — Optional local cross-modal module

Only after Version A is stable:
1. take token states excluding padding and special tokens where appropriate;
2. take ViT patch tokens excluding CLS;
3. apply cross-attention in both directions or one direction (choose one and document it);
4. pool the attended states;
5. fuse local interaction features with the global CLS features;
6. keep the same hard-negative objective.

Do not introduce Version B until Version A has passed all tests and the baseline is saved.

---

## 3. Dataset contract and split safety

Use the same manifest and frozen splits as Plan 1. Required fields:

- `sample_id`
- `text`
- `image_path`
- `label` (`0=real`, `1=fake`, fixed everywhere)
- optional `event_id`, `source_id`, `image_id`, `claim_id`, `thread_id`

Before mining:
- identify duplicates and near duplicates;
- calculate image hashes;
- identify repeated images and event/claim clusters;
- ensure train, validation, and test split isolation;
- create a `hard_negative_audit.csv` with candidate pair IDs, selection reason, labels, event/claim IDs if known, and a verification status.

**Never mine negatives from validation or test data.** The sampler must use training samples only. Validation/test are for evaluation only.

### Candidate hard-negative types

1. **Same-event, opposite-label samples:** candidate only; verify that the labels and event relation really make the pair semantically contrasting.
2. **Similar text, opposite label:** nearest text embeddings among training samples with opposite labels; inspect duplicates and event leakage.
3. **Similar images, opposite label:** nearest image embeddings among training samples with opposite labels; verify the image isn't duplicated with a noisy or incorrect label.
4. **Verified mismatched text-image pairs:** combine a text and image from different posts only when the dataset evidence supports that this is a meaningful negative pair. Do not automatically treat all cross-sample combinations as semantically false.
5. **In-batch hard negatives:** among examples in the same batch, select the highest-similarity examples with an opposite label.

### False-negative protection

- Do not label two posts as negatives solely because their sample IDs differ.
- Do not create an opposite-label pair if the label is missing, ambiguous, or inconsistent.
- Exclude known duplicates and same-claim/same-veracity pairs from negative candidates.
- If event/claim metadata is absent, mark pair quality as `unverified`; use it only in a conservative in-batch label-aware strategy or manual audit.
- Log the count of candidates, accepted pairs, rejected pairs, and rejection reasons.
- Start with a small hard-negative ratio and compare it with random-negative sampling.

---

## 4. Loss functions and exact contracts

### 4.1 Classification loss

For two-class logits `logits [B,2]` and integer labels `y [B]`:

\[
L_{\mathrm{cls}}=\mathrm{CrossEntropyLoss}(\mathrm{logits}, y)
\]

Use raw logits; do not apply softmax before `CrossEntropyLoss`.

### 4.2 Projection embeddings

Let text and image CLS vectors be `zt` and `zv`. Use separate projection heads:

\[
p_t=\operatorname{normalize}(g_t(z_t)),\qquad
p_v=\operatorname{normalize}(g_v(z_v))
\]

Choose a configurable shared dimension `d=256` initially. This is a proposed implementation choice, not a dimension specified by the source paper. Assert both outputs have shape `[B, 256]`.

### 4.3 Label-aware supervised contrastive loss

Implement supervised contrastive loss over one selected embedding per sample (initially the fused or text/image projected features). For each anchor, positives have the same label and negatives have the opposite label. Use a stable log-sum-exp implementation and temperature `τ=0.07` as an initial configurable value.

Because fake/real labels alone can group semantically unrelated posts, compare:
- `SupCon-fused`: projection of fused representation;
- `SupCon-text-image`: text and image views for the same post treated as positive views;
- optional label-aware contrastive loss.

Document the exact formula and positive/negative mask in code comments and `docs/loss_design.md`. Do not call a loss “dual contrastive” unless the implementation genuinely contains the two intended contrastive objectives.

### 4.4 Pairwise hard-negative ranking loss (optional, enable only with verified pairs)

For a matched text-image pair score `s_pos` and a verified mismatched pair score `s_neg`:

\[
L_{\mathrm{pair}}=\max(0,\ m-s_{\mathrm{pos}}+s_{\mathrm{neg}})
\]

where margin `m=0.2` is a configurable starting value. Similarity can be cosine similarity of normalized projection embeddings. This term should only use verified pairs, not arbitrary cross-batch pairs.

### 4.5 Total loss

Start with:

\[
L_{\mathrm{total}}=L_{\mathrm{cls}}+
\lambda_{\mathrm{con}}L_{\mathrm{supcon}}+
\lambda_{\mathrm{pair}}L_{\mathrm{pair}}
\]

Suggested initial config:
- `lambda_con = 0.1`
- `lambda_pair = 0.1` only when verified mismatched pairs exist; otherwise `0.0`
- `temperature = 0.07`
- `margin = 0.2`
- `hard_negative_ratio = 0.25` initially, then ablate `0.0`, `0.10`, `0.25`, `0.50`
- `projection_dim = 256`

These are starting hyperparameters for experiments, not values claimed by the source paper. Run a small validation-only tuning sweep; do not tune on the test set.

---

## 5. Training and mining schedule

### Recommended staged training

**Stage 0 — Baseline**
- Train Plan 1 architecture with classification loss only.
- Save baseline checkpoint, split manifests, and metrics.

**Stage 1 — Warm start**
- Initialize from the baseline checkpoint.
- Add projection heads.
- Train with classification + supervised contrastive loss.
- Start with conservative loss weight `0.1`.

**Stage 2 — Hard-negative mining**
- At the beginning of each epoch (or every configured `N` steps), compute/update embeddings for training samples only.
- Build a nearest-neighbor candidate list using cosine similarity.
- Filter candidates by opposite label and known duplicate/claim constraints.
- Use top-k candidates only after applying false-negative checks.
- Freeze the candidate list for the epoch so sampling is deterministic and auditable.
- Save mining metadata and the random seed.

**Stage 3 — Pairwise loss**
- Enable `L_pair` only if verified mismatched pairs exist.
- Compare no-pair-loss vs pair-loss under identical seeds and splits.

**Stage 4 — Optional local cross-attention**
- Add the local token-patch module as a separate experiment.
- Keep the hard-negative sampler and loss settings fixed while measuring its contribution.

Avoid continuously changing the mined set within each mini-batch in the first version; it makes debugging and reproducibility harder.

---

## 6. Suggested repository structure

Create a separate folder; do not overwrite Plan 1 or the user's MFND-DCL project.

```text
hn_rovit/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   └── hard_negative.yaml
├── docs/
│   ├── method.md
│   ├── loss_design.md
│   ├── mining_policy.md
│   └── experiment_protocol.md
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── validate_dataset.py
│   │   └── split_audit.py
│   ├── models/
│   │   ├── rovit_backbone.py
│   │   ├── projection_heads.py
│   │   ├── hard_pair_scorer.py
│   │   └── optional_cross_attention.py
│   ├── losses/
│   │   ├── supervised_contrastive.py
│   │   └── pairwise_ranking.py
│   ├── mining/
│   │   ├── candidate_builder.py
│   │   ├── hard_negative_sampler.py
│   │   └── audit_pairs.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_hn_rovit.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── logging.py
├── tests/
│   ├── test_pair_filtering.py
│   ├── test_loss_finite.py
│   ├── test_no_eval_mining.py
│   ├── test_no_false_negative_duplicates.py
│   ├── test_model_shapes.py
│   └── test_one_train_step.py
├── checkpoints/
├── logs/
└── reports/
```

---

## 7. Antigravity phases and acceptance gates

### Phase 1 — Audit and copy the working baseline

- Inspect Plan 1 output or existing repository.
- Do not alter its model, splits, or checkpoint.
- Copy/reuse only documented data loader and baseline model code.
- Record hashes for split manifests and config.

**Gate:** baseline can be evaluated from a clean process and matches its saved metrics within expected deterministic tolerance.

### Phase 2 — Implement projections and losses

- Add projection heads with L2 normalization.
- Add supervised contrastive loss with unit tests for positive/negative masks.
- Add pairwise ranking loss as a standalone module.
- Verify finite loss and gradients for batch size 2 and edge cases with no valid positives/negatives.
- If an anchor has no valid positives, skip it safely and log the count; do not divide by zero.

**Gate:** unit tests pass; no NaN/Inf; loss returns scalar; gradients reach intended parameters.

### Phase 3 — Implement safe hard-negative candidate mining

- Build candidates from training samples only.
- Compute similarity using frozen or current encoder embeddings; make the selected mode configurable.
- Filter same sample, exact duplicate, near duplicate, ambiguous labels, and known same-claim/same-veracity candidates.
- Store candidate pairs and reasons in CSV.
- Make mining deterministic given a seed.
- Provide `--dry-run` to generate the audit report without training.

**Gate:** tests prove no validation/test IDs enter the mining pool and every selected pair has a recorded reason.

### Phase 4 — Integrate the training objective

- Start from baseline checkpoint.
- Add `L_supcon`; keep `L_pair` disabled until verified pairs are available.
- Log every loss component separately.
- Use the same optimizer/scheduler policy as the baseline initially.
- Preserve early stopping based only on validation metrics.
- Save the configuration and mined-pair manifest with each checkpoint.

**Gate:** a short smoke run completes, checkpoints reload, and each loss component is finite.

### Phase 5 — Add pairwise loss only when justified

- Validate the pairing audit manually or with reliable metadata.
- Run matched-pair vs verified-mismatch-pair ranking loss.
- Compare `lambda_pair=0` and `lambda_pair=0.1`.
- If the dataset cannot establish reliable mismatches, leave this feature disabled and state that limitation.

**Gate:** all pairs in the loss can be traced to the audit manifest.

### Phase 6 — Evaluate hard-example performance

In addition to normal test metrics, create a **predefined hard test subset** before examining model outputs. Examples:
- high text-embedding similarity but opposite labels, after duplicate/event checks;
- high image similarity but opposite labels, after pair verification;
- verified misleading/mismatched image-text pairs;
- samples where text-only and image-only models disagree.

Do not train or mine from the test subset. It is evaluation-only.

Report:
- accuracy, macro-F1, weighted-F1, precision/recall per class
- confusion matrix
- ROC-AUC and PR-AUC where appropriate
- hard-subset accuracy, macro-F1, and per-class recall
- calibration (ECE or Brier score if implemented correctly)
- performance by event/claim group where metadata exists
- mean ± standard deviation across at least 3 seeds if compute permits
- candidate counts, verified pair counts, and rejected pair counts

**Gate:** the baseline and HN model use the exact same split manifests and test examples.

### Phase 7 — Optional local token-patch interaction

Only after the hard-negative extension is stable:
- use token states `[B,L,Dt]` and ViT patch states `[B,P,Dv]` excluding CLS for the patch-only sequence;
- project both to a common dimension `d`;
- implement a clearly documented cross-attention direction (text queries over image patches, or bidirectional);
- mask padding tokens correctly;
- pool the interaction output and fuse it with global features;
- add shape, masking, and gradient tests;
- compare against the same HN model without cross-attention.

Do not add a VAE/reliability gate simultaneously. Too many changes at once make attribution impossible.

---

## 8. Required ablation matrix

Run experiments on identical splits and seeds wherever possible.

| ID | Architecture / training | What it tests |
|---|---|---|
| A | RoViT late fusion + CE | Reproduced baseline |
| B | A + supervised contrastive loss | Effect of contrastive representation learning |
| C | A + label-aware hard-negative sampler | Effect of selecting difficult negatives |
| D | A + supervised contrastive + hard-negative sampler | Combined effect |
| E | D + verified pairwise ranking loss | Effect of explicit text-image mismatch discrimination |
| F (optional) | D/E + local token-patch cross-attention | Added value of fine-grained interaction |

For every row, report the same standard metrics and hard-subset metrics. Repeat the main comparison with at least 3 seeds if resources allow. Do not select only the best seed.

---

## 9. Failure modes Antigravity must actively check

- **False negatives:** semantically related or same-claim examples treated as negatives.
- **Label leakage:** label words, filenames, folder names, IDs, or source metadata reveal labels.
- **Split leakage:** duplicate text/image/event crosses splits.
- **Test contamination:** test samples enter nearest-neighbor index or pair mining.
- **Loss collapse:** embeddings become constant or all cosine similarities saturate.
- **Class imbalance:** sampler overrepresents one class; log class distribution of anchors and negatives.
- **No valid positives:** handle anchors with no same-label positives safely.
- **Unverified image mismatch:** do not assume cross-sample image swaps are wrong pairs.
- **Unfair comparison:** baseline and improved model use different splits or preprocessing.
- **Metric overclaim:** high accuracy alone does not prove better hard-negative discrimination.
- **Compute constraints:** use gradient accumulation or mixed precision if safe; do not silently alter the model.

---

## 10. Recommended initial configuration

```yaml
seed: 42
num_labels: 2
label_mapping:
  real: 0
  fake: 1

text_model_name: hfl/chinese-roberta-wwm-ext
image_model_name: google/vit-base-patch16-224
max_length: 128
image_size: 224
projection_dim: 256

optimizer: AdamW
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.10
dropout: 0.3
gradient_clip_norm: 1.0
batch_size: 8
max_epochs: 10
early_stopping_patience: 3

loss:
  classification_weight: 1.0
  supervised_contrastive_weight: 0.1
  pairwise_weight: 0.0
  temperature: 0.07
  pairwise_margin: 0.2

mining:
  source_split: train
  hard_negative_ratio: 0.25
  top_k: 10
  refresh_frequency: epoch
  exclude_exact_duplicates: true
  require_opposite_labels: true
  require_verified_mismatch_for_pairwise_loss: true
```

The model checkpoint name above is an implementation choice; check that the actual selected checkpoint is compatible with the intended ViT architecture and document the resolved revision. If pretrained weights cannot be downloaded in the environment, stop and ask the user rather than substituting silently.

---

## 11. Antigravity execution prompt

Paste this prompt into Antigravity after opening the intended workspace:

> Implement Plan 2, `HN-RoViT`, as a separate research extension. First inspect the repository and identify the working Plan 1 RoViT baseline and frozen dataset splits. Do not overwrite the original MFND-DCL project or Plan 1. Follow this Markdown plan phase by phase and report the files changed, tests run, and exact test outcomes after every phase. Start with the same RoBERTa + ViT CLS concatenation classifier and add projection heads plus supervised contrastive learning. Implement the hard-negative candidate builder only over training IDs, with duplicate, label, event/claim, and split-leakage safeguards. Never mine validation/test samples. Keep pairwise ranking loss disabled unless verified mismatched pairs exist and are recorded in an audit CSV. Log each loss component separately. Use the same data split, preprocessing, and baseline checkpoint for fair comparisons. Add unit tests for pair filtering, no evaluation leakage, finite losses, shapes, and checkpoint loading. Run the ablation matrix and evaluate hard-subset metrics as well as normal test metrics. Do not claim the proposed method improves results until the experiments support that claim. Do not add local cross-attention or reliability-aware fusion until the first hard-negative version is stable. Ask me only when data schema, label mapping, or evidence for a proposed mismatched pair cannot be resolved.

---

## 12. Definition of done

Plan 2 is complete when:
- Plan 1 baseline remains unchanged and reproducible;
- hard-negative candidates are generated only from training data and audited;
- false-negative and duplicate safeguards are tested;
- contrastive and optional pairwise losses are numerically stable;
- all loss terms, mining decisions, seeds, and configs are logged;
- A–D ablations are complete (E only if verified pairs exist);
- hard-subset metrics are reported on a fixed evaluation subset;
- results are compared on the same splits and across multiple seeds where feasible;
- conclusions distinguish measured gains from hypotheses;
- the paper clearly labels HN-RoViT as a proposed extension, not as the original RoViT-Detect.
