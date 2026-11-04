# Antigravity Implementation Plan 1 — Faithful Reproduction of RoViT-Detect on Weibo

> **Goal:** Implement a faithful, auditable reproduction of the paper *“Multimodal social media fake news detection using RoBERTa and vision transformer encoders with reliability aware adaptive fusion”* (Bhukya et al., 2026), focusing first on the Weibo dataset.
>
> **Important scope note:** This plan reproduces the paper's described dual-stream RoBERTa + ViT model with late fusion and an MLP classifier. The paper's title/abstract mentions reliability-aware adaptive fusion, but the detailed method text also describes concatenation/late fusion. Therefore, implement the explicitly described concatenation baseline first and isolate any adaptive-fusion interpretation behind a separate optional experiment. Do not silently invent missing equations or claim exact reproduction of unspecified details.

---

## 0. Non-negotiable rules for Antigravity

1. **Inspect the existing repository and data before editing anything.** Do not overwrite the user's existing MFND-DCL implementation or datasets.
2. Work in a new branch or a new folder named `rovit_weibo_reproduction/`.
3. Do not fabricate missing image paths, labels, text, tweet IDs, or results.
4. Do not download a replacement dataset automatically. Ask the user if the expected Weibo files cannot be found.
5. Keep all train/validation/test partitions fixed and record the split IDs.
6. Fit preprocessing, vocabulary/statistics, thresholds, and any learned transformation using training data only.
7. Deduplicate exact and near-duplicate posts before splitting. Prefer event-aware/group-aware splitting when event/source IDs exist. If reproducing a published random split is necessary, document that choice and add a group-split robustness experiment.
8. Report results obtained by our run only. **Never hard-code the paper's 98.93% accuracy as an expected output.**
9. Save random seeds, package versions, config, logs, checkpoints, metrics, and confusion matrix.
10. If a paper detail is ambiguous or inconsistent, add it to `docs/reproduction_notes.md`; do not silently choose a value and call it exact.
11. Make one change at a time and run tests before proceeding.
12. Do not expose API keys, tokens, or credentials in code, logs, or commits.

---

## 1. Paper-derived architecture and implementation contract

### Model flow

```text
Weibo post text
  -> Chinese RoBERTa tokenizer (max length 128)
  -> hfl/chinese-roberta-wwm-ext
  -> text CLS/global embedding
                                      \
                                       -> concatenate late fusion
                                      /   -> MLP -> logits [B, 2]
Weibo post image                       -> softmax probabilities
  -> resize/crop to 224 x 224
  -> ImageNet normalization
  -> ViT-Base/16
  -> image CLS/global embedding
```

### Components and dimensions

| Stage | Input | Output / contract |
|---|---|---|
| Text batch | `B` raw Chinese posts | list of `B` strings |
| Tokenizer | text, max length 128 | `input_ids: [B, 128]`, `attention_mask: [B, 128]` |
| Chinese RoBERTa | token IDs and mask | `last_hidden_state: [B, 128, 768]` for the specified base checkpoint; confirm at runtime |
| Text global feature | hidden states | `text_cls: [B, 768]` using position 0 / CLS |
| Image preprocessing | image files | `pixel_values: [B, 3, 224, 224]` |
| ViT-Base/16 | pixel values | patch sequence typically `[B, 197, 768]` including CLS; confirm from checkpoint |
| Image global feature | ViT output | `image_cls: [B, 768]` |
| Late fusion | text and image CLS | `fused: [B, 1536]` if both encoders return 768-D |
| MLP hidden layer | fused features | `hidden: [B, H]`; choose and document `H` because paper does not unambiguously fix it |
| Classifier | hidden features | `logits: [B, 2]` for real/fake |
| Softmax | logits | `probabilities: [B, 2]` |

**Dimension verification is mandatory.** At model initialization, print and assert the actual hidden sizes. If the selected RoBERTa checkpoint does not output 768 dimensions, derive fusion dimensions from `config.hidden_size`; never hard-code an incompatible dimension.

### Paper-described Weibo settings to start with

- Text encoder: `hfl/chinese-roberta-wwm-ext`
- Maximum token length: `128`
- Image input: `224 x 224`
- Image encoder: ViT-Base/16; use a documented, compatible pretrained checkpoint.
- Optimizer: AdamW
- Learning rate: `2e-5` as the paper's reported starting point
- Weight decay: `0.01`
- Scheduler: 10% linear warm-up followed by linear decay
- Dropout: `0.3`
- Gradient clipping: max norm `1.0`
- Batch size: start at `8` if memory is limited; use `16` only if it fits
- Maximum epochs: `10` with validation-based early stopping as an initial reproduction setting
- Loss: cross-entropy for the 2-class task
- Image normalization: ImageNet mean/std
- Random seed: use `42` for the first run, then repeat with at least 3 seeds for a stability report.

The paper's hyperparameter table has some formatting/wording ambiguity around epochs and early stopping. Keep the configuration explicit in `configs/weibo.yaml` and record any deviation.

---

## 2. Dataset contract — resolve before training

Antigravity must locate the existing dataset and inspect its schema before implementing dataset loading.

### Required logical fields

Each post must resolve to:

- `sample_id`: stable unique identifier
- `text`: original Chinese text
- `image_path`: path to the corresponding image, or a clearly marked missing-image state
- `label`: binary label with one consistent mapping, e.g. `0=real`, `1=fake`
- optional `event_id`, `source_id`, `tweet_id`, `image_id`, timestamp

Create `data/processed/weibo_manifest.csv` with the normalized fields. Keep the raw source files unchanged.

### Dataset validation

Create `src/data/validate_dataset.py` to report:

- number of rows and unique IDs
- label counts and label mapping
- missing/empty text
- missing image paths and unreadable images
- duplicate sample IDs
- exact duplicate text and duplicate image hashes
- image-text pair mismatch warnings where identifiers permit checking
- split counts and class counts
- overlap of sample IDs, text hashes, image hashes, and event IDs between splits

**Do not silently drop records.** Every excluded record must have a reason written to `reports/data_quality_report.json` and a row in an exclusion manifest.

### Splits

1. Prefer provided official split files if they are available and clearly documented.
2. Otherwise create train/validation/test splits with a fixed seed and stratification.
3. If `event_id` is available, also produce an event-disjoint split as a robustness test.
4. Freeze split manifests before model training. Never select checkpoints or tune hyperparameters using test results.
5. Verify that no same image or duplicate post crosses splits.

**Reproducibility warning:** the paper reports Weibo accuracy of 98.93%, but its dataset/split counts appear inconsistent in different parts of the paper. Do not assume the reported number is directly comparable until the exact split and evaluation protocol are resolved. Record this in `docs/reproduction_notes.md`.

---

## 3. Recommended project structure

```text
rovit_weibo_reproduction/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── weibo.yaml
├── docs/
│   ├── reproduction_notes.md
│   ├── data_schema.md
│   └── experiment_protocol.md
├── data/
│   ├── raw/                  # user-provided data; do not alter
│   └── processed/
│       ├── weibo_manifest.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── split_data.py
│   │   └── validate_dataset.py
│   ├── models/
│   │   ├── rovit_model.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── checkpointing.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── logging.py
├── tests/
│   ├── test_dataset.py
│   ├── test_model_shapes.py
│   ├── test_no_split_leakage.py
│   └── test_train_step.py
├── checkpoints/
├── logs/
└── reports/
```

Keep large datasets and checkpoints out of Git.

---

## 4. Implementation phases and acceptance gates

### Phase 1 — Repository and data audit

**Antigravity tasks**
1. Inspect the current workspace and list relevant files.
2. Identify the actual Weibo text, labels, image files, and existing split files.
3. Write `docs/data_schema.md` describing the actual schema, including how each image is linked to its post.
4. Create the manifest and data-quality report.
5. Do not start model training until all schema/label/image linkage ambiguities are listed.

**Acceptance gate**
- Manifest generation is repeatable.
- Label mapping is explicit.
- Every row has a stable ID.
- Missing and duplicate records are quantified.
- No invented data or labels.

### Phase 2 — Environment and config

1. Create a Python environment and pinned dependencies compatible with the available CUDA/CPU environment.
2. Use PyTorch and Hugging Face Transformers for the two encoders.
3. Add `configs/weibo.yaml` for all paths, model names, dimensions, optimizer values, seeds, batch size, and early stopping.
4. Detect CUDA availability at runtime; fall back to CPU without crashing.
5. Log package versions, GPU name, device, and config at startup.

**Acceptance gate**
- `python -m src.data.validate_dataset --config configs/weibo.yaml` runs.
- `python -m src.training.train --help` works.
- No credentials or machine-specific absolute paths are committed.

### Phase 3 — Dataset and preprocessing

1. Implement a `WeiboDataset` returning `input_ids`, `attention_mask`, `pixel_values`, `label`, and `sample_id`.
2. Use tokenizer truncation/padding to exactly 128 tokens.
3. Use deterministic image preprocessing: RGB conversion, resize/crop to 224x224, ImageNet normalization.
4. Handle unreadable/missing images only according to a documented policy. For the primary multimodal experiment, exclude them from all splits consistently and report the count; optionally evaluate a separate missing-modality robustness setting.
5. Add tests for tensor shapes, labels, and deterministic behavior.

**Acceptance gate**
- A batch has `input_ids [B,128]`, `attention_mask [B,128]`, `pixel_values [B,3,224,224]`, labels `[B]`.
- A batch can be loaded repeatedly without inconsistent labels or paths.

### Phase 4 — Implement the exact late-fusion model

Implement `RoViTDetect`:

1. Text encoder: Chinese RoBERTa.
2. Image encoder: ViT-Base/16.
3. Extract the global CLS representation from each encoder.
4. Concatenate `text_cls` and `image_cls`.
5. Apply dropout `0.3`.
6. Apply an MLP with a configurable hidden layer and ReLU.
7. Output 2 logits. Use `CrossEntropyLoss`; do **not** apply softmax before passing logits to this loss.
8. Provide a `return_features=True` option for debugging and analysis.
9. No cross-attention, contrastive loss, hard-negative mining, VAE, or adaptive gate in this exact baseline.

Pseudo-contract:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    return_features=True,
)
# outputs["logits"]: [B, 2]
# outputs["text_cls"]: [B, 768]  # confirm at runtime
# outputs["image_cls"]: [B, 768] # confirm at runtime
# outputs["fused"]: [B, 1536]     # if both branches are 768-D
```

**Important:** the article's description of reliability-aware adaptive fusion is not sufficiently unambiguous in the provided method details to infer a unique exact equation. Do not invent one inside the reproduction model. Implement concatenation first; add any clearly specified adaptive variant as a separate file/experiment only if the paper's exact formula is verified.

**Acceptance gate**
- Shape test passes for batch sizes 1 and 2.
- One forward pass returns finite logits.
- One training step produces finite loss and gradients.
- No accidental gradient flow through frozen parameters if an optional frozen mode is enabled.

### Phase 5 — Training loop

Implement:
- AdamW, LR `2e-5`, weight decay `0.01`
- warm-up for 10% of total steps and linear decay
- cross-entropy loss
- gradient clipping at 1.0
- validation each epoch
- early stopping on validation macro-F1 (or validation loss; choose one in config and document it)
- save `best.pt` and `last.pt`
- resume-from-checkpoint support
- log training/validation loss, accuracy, precision, recall, macro-F1, and learning rate per epoch.

Start with batch size 8 if GPU memory is constrained. If CUDA OOM occurs, reduce batch size or use gradient accumulation; do not silently change input resolution or model architecture.

**Acceptance gate**
- The model completes one short smoke-test epoch on a small subset.
- Checkpoints can be loaded into a fresh process.
- Resume training works.
- The test set is not evaluated during model selection.

### Phase 6 — Final evaluation

Evaluate the chosen checkpoint on the frozen test split exactly once per finalized configuration.

Report:
- accuracy
- precision, recall, F1 for each class
- macro-F1 and weighted-F1
- confusion matrix
- ROC-AUC and PR-AUC where the binary score is defined
- classification report
- inference time per sample and model parameter count
- seed and exact split manifest hashes

Save outputs under `reports/` as CSV/JSON/PNG. Do not report only accuracy.

### Phase 7 — Repeated runs and audit

1. Repeat the finalized configuration with at least three seeds if compute permits.
2. Report mean ± standard deviation.
3. Run an event-disjoint split if event identifiers are available.
4. Add text-only and image-only baselines to understand modality contribution.
5. Compare with 98.93% only as a paper-reported reference, not as a guaranteed target.
6. If results are unexpectedly near-perfect, audit duplicates, event overlap, image reuse, label leakage, and split protocol before interpreting the result.

---

## 5. Tests that must exist before calling the implementation complete

- `test_manifest_has_unique_ids`
- `test_labels_are_binary_and_mapped`
- `test_no_duplicate_ids_across_splits`
- `test_no_image_hash_overlap_across_splits` (unless documented by official split)
- `test_dataset_tensor_shapes`
- `test_model_logits_shape`
- `test_model_outputs_are_finite`
- `test_one_training_step_updates_parameters`
- `test_checkpoint_round_trip`
- `test_test_split_not_used_for_early_stopping`

Tests should fail loudly with actionable error messages.

---

## 6. Required commands and user documentation

Add README commands adapted to the actual environment, for example:

```bash
python -m src.data.validate_dataset --config configs/weibo.yaml
python -m src.data.split_data --config configs/weibo.yaml
python -m src.training.train --config configs/weibo.yaml
python -m src.training.evaluate --config configs/weibo.yaml --checkpoint checkpoints/best.pt --split test
```

If these modules use a different CLI, make the README match the implemented CLI exactly.

README must explain:
- where the dataset should be placed
- expected manifest columns and label mapping
- how to configure model checkpoints
- how to train/evaluate
- how to resume training
- how to interpret reports
- known reproduction ambiguities
- what has and has not been reproduced from the paper

---

## 7. Antigravity execution prompt

Paste this prompt into Antigravity after opening the intended workspace:

> You are implementing Plan 1: a faithful reproduction of RoViT-Detect on Weibo. First inspect the repository and dataset; do not modify or delete existing project files. Create a new isolated folder `rovit_weibo_reproduction/`. Follow this Markdown plan phase by phase. Before each phase, summarize the files you will create/change and the acceptance tests; after each phase, run the tests and report exact outcomes. Do not invent dataset fields, images, labels, model equations, or results. Implement the RoBERTa + ViT CLS concatenation late-fusion baseline first. Use `hfl/chinese-roberta-wwm-ext`, max text length 128, image size 224x224, ImageNet normalization, an MLP classifier, two-class logits, cross-entropy, AdamW, LR 2e-5, weight decay 0.01, 10% warm-up, dropout 0.3, gradient clipping 1.0, and validation-based early stopping as documented here. Confirm tensor dimensions at runtime. Keep the paper's unclear reliability-aware fusion description separate rather than inventing its equation. Audit data leakage and duplicates before training. Save configuration, seeds, checkpoints, logs, metrics, confusion matrix, and reproduction notes. Do not claim 98.93% unless our own test evaluation actually obtains it. Stop and ask me only if the actual dataset schema, label mapping, or text-image pairing cannot be resolved from files.

---

## 8. Definition of done

Plan 1 is complete only when:
- data schema and split integrity are documented;
- the model's dimensions and forward pass are tested;
- training and checkpoint resume work;
- final metrics are produced from the frozen test split;
- text-only/image-only comparisons are available or explicitly marked pending;
- results are reproducible from the README;
- paper ambiguities and any deviations are documented;
- no unsupported claim of exact numerical replication is made.
