# SAHN-DCL Implementation Plan for Antigravity

## Semantic-Ambiguity-Aware Hard-Negative Dual Contrastive Learning for MFND-DCL

**Purpose:** This document is the implementation specification to give
Google Antigravity so it can modify the existing MFND-DCL project in
small, testable steps without replacing the baseline or inventing
missing assumptions.

**Important status:** SAHN-DCL is a proposed extension to the existing
MFND-DCL implementation, not a claim that this exact named method
already exists in the cited papers. Its usefulness must be established
experimentally.

------------------------------------------------------------------------

## 1. Main objective

Improve the existing multimodal fake-news detector by targeting two
specific weaknesses:

1.  **Fine-grained semantic mismatch:** a text and image may be broadly
    about the same topic while differing in the actual event, entity,
    action, time, location, or claim.
2.  **Hard-negative quality:** ordinary negatives may be too easy, while
    visually or semantically similar examples can be confusing. The
    training process should emphasize informative negatives without
    accidentally treating a valid pair as a negative.

The proposed extension must preserve the original MFND-DCL model as a
reproducible baseline. Do not overwrite the baseline implementation.
Implement the extension behind a configuration flag and compare the
baseline and extended model using the same data splits and evaluation
protocol.

### Non-goals

-   Do not promise that semantic compatibility can be known perfectly
    from CLIP cosine similarity.
-   Do not infer text-image pair correctness from fake/real labels
    alone.
-   Do not silently change the dataset, label mapping,
    train/validation/test split, preprocessing, encoder, or evaluation
    metrics.
-   Do not claim improvement until the baseline and proposed model have
    been evaluated on the same split and seeds.
-   Do not rewrite the whole project when a minimal patch is sufficient.

------------------------------------------------------------------------

## 2. Research grounding and boundaries

Use these concepts as design inspiration, while keeping their roles
distinct:

-   **MFND-DCL baseline:** existing multimodal encoders, fusion module,
    classification objective, and dual contrastive objectives in the
    current repository.
-   **VCLMMF-style ambiguity awareness:** ambiguity can be used as a
    signal for cautious modality contribution. A modality-discrepancy
    score is not, by itself, proof that a particular text-image pair is
    false or mismatched.
-   **Bidirectional cross-modal attention approaches:** preserve
    original modality representations and their attended representations
    where the current architecture supports this, rather than replacing
    original features without an ablation.
-   **Fine-grained / local-global fusion approaches:** consider local
    evidence only if the current encoders expose appropriate token/patch
    features and there is a reliable training/evaluation path. Do not
    add local attention by assuming unavailable hidden states.

These are inspiration points, not proof that the proposed combination
will work. The actual code and data schema determine what can be
implemented safely.

------------------------------------------------------------------------

## 3. Non-negotiable implementation rules for Antigravity

Paste these rules into Antigravity before asking it to edit code.

1.  **Inspect first, edit second.** Read the repository tree, entry
    points, dataset class, model, losses, trainer, config, and
    evaluation code before changing anything.
2.  **Report actual code facts.** Identify filenames, classes,
    functions, tensor shapes, label values, loss equations, and the
    training call sites from the repository. Do not guess.
3.  **Create a baseline checkpoint in source control.** Do not delete or
    rewrite the original MFND-DCL path. Add a feature flag such as
    `use_sahn_dcl: false` by default.
4.  **Make one phase at a time.** For each phase: explain the proposed
    change, list files to edit, implement the smallest patch, run
    targeted tests, report exact commands and results, then wait for
    approval before the next phase.
5.  **No fabricated dimensions.** Derive dimensions from actual model
    outputs or assert them at runtime. Document every tensor shape from
    the real implementation.
6.  **No fabricated labels or pair matches.** Inspect the dataset schema
    and data provenance. If pair-level compatibility labels do not
    exist, explicitly state that limitation.
7.  **Do not use fake/real labels as text-image matching labels.** A
    real-news example may contain an irrelevant image, and a fake-news
    example may contain a matching image. The task labels are not
    interchangeable.
8.  **Do not treat every off-diagonal batch pair as a valid negative
    without checking duplicates or known relationships.** Use metadata
    where available; otherwise use a conservative policy and document
    its limits.
9.  **Numerical stability is mandatory.** Check for finite losses,
    finite gradients, valid masks, non-empty positive sets, and no
    division by zero.
10. **No data leakage.** Fit any learned preprocessing or thresholds
    using training/validation data only. Never mine negatives using test
    labels or tune on test performance.
11. **Keep the baseline reproducible.** Same splits, preprocessing,
    encoder weights, optimizer policy, and evaluation metrics for the
    baseline-vs-extension comparison wherever possible.
12. **Do not claim success from a single batch or a single seed.** Unit
    tests are necessary but do not replace evaluation.
13. **Never silently fall back.** If a required field or metadata is
    missing, raise a clear error or use an explicitly documented safe
    fallback.
14. **Avoid broad refactors.** Do not rename public interfaces or change
    unrelated code.
15. **Do not proceed past a failed gate.** Fix the failure and rerun the
    relevant tests before continuing.

------------------------------------------------------------------------

## 4. Phase 0 --- Repository audit (no model edits)

### Tasks

Antigravity must inspect the project and produce
`docs/SAHN_DCL_AUDIT.md` containing:

-   Repository tree, excluding large datasets, model weights, caches,
    and virtual environments.
-   Actual training entry point and exact command used to launch
    training.
-   Dataset class(es), dataset columns/fields, label encoding,
    missing-image/text handling, duplicate handling, and split logic.
-   Whether records contain a verified text-image pair-match label,
    article/event ID, image ID, source ID, or other metadata useful for
    avoiding false negatives.
-   Model architecture and actual forward pass, including tensor shapes
    for:
    -   text encoder output;
    -   image encoder output;
    -   projected text and image features;
    -   cross-attention outputs;
    -   fused representation;
    -   classifier logits;
    -   both existing contrastive projections, if present.
-   Existing losses and their exact equations as implemented, not just
    as described in comments.
-   Optimizer, scheduler, temperature, batch size, gradient
    accumulation, AMP settings, random seeds, checkpoint selection,
    early stopping, and evaluation metrics.
-   Current baseline performance if a valid run/checkpoint exists.
    Record the command, configuration, split, seed, and metrics. Do not
    invent missing results.
-   Current bugs, shape inconsistencies, suspicious data leakage, or
    differences between paper and implementation.
-   A proposed file-by-file minimal change plan.

### Audit gate

Do not implement SAHN-DCL until the audit confirms: - which feature
tensors are available; - which labels and metadata exist; - how the
current two contrastive losses work; - how a baseline run can be
reproduced.

If source code, data, or run configuration is missing, list the exact
missing item and stop. Do not fill gaps with assumed code.

------------------------------------------------------------------------

## 5. Phase 1 --- Freeze and validate the baseline

### Tasks

1.  Preserve the original baseline path.
2.  Save the resolved configuration and random seed.
3.  Run existing tests, if present.
4.  Run a small smoke test: one training batch, forward pass, loss
    calculation, backward pass, optimizer step.
5.  Check that all baseline loss values and gradients are finite.
6.  Save baseline predictions and evaluation metrics for the unchanged
    test split, if a valid checkpoint can be produced.
7.  Add or update a results CSV with fields such as:
    `model, seed, split, accuracy, macro_f1, precision, recall, auroc, auprc, checkpoint, config_path`.
    Include only metrics supported by the actual evaluation setup; do
    not invent unavailable values.

### Acceptance criteria

-   Baseline still runs without the SAHN-DCL flag.
-   No dataset or label behavior changed.
-   Baseline smoke test passes.
-   Baseline configuration, command, and metrics are recorded.
-   The original baseline can be rerun independently.

------------------------------------------------------------------------

## 6. Phase 2 --- Build reliable negative masks before adding semantic scoring

### Principle

Hard-negative mining is only useful if the selected negatives are
genuinely likely to be negative. A high text-image similarity score does
not prove a match, and a low score does not prove a mismatch.

### Tasks

1.  Identify the exact batch structure and the sample IDs for text and
    image features.
2.  Build a batch-level candidate matrix with shape `[B, B]`, where row
    `i` is text sample `i` and column `j` is image sample `j`.
3.  Create a **valid-negative mask** using only defensible information:
    -   Exclude the diagonal (the known paired sample) from negatives.
    -   If reliable IDs or metadata identify duplicate/related samples,
        exclude them from negatives unless their relationship is
        understood.
    -   Do not use real/fake labels as if they were pair-match labels.
    -   If no relationship metadata exists, explicitly mark off-diagonal
        candidates as *unverified negatives*, not guaranteed mismatches.
4.  Create tests for mask shape, diagonal exclusion, duplicate exclusion
    (when metadata exists), device placement, dtype, and empty-negative
    rows.
5.  If a row has no valid negatives, skip its contrastive contribution
    safely and report the count. Never divide by zero.

### Acceptance criteria

-   Every mask entry has a documented meaning.
-   Tests verify known positives are never selected as negatives.
-   No test data is used to create training masks.
-   Logs report the number of eligible negatives per batch and how often
    rows are skipped.

------------------------------------------------------------------------

## 7. Phase 3 --- Define semantic ambiguity without overclaiming

### Design intent

Estimate how uncertain the cross-modal relationship appears, and use
that uncertainty to avoid blindly trusting questionable pairs. This is a
training signal, not a ground-truth compatibility label.

### Candidate signals

Use only signals that the actual implementation can produce:

-   normalized global text-image embedding similarity;
-   discrepancy between text-conditioned image features and
    image-conditioned text features, if bidirectional attention already
    exists or can be added with a minimal patch;
-   local/global similarity, only if token/patch-level features are
    available and their shapes are verified;
-   an optional learned compatibility head, **only if it has a valid
    supervision strategy**.

### Mandatory supervision check

Before implementing a learned compatibility scorer, answer:

1.  Does the dataset provide a verified matching/mismatching text-image
    label?
2.  Does it provide article/event IDs that can support safe positive and
    negative construction?
3.  Is there a trustworthy external annotation process?
4.  If none exist, what independent signal trains the scorer without
    simply learning the fake/real label or circularly learning from the
    same similarity score it is supposed to validate?

If there is no reliable pair-level supervision, do **not** present a
learned score as a calibrated probability of pair correctness. Start
with a fixed, normalized diagnostic score or an uncertainty proxy and
label it accordingly. Compare it against simple baselines.

### Avoid circularity

Do not define "hard negative" as "high similarity" and then claim that
the same high similarity independently proves semantic ambiguity. Keep
candidate selection, ambiguity scoring, and downstream evaluation
conceptually distinct.

### Acceptance criteria

-   The score is named accurately (e.g., `ambiguity_score` or
    `compatibility_proxy`, not `pair_match_probability` unless
    calibrated and validated).
-   The formula and range are documented.
-   Tests cover extreme values, missing modality cases, and numerical
    stability.
-   The score is not trained using test data.
-   The audit report states what the score can and cannot mean.

------------------------------------------------------------------------

## 8. Phase 4 --- Add conservative hard-negative weighting

### Goal

Give more training emphasis to informative, valid candidate negatives
while limiting the influence of uncertain candidates.

### Recommended implementation strategy

-   Begin with a **fixed, bounded weighting rule** and a configuration
    flag. Avoid introducing a new learnable temperature and several
    trainable modules at the same time.
-   Compute pairwise similarities from normalized embeddings.
-   Apply the valid-negative mask before selecting or weighting
    negatives.
-   Use a numerically stable contrastive formulation (log-sum-exp /
    `log_softmax` style operations).
-   Normalize weights over the eligible negatives for each anchor.
-   Clamp or bound any weights and temperature to documented safe
    ranges.
-   Make the no-valid-negative case explicit.
-   Keep positive-pair handling separate from negative weighting.
-   Keep the original contrastive loss available unchanged for baseline
    comparisons.

### Loss implementation requirements

Antigravity must first derive the existing loss equation from the code
and state what each term means. Then propose the modified equation in
`docs/SAHN_DCL_MATH.md` before coding.

The document must specify:

-   anchor and candidate definitions;
-   positive mask;
-   valid-negative mask;
-   hard-negative score;
-   ambiguity signal and whether it affects candidate eligibility or
    only weighting;
-   per-anchor normalization;
-   temperature definition and constraints;
-   reduction over anchors;
-   treatment of anchors with no valid negatives;
-   gradient flow and which tensors are detached, if any.

Do not simply add `log(weight)` to logits unless the intended
probabilistic derivation justifies it. Do not silently change the
original loss semantics.

### Acceptance criteria

-   Unit tests compare the original loss with its unmodified reference
    behavior.
-   With hard-negative weighting disabled, the loss matches the baseline
    implementation within a defined numerical tolerance.
-   Modified loss is finite and gradients are finite on synthetic and
    real batches.
-   A tiny synthetic example verifies that a valid, deliberately hard
    negative receives the intended relative emphasis.
-   Masking prevents invalid candidates from affecting the loss.

------------------------------------------------------------------------

## 9. Phase 5 --- Integrate into both contrastive branches carefully

MFND-DCL is described in this project context as using sample-level and
modality-level contrastive objectives, but Antigravity must verify their
exact names and semantics in the code before integrating anything.

### Tasks

1.  Map the existing loss calls and the tensors passed to them.
2.  Decide whether hard-negative weighting applies to:
    -   sample-level contrastive loss;
    -   modality-level consistency loss;
    -   both, but only if each branch has a defensible
        candidate/positive definition.
3.  Do not assume the same mask or positive semantics apply to both
    branches.
4.  Add independent flags for each branch so ablations can isolate the
    effect:
    -   `use_sahn_sample_loss`
    -   `use_sahn_modality_loss`
5.  Preserve the original losses as the default when the flags are
    false.
6.  Log each loss component separately, not just a total loss.
7.  Check loss scale and gradients before tuning coefficients.

### Acceptance criteria

-   Every loss branch has a documented input shape and target semantics.
-   Flags allow baseline, sample-only, modality-only, and combined
    configurations.
-   No branch silently changes its positive-pair definition.
-   The total loss is finite and backward propagation succeeds.

------------------------------------------------------------------------

## 10. Phase 6 --- Fine-grained semantics: optional second-stage experiment

Do not implement this phase until Phases 0--5 pass.

### First inspect what the encoders expose

-   If only pooled embeddings are available, do not claim patch-level or
    token-level fine-grained alignment.
-   If token/patch embeddings are available, document their actual
    shapes, masks, pooling, and computational cost.
-   Preserve the original global representation for comparison.

### Minimal experiment

If the encoder supports token/patch features: 1. Compute a
local-to-global or token-to-region similarity summary using the existing
features. 2. Keep padding tokens and invalid patches out of the
computation using the actual attention masks. 3. Define one
interpretable local/global discrepancy statistic. 4. Test it
independently before allowing it to affect the loss. 5. Compare
global-only against global-plus-local features. 6. Add no new attention
block unless the simpler feature-based experiment provides evidence that
it is needed.

### Acceptance criteria

-   Shapes and masks are unit-tested.
-   The method is not described as object grounding unless object/region
    annotations or a valid grounding evaluation exist.
-   Runtime and GPU memory are measured.
-   The local/global branch is independently switchable and ablated.

------------------------------------------------------------------------

## 11. Phase 7 --- Training integration and safety checks

### Configuration

Put new parameters in the project's existing configuration system. If
there is no configuration system, use one clearly defined configuration
object rather than scattered hard-coded constants.

Suggested settings (names may be adapted to existing conventions):

``` yaml
use_sahn_dcl: false
use_sahn_sample_loss: false
use_sahn_modality_loss: false
use_fine_grained_branch: false

# These are placeholders to be selected through validation,
# not claimed optimal values.
hard_negative_weighting: "bounded"
ambiguity_weighting: false
```

Do not add arbitrary numerical defaults for temperature, margins, or
coefficients until the existing implementation is audited and a
controlled validation protocol is defined.

### Training-time assertions/logging

-   Input batch size and feature dimensions are consistent.
-   Similarity matrix shape is `[B, B]`.
-   Masks have the expected shape and boolean dtype.
-   Each eligible anchor has a valid positive.
-   Empty-negative anchors are handled safely.
-   Loss components and total loss are finite.
-   Gradients are finite before the optimizer step.
-   Temperature is positive and within its configured bounds.
-   Log skipped anchors and invalid-candidate counts.
-   Save configuration and random seeds with each checkpoint.
-   Use training data for parameter fitting and validation data for
    model selection; never use test results to select hyperparameters.

### Checkpointing

Use distinct filenames or run directories, for example: - `baseline/` -
`sahn_sample/` - `sahn_modality/` - `sahn_combined/` -
`sahn_fine_grained/`

Do not overwrite baseline checkpoints.

------------------------------------------------------------------------

## 12. Phase 8 --- Required tests

Add tests to the repository's existing test framework. If none exists,
use `pytest` and keep tests small and deterministic.

### Unit tests

-   Candidate similarity matrix has the expected shape.
-   Embeddings are normalized as intended.
-   Positive and negative masks are correct.
-   Diagonal and known duplicate pairs are not mined as negatives.
-   A row with no eligible negatives is handled without NaN or infinity.
-   Loss is finite for batch size 1 and for the smallest supported batch
    size.
-   Loss and gradients are finite for normal synthetic batches.
-   Hard-negative weighting disabled reproduces baseline loss.
-   Very high/low similarities do not overflow.
-   Missing text/image inputs follow the repository's explicit policy.
-   Model state dict loads and saves correctly.
-   Default config uses the baseline path.
-   Each experiment flag changes only its intended branch.

### Integration tests

-   One training batch completes.
-   One validation batch completes without gradients.
-   One checkpoint can be saved and reloaded.
-   Prediction output dimensions and label order match the baseline.
-   Evaluation uses the same test set and metric definitions as the
    baseline.

### Tiny overfit test

Train on a tiny, fixed subset only as a debugging test. It can catch
implementation bugs, but it is not evidence of generalization or
scientific improvement.

------------------------------------------------------------------------

## 13. Phase 9 --- Experimental protocol

Run these configurations on the same data split and comparable compute
budget:

  -----------------------------------------------------------------------
  ID                      Configuration           Purpose
  ----------------------- ----------------------- -----------------------
  E0                      Original MFND-DCL       Reproducible baseline

  E1                      Baseline +              Isolate candidate
                          valid-negative masking  filtering
                          only                    

  E2                      E1 + hard-negative      Test sample-level
                          weighting in            mining
                          sample-level loss       

  E3                      E1 + hard-negative      Test modality branch
                          weighting in            
                          modality-level loss     

  E4                      E1 + both contrastive   Test combined SAHN-DCL
                          branches                

  E5                      E4 + ambiguity proxy    Test ambiguity signal

  E6                      E5 + local/global       Test fine-grained
                          feature branch, only if feature signal
                          supported               
  -----------------------------------------------------------------------

Use multiple random seeds (preferably at least three, if compute
permits). Report mean and standard deviation, not only the best run. Use
the validation split for model selection and reserve the test split for
final comparison.

### Report these metrics where applicable

-   Accuracy
-   Macro-F1 (important for class imbalance)
-   Per-class precision, recall, and F1
-   Confusion matrix
-   AUROC and AUPRC if the output scores and label setup support them
-   Parameter count and inference latency, if model complexity changes
-   Peak GPU memory and training time
-   Number of valid negatives and skipped anchors
-   Results by ambiguity bucket only if the bucket definition is fixed
    using training/validation data and evaluated without leakage

Do not compare a paper's reported numbers to a locally reproduced result
unless the dataset split, preprocessing, label mapping, and evaluation
protocol are sufficiently comparable.

------------------------------------------------------------------------

## 14. Phase 10 --- Ablation and failure analysis

At minimum, answer these questions:

1.  Does valid-negative filtering alone change results?
2.  Does hard-negative weighting help over the filtered baseline?
3.  Does it improve Macro-F1 and minority-class behavior, or only
    accuracy?
4.  Does ambiguity weighting help consistently across seeds?
5.  Does the fine-grained branch add value beyond global embeddings?
6.  Does the extension help on examples with likely text-image
    disagreement?
7.  Are gains caused by leakage, duplicated records, changed splits, or
    altered preprocessing?
8.  How much do training time and memory increase?
9.  Are improvements statistically or practically meaningful given
    run-to-run variation?
10. Which examples become worse, and why?

Inspect representative errors manually, but do not select or modify the
final test set based on those observations.

------------------------------------------------------------------------

## 15. Required outputs

Antigravity should create/update these artifacts, adapting filenames to
repository conventions:

-   `docs/SAHN_DCL_AUDIT.md` --- factual repository and dataset audit.
-   `docs/SAHN_DCL_MATH.md` --- exact baseline and proposed loss
    equations.
-   `docs/SAHN_DCL_EXPERIMENTS.md` --- configuration matrix, commands,
    seeds, and results.
-   `tests/test_sahn_masks.py` --- masks and negative eligibility.
-   `tests/test_sahn_losses.py` --- loss equivalence, edge cases, and
    gradients.
-   `configs/sahn_dcl.yaml` or equivalent --- only if compatible with
    the existing config system.
-   Separate baseline and experiment checkpoints/logs.
-   A results CSV with actual measured metrics.
-   A final summary listing changed files, commands run, test outcomes,
    known limitations, and next actions.

Do not create empty documents and claim completion. Each document must
contain actual repository-derived facts or measured results.

------------------------------------------------------------------------

## 16. Antigravity execution prompts

Use the following prompts one at a time. Do not ask Antigravity to
execute all phases in a single uncontrolled edit.

### Prompt A --- Audit only

> Read the entire existing MFND-DCL repository before editing anything.
> Do not modify code. Identify the training entry point, dataset schema
> and label mapping, split logic, model forward pass, actual tensor
> shapes, both contrastive losses and their equations,
> optimizer/training loop, evaluation metrics, and available pair-level
> metadata. Determine whether verified text-image matching labels exist.
> Write `docs/SAHN_DCL_AUDIT.md` with facts tied to exact filenames and
> functions. List uncertainties and missing files explicitly. Do not
> assume dimensions or infer pair matching from fake/real labels. Stop
> after the audit and show me the report.

### Prompt B --- Baseline protection

> Using the audit, preserve the original MFND-DCL implementation and
> establish a reproducible baseline run. Do not implement SAHN-DCL yet.
> Add only minimal tests or logging needed to verify a forward pass,
> both current losses, backward pass, finite gradients, checkpoint
> save/load, and the current evaluation path. Record the exact command,
> configuration, split, seed, and actual metrics. Do not change data
> preprocessing or model behavior. Stop and report all results.

### Prompt C --- Negative-mask design

> Implement only a tested candidate-validity mask for contrastive
> learning, based on metadata confirmed in `docs/SAHN_DCL_AUDIT.md`.
> Exclude the known diagonal positive and any duplicates/related pairs
> that can be identified reliably. Do not use fake/real labels as
> pair-match labels. If pair metadata is absent, describe off-diagonal
> candidates as unverified negatives and do not claim they are true
> mismatches. Add deterministic unit tests, finite-value checks, and
> handling for anchors with no eligible negatives. Keep the baseline
> path unchanged. Run the tests and stop.

### Prompt D --- Loss mathematics before coding

> Before changing a loss, write `docs/SAHN_DCL_MATH.md`. Copy the actual
> baseline loss definitions from the implementation in mathematical
> notation, then propose a conservative hard-negative-weighted variant.
> Define anchors, positives, candidate negatives, masks, score, weights,
> temperature, normalization, reduction, and empty-negative handling.
> Explain why the weighting is mathematically appropriate and how it
> avoids invalid candidates. Do not introduce learnable temperature or
> multiple new modules in this first version. Do not code the new loss
> until I approve the equation.

### Prompt E --- Implement sample-level loss

> Implement only the approved sample-level SAHN-DCL loss variant. Keep
> the original loss callable and unchanged. Add a configuration flag
> defaulting to false. Verify that the disabled path reproduces the
> original loss within a stated tolerance. Add tests for shapes, masks,
> finite loss, finite gradients, batch-size edge cases, and the intended
> relative emphasis on a synthetic hard negative. Run all relevant tests
> and one-batch training smoke test. Do not modify the modality-level
> loss yet. Stop and report.

### Prompt F --- Modality-level loss

> Inspect the actual modality-level contrastive/consistency loss and
> explain its positive/negative semantics. Do not reuse the sample-level
> mask unless the code and objective justify it. Propose the smallest
> compatible extension, document the equation, and wait for approval
> before editing. Once approved, implement it behind its own flag and
> add equivalence, edge-case, and gradient tests. Keep the baseline path
> unchanged.

### Prompt G --- Ambiguity proxy

> Based on the actual available features and metadata, propose one
> interpretable ambiguity proxy. Do not call it a pair-match probability
> unless verified pair-level supervision and calibration exist. Explain
> its formula, range, limitations, and whether it is derived from the
> same similarity used for mining. Avoid circular reasoning. Write the
> design first and wait for approval. If there is no defensible signal,
> state that and skip this component rather than inventing one.

### Prompt H --- Full experiment matrix

> Run the approved experiment matrix with the same split and evaluation
> code for every configuration. Use multiple seeds if compute permits.
> Never tune on the test set. Save each configuration, seed, logs, and
> checkpoint separately. Produce a results table with actual values,
> mean and standard deviation where available, compute cost, and failed
> runs. Do not fabricate missing results or claim improvement unless the
> measured evidence supports it.

### Prompt I --- Final review

> Review the final diff against the original MFND-DCL baseline. Look for
> accidental dataset changes, data leakage, invalid negative masks,
> label misuse, incorrect loss reductions, non-finite values, checkpoint
> incompatibility, untested branches, and claims not supported by
> results. Run all tests and provide a final report with exact changed
> files, commands, results, limitations, and whether each acceptance
> criterion passed. Do not make further refactors during this review.

------------------------------------------------------------------------

## 17. Definition of done

The project is not complete until all are true:

-   [ ] Original MFND-DCL remains runnable and reproducible.
-   [ ] Dataset labels and pair-level metadata have been audited.
-   [ ] Negative masks have documented semantics and passing tests.
-   [ ] The modified loss equations are documented and approved.
-   [ ] Baseline equivalence is verified when the extension is disabled.
-   [ ] All losses and gradients remain finite on tested edge cases.
-   [ ] Sample-level and modality-level branches are independently
    configurable.
-   [ ] No test-set leakage or use of fake/real labels as pair-match
    labels.
-   [ ] The experiment matrix has been run and results are recorded
    honestly.
-   [ ] Multiple seeds and ablations are reported where compute permits.
-   [ ] Runtime/memory cost and failure cases are documented.
-   [ ] Final conclusions match the evidence; no unsupported accuracy
    claims.

------------------------------------------------------------------------

## 18. Final instruction to Antigravity

**Work incrementally. Audit first, preserve the baseline, document the
math, test each change, and stop at every phase boundary for review.
Correctness and defensible experiments matter more than adding
architectural complexity. If the repository or dataset does not support
a proposed component, say so explicitly and do not invent the missing
supervision, dimensions, labels, or results.**
