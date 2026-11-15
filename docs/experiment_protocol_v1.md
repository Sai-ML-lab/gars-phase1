# Experiment protocol V1 — DeBERTa-v3-base / GLUE RTE

## Purpose

Establish one reproducible small-model experiment that compares allocation policies on an identical trained LoRA parent adapter. The primary comparison is at a matched physical A/B parameter budget.

## Dataset and task

- Dataset: GLUE, configuration rte.
- Training split: official GLUE RTE train split.
- Validation split: official GLUE RTE validation split.
- Metric: accuracy (primary) and macro-F1 (secondary).
- Maximum sequence length: 256.
- Do not use the hidden test split for tuning or claims.

## Parent adapter

- Base model: microsoft/deberta-v3-base.
- LoRA target modules: query_proj, value_proj (verify module names against the loaded model before training).
- Initial rank: 32.
- LoRA alpha: 64.
- LoRA dropout: 0.05.
- Classifier and pooler are included in modules_to_save for sequence classification.
- Optimizer: AdamW.
- Learning rate: 2e-4.
- Weight decay: 0.01.
- Epochs: 3.
- Per-device batch size: 8.
- Gradient accumulation: 2.
- Warmup ratio: 0.10.
- Gradient clipping: 1.0.
- Maximum token length: 256.
- Seeds: 42, 123, 999.

These are pilot starting values, not claims of optimal hyperparameters. If hardware constraints require changing them, record the change and use identical settings across seeds/policies.

## Gradient capture

- Collect gradients only at optimizer-update boundaries.
- Capture the final 50 optimizer updates.
- Aggregate all LoRA A/B gradients in one Transformer block into a joint Frobenius norm.
- Store the arithmetic mean of the step-level block norms.
- Do not use EMA or square the gradient score in V1.

## Allocation comparison

For each seed, train one parent adapter and then produce:
- uncompressed parent LoRA;
- uniform local squared-energy pruning;
- global singular-value cutoff;
- GARS gradient-conditioned squared-energy pruning.

Target compressed budget: 60% of the parent A/B factor-parameter count. Record the realized budget; integer-rank constraints may prevent exact matching.

## Required checks before reporting metrics

1. QR-SVD reconstruction at full rank matches the original dense B @ A within numerical tolerance.
2. Rank-k QR-SVD reconstruction matches a direct dense truncated SVD on small matrices.
3. The squared-energy retention ratio meets the selected threshold, except where the rank floor binds.
4. Exported A/B shapes physically shrink to the selected ranks.
5. Export reloads into an equivalent adapter wrapper without shape mismatch.
6. For a fixed test batch, reloaded output matches the in-memory truncated adapter within a declared numerical tolerance.
7. All methods are evaluated on the same validation examples.
8. Parent performance and all seed-level results are retained even if pruning harms accuracy.

## Main reported metrics

- Validation accuracy and macro-F1.
- Absolute performance difference from the parent adapter.
- Total physical LoRA A/B parameters.
- Parameter reduction relative to the parent.
- Adapter serialized size in bytes.
- Per-block gradient scores and per-projection squared-energy entropy.
- Per-module parent and retained ranks.
- Export/load status and output-equivalence error.
- Inference latency only after export is verified; measure merged and unmerged modes separately.

## Statistical interpretation

For each policy, report the three seed-level observations, mean and standard deviation. With only three seeds, confidence intervals and significance tests have low power. Treat this as a Phase 1 pilot; do not claim strong statistical significance from a tiny seed count. Pair methods by the same parent-training seed.


## Physical export validation

The post-hoc exporter creates three physical adapter variants from each common parent: `uniform_local_energy`, `global_singular_value`, and `gars`. It retains classifier/pooler `modules_to_save` tensors and tokenizer files, while replacing each LoRA A/B tensor with its rank-k SVD factors.

The target remains 60% of the parent LoRA A/B factor count. Every method records its realized physical count, budget error, per-module ranks, retained squared-energy fraction, spectral entropy, weight-file bytes and validation metrics. Integer ranks can make exact matching impossible; report actual counts rather than describing them as exact matches.

The export adjusts each module's PEFT `alpha_pattern` so the original LoRA multiplier is unchanged after rank reduction. Before metrics are reported, reload the adapter on a fresh FP32 base, verify every instantiated A/B rank and runtime scaling multiplier, and evaluate the same RTE validation examples. The parent model weights must be pinned to an immutable Hugging Face commit SHA for this comparison; a floating `main` revision is not acceptable for a final reproducibility claim.


## Base-model snapshot and provenance

The V1 config pins `microsoft/deberta-v3-base` to Hub revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. Before the first pruning comparison, the local Hugging Face cache contained two partial snapshot directories: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` and `de19fe7db5162df5f3d8f0b41321c0267288fd74`.

A direct local comparison established:
- 210 PyTorch checkpoint tensors and 210 safetensors tensors;
- all 210 common tensors have matching shapes and are exactly equal after loading;
- maximum absolute tensor difference is 0.0;
- the two snapshot config files are byte-identical, with SHA-256 `649f6a1ec33c6bdd9a648d6c66019d461139e54957073eafe9bbc2d34c75b0b`.

This establishes identical model-weight and model-config contents across the two cached snapshots. It does **not** establish a cryptographic per-run record of which commit each of the three completed parent-training processes resolved, because their historical `base_model_revision` metadata fields were null. The parent runs are retained with this limitation disclosed; do not rewrite their historical metadata or claim that the immutable commit was recorded at training time. For subsequent pruning reloads, use the pinned `8ccc9...` revision, whose model weights/config were shown to be identical to the `de19...` snapshot that held the safetensors file observed during training.

The parent-training runner now falls back to the configured revision when Transformers does not populate `config._commit_hash`, so future metadata records the pinned config revision rather than null.


## Gradient-signal reliability and spectral-concentration mechanism analysis

After all three seed-level pruning comparisons complete, run `scripts/analyze_gradient_spectral_mechanism.py`. The analysis consumes the already-saved parent `gradient_importance.json` files and the three policy `allocation.json` files per seed. It does not retrain models or recompute SVDs.

### A. Gradient-signal reliability across seeds

For each seed, normalize its 12 block-level gradient scores by that seed's mean score:

[
\tilde g_l^{(s)} = \frac{g_l^{(s)}}{\frac{1}{L}\sum_j g_j^{(s)}}.
]

This removes seed-wide magnitude differences while preserving relative block importance. Compare:
- pairwise Spearman and Kendall rank correlations of the normalized 12-dimensional scores across seeds;
- pairwise correlations of the corresponding layer-importance rank positions;
- per-block mean and sample standard deviation of normalized importance and rank across seeds.

A score of 1.0 in the normalized representation is the within-seed average block importance. If every block score is zero, use the neutral value 1.0 for each block, consistent with the V1 threshold-policy fallback.

### B. Gradient importance versus spectral concentration

For every adapted projection, use the saved normalized squared-spectral entropy:

[
H_{\mathrm{sq}}=-\frac{1}{\log r}\sum_i p_i\log(p_i+\epsilon),
\qquad p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2}.
]

Average the two target-projection entropy values within each Transformer block, yielding 12 block-level entropy observations per seed. Compute descriptive Pearson, Spearman and Kendall coefficients between block gradient importance and mean block entropy separately for each seed. Do not pool all 36 block/seed rows as if they were independent observations.

Also compute module-level entropy-versus-retained-rank correlations for each policy. These describe how spectral concentration relates to the rank actually selected by each rule; they are not a causal test of optimal rank.

### C. Rank-allocation stability

Compare the 24 per-module retained ranks across seed pairs for each policy. Report correlations, exact-rank agreement, mean absolute rank difference and maximum absolute rank difference. Repeat at block level using the mean of the query/value ranks per block. Report per-block mean and sample standard deviation of retained rank across seeds.

GARS rank-versus-gradient correlations are policy-coupled by construction because gradient importance is an input to GARS thresholds. They should be treated as allocation diagnostics, not independent validation that gradient scores identify the optimal rank.

### Statistical boundaries

All correlations are descriptive. Each within-seed relationship uses only 12 blocks or 24 projection modules; the cross-seed reliability comparison has only three seeds. The script intentionally emits no p-values and supports no significance claim. The analysis is a mechanism diagnostic for the DeBERTa-v3-base/RTE pilot, not evidence of generality across model scales or tasks.

### Execution and output

Run after the local quality gates pass:

```bash
python scripts/analyze_gradient_spectral_mechanism.py \
  --output results/v1_pruned/mechanism_analysis.json
```

The output includes seed-level gradient scores/ranks, threshold-bound counts, gradient/entropy correlations, policy-specific rank diagnostics, pairwise cross-seed rank stability, per-block summaries, and interpretation notes. Preserve the generated JSON with the Phase 1 experiment artifacts.


## Beta sensitivity ablation (post-hoc, same parent)

After the initial V1 comparison, assess whether the gradient-conditioning strength `beta` materially changes the result. Use the existing trained parent adapters and spectra; do not retrain the parent for this sensitivity analysis.

Run `scripts/run_gars_beta_sweep.py` for `beta ∈ {0.00, 0.25, 0.50, 0.75, 1.00}` on seeds 42, 123 and 999. Every beta is independently recalibrated to the same 708,096 physical A/B factor parameters. Only the GARS policy is exported per beta; the baseline methods are not recomputed in each sweep run.

Two equivalence checks are enforced:
- `beta=0` makes the gradient ratio exponent zero, so every layer receives the same energy threshold. The resulting per-module ranks must match the already saved uniform-local-energy allocation for that seed.
- `beta=0.5` must reproduce the original V1 GARS allocation for that seed.

The sweep records full-precision validation metrics, paired changes versus parent, rank histograms, cross-seed allocation correlations, rank changes relative to beta=0, and the number of layer thresholds clipped at each bound. This is a sensitivity analysis, not a hyperparameter-selection exercise: do not choose beta using the validation split and then describe the same validation result as an unbiased estimate. The fixed beta=0.5 remains the V1 configuration; alternative values are exploratory.


## Cyclic gradient-to-layer alignment ablation

The beta sensitivity sweep established that changing beta changes the allocated ranks more than the aggregate RTE validation metrics. The next controlled diagnostic asks whether the *observed assignment* of gradient importance to Transformer blocks carries useful information, rather than merely the gradient-score distribution.

### Controlled reassignment

Use the same three trained parent adapters (seeds 42, 123 and 999), their existing spectra and the frozen V1 beta (0.5). For sorted layer IDs \(L=(l_0,\ldots,l_{11})\), define cyclic shift \(s\) by assigning:

\[
g'_{l_i}=g_{l_{(i-s)\bmod 12}}, \qquad s\in\{1,\ldots,11\}.
\]

Each non-identity shift preserves the exact 12-score multiset and its mean, but changes which Transformer block receives each observed score. Across all 11 non-identity shifts, every source-layer score is assigned to every other destination layer exactly once.

For each seed and shift, rerun only the GARS physical export, reload and RTE validation. Keep unchanged:
- the trained parent adapter and all 24 projection spectra;
- beta=0.5, threshold clipping [0.80, 0.99], and minimum rank 2;
- the 60% target physical A/B parameter fraction;
- the pinned DeBERTa-v3-base revision, tokenizer, validation examples and evaluation code;
- PEFT rank/alpha export and runtime scaling checks.

This creates 33 matched-budget GARS exports (11 shifts × 3 seeds), referenced against the already completed observed-alignment GARS allocations and validation scores. The expected physical factor count is the same as the original matched-budget GARS runs.

### Implementation and checks

- The cyclically_shift_importance helper performs reassignment deterministically and validates layer IDs, shift and finite non-negative scores.
- tests/test_importance_ablation.py checks score-multiset and mean preservation, cyclic wraparound and exhaustive source-to-other-layer coverage across all non-identity shifts.
- scripts/run_pruning_comparison.py --importance-cyclic-shift N uses the reassigned importance map for GARS and records original/assigned scores plus the destination-to-source layer mapping in allocation.json.
- scripts/run_gars_gradient_permutation.py orchestrates the 11 shifts across seeds, verifies revision, assignment metadata, physical budget and reload status, then records per-seed and across-seed paired validation and rank-allocation comparisons.

Run after the local quality gates pass:

\`\`\`bash
ruff check scripts src/gars tests experiments
python -m pytest -q
python scripts/run_gars_gradient_permutation.py --device mps
\`\`\`

The default output is results/v1_gradient_permutation/gradient_permutation_summary.json. The runner refuses a non-empty output directory unless --overwrite is explicitly passed. For a completed run, verify that run_count is 33, all_requested_runs_completed is true, every shift/seed reload status is passed, and every export has the same physical A/B factor count.

### Interpretation boundary

This is a deterministic cyclic-alignment control, not a draw from all possible permutations and not a formal permutation test. It can show whether the observed layer-score alignment behaves differently from a structured set of reassigned alignments in this particular three-seed RTE pilot. It cannot establish statistical significance, causality or generalization. Validation metrics remain descriptive and must not be used to retune beta. Report the full per-seed spread and rank-allocation changes; do not report only the best shift.
