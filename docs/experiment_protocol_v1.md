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
