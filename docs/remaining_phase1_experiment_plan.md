# Remaining Phase 1 experiment plan

**Planning date:** 16 November 2026  
**Submission deadline:** 3 December 2026, 5:30 PM IST  
**Completed evidence:** three-seed DeBERTa-v3-base/RTE and DeBERTa-v3-large/RTE parent/pruning studies; large-model final-window temporal gradient diagnostics; gradient/spectral and cyclic-alignment diagnostics; beta sensitivity; full three-seed merged/unmerged inference-latency benchmark.

## Current empirical position

### Cross-scale compression and classification metrics

The DeBERTa-v3-base/RTE and DeBERTa-v3-large/RTE experiments both achieved approximately 39.974% reduction in physical LoRA A/B factor parameters at the matched budget. Across both model scales and all three seeds, the pruned methods retained parent accuracy and macro-F1 at the reported precision.

The large-model three-seed study used revision `8869586e1c6f5654a9a2a5c517fe710e24a488d0`; all nine method/seed adapter reload validations passed. Its mean validation losses were parent 0.371230, GARS 0.371468, global singular-value 0.371283 and uniform local energy 0.370233. GARS loss was higher than both spectrum-only baselines in all three paired seeds. The base-model study had shown the reverse ordering for the mean GARS-minus-baseline loss differences. Therefore, the physical factor reduction and unchanged classification metrics transfer in the tested RTE setups, but a GARS-specific loss advantage does not transfer consistently across model scales.

See:
- [DeBERTa-v3-large three-seed validation and parameter results](v1_deberta_large_seeded_results.md)
- [DeBERTa-v3-large seed-42 feasibility record](v1_deberta_large_seed42_pilot.md)

### Inference latency

The full three-seed benchmark compared parent, GARS, uniform local-energy and global singular-value adapters in merged and unmerged modes at batch sizes 1, 8 and 32. It does not demonstrate a robust 15–20% latency improvement:
- Unmerged batch 8: GARS was faster in all three seeds; mean per-seed latency reduction was approximately 10.1%.
- Merged batch 32: GARS was slower in all three seeds; mean per-seed slowdown was approximately 9.0%.
- Other conditions were mixed or modest.

The benchmark measures synchronized model-forward latency only. Tokenization, collation, host-to-device transfer, model loading and adapter merge time are excluded. It is hardware/runtime-specific and does not establish production serving performance.

See [the full latency benchmark report](v1_latency_benchmark_results.md).

### Temporal gradient reliability

The temporal analyzer successfully ran on the existing DeBERTa-v3-large/RTE parent artifacts for all three seeds; no retraining was needed. It verified that the 50 saved stepwise profiles reproduce each seed's stored arithmetic-mean importance profile.

Across seeds, the mean within-window all-pair Spearman rank correlation was 0.8244 (sample SD 0.1051), and the mean adjacent-step correlation was 0.8383 (sample SD 0.0867). The signal is moderately to strongly persistent on average, but seed-dependent: all-pair rank rho was 0.8374, 0.7134 and 0.9224 for seeds 42, 123 and 999. Adjacent top-3 Jaccard was lower (0.2918, 0.3061 and 0.6143 respectively), showing that the identity of the very highest-ranked layers is less stable than the full ranking.

A diagnostic recomputation of thresholds at each captured step, using each seed's fixed calibrated `tau_base`, beta 0.5 and clipping bounds [0.80, 0.99], found that 82.28% of the 3,600 step/layer thresholds landed on a clipping boundary (25.03% lower-bound and 57.25% upper-bound hits). This is a diagnostic only: the actual V1 allocation still uses the arithmetic-mean gradient profile, not instantaneous dynamic pruning. The result characterizes saturation of the bounded threshold map, not rank changes during training or an observed performance effect.

See [the detailed temporal gradient diagnostic report](v1_deberta_large_temporal_gradient_results.md). Optimizer-step pairs are sequential/autocorrelated; these descriptive correlations are not independent samples and do not support p-values or significance claims.

## MRPC dataset identifier compatibility fix

The initial seed-42 MRPC attempt stopped during `load_dataset`, before training began. The MRPC config used the legacy unnamespaced identifier `glue`, which the current strict Hugging Face URI parser rejects when it constructs an `hf://datasets/glue@...` URI. The three GLUE experiment configs now use the canonical repository ID `nyu-mll/glue`; no task subset, model revision or training/policy hyperparameter was changed. This canonical public dataset repository includes both MRPC and RTE. See the work-log entry for the diagnostic check to run before restarting training.

## Priority 1 — second task: DeBERTa-v3-base on GLUE MRPC

The MRPC config already exists at `configs/v1_deberta_mrpc.json`. It keeps the base model revision, LoRA rank/alpha, target modules, training schedule, final 50-step gradient capture, beta and physical budget consistent with the RTE base experiment while changing the task to MRPC.

Begin with seed 42:

```bash
python experiments/train_parent_rte.py \
  --config configs/v1_deberta_mrpc.json \
  --seed 42 --device mps \
  --output-root results/v1_parent_mrpc
```

Validate the parent:

```bash
python scripts/validate_parent_adapter.py \
  --config configs/v1_deberta_mrpc.json \
  --seed-dir results/v1_parent_mrpc/seed_42 \
  --device mps
```

Run all three pruning methods at the matched physical budget:

```bash
python scripts/run_pruning_comparison.py \
  --config configs/v1_deberta_mrpc.json \
  --seed 42 --device mps \
  --parent-root results/v1_parent_mrpc \
  --output-root results/v1_pruned_mrpc
```

If seed 42 completes training, parent reload validation and all three policy exports, continue with seeds 123 and 999. Then aggregate:

```bash
python scripts/aggregate_pruning_results.py \
  --results-root results/v1_pruned_mrpc \
  --seeds 42 123 999 \
  --output results/v1_pruned_mrpc/aggregate.json
```

This replication tests whether matched-budget compression and the policy comparisons observed on RTE extend to another sentence-pair classification task. Do not assume that MRPC has the same accuracy scale or that an unchanged score is guaranteed. Report all observed metrics and the actual parameter reduction.

## Optional follow-up — threshold saturation sensitivity

The temporal diagnostic found heavy instantaneous threshold clipping. If time remains after the MRPC feasibility run, consider a controlled *post-hoc* sensitivity study using the saved large-model parent spectra and final-window gradient means. Specify the alternative clipping/saturation policy before evaluating it, match the same physical A/B budget, retain the current V1 result as the frozen reference and label all validation comparisons exploratory. Do not silently retune beta or clipping bounds on the same validation set and report the resulting best value as unbiased performance. This is optional, and MRPC task replication has higher priority.

## Not recommended before submission

Do not prioritize dynamic in-training pruning, optimizer-state surgery, a large random-permutation search, another broad latency sweep, or additional beta tuning. They expand scope without directly testing cross-task transfer or resolving a well-defined evidence gap.

## Decision gates and reporting discipline

- Keep binaries, checkpoints and generated benchmark data local and ignored by Git. Commit source, configs and reports only.
- Do not overwrite existing result directories. Use separate roots for the MRPC task or any exploratory sensitivity study.
- Each task/model result must include the parent baseline, all three policies, paired seed metrics, physical A/B parameter counts, reload status and exact pinned base revision.
- A target is not an achievement: report actual accuracy changes and actual parameter reduction.
- Latency is separate from factor-parameter reduction. Report measured values; do not infer speedup from parameter counts.
- Keep all claims limited to the models, tasks, seeds, devices and splits actually measured. Validation metrics are not test-set generalization.

## 16 November 2026 — large-model temporal diagnostics completed

The student supplied the large-model temporal-analysis output for seeds 42, 123 and 999. Each seed contained 50 saved optimizer-step gradient profiles over optimizer steps 419–468, and each stepwise arithmetic mean reproduced the stored layer-importance profile.

The mean within-seed all-pair rank correlation across the three seeds was 0.8243916 (sample SD 0.1050704); mean adjacent-step rank correlation was 0.8383141 (sample SD 0.0867266). Seed-level all-pair rho values were 0.8373934, 0.7134254 and 0.9223560 for seeds 42, 123 and 999. Adjacent top-3 Jaccard values were 0.2918367, 0.3061224 and 0.6142857. This shows average persistence of the layer ranking but meaningful seed heterogeneity and weaker stability of the top-three set.

At each captured optimizer step, recalculating thresholds using the seed's fixed calibrated `tau_base`, beta 0.5 and clipping bounds [0.80, 0.99] produced 3,600 step/layer values. A total of 901 (25.03%) hit the lower bound and 2,061 (57.25%) hit the upper bound: 82.28% landed at a clipping boundary. This is a diagnostic of instantaneous threshold saturation, not evidence of dynamic pruning during training; the actual V1 policy used the mean profile over the capture window.

The full findings and limitations are recorded in [v1_deberta_large_temporal_gradient_results.md](v1_deberta_large_temporal_gradient_results.md).

**Next action:** begin the GLUE MRPC/base-model replication with seed 42 and the feasibility gates above. The remaining submission time should be used to strengthen task transfer rather than rerun already completed RTE experiments.
