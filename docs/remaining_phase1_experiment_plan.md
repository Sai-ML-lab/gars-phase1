# Remaining Phase 1 experiment plan

**Planning date:** 16 November 2026  
**Submission deadline:** 3 December 2026, 5:30 PM IST  
**Completed evidence:** three-seed DeBERTa-v3-base/RTE parent and matched-budget pruning study; gradient/spectral diagnostics; beta sensitivity; 33-export cyclic alignment control; full three-seed merged/unmerged inference-latency benchmark; DeBERTa-v3-large/RTE seed-42 feasibility run.

## Research gaps that remain

The empirical study currently covers one model scale with three seeds and a second model scale with one seed. The DeBERTa-v3-base/RTE study demonstrates approximately 40% reduction in physical LoRA A/B factor parameters, while retaining parent-level validation accuracy and macro-F1 at the reported precision. The DeBERTa-v3-large/RTE seed-42 pilot also achieved approximately 40% factor reduction with unchanged validation accuracy and macro-F1 at reported precision, but it is not yet a multi-seed replication.

Inference latency has now been measured, but the results do **not** demonstrate a robust 15–20% improvement. GARS is faster than the parent in all three seeds for unmerged batch size 8 (mean per-seed latency reduction approximately 10.1%), but it is slower in all three seeds for merged batch size 32 (mean per-seed slowdown approximately 9.0%). Other conditions are mixed or modest. See [the full latency benchmark report](v1_latency_benchmark_results.md).

The existing mechanism work also lacks a temporal reliability result: historical V1 gradient artifacts contain only final-window means, not the individual per-step layer scores.

## Current priority order

### Priority 1 — complete the second model scale: DeBERTa-v3-large on RTE

The seed-42 feasibility gate has passed: parent validation was produced, all three matched-budget pruning methods exported and reloaded successfully, and each reached 1,888,256 physical LoRA A/B parameters (39.974% reduction from 3,145,728). Parent, GARS, global singular-value and uniform local-energy accuracy/F1 are identical at reported precision for this seed. GARS validation loss is slightly higher than both spectrum-only baselines, so no GARS superiority claim is supported by this one-seed result. See [the seed-42 pilot report](v1_deberta_large_seed42_pilot.md).

Now complete seeds 123 and 999 using the same config, one seed at a time. For each seed, train the parent:

```bash
python experiments/train_parent_rte.py \
  --config configs/v1_deberta_v3_large_rte.json \
  --seed 123 --device mps \
  --output-root results/v1_parent_deberta_large_rte
```

Validate the seed-123 parent before pruning:

```bash
python scripts/validate_parent_adapter.py \
  --config configs/v1_deberta_v3_large_rte.json \
  --seed-dir results/v1_parent_deberta_large_rte/seed_123 \
  --device mps
```

Run the three matched-budget pruning exports:

```bash
python scripts/run_pruning_comparison.py \
  --config configs/v1_deberta_v3_large_rte.json \
  --seed 123 --device mps \
  --parent-root results/v1_parent_deberta_large_rte \
  --output-root results/v1_pruned_deberta_large_rte
```

Only proceed to seed 999 after seed 123 completes training, parent reload validation and all three physical exports successfully. Repeat the same commands with `--seed 999` and `--seed-dir .../seed_999`.

After all three seeds, aggregate:

```bash
python scripts/aggregate_pruning_results.py \
  --results-root results/v1_pruned_deberta_large_rte \
  --seeds 42 123 999 \
  --output results/v1_pruned_deberta_large_rte/aggregate.json
```

Use the same RTE task, target projections, LoRA rank/alpha, beta, squared-energy criterion and physical-budget objective to test model-scale transfer while limiting task-related confounding. The config uses:
- `microsoft/deberta-v3-large`, pinned to revision `8869586e1c6f5654a9a2a5c517fe710e24a488d0`;
- 24 Transformer layers and hidden size 1024;
- rank 32, alpha 64, query/value projections;
- batch size 4 × accumulation 4, preserving the same effective batch size of 16 as the base experiment;
- 3 epochs, 50 final optimizer-step gradient captures, and the 60% physical A/B factor target.

Each scale result must include its parent baseline, all three policies, paired seed metrics, physical A/B parameter counts, reload status and exact pinned base revision. If compute or deadline pressure prevents all three seeds, report the completed runs as a pilot; do not present one or two seeds as a three-seed result.

### Priority 2 — temporal gradient-signal reliability (optional instrumented replication)

The current parent trainer writes `stepwise_importance_by_optimizer_step` in `gradient_importance.json`. The temporal analyzer checks that the arithmetic mean of the stored scores reproduces the saved final-window block-importance profile, then measures:
- all-pairs and lagged rank stability over the final 50 optimizer updates;
- adjacent-step top-1/top-3 stability;
- per-layer coefficient of variation and normalized-importance drift;
- per-step threshold dynamics using the fixed V1 calibrated threshold base, beta and clipping limits.

The historical parent outputs do not contain this field. The earlier run of:

```bash
python scripts/analyze_gars_gradient_temporal_stability.py
```

therefore stopped at seed 42 with an explicit missing-data error. Do not infer temporal reliability from the saved means.

If time and compute remain after the large-model run, train fresh base/RTE parents into a separate root, generate matched pruning exports into another separate root, and run the temporal analyzer. This is a new replicated training study, not reconstruction of the historical V1 training trajectory. Do not overwrite original artifacts or mix new gradient traces with old pruning allocations.

Optional instrumented base/RTE rerun commands:

```bash
python experiments/train_parent_rte.py --config configs/v1_deberta_rte.json --seed 42 --device mps --output-root results/v1_parent_rte_temporal
python experiments/train_parent_rte.py --config configs/v1_deberta_rte.json --seed 123 --device mps --output-root results/v1_parent_rte_temporal
python experiments/train_parent_rte.py --config configs/v1_deberta_rte.json --seed 999 --device mps --output-root results/v1_parent_rte_temporal

python scripts/run_pruning_comparison.py --config configs/v1_deberta_rte.json --seed 42 --device mps --parent-root results/v1_parent_rte_temporal --output-root results/v1_pruned_rte_temporal
python scripts/run_pruning_comparison.py --config configs/v1_deberta_rte.json --seed 123 --device mps --parent-root results/v1_parent_rte_temporal --output-root results/v1_pruned_rte_temporal
python scripts/run_pruning_comparison.py --config configs/v1_deberta_rte.json --seed 999 --device mps --parent-root results/v1_parent_rte_temporal --output-root results/v1_pruned_rte_temporal

python scripts/analyze_gars_gradient_temporal_stability.py --config configs/v1_deberta_rte.json --parent-root results/v1_parent_rte_temporal --pruned-root results/v1_pruned_rte_temporal --output results/v1_pruned_rte_temporal/temporal_gradient_analysis.json
```

### Priority 3 — second task (only after scale experiment)

If compute and time remain after the second-scale experiment, repeat the base-scale parent/pruning study on GLUE MRPC using the same sentence-pair training pipeline, three seeds and matched physical budget. This tests whether the RTE result is task-specific. It is useful but secondary to demonstrating a second model scale.

## Completed inference-latency benchmark

The three-seed benchmark compared parent, GARS, uniform local-energy and global singular-value adapters in merged and unmerged modes at batch sizes 1, 8 and 32. All 12 method/seed runs completed with 10 warm-up and 30 timed forwards per representative batch.

The key result is workload dependence:
- Unmerged batch 8: GARS was faster in all three seeds; mean per-seed latency reduction was approximately 10.1%, with reductions of 21.0%, 2.1% and 7.0%.
- Merged batch 32: GARS was slower in all three seeds; mean per-seed slowdown was approximately 9.0%.
- Other GARS mode/batch combinations were mixed or showed small changes.

This does not establish the proposal's 15–20% latency target. The benchmark measures synchronized model forward latency only; tokenization, collation, host-to-device transfer, model loading and adapter merge time are excluded. It is hardware/runtime-specific and does not establish production serving performance.

See [the detailed latency benchmark report](v1_latency_benchmark_results.md) for aggregate medians, paired seed ratios, variability and claims boundaries.

## Not recommended before submission

Do not prioritize dynamic in-training pruning, optimizer-state surgery, a large random-permutation search, more beta tuning, or additional seeds on the existing base RTE setup before completing the second-scale study. They expand scope without addressing the most important remaining evidence gap: model-scale transfer.

## Decision gates and reporting discipline

- Keep binaries, checkpoints and generated benchmark data local and ignored by Git. Commit source code, configs and reports only.
- Do not overwrite existing result directories. Use separate output roots for smoke tests and new replications.
- Each model-scale result must show its parent baseline, all three policies, paired seed metrics, physical A/B parameter counts, reload status and exact pinned base revision.
- A target is not an achievement: report actual accuracy drop and actual parameter reduction. If a scale misses the ≤1 percentage-point accuracy target, state that plainly.
- Latency is separate from factor-parameter reduction. Report observed values; do not infer speedup from parameter counts.
- Keep all claims limited to the models, tasks, seeds, devices and splits actually measured.

## 16 November 2026 — temporal capture availability, latency and large-model seed-42 pilot

The temporal analyzer was run against the historical V1 parent outputs and stopped at seed 42 because the saved gradient artifact has no `stepwise_importance_by_optimizer_step` field. The existing parent results therefore support the 50-step arithmetic-mean gradient profile and cross-seed profile comparisons, but not within-window temporal reliability.

The student reported that the full local test suite passes (69 tests). The temporal script's explicit missing-data error is expected and does not invalidate those tests or the original V1 results.

The full inference benchmark completed: 12/12 method/seed runs, 3 seeds, 4 adapter variants, merged/unmerged, batch sizes 1/8/32. Its findings and measurement boundary are documented in [v1_latency_benchmark_results.md](v1_latency_benchmark_results.md).

The DeBERTa-v3-large/RTE seed-42 feasibility run then passed the training/parent-validation/pruning-export path. All three policies reloaded successfully and achieved 1,888,256 A/B factor parameters against a parent count of 3,145,728 (39.974% reduction). Accuracy and macro-F1 matched the parent at reported precision for all policies. GARS validation loss was +0.00034262 versus parent and slightly higher than both spectrum-only baselines (+0.0001581). These are one-seed validation observations and do not establish superiority. See [v1_deberta_large_seed42_pilot.md](v1_deberta_large_seed42_pilot.md).

**Next action:** complete DeBERTa-v3-large/RTE seeds 123 and 999, checking parent reload validation and all three physical exports after each seed, then aggregate seeds 42/123/999. Only after the multi-seed model-scale result should optional temporal replication or MRPC be considered.

The DeBERTa-v3-large configuration uses the pinned Hugging Face revision `8869586e1c6f5654a9a2a5c517fe710e24a488d0`. The pinned config reports hidden size 1024 and 24 Transformer layers: https://huggingface.co/microsoft/deberta-v3-large/blob/8869586e1c6f5654a9a2a5c517fe710e24a488d0/config.json
