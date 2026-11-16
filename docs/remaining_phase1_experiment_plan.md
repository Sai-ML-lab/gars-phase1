# Remaining Phase 1 experiment plan

**Planning date:** 16 November 2026  
**Submission deadline:** 3 December 2026, 5:30 PM IST  
**Completed evidence:** three-seed DeBERTa-v3-base/RTE and DeBERTa-v3-large/RTE parent/pruning studies; gradient/spectral diagnostics; beta sensitivity; 33-export cyclic alignment control; three-seed merged/unmerged inference-latency benchmark.

## Current empirical position

The two RTE model scales both achieved approximately 39.974% reduction in physical LoRA A/B factor parameters while preserving parent validation accuracy and macro-F1 at the reported precision.

The large-model study is complete for seeds 42, 123 and 999. All nine policy/seed adapter reload validations passed. On DeBERTa-v3-large, GARS has mean validation loss 0.371468, global singular-value pruning 0.371283, and uniform local-energy pruning 0.370233, versus parent 0.371230. GARS is worse than both spectrum-only baselines on loss in all three seeds. The earlier DeBERTa-v3-base result favored GARS over uniform local energy and slightly over global singular-value pruning on mean validation loss. This difference in baseline ordering across model scales is an important finding: parameter reduction and unchanged classification metrics transfer, but a validation-loss advantage unique to GARS is not established. See [the full large-model results](v1_deberta_large_seeded_results.md) and [the seed-42 pilot record](v1_deberta_large_seed42_pilot.md).

The three-seed latency benchmark also completed. It does not demonstrate a robust 15–20% latency improvement: unmerged batch 8 is faster in all seeds (mean per-seed reduction about 10.1%), while merged batch 32 is slower in all seeds (mean per-seed increase about 9.0%). Other conditions are mixed. See [the latency benchmark report](v1_latency_benchmark_results.md).

## Priority 1 — attempt temporal gradient analysis on the existing large-model runs

The current parent trainer stores `stepwise_importance_by_optimizer_step` for the final 50 optimizer updates. The large-model seeds were trained with this instrumented trainer, so inspect the existing files first. This may produce the temporal reliability evidence without any retraining.

Check all three artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path

for seed in (42, 123, 999):
    path = Path(f"results/v1_parent_deberta_large_rte/seed_{seed}/gradient_importance.json")
    data = json.loads(path.read_text())
    steps = data.get("stepwise_importance_by_optimizer_step")
    print(f"seed={seed}: captured_steps={len(steps) if isinstance(steps, list) else None}")
PY
```

If all seeds contain the 50 stepwise observations, run:

```bash
python scripts/analyze_gars_gradient_temporal_stability.py \
  --config configs/v1_deberta_v3_large_rte.json \
  --parent-root results/v1_parent_deberta_large_rte \
  --pruned-root results/v1_pruned_deberta_large_rte \
  --output results/v1_pruned_deberta_large_rte/temporal_gradient_analysis.json
```

The analyzer verifies that the stepwise arithmetic means reproduce the saved per-layer importance profile. It then measures all-pairs/lagged rank stability, adjacent top-rank stability, per-layer variability and per-step threshold dynamics using each seed's calibrated GARS threshold base.

If the field is absent, do not infer temporal stability from the aggregate layer means and do not automatically rerun training. Record the limitation and prioritize the second task below. The temporal diagnostics are descriptive; sequential optimizer steps are autocorrelated and must not be treated as independent samples or used for significance claims.

## Priority 2 — second task: DeBERTa-v3-base on GLUE MRPC

The MRPC config already exists at `configs/v1_deberta_mrpc.json`. It keeps the base model revision, LoRA rank/alpha, target modules, training schedule, final 50-step gradient capture, beta and physical budget consistent with the RTE base experiment while changing the task to MRPC.

Start with seed 42:

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

If seed 42 completes training, reload validation and all policy exports, continue with seeds 123 and 999. Then aggregate:

```bash
python scripts/aggregate_pruning_results.py \
  --results-root results/v1_pruned_mrpc \
  --seeds 42 123 999 \
  --output results/v1_pruned_mrpc/aggregate.json
```

This replication tests whether the matched-budget compression and policy comparisons observed on RTE extend to another sentence-pair classification task. Do not assume that MRPC has the same accuracy scale or that an unchanged score is guaranteed. Report all observed metrics and the actual parameter reduction.

## Not recommended before submission

Do not prioritize dynamic in-training pruning, optimizer-state surgery, more beta tuning, a large random-permutation search, or a new broad latency sweep. The highest-value remaining steps are the no-retraining temporal check on the large-model artifacts and the MRPC task replication.

## Decision gates and reporting discipline

- Keep binaries, checkpoints and generated benchmark data local and ignored by Git. Commit source, configs and reports only.
- Do not overwrite existing result directories. Use separate roots for any new task or replication.
- Each task/model result must include the parent baseline, all three policies, paired seed metrics, physical A/B parameter counts, reload status and exact pinned base revision.
- A target is not an achievement: report actual accuracy changes and actual parameter reduction.
- Latency is separate from factor-parameter reduction. Report measured values; do not infer speedup from parameter counts.
- Keep all claims limited to the models, tasks, seeds, devices and splits actually measured. Validation metrics are not test-set generalization.

## 16 November 2026 — large-model three-seed result complete

The DeBERTa-v3-large/RTE experiment completed for seeds 42, 123 and 999, all using pinned revision `8869586e1c6f5654a9a2a5c517fe710e24a488d0`. All nine policy/seed exports passed reload validation. Parent A/B factor count was 3,145,728; every policy/seed reached 1,888,256 parameters, a 39.9739583% reduction.

Across seeds, all methods reproduce parent accuracy (0.867629 mean) and macro-F1 (0.866789 mean) at reported precision. Mean validation losses were parent 0.371230, GARS 0.371468, global singular-value 0.371283 and uniform local energy 0.370233. GARS loss was higher than global and uniform in each paired seed; uniform had the lowest mean loss. Compared with the DeBERTa-v3-base/RTE result, the GARS-vs-baseline loss ordering is not stable across model scales. This supports cross-scale transfer of the physical factor reduction and unchanged accuracy/F1 in the tested setup, not unique predictive superiority for GARS.

Full values and paired differences are recorded in [v1_deberta_large_seeded_results.md](v1_deberta_large_seeded_results.md).

**Next:** inspect the large-model parent gradient artifacts for saved 50-step profiles and, if present, run the temporal analyzer using the existing large-model parent/pruned roots. Then proceed to MRPC as the second-task replication, beginning with a seed-42 feasibility gate.

