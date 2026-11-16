# Remaining Phase 1 experiment plan

**Planning date:** 16 November 2026  
**Submission deadline:** 3 December 2026, 5:30 PM IST

## Completed empirical work

The current Phase 1 evidence includes three three-seed experiments:

| Model | Task | Seeds | Parent A/B params | Pruned A/B params | Reduction |
|---|---|---:|---:|---:|---:|
| DeBERTa-v3-base | GLUE RTE | 42, 123, 999 | 1,179,648 | 708,096 | 39.974% |
| DeBERTa-v3-large | GLUE RTE | 42, 123, 999 | 3,145,728 | 1,888,256 | 39.974% |
| DeBERTa-v3-base | GLUE MRPC | 42, 123, 999 | 1,179,648 | 708,096 | 39.974% |

All three policies (GARS, uniform local squared-energy and global singular-value cutoff) reached the same matched physical budget within each experiment. All 27 policy/seed adapter exports across the three experiments passed reload validation.

### Classification metrics and loss

All pruned methods retained parent accuracy and macro-F1 at reported precision on both RTE model scales. On MRPC, global singular-value pruning matched the parent metrics; GARS and uniform local energy had a small aggregate accuracy/F1 gain arising only from seed 42.

A GARS-specific validation-loss advantage did not transfer consistently:
- Base/RTE: GARS had lower mean loss than uniform local energy and a small lower mean loss than global singular-value pruning.
- Large/RTE: GARS loss was higher than both spectrum-only baselines in all three paired seeds; uniform local energy had the lowest mean loss.
- Base/MRPC: GARS and global singular-value pruning were nearly tied on mean loss; uniform local energy had slightly lower mean loss than GARS.

The complete side-by-side evidence is recorded in [the cross-experiment synthesis](phase1_cross_experiment_synthesis.md), with detailed reports for [base/RTE](v1_seeded_results.md), [large/RTE](v1_deberta_large_seeded_results.md) and [base/MRPC](v1_mrpc_seeded_results.md).

### Gradient temporal diagnostics

The DeBERTa-v3-large/RTE temporal analysis completed using the existing 50-step per-seed profiles, with no retraining. Mean within-seed all-pair rank rho was 0.8244 and adjacent-step rho was 0.8383; the result was seed-dependent and top-three identities were less stable. Instantaneous threshold recomputation placed 82.28% of step/layer values at a clipping boundary. See [large/RTE temporal diagnostics](v1_deberta_large_temporal_gradient_results.md).

### Inference latency

The full three-seed base/RTE benchmark compared parent, GARS, uniform and global adapters in merged/unmerged modes and batch sizes 1, 8 and 32. The measured effects were workload-dependent and did not establish a robust 15–20% latency improvement. Unmerged batch 8 was faster for GARS in all three seeds (mean per-seed reduction about 10.1%); merged batch 32 was slower in all three seeds (mean per-seed slowdown about 9.0%). See [the latency benchmark report](v1_latency_benchmark_results.md).

## Priority 1 — post-hoc MRPC temporal and mechanism diagnostics

The three MRPC parent runs were trained with the current instrumented trainer, which writes `stepwise_importance_by_optimizer_step` in each `gradient_importance.json`. Check the three files first:

```bash
python - <<'PY'
import json
from pathlib import Path

for seed in (42, 123, 999):
    path = Path(f"results/v1_parent_mrpc/seed_{seed}/gradient_importance.json")
    data = json.loads(path.read_text())
    rows = data.get("stepwise_importance_by_optimizer_step")
    print(f"seed={seed}: captured_steps={len(rows) if isinstance(rows, list) else None}")
PY
```

If all three contain the expected 50 steps, run the temporal analyzer:

```bash
python scripts/analyze_gars_gradient_temporal_stability.py \
  --config configs/v1_deberta_mrpc.json \
  --parent-root results/v1_parent_mrpc \
  --pruned-root results/v1_pruned_mrpc \
  --output results/v1_pruned_mrpc/temporal_gradient_analysis.json
```

Then run the cross-seed mechanism analysis:

```bash
python scripts/analyze_gars_mechanisms.py \
  --config configs/v1_deberta_mrpc.json \
  --parent-root results/v1_parent_mrpc \
  --pruned-root results/v1_pruned_mrpc \
  --output results/v1_pruned_mrpc/mechanism_analysis.json
```

These analyses do not retrain models. The temporal analyzer verifies that the 50 stepwise profiles reproduce the stored mean profile and reports within-window rank reliability and threshold dynamics. The mechanism analyzer compares gradient/spectral profile consistency and allocation associations across seeds. If temporal fields are missing, do not infer temporal stability from the aggregate mean; report the limitation.

Treat all correlation and temporal statistics as descriptive. Optimizer steps are sequential/autocorrelated, and three training seeds are a small sample. Do not add significance or causal claims.

## Priority 2 — freeze the Phase 1 evidence and finalize the report

Once the optional MRPC diagnostics are complete:
1. Freeze the experimental configs and all reported metrics. Do not tune beta or clipping thresholds against the same validation splits.
2. Reconcile the final report with [the cross-experiment synthesis](phase1_cross_experiment_synthesis.md) and the detailed per-experiment reports.
3. Clearly separate physical LoRA A/B factor reduction from full-model size, forward latency and production serving performance.
4. State that the strongest transferable result is matched-budget factor reduction with preserved classification accuracy/F1 in the tested model/task conditions.
5. Explicitly report that GARS does not consistently beat spectrum-only baselines on validation loss, that latency effects are workload-specific, and that threshold saturation is substantial in the instantaneous diagnostic.
6. Include limitations: three seeds, small GLUE validation splits, one hardware/runtime for latency, no independent test-set generalization, and no causal attribution.
7. Run final `ruff check scripts src/gars tests experiments` and `python -m pytest -q` after pulling the final source/config changes.

## Not recommended before submission

Do not add another full training task, another model scale, dynamic in-training pruning, optimizer-state surgery, a large random-permutation search, broad latency retuning, or further beta/clipping optimization unless the mentor explicitly asks for it. The three matched-budget experiments already address the main cross-scale and cross-task feasibility questions. The remaining high-value work is post-hoc mechanism analysis and rigorous report synthesis.

## Reporting discipline

- Keep binaries, checkpoints and generated benchmark data local and ignored by Git. Commit source, configs and reports only.
- Do not overwrite existing result directories. Any exploratory sensitivity study must use a new output root and remain clearly labeled exploratory.
- Report actual per-seed results, aggregate mean ± sample SD, paired differences, physical A/B parameter counts, revision and reload status.
- A target is not an achievement: report actual accuracy changes and actual parameter reduction.
- Validation metrics are not test-set generalization.
- Latency measures in the current benchmark are synchronized model-forward times only; they exclude tokenization, collation, host-to-device transfer, model loading and adapter merge time.

## 16 November 2026 — MRPC three-seed experiment completed

The DeBERTa-v3-base/MRPC study completed for seeds 42, 123 and 999 using the pinned base revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. All nine policy/seed adapter reload validations passed.

- Parent A/B factor count: 1,179,648.
- Pruned A/B factor count: 708,096 for every policy and seed.
- Reduction: 39.9739583%.
- Validation examples: 408 per seed.
- Parent means: accuracy 0.897059, macro-F1 0.880135, loss 0.267556.
- GARS means: accuracy 0.897876, macro-F1 0.880991, loss 0.267861.
- Global means: accuracy 0.897059, macro-F1 0.880135, loss 0.267846.
- Uniform local-energy means: accuracy 0.897876, macro-F1 0.880991, loss 0.267710.

GARS and uniform local energy had identical per-seed accuracy/F1; their small mean accuracy/F1 gain over parent came only from seed 42. GARS-vs-global mean loss difference was +0.000015, while GARS-vs-uniform was +0.000151. These small differences do not establish superiority. The full report is [v1_mrpc_seeded_results.md](v1_mrpc_seeded_results.md).

The canonical dataset identifier `nyu-mll/glue` was used after the legacy alias `glue` failed under the local strict Hugging Face URI parser. The identifier fix changed no task subset or training/policy hyperparameter.

**Next action:** freeze the experiments and finalize the Phase 1 report. The MRPC temporal and mechanism diagnostics have now completed; no further training run is currently recommended.


## 16 November 2026 — MRPC temporal and mechanism diagnostics completed

The student supplied successful output from both `analyze_gars_gradient_temporal_stability.py` and `analyze_gars_mechanisms.py` for the existing MRPC artifacts. All three seeds had 50 final-window profiles and the temporal analyzer completed without retraining.

- Mean within-seed all-pair temporal rank rho: 0.896749 (sample SD 0.028350).
- Mean adjacent-step rank rho: 0.901384 (sample SD 0.036758).
- Cross-seed raw gradient profile Spearman rho: 0.946387 (sample SD 0.010682).
- Cross-seed module spectral-entropy rho: 0.861449 (sample SD 0.048449).
- Cross-seed GARS module-rank allocation rho: 0.862844, compared with 0.929285 for global singular-value and 0.937712 for uniform local energy.
- Instantaneous thresholds hit the upper clipping bound in 1,527/1,800 observations (84.83%), lower bound in 30/1,800 observations (1.67%), total boundary clipping 86.50%.
- Mean-profile thresholds used by the actual V1 allocation hit the upper bound in 29/36 layer/seed cases (80.56%); none hit the lower bound.

The raw gradient/spectral-entropy association (-0.8508 mean rho) weakened to -0.1391 after controlling descriptively for layer depth (sample SD 0.3550). The gradient association with GARS-minus-uniform layer-rank uplift remained positive after depth adjustment (+0.5663 mean partial rho; sample SD 0.1011). These are descriptive correlations over 12 layers and three seeds, not significance tests or causal evidence.

Detailed outputs are recorded in [v1_mrpc_temporal_mechanism_results.md](v1_mrpc_temporal_mechanism_results.md), and combined with the previous temporal analysis in [phase1_cross_experiment_synthesis.md](phase1_cross_experiment_synthesis.md).

## Phase 1 experiment freeze recommendation

The planned three-seed experiments on base/RTE, large/RTE and base/MRPC are complete. The three matched-budget studies achieved approximately 39.974% physical LoRA A/B factor reduction, and all 27 pruning exports passed reload validation. Latency and temporal/mechanism analyses are complete for the planned scope. No additional model training or validation-set tuning is recommended before report submission.

Remaining work is report assembly and quality review:
1. Ensure the submitted methodology and results distinguish physical A/B factor parameter reduction from whole-model size, total compute and inference latency.
2. Include the cross-model/task table and disclose that GARS does not consistently outperform spectrum-only baselines on validation loss.
3. Describe the threshold clipping saturation and lower cross-seed rank-allocation stability of GARS as limitations/mechanism observations.
4. State the latency benchmark boundary and its workload-dependent findings; do not claim the 15–20% latency target was met.
5. State that all metrics are from validation splits and that the three-seed statistics are descriptive, not significance tests.
6. Run final local code quality gates after pulling all changes: `ruff check scripts src/gars tests experiments` and `python -m pytest -q`.

Do not add further tuning or experiments unless required by the mentor or rubric. The priority is a coherent, transparent report supported by the completed evidence.
