# GARS Phase 1 — cross-experiment synthesis

## Study matrix

The completed Phase 1 comparative study currently includes three three-seed experiments:

| Base model | Task | Seeds | Parent A/B parameters | Pruned A/B parameters | Reduction |
|---|---|---:|---:|---:|---:|
| DeBERTa-v3-base | GLUE RTE | 42, 123, 999 | 1,179,648 | 708,096 | 39.974% |
| DeBERTa-v3-large | GLUE RTE | 42, 123, 999 | 3,145,728 | 1,888,256 | 39.974% |
| DeBERTa-v3-base | GLUE MRPC | 42, 123, 999 | 1,179,648 | 708,096 | 39.974% |

Each experiment compares the common parent adapter, GARS, uniform local squared-energy pruning and a global singular-value cutoff. All three pruning policies use the same physical LoRA A/B parameter budget within each seed. The 39.974% reduction applies only to A/B factor parameters, not the entire base model. All nine policy/seed exports passed reload validation in each experiment.

## Aggregate classification and loss metrics

Values are mean ± sample SD over the three training seeds.

### DeBERTa-v3-base / GLUE RTE

| Method | Accuracy | Macro-F1 | Validation loss |
|---|---:|---:|---:|
| Parent | 0.831528 ± 0.012678 | 0.829541 ± 0.013702 | 0.444437 ± 0.009443 |
| GARS | 0.831528 ± 0.012678 | 0.829541 ± 0.013702 | 0.445440 ± 0.010439 |
| Global singular-value | 0.830325 ± 0.014440 | 0.828254 ± 0.015627 | 0.445873 ± 0.011321 |
| Uniform local energy | 0.831528 ± 0.012678 | 0.829349 ± 0.013778 | 0.448445 ± 0.012842 |

Paired mean loss differences: GARS − parent +0.001003; GARS − global -0.000433; GARS − uniform -0.003004. GARS preserved parent accuracy/F1 at displayed precision and had lower mean loss than both baselines, though the global-baseline loss difference was small and mixed by seed.

### DeBERTa-v3-large / GLUE RTE

| Method | Accuracy | Macro-F1 | Validation loss |
|---|---:|---:|---:|
| Parent | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.371230 ± 0.046347 |
| GARS | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.371468 ± 0.046416 |
| Global singular-value | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.371283 ± 0.046370 |
| Uniform local energy | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.370233 ± 0.045922 |

Paired mean loss differences: GARS − parent +0.000238; GARS − global +0.000185; GARS − uniform +0.001235. GARS retained accuracy/F1 but had higher loss than both spectrum-only methods in all three seeds. Uniform local-energy pruning had the lowest mean loss.

### DeBERTa-v3-base / GLUE MRPC

| Method | Accuracy | Macro-F1 | Validation loss |
|---|---:|---:|---:|
| Parent | 0.897059 ± 0.004245 | 0.880135 ± 0.003305 | 0.267556 ± 0.011071 |
| GARS | 0.897876 ± 0.002830 | 0.880991 ± 0.001823 | 0.267861 ± 0.011082 |
| Global singular-value | 0.897059 ± 0.004245 | 0.880135 ± 0.003305 | 0.267846 ± 0.010832 |
| Uniform local energy | 0.897876 ± 0.002830 | 0.880991 ± 0.001823 | 0.267710 ± 0.010849 |

Paired mean loss differences: GARS − parent +0.000305; GARS − global +0.000015; GARS − uniform +0.000151. GARS and uniform have identical accuracy/F1 at reported precision; their small aggregate gain versus parent comes from seed 42 alone (one additional correctly classified validation example in that seed). GARS and global are nearly tied on mean loss, while uniform has slightly lower mean loss than GARS.

## Main cross-experiment findings

### Finding 1 — matched-budget A/B-factor compression transfers across the tested models and tasks

All three experiments achieved 39.9739583% physical LoRA A/B factor reduction at matched budgets. This includes a scale extension from 12-layer DeBERTa-v3-base to 24-layer DeBERTa-v3-large, and a task extension from RTE to MRPC on DeBERTa-v3-base. All 27 policy/seed exports across the three experiments passed reload validation.

This supports the feasibility and reproducibility of the post-hoc physical factor-pruning pipeline in the tested conditions. It does not mean the full model is 40% smaller or that total inference compute/latency is reduced by 40%.

### Finding 2 — classification accuracy and macro-F1 are preserved at reported precision

On both RTE model scales, all three pruned methods retained parent accuracy and macro-F1 at reported precision. On MRPC, global singular-value pruning matched parent accuracy/F1; GARS and uniform local-energy pruning had a small aggregate improvement that was entirely due to seed 42. These results are descriptive validation metrics on 277-example RTE and 408-example MRPC validation splits, not independent test-set generalization.

### Finding 3 — a GARS-specific validation-loss advantage is not robust across scale/task

GARS had lower mean loss than uniform local-energy pruning on base/RTE, but higher mean loss than uniform on large/RTE and base/MRPC. Its comparison with global singular-value pruning was small and/or mixed in all three settings. Therefore, the evidence does not support a general claim that gradient-conditioned thresholds improve validation loss over spectrum-only policies. This variation across conditions is itself an important research result.

### Finding 4 — the temporal gradient signal is persistent on average, but seed-dependent

On large/RTE, final-window within-seed all-pair and adjacent-step rank correlations averaged 0.8244 and 0.8383. Seed 123 was less stable than seeds 42 and 999, and adjacent top-three Jaccard was lower than full-ranking correlation. Instantaneous threshold recomputation hit one of the clipping boundaries for 82.28% of step/layer observations. This characterizes temporal variation and saturation in the existing bounded threshold mapping; it does not demonstrate dynamic pruning or a causal performance effect.

### Finding 5 — latency effects are workload-dependent and below the proposal target as a general result

On the three-seed base/RTE MPS forward benchmark, unmerged batch 8 was faster for GARS in all three seeds with a mean per-seed latency reduction of approximately 10.1%. Merged batch 32 was slower in all three seeds with a mean per-seed slowdown of approximately 9.0%. Other mode/batch conditions were mixed or modest. This does not establish the proposed 15–20% general latency improvement or production-serving performance.

## Overall defensible Phase 1 conclusion

Across DeBERTa-v3-base/RTE, DeBERTa-v3-large/RTE and DeBERTa-v3-base/MRPC, the implemented post-hoc rank-pruning pipeline consistently removed approximately 39.974% of physical LoRA A/B factor parameters at a matched budget, while preserving parent validation accuracy and macro-F1 at reported precision (with a small seed-42-only improvement for GARS/uniform on MRPC). The pipeline and PEFT rank-pattern exports passed reload validation across all three-seed experiments.

The evidence supports cross-scale and cross-task feasibility of the compression procedure. It does not establish a consistent validation-loss advantage for GARS over uniform local-energy or global singular-value pruning: the loss ordering changes across the three model/task conditions. Temporal analysis indicates persistent but seed-dependent final-window gradient rankings and substantial clipping saturation. Latency measurements are likewise condition-dependent and do not show a robust 15–20% improvement.

The main research contribution should therefore be framed as a controlled empirical comparison and characterization of gradient-conditioned rank scheduling under matched physical budgets—not as proof that GARS is universally more accurate, lower-latency, or causally superior. No test-set generalization or production deployment claim is established by these validation and forward-only benchmark results.

## Remaining optional analysis before report freeze

The MRPC parent runs used the current trainer, which persists the 50 per-step layer gradient scores. The existing MRPC parent/pruned artifacts can therefore be checked for:
- temporal stability across the final 50 updates, using `scripts/analyze_gars_gradient_temporal_stability.py`;
- cross-seed gradient/spectral/rank-allocation mechanism diagnostics, using `scripts/analyze_gars_mechanisms.py`.

These are post-hoc analyses and do not require model retraining if all three MRPC gradient artifacts contain the expected stepwise field. They are optional strengthening analyses; do not expand into additional tuning on the MRPC validation split.


## MRPC temporal and mechanism diagnostics

The existing MRPC parent/pruned artifacts also supported both temporal and cross-seed mechanism analysis without retraining. All three seeds had 50 final-window profiles, and the temporal mean-profile integrity check passed.

### Temporal reliability

| Seed | All-pair within-window rho | Adjacent-step rho |
|---:|---:|---:|
| 42 | 0.920390 | 0.936060 |
| 123 | 0.904540 | 0.905240 |
| 999 | 0.865320 | 0.862850 |
| Mean ± sample SD | 0.896749 ± 0.028350 | 0.901384 ± 0.036758 |

The MRPC within-window rankings were relatively persistent across all three seeds. This is higher on average than the large/RTE temporal analysis (all-pair rho 0.8244; adjacent rho 0.8383), but these two analyses differ in both task and model scale, so the difference cannot be attributed to task alone.

Instantaneous threshold recomputation produced 1,800 step/layer observations: 30 lower-bound hits (1.67%) and 1,527 upper-bound hits (84.83%), or 86.50% total clipping. The mean-gradient profiles used by the actual V1 allocation also had upper-bound threshold clipping in 29 of 36 layer/seed cases (80.56%), with no lower-bound hits. This is a diagnostic of the bounded threshold mapping; it is not evidence of dynamic pruning during training.

### Cross-seed gradient, spectral and rank-allocation profiles

| Profile | Mean pairwise Spearman rho | Sample SD |
|---|---:|---:|
| Gradient importance (12 layers) | 0.946387 | 0.010682 |
| Module spectral entropy (24 modules) | 0.861449 | 0.048449 |
| Layer-mean spectral entropy (12 layers) | 0.762238 | 0.122926 |
| GARS module rank allocation | 0.862844 | 0.072860 |
| Global singular-value module rank allocation | 0.929285 | 0.024739 |
| Uniform local-energy module rank allocation | 0.937712 | 0.035853 |

Although the gradient profile ordering is highly concordant across seeds, GARS's discrete module-rank allocations are less stable across seeds than either spectrum-only baseline. This reinforces the point that stable importance ordering does not automatically produce stable integer allocations.

### Depth-adjusted within-seed associations

The mean raw Spearman association between gradient importance and layer-mean spectral entropy was -0.8508, but the mean partial Spearman after controlling for layer depth was -0.1391 (sample SD 0.3550). The raw relationship is therefore substantially attenuated by depth adjustment and is not robustly independent of layer depth in this three-seed/12-layer descriptive analysis.

The association between gradient importance and GARS-minus-uniform layer-rank uplift was +0.6190 raw and +0.5663 after depth adjustment (sample SD of partial correlations 0.1011). This is a descriptive association with the allocation difference, not evidence of causality or predictive advantage.

The full methods, per-seed profiles, threshold details and limitations are documented in [the MRPC temporal/mechanism report](v1_mrpc_temporal_mechanism_results.md).

### Combined mechanism takeaway

Across large/RTE and base/MRPC, the gradient signal is persistent on average within the measured final windows, but seed-level variation remains. Threshold saturation is substantial in both: 82.28% of instantaneous thresholds clipped in large/RTE and 86.50% in base/MRPC. On MRPC, most clipping is at the upper bound. These results make clipping and integer-allocation stability important design limitations to discuss explicitly. They do not establish a causal benefit from gradient conditioning.
