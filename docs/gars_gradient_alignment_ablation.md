# GARS cyclic gradient-to-layer alignment ablation

**Experiment:** matched-budget cyclic reassignment of the final-window gradient-importance scores  
**Base/task:** DeBERTa-v3-base / GLUE RTE  
**Seeds:** 42, 123, 999  
**Beta:** 0.50 (frozen V1)  
**Physical A/B factor budget:** 708,096 parameters for every export  
**Exports:** 11 non-identity cyclic shifts × 3 seeds = 33  
**Reload validation:** 33/33 passed

## 1. Design

For each seed, the original 12 layer-level gradient scores were cyclically reassigned to Transformer blocks using each of the 11 non-identity rotations. Every rotation preserves the exact score multiset and mean, while changing the layer receiving each score. The parent adapters, per-module spectra, beta, threshold clipping, minimum rank, parameter budget, base-model revision, validation split, and export/reload checks remain unchanged.

This is a deterministic cyclic-alignment control. It is not a sample from all possible assignments and is not a formal permutation test.

## 2. Seed 42

Observed GARS validation loss: **0.453595200**. The observed alignment ranks 7th of 12 (rank 1 is lowest loss); six cyclic controls have lower loss.

| Shift | Validation loss | Delta vs observed GARS | Modules with changed rank | Rank rho vs observed ranks |
|---:|---:|---:|---:|---:|
| 1 | 0.453283456 | -0.000311744 | 10 | 0.463121 |
| 2 | 0.453335733 | -0.000259467 | 10 | 0.329978 |
| 3 | 0.454503635 | +0.000908435 | 14 | 0.128615 |
| 4 | 0.456202052 | +0.002606852 | 16 | -0.183589 |
| 5 | 0.451529523 | -0.002065677 | 16 | -0.039194 |
| 6 | 0.450015322 | -0.003579878 | 16 | 0.041228 |
| 7 | 0.456358849 | +0.002763649 | 16 | 0.129565 |
| 8 | 0.437060610 | -0.016534590 | 15 | 0.298859 |
| 9 | 0.457202768 | +0.003607569 | 11 | 0.419933 |
| 10 | 0.439902617 | -0.013692583 | 9 | 0.565124 |
| 11 | 0.454839952 | +0.001244752 | 7 | 0.919939 |

Across the 11 rotations: mean 0.451294047; sample SD 0.006705974; range [0.437060610, 0.457202768]. Mean number of modules whose rank differs from observed GARS: 12.73 of 24. Mean rank-profile Spearman rho versus observed GARS: 0.2794.

## 3. Seed 123

Observed GARS validation loss: **0.433674704**. The observed alignment ranks 5th of 12; four cyclic controls have lower loss.

| Shift | Validation loss | Delta vs observed GARS | Modules with changed rank | Rank rho vs observed ranks |
|---:|---:|---:|---:|---:|
| 1 | 0.433575086 | -0.000099618 | 10 | 0.737684 |
| 2 | 0.433060878 | -0.000613826 | 12 | 0.463340 |
| 3 | 0.434263851 | +0.000589147 | 14 | 0.209710 |
| 4 | 0.434281877 | +0.000607172 | 16 | 0.151456 |
| 5 | 0.435647363 | +0.001972659 | 16 | 0.195052 |
| 6 | 0.432393321 | -0.001281383 | 16 | 0.313691 |
| 7 | 0.434164041 | +0.000489337 | 16 | 0.423372 |
| 8 | 0.425246308 | -0.008428396 | 15 | 0.529350 |
| 9 | 0.434142815 | +0.000468111 | 13 | 0.511313 |
| 10 | 0.434391858 | +0.000717154 | 11 | 0.509752 |
| 11 | 0.434069593 | +0.000394889 | 9 | 0.800263 |

Across the 11 rotations: mean 0.433203363; sample SD 0.002763307; range [0.425246308, 0.435647363]. Mean number of modules whose rank differs from observed GARS: 13.45 of 24. Mean rank-profile Spearman rho versus observed GARS: 0.4405.

## 4. Seed 999

Observed GARS validation loss: **0.449050232**. The observed alignment ranks 2nd of 12; one cyclic control has lower loss.

| Shift | Validation loss | Delta vs observed GARS | Modules with changed rank | Rank rho vs observed ranks |
|---:|---:|---:|---:|---:|
| 1 | 0.448905932 | -0.000144301 | 10 | 0.816077 |
| 2 | 0.449576838 | +0.000526606 | 12 | 0.581624 |
| 3 | 0.449289779 | +0.000239546 | 14 | 0.398601 |
| 4 | 0.450180682 | +0.001130449 | 16 | 0.382314 |
| 5 | 0.451679574 | +0.002629342 | 16 | 0.372900 |
| 6 | 0.449668852 | +0.000618619 | 16 | 0.495088 |
| 7 | 0.464932710 | +0.015882478 | 16 | 0.567870 |
| 8 | 0.449268973 | +0.000218740 | 15 | 0.625146 |
| 9 | 0.456442822 | +0.007392589 | 13 | 0.594744 |
| 10 | 0.459446685 | +0.010396453 | 9 | 0.656182 |
| 11 | 0.449123447 | +0.000073215 | 9 | 0.852055 |

Across the 11 rotations: mean 0.452592390; sample SD 0.005347492; range [0.448905932, 0.464932710]. Mean number of modules whose rank differs from observed GARS: 13.27 of 24. Mean rank-profile Spearman rho versus observed GARS: 0.5766.

## 5. Cross-seed summary

| Seed | Observed GARS loss | Cyclic mean loss | Cyclic sample SD | Cyclic range | Cyclic controls beating observed | Observed rank among 12 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.453595200 | 0.451294047 | 0.006705974 | 0.437060610–0.457202768 | 6/11 | 7 |
| 123 | 0.433674704 | 0.433203363 | 0.002763307 | 0.425246308–0.435647363 | 4/11 | 5 |
| 999 | 0.449050232 | 0.452592390 | 0.005347492 | 0.448905932–0.464932710 | 1/11 | 2 |

Across the 33 seed-by-shift comparisons, 11 cyclic controls have lower loss than their seed's observed GARS result and 22 have higher loss. The equally weighted mean of the 33 cyclic observations is approximately 0.445696600, compared with the original GARS three-seed mean of 0.445440045. Thus the observed alignment's average loss advantage over the average cyclic control is approximately 0.000256555. This is a small descriptive difference; the 33 control observations are not independent because they reuse the same three parent adapters and consist of structured rotations.

The observed alignment is not uniformly best: it ranks 7th, 5th and 2nd across the three seeds. Its per-seed difference from the cyclic-control mean changes sign. Cyclic shifts also alter 7–16 module ranks relative to observed GARS (mean 13.15 of 24 across 33 exports), with rank-profile correlation versus observed allocations varying substantially.

## 6. Interpretation and claims boundary

The evidence supports these bounded findings:

1. **The assignment affects allocation.** Reassigning the same gradient-score values changes a substantial portion of the 24 module ranks.
2. **The assignment affects validation loss in this pilot.** Some rotations improve loss and others worsen it; the spread is seed-dependent.
3. **The naturally observed assignment is not consistently superior.** It is better than most cyclic controls for seed 999, but worse than the cyclic-control mean for seeds 42 and 123.
4. **No optimal-alignment or causal claim is established.** The original alignment has a small aggregate mean-loss advantage over all cyclic controls, but it does not rank first in any of the three seed-specific comparisons.
5. **The control set is limited.** Eleven rotations are not all possible permutations of 12 layer scores, and the observations are not independent random draws. Do not calculate or report a permutation-test p-value from these rotations.

All results are validation-set measurements on one base-model scale, one task and three parent-training seeds. They do not establish test-set generalization or broad effectiveness across models/tasks. No latency or end-to-end system-size effect is inferred.

## 7. Reproducibility checks

The runner reported:
- run_count = 33;
- all_requested_runs_completed = true;
- physical A/B factor parameters = 708,096 for every export;
- reload status = passed for all 33 exports.

The full-precision machine-readable results are retained locally at results/v1_gradient_permutation/gradient_permutation_summary.json. The repository contains the source code and this report; generated model/export artifacts remain local and are not committed.
