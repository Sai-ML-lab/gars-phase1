# DeBERTa-v3-large / GLUE RTE — final-window gradient temporal diagnostics

## Scope and integrity checks

The temporal analyzer was run on the existing DeBERTa-v3-large/RTE parent and matched pruning artifacts for seeds 42, 123 and 999. No training rerun was required.

- Base model: `microsoft/deberta-v3-large`
- Pinned revision: `8869586e1c6f5654a9a2a5c517fe710e24a488d0`
- Dataset: GLUE RTE
- Capture window: final 50 optimizer updates, steps 419–468 for each seed
- Layers: 24
- Observations: 50 layer-importance profiles per seed
- Each seed's arithmetic mean of its 50 saved profiles reproduced the stored `importance_by_transformer_block` profile.
- The threshold diagnostic used the actual seed-specific calibrated GARS `tau_base`, beta 0.5 and clipping bounds [0.80, 0.99].

The analyzer computes descriptive statistics. Optimizer steps are sequential and autocorrelated; the step pairs are not independent observations. No p-values or significance claims are made.

## Within-window rank stability

Spearman correlations compare the 24-layer importance rankings at different captured optimizer updates. “All-pair” averages every pair of the 50 step profiles (1,225 pairs per seed). “Adjacent-step” averages the 49 consecutive-step correlations.

| Seed | Mean all-pair Spearman rho | Mean adjacent-step rho | Mean adjacent top-3 Jaccard | CV of global gradient magnitude |
|---:|---:|---:|---:|---:|
| 42 | 0.837393 | 0.846034 | 0.291837 | 0.432351 |
| 123 | 0.713425 | 0.747986 | 0.306122 | 0.735850 |
| 999 | 0.922356 | 0.920923 | 0.614286 | 0.399890 |
| **Mean across seeds** | **0.824392** | **0.838314** | **0.404082** | **0.522697** |

Sample SD across the three seed-specific means:
- All-pair rank rho: 0.105070.
- Adjacent-step rank rho: 0.086727.

### Interpretation

The overall layer ranking is moderately to strongly stable on average in this final window, but its stability differs by seed. Seed 999 is highly stable (all-pair rho 0.9224; adjacent rho 0.9209); seed 42 is intermediate; seed 123 is less stable (all-pair rho 0.7134; adjacent rho 0.7480).

The top-three layer set is less stable than the full ranking: mean adjacent top-3 Jaccard ranges from 0.2918 to 0.6143 across seeds. This suggests that broad relative ordering can persist while the identity of the highest-ranked few layers changes from one update to the next. The global gradient magnitude also varies across the capture window; its coefficient of variation ranges from 0.3999 to 0.7358.

For additional context, the minimum pairwise step-rank correlation is negative for seeds 42 and 123 (-0.6652 and -0.4426 respectively), while seed 999's minimum is positive (0.5313). Thus, the average correlations should not be interpreted as every pair of optimizer steps having similar rankings.

A bounded conclusion is that final-window gradient rankings contain a persistent layer-level signal, but temporal reliability is seed-dependent and the top-ranked layers are not uniformly stable. The arithmetic-mean profile compresses this temporal variability into one allocation input.

## Instantaneous threshold saturation diagnostic

For each captured optimizer step, the analyzer recomputed the GARS threshold profile using the seed's fixed calibrated `tau_base`, beta 0.5 and clipping bounds [0.80, 0.99]. This is a diagnostic of what the threshold mapping would produce from each instantaneous gradient profile. It does **not** change the original experiment's allocation, which uses the arithmetic-mean gradient profile over the final capture window.

Each seed contributes 50 × 24 = 1,200 step/layer threshold values.

| Seed | Fixed calibrated tau_base | Lower-bound hits (0.80) | Upper-bound hits (0.99) | Total values clipped |
|---:|---:|---:|---:|---:|
| 42 | 1.156494 | 272 / 1,200 (22.67%) | 701 / 1,200 (58.42%) | 81.08% |
| 123 | 1.098069 | 232 / 1,200 (19.33%) | 692 / 1,200 (57.67%) | 77.00% |
| 999 | 1.208237 | 397 / 1,200 (33.08%) | 668 / 1,200 (55.67%) | 88.75% |
| **All seeds** | — | **901 / 3,600 (25.03%)** | **2,061 / 3,600 (57.25%)** | **82.28%** |

Thus, approximately 82.3% of the diagnostic instantaneous thresholds land on one of the two clipping boundaries. The upper bound is hit more frequently than the lower bound for every seed. This indicates substantial saturation of the bounded threshold mapping when applied to individual step profiles with the fixed calibrated `tau_base`.

### Interpretation and caution

The clipping result is important for understanding the implemented schedule: most instantaneous layer thresholds are constrained by either the 0.80 floor or the 0.99 ceiling. Consequently, the raw gradient ratio is not translated linearly into a distinct threshold for most step/layer observations.

However, this diagnostic must not be confused with dynamic pruning during training. The actual V1 GARS allocation is computed post hoc from the final-window arithmetic-mean gradient scores and the saved calibration. The stepwise threshold counts do not show that ranks were changing during training, nor do they alone demonstrate that clipping harms validation performance. They do motivate reporting clipping as a structural property of the threshold mapping and considering a separately specified sensitivity experiment only if time permits.

## Mechanism-level conclusion

Taken together, the large-model temporal diagnostics show:
1. The 50-step saved profiles are internally consistent with the stored mean importance profiles.
2. Full layer rankings are reasonably persistent on average, but temporal rank stability varies materially across seeds.
3. The identity of the top-three layers is less stable than the full ranking, especially for seeds 42 and 123.
4. The fixed-`tau_base` instantaneous threshold diagnostic is heavily clipped: 82.28% of values fall on the lower or upper threshold boundary.
5. These observations do not establish that gradient magnitude causally improves rank allocation or validation performance. They characterize reliability and saturation of the signal used by the current policy.

Combined with the three-seed DeBERTa-v3-large pruning study, the evidence supports feasibility of the gradient-conditioned pipeline and approximately 40% physical LoRA A/B factor reduction with unchanged validation accuracy/F1. It does not support a GARS-specific validation-loss advantage on this model scale: uniform local-energy pruning has the lower mean loss.

The analysis is limited to the final 50 optimizer updates of three DeBERTa-v3-large/RTE runs. It does not establish stability across the full training trajectory, other tasks, architectures, or independent test data.
