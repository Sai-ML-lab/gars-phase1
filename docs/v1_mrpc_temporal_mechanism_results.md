# DeBERTa-v3-base / GLUE MRPC — temporal and mechanism diagnostics

## Scope and data checks

The temporal and mechanism analyzers ran on the existing three-seed DeBERTa-v3-base/MRPC artifacts. No retraining was required.

- Base model: `microsoft/deberta-v3-base`
- Pinned revision: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
- Task: GLUE MRPC
- Seeds: 42, 123, 999
- Transformer blocks: 12
- Gradient observations: 50 per seed from the final 50 optimizer updates
- Each stepwise profile's arithmetic mean reproduces the stored block-importance profile (temporal analyzer integrity check passed).
- The pruning allocations use the common matched physical A/B factor budget documented in the MRPC seeded-results report.

The analyses are descriptive. Cross-seed correlations use three seed-pair comparisons, and within-seed correlations are based on 12 Transformer blocks. Captured optimizer updates are sequential/autocorrelated. No p-values, significance claims or causal conclusions are made.

## 1. Within-window gradient temporal reliability

Spearman correlations compare the 12-block gradient-importance rankings from different optimizer updates within the final 50-update capture window.

| Seed | Mean all-pair Spearman rho | Mean adjacent-step Spearman rho |
|---:|---:|---:|
| 42 | 0.920390 | 0.936060 |
| 123 | 0.904540 | 0.905240 |
| 999 | 0.865320 | 0.862850 |
| **Mean across seeds** | **0.896749** | **0.901384** |
| **Sample SD across seeds** | **0.028350** | **0.036758** |

The layer ranking is relatively persistent across the final 50 updates for all three seeds. Seed 42 has the highest within-window correlation and seed 999 the lowest, but even the latter remains around 0.86. The all-pair and adjacent-step averages are close, so the ranking does not exhibit a large average loss of concordance between adjacent updates in this window.

This is stronger average within-window rank stability than the DeBERTa-v3-large/RTE temporal analysis (mean all-pair rho 0.8244 and adjacent-step rho 0.8383). This comparison is descriptive across two different model/task settings; it does not isolate task as the sole reason for the difference.

## 2. Instantaneous threshold clipping

The temporal analyzer recomputed a threshold for each layer at each captured optimizer step, using the seed-specific fixed calibrated GARS `tau_base`, beta 0.5 and clipping interval [0.80, 0.99]. Each seed contributes 50 × 12 = 600 step/layer observations.

| Seed | Lower-bound hits (0.80) | Upper-bound hits (0.99) | Total clipped |
|---:|---:|---:|---:|
| 42 | 5 / 600 (0.83%) | 525 / 600 (87.50%) | 88.33% |
| 123 | 18 / 600 (3.00%) | 491 / 600 (81.83%) | 84.83% |
| 999 | 7 / 600 (1.17%) | 511 / 600 (85.17%) | 86.33% |
| **All seeds** | **30 / 1,800 (1.67%)** | **1,527 / 1,800 (84.83%)** | **86.50%** |

The instantaneous diagnostic is strongly dominated by upper-bound clipping: 84.83% of all step/layer values hit 0.99, while 1.67% hit 0.80. This indicates that, with the fixed calibrated `tau_base` values, most instantaneous gradient profiles map to the maximum allowed energy-retention threshold.

A separate cross-seed mechanism diagnostic reports the threshold profile computed from the **mean** gradient importance profile used by the actual V1 allocation. It finds 10, 9 and 10 of the 12 layer thresholds at the upper bound for seeds 42, 123 and 999 respectively (29/36 = 80.56%); no mean-profile layer threshold hits the lower bound in these runs. These mean-profile and instantaneous-profile counts have different denominators and must not be conflated.

This is a property of the bounded threshold mapping. It does not mean that V1 dynamically prunes or changes ranks during training: the implemented allocation is post-hoc and uses the arithmetic-mean gradient profile.

## 3. Cross-seed gradient and spectral-profile reliability

The mechanism analyzer compares per-layer/per-module profiles between the three training seeds. The values below are the mean and sample SD across the three seed-pair Spearman correlations.

| Profile | Mean pairwise Spearman rho | Sample SD |
|---|---:|---:|
| Raw gradient-importance profile (12 layers) | 0.946387 | 0.010682 |
| Module squared-spectral-entropy profile (24 modules) | 0.861449 | 0.048449 |
| Layer-mean squared-spectral-entropy profile (12 layer means) | 0.762238 | 0.122926 |

The relative gradient-importance ordering is highly concordant across seeds. Module-level spectral entropy is also reasonably concordant; layer-mean entropy is less stable across seed pairs.

## 4. Cross-seed rank-allocation stability

| Allocation profile | Mean module-rank rho | Sample SD | Mean layer-mean-rank rho | Sample SD |
|---|---:|---:|---:|---:|
| GARS | 0.862844 | 0.072860 | 0.798902 | 0.066120 |
| Global singular-value | 0.929285 | 0.024739 | 0.858691 | 0.085680 |
| Uniform local energy | 0.937712 | 0.035853 | 0.862348 | 0.075199 |

The integer rank allocation is less stable across seeds for GARS than for either spectrum-only baseline, both at module level and after averaging ranks within layers. This occurs despite the high cross-seed concordance of the raw gradient-importance profile. Thus, stable gradient ranking does not automatically yield equally stable GARS allocation: the layer- and module-specific spectra, clipping, rank floor and matched-budget calibration also shape the final discrete ranks.

## 5. Within-seed associations and layer-depth adjustment

Mean Spearman associations averaged over the three seeds:

| Association | Raw mean rho | Depth-adjusted partial mean rho |
|---|---:|---:|
| Gradient importance vs layer-mean spectral entropy | -0.850816 | -0.139095 |
| Gradient importance vs GARS layer-mean rank | -0.624843 | +0.290483 |
| Gradient importance vs GARS-minus-uniform layer-mean rank | +0.619026 | +0.566311 |
| Layer-mean spectral entropy vs GARS layer-mean rank | +0.810386 | +0.263133 |

Sample SDs of the depth-adjusted partial correlations across the three seeds are 0.355008, 0.159238, 0.101056 and 0.359050 respectively. The gradient/entropy raw association weakens substantially after controlling for layer depth, and the adjusted values vary in sign across seeds. A strong raw association should therefore not be interpreted as an independent gradient/entropy relationship.

The gradient association with GARS's relative layer-rank uplift over uniform local-energy pruning remains positive after depth adjustment (mean partial rho +0.5663; sample SD 0.1011). This is a descriptive relationship across 12 layers per seed, not evidence that gradient conditioning causes better predictive performance. The module-level entropy association with GARS-minus-uniform rank difference has raw mean rho -0.4891 (sample SD 0.0342).

Layer depth itself is strongly associated with several profiles in this MRPC pilot: gradient-importance depth rho is +0.9161, +0.9231 and +0.9161 for seeds 42, 123 and 999; layer-mean entropy depth rho is -0.9371, -0.7063 and -0.9860. These trends reinforce the need to distinguish raw correlations from depth-adjusted descriptive checks.

## 6. Mechanism-level interpretation

The MRPC diagnostics support four bounded observations:

1. **Temporal persistence:** final-window gradient-importance rankings are relatively stable on average (all-pair rho 0.8967; adjacent rho 0.9014) across these three runs.
2. **Cross-seed consistency:** the mean gradient profile ranking is highly concordant across seeds (rho 0.9464), and module spectral-entropy profiles show moderate-to-high concordance (rho 0.8614).
3. **Allocation stability:** GARS module rank allocation has lower cross-seed stability (rho 0.8628) than uniform local energy (0.9377) and global singular-value cutoff (0.9293).
4. **Threshold saturation:** most instantaneous thresholds hit the upper clipping boundary (84.83%); the mean-profile thresholds used by the actual V1 policy also hit the upper boundary in 29 of 36 layer-seed observations (80.56%).

These observations characterize the signal and policy behavior; they do not prove a causal relationship between gradients and predictive performance. The MRPC validation comparison itself showed very small, mixed loss differences among pruning policies, with no established GARS-specific advantage.

## Limitations

- Three training seeds and one 408-example validation split.
- Within-seed association analyses use 12 Transformer blocks; module-level profile analyses use 24 adapted projection modules.
- Optimizer steps are temporally dependent and are not independent statistical samples.
- Rank correlations and partial correlations are descriptive, not inferential tests.
- Instantaneous threshold clipping is recomputed for diagnosis; no dynamic in-training pruning was performed.
- The results do not establish independent test-set generalization or causal benefit from gradient conditioning.

## Defensible conclusion

On DeBERTa-v3-base/MRPC, the final-window layer-wise gradient signal is relatively persistent within each seed and highly concordant in its mean ordering across seeds. However, GARS's final discrete rank allocation is less stable across seeds than the spectrum-only baselines, and its bounded threshold schedule saturates at the upper clipping boundary for most instantaneous observations. The positive association between gradient importance and GARS's relative rank uplift over uniform pruning remains after descriptive layer-depth adjustment, but it does not establish causality or a validation-performance advantage.

The MRPC mechanism findings complement the large-model/RTE diagnostics. They suggest that temporal and allocation behavior depends on model/task setting and that threshold saturation is a recurring property of the frozen V1 mapping. The results motivate explicit discussion of clipping and allocation stability in the report rather than further validation-set tuning.
