# GARS V1 preliminary results — DeBERTa-v3-base / GLUE RTE

## Scope and protocol

This document records the three-seed Phase 1 pilot comparing post-hoc LoRA rank-allocation policies on the same trained parent adapter for each seed.

- Base model: `microsoft/deberta-v3-base`
- Pinned revision used by the pruning reloads: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
- Task: GLUE RTE; official validation split, 277 examples
- Seeds: 42, 123, 999
- Parent LoRA rank: 32; target modules: query/value projections in 12 Transformer blocks (24 LoRA modules)
- Compared policies: GARS, uniform local squared-energy, global singular-value cutoff
- Target: 60% of parent LoRA A/B factor parameters (40% reduction)
- Each method's physical factor count: 708,096 versus 1,179,648 parent A/B parameters
- Achieved reduction: 39.9739583% (same for every method and seed)
- All nine policy/seed reload validations are recorded as passed.

The reduction percentage applies to LoRA A/B factor parameters, not to all parameters in the base model or the complete end-to-end system.

## Three-seed validation metrics

Values are mean ± sample standard deviation across the three training seeds.

| Method | Accuracy | Macro-F1 | Validation loss |
|---|---:|---:|---:|
| Unpruned parent | 0.831528 ± 0.012678 | 0.829541 ± 0.013702 | 0.444437 ± 0.009443 |
| GARS | 0.831528 ± 0.012678 | 0.829541 ± 0.013702 | 0.445440 ± 0.010439 |
| Global singular-value | 0.830325 ± 0.014440 | 0.828254 ± 0.015627 | 0.445873 ± 0.011321 |
| Uniform local energy | 0.831528 ± 0.012678 | 0.829349 ± 0.013778 | 0.448445 ± 0.012842 |

## Paired changes versus the unpruned parent

Changes are computed within each seed first, then averaged across seeds. Negative loss change is better; positive accuracy/F1 change is better.

| Method | Accuracy change | Macro-F1 change | Loss change |
|---|---:|---:|---:|
| GARS | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | +0.001003 ± 0.001003 |
| Global singular-value | -0.001203 ± 0.002084 | -0.001287 ± 0.002230 | +0.001436 ± 0.001905 |
| Uniform local energy | 0.000000 ± 0.000000 | -0.000192 ± 0.000186 | +0.004007 ± 0.003792 |

## Paired GARS-minus-baseline differences

These are paired within seed. For accuracy and macro-F1, positive values favor GARS. For loss, negative values favor GARS.

| Comparison | Accuracy difference | Macro-F1 difference | Loss difference |
|---|---:|---:|---:|
| GARS − global singular-value | +0.001203 ± 0.002084 | +0.001287 ± 0.002230 | -0.000433 ± 0.000994 |
| GARS − uniform local energy | 0.000000 ± 0.000000 | +0.000192 ± 0.000186 | -0.003004 ± 0.003062 |

### Paired loss differences by seed

| Seed | GARS − global loss | GARS − uniform loss |
|---|---:|---:|
| 42 | -0.001520 | -0.006531 |
| 123 | +0.000430 | -0.001018 |
| 999 | -0.000208 | -0.001464 |

## Interpretation

1. **Physical budget:** all nine exports used exactly the same physical A/B factor count, 708,096, corresponding to a 39.974% reduction. The budget is closely matched to the 40% target.
2. **Classification metrics:** GARS reproduces the parent accuracy and macro-F1 at full reported precision for each of the three seeds. Uniform local energy also preserves accuracy at these metric resolutions, while its mean macro-F1 is slightly lower. Global singular-value pruning has a small negative mean change against the parent.
3. **Against uniform local energy:** GARS has lower validation loss on all three seeds. The mean paired loss difference is -0.0030045.
4. **Against global singular-value pruning:** GARS has a small mean advantage in accuracy and macro-F1, and a mean loss difference of -0.0004328. However, the paired loss difference is mixed: GARS has lower loss on seeds 42 and 999, but higher loss on seed 123. The result does not establish consistent superiority over the global baseline.
5. **Against the parent:** GARS retains accuracy and macro-F1 at the reported precision, but mean validation loss is 0.001003 higher. It therefore does not preserve every aspect of the parent metric profile.

## Limitations and claims boundary

- This is a pilot on one base-model scale and one task, with three seeds and 277 validation examples per seed.
- Three seeds provide only a small estimate of across-seed variability. The reported sample standard deviations are descriptive; no significance claim is made.
- The aggregate comparison uses validation metrics. It does not establish test-set generalization.
- No inference latency, throughput, memory, or end-to-end system-size measurements have been made.
- The parameter reduction is specific to LoRA A/B factors, not the complete model.
- The original parent-run metadata did not record an immutable pretrained revision. The two cached snapshots were later found to contain exactly matching model tensors and byte-identical config, and the pruning reloads now use the pinned `8ccc9...` revision. This resolves the content-equivalence check but does not retroactively create a per-run revision attestation for historical parent training.
- The observed advantage over uniform local-energy pruning is consistent on validation loss in this pilot. The advantage over global singular-value pruning is small and mixed. No causal mechanism or broad generality is established.

## Defensible Phase 1 conclusion

In the DeBERTa-v3-base / GLUE RTE three-seed pilot, GARS achieved a 39.974% reduction in physical LoRA A/B factor parameters while retaining parent-level validation accuracy and macro-F1 at the reported precision. It consistently reduced validation loss relative to uniform local squared-energy pruning across the three seeds. Its advantage over a global singular-value cutoff was smaller and not consistent on every seed. These results are preliminary and motivate further tests of gradient-signal reliability and spectral concentration across tasks and model scales.


## Mechanism diagnostics from the three-seed pilot

The saved mechanism analysis compares the 12 block-level final-window gradient scores across seeds and relates these scores to spectral entropy and allocation. Correlations are descriptive and are not significance tests.

### Cross-seed reliability

| Diagnostic | Pairwise mean Spearman rho | Pairwise sample SD | Units |
|---|---:|---:|---:|
| Gradient-importance ranking | 0.9580 | 0.0140 | 12 Transformer blocks |
| Mean squared-spectral-entropy ranking | 0.9324 | 0.0265 | 12 layer means |
| Module squared-spectral-entropy ranking | 0.8330 | 0.0370 | 24 projection modules |
| GARS retained-rank ranking | 0.8288 | 0.1090 | 24 projection modules |
| Uniform retained-rank ranking | 0.9320 | 0.0127 | 24 projection modules |
| Global retained-rank ranking | 0.9525 | 0.0026 | 24 projection modules |

Gradient layer rankings are highly concordant across the three seed pairs. In contrast, GARS module-level rank allocation is less stable across seeds than the two spectrum-only allocations. Stable ordering of the gradient signal therefore does not automatically imply equally stable integer rank assignments; spectral profile differences and discrete budget calibration also influence the final allocation.

### Within-seed associations and layer-depth adjustment

| Association | Raw mean rho | Partial Spearman controlling for layer depth |
|---|---:|---:|
| Gradient importance vs layer-mean spectral entropy | -0.9394 | -0.5958 |
| Gradient importance vs GARS layer-mean rank | +0.0233 | +0.2090 |
| Gradient importance vs GARS-minus-uniform layer rank | +0.7176 | +0.3622 |
| Spectral entropy vs GARS layer-mean rank | +0.0375 | -0.0226 |

The gradient/entropy association weakens after controlling for depth, but remains moderately negative in this descriptive analysis. The association between gradient importance and GARS's relative rank uplift over uniform pruning also weakens, from +0.7176 to +0.3622. The entropy/GARS-rank relationship becomes essentially zero after depth adjustment. This cautions against interpreting the raw spectral-entropy/rank relationship as a direct allocation mechanism.

The partial Spearman calculation ranks each variable and layer index, linearly residualizes the ranked variables against ranked layer depth, and correlates the residuals. It is a descriptive sensitivity check with 12 blocks per seed, not causal adjustment and not a significance test. The three-seed average of partial correlations is itself descriptive.

### Interpretation

The current evidence is consistent with:
- a reproducible relative gradient-importance ordering across seeds;
- a relationship between gradient magnitude and spectral concentration that is not fully explained by monotonic layer depth in this pilot;
- some association between gradient importance and the additional rank assigned by GARS over uniform local-energy pruning, though this association is attenuated by depth;
- a less stable module-level GARS integer-rank allocation across seeds than the spectrum-only baselines.

It does **not** yet establish that gradient magnitude is the causal reason for an accuracy or loss advantage. The GARS allocation depends jointly on the gradient-conditioned threshold, each module's spectral energy curve, rank floor, and global matched-budget calibration.

### Remaining mechanism limitation

The current parent runs store only the arithmetic mean of each layer's gradient norm over the final 50 optimizer updates. They do not store the 50 individual step-level values. Within-window gradient variance, autocorrelation, and ranking stability over training time cannot be estimated from these artifacts. A future instrumented run must persist each update's per-layer score before those temporal questions can be answered.
