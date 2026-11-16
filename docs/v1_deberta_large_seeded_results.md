# DeBERTa-v3-large / GLUE RTE — three-seed results

## Scope and provenance

The DeBERTa-v3-large / RTE follow-up completed for seeds 42, 123 and 999.

- Base model: `microsoft/deberta-v3-large`
- Pinned revision: `8869586e1c6f5654a9a2a5c517fe710e24a488d0` (all three seeds)
- Task: GLUE RTE validation split, 277 examples per seed
- Transformer depth: 24 layers
- Adapted projections: query and value in each layer (48 LoRA modules)
- Parent LoRA rank/alpha: 32/64
- Training: 3 epochs; effective batch size 16 (per-device batch 4 × gradient accumulation 4)
- Seeds: 42, 123 and 999
- Parent physical A/B factor parameters: 3,145,728
- Target parameter fraction: 0.60
- Achieved pruned A/B factor parameters: 1,888,256 for every method and seed
- Reduction: 39.9739583% for every method and seed
- All nine method/seed reload validations passed.

The achieved count is 819 parameters above the nominal integer target of 1,887,437, approximately 0.0434% of that target. The actual factor reduction is 39.974%, closely matching the intended 40%. These counts apply only to LoRA A/B factors, not the full model.

## Three-seed validation results

Values are mean ± sample standard deviation across seeds 42, 123 and 999.

| Method | Accuracy | Macro-F1 | Validation loss |
|---|---:|---:|---:|
| Parent | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.371230 ± 0.046347 |
| GARS | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.371468 ± 0.046416 |
| Global singular-value | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.371283 ± 0.046370 |
| Uniform local energy | 0.867629 ± 0.036341 | 0.866789 ± 0.037078 | 0.370233 ± 0.045922 |

All methods reproduce parent accuracy and macro-F1 at the reported precision for every seed. Paired accuracy and macro-F1 changes versus parent are exactly zero in the supplied aggregate.

## Paired loss differences

Positive values mean the first method has higher (worse) loss; negative values mean lower (better) loss.

| Paired comparison | Mean loss difference | Sample SD |
|---|---:|---:|
| GARS − parent | +0.000238108 | 0.000136953 |
| Global singular-value − parent | +0.000053034 | 0.000105655 |
| Uniform local energy − parent | -0.000997075 | 0.001008156 |
| GARS − global singular-value | +0.000185075 | 0.000049088 |
| GARS − uniform local energy | +0.001235184 | 0.000937956 |

GARS has higher validation loss than global singular-value pruning and uniform local-energy pruning in all three paired seeds. The per-seed GARS-minus-global loss differences are +0.000136269 (seed 123), +0.000184515 (seed 42) and +0.000234440 (seed 999). The per-seed GARS-minus-uniform differences are +0.001532658, +0.000184563 and +0.001988330 for seeds 123, 42 and 999 respectively.

Uniform local-energy pruning has the lowest mean validation loss. It improves on parent loss in seeds 123 and 999 and is slightly worse in seed 42. The global method has a small positive mean loss delta versus parent, with mixed seed-level direction. These are descriptive validation results over three seeds, not significance tests.

## Cross-scale interpretation

The DeBERTa-v3-base/RTE study also achieved approximately 39.974% LoRA A/B factor reduction while preserving parent-level accuracy and macro-F1 at reported precision. On that base-scale experiment, GARS had a lower mean validation loss than uniform local-energy pruning (paired GARS-minus-uniform loss difference approximately -0.003004) and a small lower mean loss than global singular-value pruning (approximately -0.000433).

On DeBERTa-v3-large/RTE, the direction reverses: GARS has higher mean validation loss than both spectrum-only baselines, especially uniform local-energy pruning (paired difference +0.001235). Therefore:
- the approximately 40% physical A/B factor reduction with unchanged validation accuracy/F1 transfers to this second model scale in the tested setup;
- the validation-loss advantage of GARS over spectrum-only baselines observed on DeBERTa-v3-base does **not** transfer to DeBERTa-v3-large;
- these results do not demonstrate a general predictive or validation-loss advantage unique to gradient-conditioned allocation.

All four methods obtain identical classification accuracy and macro-F1 at the reported precision in the large-model study. The observed loss ordering should therefore be reported as a small, descriptive difference and not inflated into a broad claim.

## Defensible Phase 1 conclusion

Across DeBERTa-v3-base and DeBERTa-v3-large on GLUE RTE, post-hoc physical LoRA rank pruning achieved approximately 40% reduction in A/B factor parameters while preserving parent accuracy and macro-F1 at the reported precision. The large-model three-seed result supports transfer of the compression-and-classification-metric outcome to a 24-layer model. However, GARS did not outperform the spectrum-only baselines on large-model validation loss; uniform local-energy pruning had the lowest mean loss. The base-model GARS loss advantage over the spectrum-only baselines is therefore model-scale dependent in this pilot.

The current evidence supports feasibility and matched-budget compression with unchanged classification metrics, but it does not establish a consistent loss advantage or causal benefit of gradient conditioning. Results remain limited to RTE validation, two DeBERTa model scales, three seeds, and the tested training/pruning setup. Do not claim test-set generalization.

## Remaining mechanism question

The current parent trainer writes the final-window per-optimizer-step gradient profiles. The large-model parent runs were trained with this instrumented trainer, but the actual saved `gradient_importance.json` files must be checked locally for `stepwise_importance_by_optimizer_step` before temporal stability is claimed. If present, the existing temporal analyzer can be run with the large-model config and these parent/pruned roots, without retraining. If absent, retain the temporal limitation rather than inferring stability from the per-layer means.
