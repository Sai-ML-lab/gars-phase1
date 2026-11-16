# DeBERTa-v3-large / GLUE RTE — seed 42 feasibility result

**Status:** seed-42 parent/pruning pipeline completed and passed the seed-level feasibility gate. This is a single-seed pilot, not a three-seed result.

## Configuration and provenance

- Base model: `microsoft/deberta-v3-large`
- Pinned revision: `8869586e1c6f5654a9a2a5c517fe710e24a488d0`
- Task: GLUE RTE validation split (277 examples)
- Transformer depth: 24 layers
- Adapted projections: query and value in every layer (48 LoRA modules)
- Seed: 42
- Parent A/B factor parameters: 3,145,728
- Target parameter fraction: 0.60
- Nominal integer target: 1,887,437
- Achieved pruned A/B factor parameters: 1,888,256
- Actual parameter reduction: 39.9739583%
- All three policy exports report `reload_validation: passed`.

The achieved count exceeds the nominal integer target by 819 parameters, approximately 0.0434% of the target. The actual reduction remains close to the intended 40%. The count refers only to LoRA A/B factor parameters, not the full model.

## Parent and pruned validation metrics

| Method | Accuracy | Macro-F1 | Validation loss | Loss change vs parent |
|---|---:|---:|---:|---:|
| Parent | 0.86281588 | 0.86241242 | 0.35432420 | Reference |
| GARS | 0.86281588 | 0.86241242 | 0.35466682 | +0.00034262 |
| Global singular-value | 0.86281588 | 0.86241242 | 0.35448230 | +0.00015810 |
| Uniform local energy | 0.86281588 | 0.86241242 | 0.35448225 | +0.00015805 |

All methods reproduce parent accuracy and macro-F1 at the reported precision. Validation loss is slightly higher for all pruned methods. For this seed, GARS has a slightly higher (worse) validation loss than either spectrum-only baseline: approximately +0.0001845 versus global singular-value and +0.0001846 versus uniform local energy. These are small descriptive differences on one validation split; they do not establish a general ranking of methods.

## Gradient assignment and artifact checks

The metadata reports 24 block importance scores. The assignment is the observed gradient-to-layer alignment (`cyclic_shift: 0`), with each destination layer mapped to its same-numbered source layer. This is the intended unpermuted GARS seed-42 run.

All three policies have 1,888,256 physical A/B parameters and successful reload validation. The export note confirms post-hoc SVD truncation of the same parent adapter and checks that reloaded physical ranks and LoRA scaling preserve the original parent multiplier.

## Interpretation

This first large-model result is encouraging for **feasibility and parameter reduction**: the 24-layer model completed the parent/pruning path and reached approximately 40% A/B-factor reduction while preserving validation accuracy and macro-F1 at reported precision.

It does not yet show that GARS is better than the spectrum-only baselines. On this seed all three methods have identical accuracy/F1, and GARS has a marginally higher validation loss than both other pruned methods.

The cross-scale claim must remain provisional until seeds 123 and 999 are completed and the three-seed aggregate is reviewed. Do not report this one-seed result as a multi-seed finding or as proof of generalization.

## Next actions

Run seed 123 and seed 999 through parent training, parent reload validation and all three pruning exports using the same config and separate per-seed directories. Then aggregate all three seeds with `scripts/aggregate_pruning_results.py`. Compare parent-relative accuracy/F1/loss changes, physical A/B counts, reduction percentage and reload status across all seeds.
