# DeBERTa-v3-base / GLUE MRPC — three-seed results

## Scope and provenance

The second-task replication completed for seeds 42, 123 and 999.

- Base model: `microsoft/deberta-v3-base`
- Pinned revision: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` (all three seeds)
- Dataset: GLUE MRPC validation split, 408 examples per seed
- Parent LoRA rank/alpha: 32/64
- Seeds: 42, 123 and 999
- Parent physical LoRA A/B factor parameters: 1,179,648
- Target parameter fraction: 0.60
- Achieved pruned A/B factor parameters: 708,096 for every method and seed
- Reduction: 39.9739583% for every method and seed
- All nine method/seed reload validations passed.

The physical factor count applies only to LoRA A/B factors, not the full base model. All policies and seeds use the same matched budget.

## Three-seed validation results

Values are mean ± sample standard deviation across seeds 42, 123 and 999.

| Method | Accuracy | Macro-F1 | Validation loss |
|---|---:|---:|---:|
| Parent | 0.897059 ± 0.004245 | 0.880135 ± 0.003305 | 0.267556 ± 0.011071 |
| GARS | 0.897876 ± 0.002830 | 0.880991 ± 0.001823 | 0.267861 ± 0.011082 |
| Global singular-value | 0.897059 ± 0.004245 | 0.880135 ± 0.003305 | 0.267846 ± 0.010832 |
| Uniform local energy | 0.897876 ± 0.002830 | 0.880991 ± 0.001823 | 0.267710 ± 0.010849 |

GARS and uniform local-energy pruning have identical accuracy and macro-F1 at the reported precision for every seed. Both have a small aggregate improvement over parent (+0.000817 accuracy and +0.000855 macro-F1), which comes entirely from seed 42: that seed gains one validation example in accuracy and a corresponding macro-F1 increase; seeds 123 and 999 have no accuracy/F1 change versus parent. Global singular-value pruning matches parent accuracy/F1 in all three seeds.

## Paired changes versus the unpruned parent

Positive accuracy/F1 differences are better; positive loss differences are worse.

| Method | Accuracy change | Macro-F1 change | Loss change |
|---|---:|---:|---:|
| GARS | +0.000817 ± 0.001415 | +0.000855 ± 0.001482 | +0.000305 ± 0.000518 |
| Global singular-value | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | +0.000290 ± 0.000340 |
| Uniform local energy | +0.000817 ± 0.001415 | +0.000855 ± 0.001482 | +0.000154 ± 0.000330 |

All three methods have slightly higher mean loss than the parent. GARS and global are nearly tied on mean loss relative to each other; uniform local energy has the lowest mean loss among the three pruned methods.

## Paired GARS-versus-baseline differences

Positive accuracy/F1 favors GARS. For loss, positive values mean GARS has higher (worse) loss.

| Comparison | Accuracy difference | Macro-F1 difference | Loss difference |
|---|---:|---:|---:|
| GARS − global singular-value | +0.000817 ± 0.001415 | +0.000855 ± 0.001482 | +0.000015 ± 0.000369 |
| GARS − uniform local energy | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | +0.000151 ± 0.000355 |

The GARS-minus-global loss difference is mixed by seed: -0.000283 (GARS lower loss) for seed 123, -0.000100 for seed 42, and +0.000428 for seed 999. GARS and uniform local-energy loss differences are also mixed: -0.000127 for seed 123, +0.000029 for seed 42, and +0.000551 for seed 999.

These differences are very small on the 408-example validation split and do not establish statistically significant superiority. No significance tests are claimed.

## Task-transfer interpretation

The MRPC replication supports the following bounded findings:
1. The approximately 39.974% physical LoRA A/B factor reduction achieved on RTE also holds on MRPC at the same matched budget.
2. On MRPC, all three pruning methods preserve or slightly improve parent validation accuracy/F1 at reported precision; the small aggregate gains for GARS/uniform are driven by seed 42 alone.
3. GARS does not show a clear loss advantage over the spectrum-only baselines on MRPC. Its mean loss is nearly identical to global singular-value pruning and slightly higher than uniform local-energy pruning.
4. This aligns with the large-model/RTE outcome: a GARS-specific loss advantage is not stable across task/model conditions. In the base-model/RTE experiment GARS had lower mean loss than uniform local energy and a small lower mean loss than global singular-value pruning, but that ordering did not reproduce on large-model/RTE or base-model/MRPC.

The results support task transfer of matched-budget physical factor reduction and preservation of classification metrics in the tested experiments. They do not establish that gradient-conditioned allocation is generally more accurate or has lower validation loss than spectrum-only allocation.

## Defensible Phase 1 MRPC conclusion

On DeBERTa-v3-base / GLUE MRPC, all three post-hoc pruning policies reduced physical LoRA A/B factor parameters by 39.974% under an identical budget. All nine exports passed reload validation. GARS preserved parent accuracy and macro-F1 at reported precision for seeds 123 and 999 and improved both slightly for seed 42; uniform local-energy pruning produced the same classification metrics. Mean validation losses were close, with uniform local energy slightly lower than GARS and global singular-value pruning nearly tied with GARS. This supports cross-task feasibility of the compression pipeline but does not establish a unique predictive advantage for GARS.

The study is limited to three training seeds and one validation split of 408 MRPC examples. These validation results do not establish independent test-set generalization. Across-seed sample standard deviations are descriptive and not confidence intervals or significance tests.
