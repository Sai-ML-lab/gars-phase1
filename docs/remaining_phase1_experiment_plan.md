# Remaining Phase 1 experiment plan

**Planning date:** 16 November 2026  
**Submission deadline:** 3 December 2026, 5:30 PM IST  
**Current completed evidence:** three-seed DeBERTa-v3-base/RTE parent and matched-budget pruning study; gradient/spectral diagnostics; beta sensitivity; 33-export cyclic alignment control.

## Research gaps that remain

The current empirical study covers one model scale and one task. It demonstrates approximately 40% reduction in LoRA A/B factor parameters on the DeBERTa-v3-base/RTE pilot, but it does not yet establish the proposal's multi-scale objective. No inference latency or serving-throughput result has been measured. The existing mechanism work also lacks a temporal reliability result unless the individual final-window optimizer-step gradients exist in the local parent artifacts.

## Priority order

### Priority 1 — temporal gradient-signal reliability (no retraining if saved stepwise fields exist)

The current parent trainer writes the field stepwise_importance_by_optimizer_step in gradient_importance.json. The temporal analyzer checks that the arithmetic mean of all stored steps exactly reproduces the existing V1 block-importance profile, then measures:
- all-pairs and lagged rank stability over the final 50 optimizer updates;
- adjacent-step top-1/top-3 stability;
- per-layer coefficient of variation and normalized-importance drift;
- the per-step threshold dynamics obtained using the fixed V1 calibrated tau_base, beta and clipping limits.

Run:

\`\`\`bash
python scripts/analyze_gars_gradient_temporal_stability.py
\`\`\`

If historical parent artifacts do not contain the stepwise field, the script stops with a clear error. In that case, do not silently infer temporal stability from the saved means. Decide whether to rerun one or all parent seeds with the current trainer; this can be a targeted follow-up rather than an automatic rerun.

### Priority 2 — merged/unmerged inference latency and throughput

Compare the same four adapter types on each seed:
- unpruned parent;
- GARS;
- uniform local squared-energy;
- global singular-value cutoff.

For both unmerged PEFT and merged weights, use batch sizes 1, 8 and 32, with three representative sequence-length batches from the RTE validation split. Timed forwards are synchronized on MPS. Tokenization, collation, host-to-device transfer, model loading and merge time are excluded from the forward-latency statistic and separately identified where relevant. Merged/unmerged logits are checked for equivalence before timing.

Smoke test first:

\`\`\`bash
python scripts/benchmark_inference_latency.py \\
  --device mps --seeds 42 --warmup-steps 3 --repeats 10 \\
  --batches-per-size 2 --output-root results/v1_latency_smoke
\`\`\`

Then run the full three-seed benchmark with defaults:

\`\`\`bash
python scripts/benchmark_inference_latency.py --device mps
\`\`\`

The result is a hardware-specific forward benchmark, not a production-serving claim. Report batch-1 latency and batch-8/32 throughput separately. Do not assume a 40% adapter-factor reduction yields a proportional latency reduction.

### Priority 3 — second model scale: DeBERTa-v3-large on RTE

Use the same RTE task, target projections, LoRA rank/alpha, beta, squared-energy criterion and physical-budget objective to isolate model-scale transfer. The added config uses:
- microsoft/deberta-v3-large, pinned to revision 8869586e1c6f5654a9a2a5c517fe710e24a488d0;
- 24 Transformer layers and hidden size 1024;
- rank 32, alpha 64, query/value projections;
- batch size 4 × accumulation 4, preserving the same effective batch size of 16 as the base experiment;
- 3 epochs, 50 final optimizer-step gradient captures, and the 60% physical A/B factor target.

Start with seed 42 only:

\`\`\`bash
python - <<'PY'
from transformers import AutoConfig

revision = "8869586e1c6f5654a9a2a5c517fe710e24a488d0"
config = AutoConfig.from_pretrained("microsoft/deberta-v3-large", revision=revision)
print("hidden_size:", config.hidden_size)
print("num_hidden_layers:", config.num_hidden_layers)
assert config.hidden_size == 1024
assert config.num_hidden_layers == 24
PY

python experiments/train_parent_rte.py \\
  --config configs/v1_deberta_v3_large_rte.json \\
  --seed 42 --device mps \\
  --output-root results/v1_parent_deberta_large_rte
\`\`\`

Validate the parent before pruning:

\`\`\`bash
python scripts/validate_parent_adapter.py \\
  --config configs/v1_deberta_v3_large_rte.json \\
  --seed-dir results/v1_parent_deberta_large_rte/seed_42 \\
  --device mps

python scripts/run_pruning_comparison.py \\
  --config configs/v1_deberta_v3_large_rte.json \\
  --seed 42 --device mps \\
  --parent-root results/v1_parent_deberta_large_rte \\
  --output-root results/v1_pruned_deberta_large_rte
\`\`\`

Only proceed to seeds 123 and 999 after seed 42 finishes training, reload validation and all three physical exports successfully. Then aggregate:

\`\`\`bash
python scripts/aggregate_pruning_results.py \\
  --results-root results/v1_pruned_deberta_large_rte \\
  --seeds 42 123 999 \\
  --output results/v1_pruned_deberta_large_rte/aggregate.json
\`\`\`

This second-scale experiment is the most important additional scientific run because the existing evidence currently covers only DeBERTa-v3-base. It uses the same task to reduce confounding from task changes.

### Priority 4 — second task (only after scale and latency experiments)

If compute and time remain, repeat the base-scale parent/pruning study on GLUE MRPC using the same sentence-pair training pipeline, same three seeds and same matched physical budget. This tests whether the result is task-specific. It is useful but secondary to demonstrating a second model scale and measuring the proposal's latency objective.

### Not recommended before submission

Do not prioritize dynamic in-training pruning, optimizer-state surgery, a large random-permutation search, more beta tuning, or more seeds on the existing base RTE setup before completing the above. They expand scope without addressing the two largest current evidence gaps: multi-scale transfer and measured inference performance.

## Decision gates and reporting discipline

- Keep binaries, checkpoints and generated benchmark data local and ignored by Git. Commit source code, configs and reports only.
- Do not overwrite existing result directories. Use separate output roots for smoke tests and full runs.
- Each model-scale result must show its parent baseline, all three policies, paired seed metrics, physical A/B parameter counts, reload status and the exact pinned base revision.
- A target is not an achievement: report the actual accuracy drop and actual parameter reduction. If a scale misses the ≤1 percentage-point accuracy target, state that plainly.
- Latency is a separate measurement from factor-parameter reduction. Report observed values, not inferred speedups.
- Keep all claims limited to the models, tasks, seeds, devices and splits actually measured.
