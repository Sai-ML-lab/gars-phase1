# Phase 1 experiment roadmap — 15 November 2026

## Current evidence and remaining gaps

Completed on DeBERTa-v3-base / GLUE RTE:
- Three-seed parent training and physical post-hoc pruning comparison.
- Matched A/B factor budget: 708,096 from 1,179,648 parent parameters (39.974% reduction).
- Parent, GARS, uniform-local-energy and global-singular-value validation comparisons.
- Gradient/spectral mechanism diagnostics, beta sensitivity and 33-export cyclic gradient-to-layer alignment controls.
- Cyclic experiment: 33/33 reloads passed.

Still not measured or established:
- Merged/unmerged inference latency, throughput and serving-time memory.
- Transfer to another task.
- Transfer to another model scale.
- Step-level gradient variability within the final 50-update window for the original RTE runs (those historical files contain only final-window means).

## Priority 1 — inference efficiency benchmark

**Question:** Does reducing the LoRA factor rank produce measurable inference-time improvements, and does this depend on merged versus unmerged execution?

Implementation: \`scripts/benchmark_gars_inference.py\`.

Conditions:
- Variants: parent, GARS, uniform local energy, global singular value.
- Seeds: 42, 123, 999.
- Modes: PEFT unmerged and dense merged model.
- Batch sizes: 1, 8 and 32.
- Fixed first 256 examples from the same GLUE validation split; drop incomplete batches.
- FP32 on MPS; model evaluation and inference mode.
- Ten warmup and 100 timed forwards per condition by default.
- Synchronize MPS after every forward; input transfer is included.
- Model loading and adapter-merge time are excluded.
- Verify merged/unmerged logits within absolute tolerance 1e-4 before timing.
- Report median, mean, p90, p95 and sample SD of per-batch latency, examples/second, and current allocated device-memory snapshots.

Run a short smoke condition first:

\`\`\`bash
python scripts/benchmark_gars_inference.py \\
  --seeds 42 \\
  --variants parent gars \\
  --batch-sizes 1 8 32 \\
  --sample-count 256 \\
  --warmup-steps 2 \\
  --timed-steps 10 \\
  --device mps \\
  --output results/v1_latency_smoke/latency_benchmark.json
\`\`\`

Then run all 12 adapter/seed combinations with the default settings:

\`\`\`bash
python scripts/benchmark_gars_inference.py --device mps
\`\`\`

Full output: \`results/v1_latency/latency_benchmark.json\`.

Do not interpret lower adapter parameter count as a latency improvement until the benchmark shows it. In merged mode, the runtime graph is a dense model, so rank reductions may not produce a corresponding reduction in dense-matmul cost. Report merged and unmerged modes separately.

## Priority 2 — second task: GLUE MRPC on DeBERTa-v3-base

**Question:** Do the matched-budget allocation and validation observations transfer beyond RTE?

A separate config \`configs/v1_deberta_mrpc.json\` preserves the V1 base model, pinned revision, LoRA settings, optimizer settings, training schedule, seeds and target budget while changing only the GLUE task to MRPC.

The parent-training runner now persists all 50 individual layer-gradient observations for future temporal analysis. Use separate output roots; do not overwrite RTE artifacts.

Train:

\`\`\`bash
python experiments/train_parent_rte.py \\
  --config configs/v1_deberta_mrpc.json \\
  --output-root results/v1_parent_mrpc \\
  --device mps
\`\`\`

Run the matched three-policy exporter once per seed:

\`\`\`bash
python scripts/run_pruning_comparison.py \\
  --config configs/v1_deberta_mrpc.json \\
  --seed 42 \\
  --parent-root results/v1_parent_mrpc \\
  --output-root results/v1_pruned_mrpc \\
  --device mps

python scripts/run_pruning_comparison.py \\
  --config configs/v1_deberta_mrpc.json \\
  --seed 123 \\
  --parent-root results/v1_parent_mrpc \\
  --output-root results/v1_pruned_mrpc \\
  --device mps

python scripts/run_pruning_comparison.py \\
  --config configs/v1_deberta_mrpc.json \\
  --seed 999 \\
  --parent-root results/v1_parent_mrpc \\
  --output-root results/v1_pruned_mrpc \\
  --device mps
\`\`\`

Aggregate and inspect within-window temporal reliability:

\`\`\`bash
python scripts/aggregate_pruning_results.py \\
  --results-root results/v1_pruned_mrpc \\
  --output results/v1_pruned_mrpc/aggregate_results.json

python scripts/analyze_gradient_temporal.py \\
  --parent-root results/v1_parent_mrpc \\
  --seeds 42 123 999 \\
  --output results/v1_parent_mrpc/temporal_gradient_analysis.json
\`\`\`

Treat MRPC as an independent task replication, not as extra RTE validation examples. Report all seeds and the same limitations.

## Priority 3 — second model scale: DeBERTa-v3-large / GLUE RTE

**Question:** Does the same method and matched-budget comparison work on a deeper/wider model?

A separate config \`configs/v1_deberta_large_rte.json\` is committed with a pinned Hub revision and 24 Transformer layers. The DeBERTa-v3-large model configuration specifies 24 layers and hidden size 1024. The model card describes a 304M-parameter backbone plus its 131M-parameter embedding component. The exporter, parent validator and mechanism analyzer now read configured layer depth instead of assuming 12 layers.

First run the numerical preflight; do not launch training unless the model load, target-module matching, forward, backward and initial optimizer steps all pass:

\`\`\`bash
python scripts/diagnose_deberta_rte.py \\
  --config configs/v1_deberta_large_rte.json \\
  --device mps \\
  --steps 3
\`\`\`

If the preflight succeeds, train and prune into separate large-model roots:

\`\`\`bash
python experiments/train_parent_rte.py \\
  --config configs/v1_deberta_large_rte.json \\
  --output-root results/v1_parent_large_rte \\
  --device mps
\`\`\`

Then run \`scripts/run_pruning_comparison.py\` for seeds 42, 123 and 999, using:
- config: \`configs/v1_deberta_large_rte.json\`
- parent root: \`results/v1_parent_large_rte\`
- output root: \`results/v1_pruned_large_rte\`

Aggregate:

\`\`\`bash
python scripts/aggregate_pruning_results.py \\
  --results-root results/v1_pruned_large_rte \\
  --output results/v1_pruned_large_rte/aggregate_results.json

python scripts/analyze_gars_mechanisms.py \\
  --config configs/v1_deberta_large_rte.json \\
  --parent-root results/v1_parent_large_rte \\
  --pruned-root results/v1_pruned_large_rte \\
  --output results/v1_pruned_large_rte/mechanism_analysis.json
\`\`\`

Keep the large-model experiment on the same RTE task as base so model scale is the principal changed dimension. Do not assume it will fit or converge on MPS; the preflight is the decision gate. If memory is insufficient, record the failure and adjust per-device batch size with gradient accumulation to preserve the effective batch size, documenting the deviation.

## Priority 4 — optional stronger alignment null

The 11 cyclic rotations are structured controls, not random permutations. If priorities 1–3 finish with enough time, consider a pre-seeded Monte Carlo set of random layer-score permutations per seed. Predefine the number of permutations and random seeds before execution, preserve the same score multiset and physical budget, and report the full distribution. Do not choose a favorable permutation and do not describe the cyclic experiment as a formal permutation test.

## Quality gates and operating rules

Before running new scripts:

\`\`\`bash
ruff check scripts src/gars tests experiments
python -m pytest -q
\`\`\`

- Do not run multiple model training/benchmark processes concurrently on MPS.
- Use separate output roots for RTE, MRPC, large-model and latency experiments.
- Never use \`--overwrite\` unless deliberately replacing the corresponding experiment outputs.
- Keep model/adapter binaries local and ignored by Git; commit source, configs and text reports only.
- Preserve the original V1 beta=0.5 and hyperparameters for replication. Any resource-driven change must be isolated and documented.
- Do not use hidden GLUE test labels for tuning or claims.
- Report validation results as validation results; three seeds do not support strong significance claims.
- The 40% factor-parameter reduction target and 15–20% latency target are proposal targets, not achieved results until independently measured.

## Suggested execution order

1. Quality gates for the new scripts.
2. Latency smoke test, then full latency benchmark.
3. MRPC three-seed replication and temporal gradient analysis.
4. DeBERTa-v3-large numerical preflight, then its three-seed RTE experiment if the preflight passes.
5. Optional random-alignment controls only after the primary replication and scale experiments are secure.
6. Freeze results, update the Phase 1 report and prepare the presentation.
