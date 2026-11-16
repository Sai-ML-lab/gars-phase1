# V1 inference latency benchmark — DeBERTa-v3-base / GLUE RTE

## Scope and protocol

The full latency benchmark completed for all 12 method/seed runs.

- Base model: `microsoft/deberta-v3-base`, using the same pinned V1 revision and RTE adapters as the three-seed pruning study.
- Seeds: 42, 123 and 999.
- Adapter conditions: unpruned parent, GARS, uniform local squared-energy and global singular-value cutoff.
- Inference modes: unmerged PEFT adapter and merged model weights.
- Batch sizes: 1, 8 and 32.
- Device: MPS.
- Precision and execution: FP32, `model.eval()`, `torch.inference_mode()`, with accelerator synchronization around timed forwards.
- Per batch size: 3 representative fixed validation batches; 10 warm-up forwards and 30 timed forwards per distinct batch.
- Total: 12 method/seed runs (4 methods × 3 seeds), all completed.
- Merged/unmerged logits were checked for equivalence before timing; the benchmark enforces a maximum absolute error tolerance of `5e-4`.

The latency statistic covers synchronized model-forward execution only. Tokenization, collation, host-to-device transfer, model loading and adapter merge time are excluded. These measurements are hardware/runtime-specific and are not production-serving or end-to-end request-latency measurements.

The aggregate latency column below is the mean of the three seed-specific median latencies, shown as mean ± sample SD across seeds. The speedup column is the arithmetic mean of the *paired per-seed ratios* (parent median latency divided by the method's median latency). Consequently, the displayed aggregate latency ratio need not equal the reported mean paired speedup.

## Aggregate results

### Unmerged inference

| Method | Batch size | Parent median mean (ms) | Method median mean ± across-seed SD (ms) | Mean paired speedup vs parent |
|---|---:|---:|---:|---:|
| GARS | 1 | 12.465 | 12.160 ± 0.292 | 1.0258× |
| GARS | 8 | 22.456 | 20.079 ± 2.157 | 1.1211× |
| GARS | 32 | 66.541 | 64.235 ± 11.718 | 1.0361× |
| Uniform local energy | 1 | 12.465 | 12.162 ± 0.473 | 1.0263× |
| Uniform local energy | 8 | 22.456 | 20.816 ± 3.030 | 1.0886× |
| Uniform local energy | 32 | 66.541 | 70.534 ± 7.960 | 0.9378× |
| Global singular value | 1 | 12.465 | 12.166 ± 0.116 | 1.0248× |
| Global singular value | 8 | 22.456 | 22.500 ± 0.591 | 1.0000× |
| Global singular value | 32 | 66.541 | 67.858 ± 0.192 | 0.9803× |

### Merged inference

| Method | Batch size | Parent median mean (ms) | Method median mean ± across-seed SD (ms) | Mean paired speedup vs parent |
|---|---:|---:|---:|---:|
| GARS | 1 | 10.979 | 10.870 ± 0.218 | 1.0103× |
| GARS | 8 | 19.857 | 19.527 ± 1.919 | 1.0166× |
| GARS | 32 | 60.222 | 65.275 ± 7.798 | 0.9190× |
| Uniform local energy | 1 | 10.979 | 11.287 ± 0.540 | 0.9736× |
| Uniform local energy | 8 | 19.857 | 21.418 ± 1.779 | 0.9256× |
| Uniform local energy | 32 | 60.222 | 67.835 ± 2.726 | 0.8849× |
| Global singular value | 1 | 10.979 | 11.135 ± 0.384 | 0.9872× |
| Global singular value | 8 | 19.857 | 21.097 ± 1.777 | 0.9513× |
| Global singular value | 32 | 60.222 | 69.643 ± 5.229 | 0.8734× |

A speedup above 1 indicates lower latency than the parent; a speedup below 1 indicates higher latency.

## GARS paired per-seed results

The per-seed speedup ratios below are paired within seed. Seed order is 123, 42, 999.

| Mode | Batch size | Seed 123 | Seed 42 | Seed 999 | Direction |
|---|---:|---:|---:|---:|---|
| Unmerged | 1 | 1.0477× | 0.9787× | 1.0510× | Faster in 2/3 seeds |
| Unmerged | 8 | 1.2665× | 1.0212× | 1.0757× | Faster in 3/3 seeds |
| Unmerged | 32 | 0.9846× | 1.0121× | 1.1117× | Faster in 2/3 seeds |
| Merged | 1 | 0.9781× | 1.0058× | 1.0471× | Faster in 2/3 seeds |
| Merged | 8 | 0.9767× | 0.9900× | 1.0831× | Faster in 1/3 seeds |
| Merged | 32 | 0.9581× | 0.8633× | 0.9357× | Slower in 3/3 seeds |

### Interpretation

1. **Unmerged batch size 8 is the strongest GARS condition.** GARS is faster than the parent in all three paired seeds. The individual latency reductions implied by the paired ratios are approximately 21.0%, 2.1% and 7.0%; their arithmetic mean is approximately 10.1%, and the median is approximately 7.0%. The three-seed mean speedup ratio is 1.1211×. The size of the gain varies substantially, and seed 123 contributes the largest improvement.
2. **Merged batch size 32 is a consistent GARS regression.** GARS is slower in all three seeds, with paired slowdowns of approximately 4.4%, 15.8% and 6.9%; the arithmetic mean slowdown is approximately 9.0%.
3. **Other GARS conditions are modest or mixed.** Unmerged batch sizes 1 and 32 are faster in two of three seeds. Merged batch sizes 1 and 8 do not show a consistent, material improvement.
4. **There is no general latency winner among pruning policies.** Uniform local-energy and global singular-value results also vary by batch size, mode and seed. The results do not justify claiming that gradient-conditioned rank allocation is intrinsically faster than the other pruning policies.
5. **The latency target is not established.** The proposal's 15–20% latency improvement objective is not demonstrated as a robust result. The 21.0% reduction for unmerged batch 8 on seed 123 is one seed/condition, not a general result; the mean per-seed reduction in that condition is about 10.1%.

## Limitations and claims boundary

- This benchmark has three seeds on one model scale, one task, one local MPS hardware/runtime setup and representative RTE validation batches.
- Across-seed sample SD is descriptive with only three observations. It is not a confidence interval or significance test.
- The benchmark measures model forward execution only. It excludes tokenization, collation, host-to-device transfer, model loading, merge time and the rest of an application serving stack.
- Results can depend on batch size, sequence-length mix, merged/unmerged deployment mode, hardware, runtime and execution order.
- No production throughput, tail-latency SLA, concurrency, memory/energy, or end-to-end serving claim follows from this experiment.
- The approximately 40% reduction reported elsewhere applies to physical LoRA A/B factor parameters; it does not imply a 40% reduction in total model compute or latency.

## Defensible Phase 1 latency conclusion

On the DeBERTa-v3-base / GLUE RTE MPS benchmark, GARS showed a repeatable direction of improvement for unmerged batch size 8 across the three seeds, but the mean per-seed latency reduction was approximately 10.1% and its magnitude varied. GARS was slower than the parent in all three seeds for merged batch size 32. Other mode/batch conditions were mixed or showed only small changes. Therefore, the experiment provides preliminary evidence that the pruning policy can affect inference latency under particular execution conditions, but it does not demonstrate a general 15–20% latency improvement or establish GARS as a consistently faster inference method.
