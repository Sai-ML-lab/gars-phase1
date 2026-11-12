# Implementation audit — IGU-LoRA and Aletheia public code

**Audit date:** 12 November 2026  
**Purpose:** Compare the public default-branch implementations with the corresponding paper descriptions. This is a source-code snapshot audit, not a full independent reproduction. Code can lag a paper version; all findings below are scoped to the inspected public repositories and commits.

## 1. IGU-LoRA

Paper: https://openreview.net/pdf?id=MnToYQx9My  
Public repository: https://github.com/Cuixuans/igulora  
Inspected commit history: two commits, both dated 26 February 2026 (78e21e9, 3f5ba10). The arXiv paper appeared 14 March 2026 and is listed in ICLR 2026 proceedings.

### Files inspected
- iglora/rank_allocator.py
- iglora/lora_module.py
- iglora/run_pruning_sft.py
- iglora/run_final_sft.py
- README.md

### What the visible code actually does

The public RankAllocator implementation has an explicit class docstring reference to AdaLoRA. Its visible per-parameter sensitivity is:

\[
I_p(t)=|p\odot \nabla_p L|.
\]

It tracks an EMA of this signal (\(\beta_1\)) and a second EMA of the absolute deviation from the smoothed score (\(\beta_2\)). It constructs component-level importance by combining:
- the mean importance across the corresponding A-factor row;
- the mean importance across the corresponding B-factor column;
- the importance of the corresponding trainable diagonal singular-value parameter E.

A global threshold is determined from all component scores to meet a target total rank schedule. The public pruning script calls RankAllocator.update_and_mask(model, completed_steps) after optimizer.step(), writes a ranks_to_mask.json file, and then zeroes the E singular-value parameters below the threshold.

The second-stage script loads ranks_to_mask.json, writes zeros into each module's lora_mask at the pruned component positions, and applies the mask in the forward path through lora_E * lora_mask.

### Important paper–code caveat

I did **not** find a parameter-path integration calculation in the inspected public code path:
- no integration baseline/path;
- no quadrature/sample-alpha calculation;
- no accumulated path-integral estimator;
- the visible allocator score is abs(p * p.grad), which is the AdaLoRA-style first-order parameter sensitivity.

The public repository's save_lora_modules function simply selects all keys containing lora and saves the state dict. It does not remove rank columns/rows or construct physically smaller A/B tensors. The second-stage mask makes components functionally inactive, but the stored A/B/E parameter tensors retain their original shapes.

**Interpretation:** this is a real mismatch between the paper's parameter-space Integrated Gradients description and the inspected public repository snapshot. Because the public repository history has only two February commits and predates the March paper version, the safest interpretation is that the default-branch code may be an earlier/incomplete implementation. Do not claim that the paper's method is merely AdaLoRA; do say that the currently inspectable public code does not demonstrate the paper's path-integral mechanism. Do not treat the public code as a faithful reproduction until the authors provide a matching release or explain the discrepancy.

### Implications for GARS
- A literature review should describe the paper's method as Integrated Gradients + uncertainty-aware scoring.
- A reproducibility appendix/notes section may point out that the current public code snapshot implements a p·grad EMA/uncertainty allocator and a full-shape mask export.
- If IGU-LoRA is used as a baseline, implement from the paper equations or obtain an updated author-confirmed implementation. Do not silently use this snapshot as if it were the published IGU algorithm.
- For GARS's export objective, make the difference between masked full-shape tensors and physically truncated factor tensors explicit.

## 2. Aletheia

Paper: https://arxiv.org/abs/2604.15351  
Public repository: https://github.com/aletheiaprotocol-ai/aletheia-lora  
Public method card: https://huggingface.co/aletheiaprotocol/aletheia-lora

### Files inspected
- src/aletheia_lora/probe.py
- src/aletheia_lora/selection.py
- src/aletheia_lora/config.py
- README.md

### What the visible code does

1. Samples probe_batches examples from a seeded, shuffled DataLoader (default batch size 1).
2. Iterates over transformer layers in chunks (default chunk size 8).
3. Enables gradients only for parameters in the current chunk, runs a causal-LM forward/backward on each sampled batch, and accumulates param.grad.float().norm().item() for each parameter tensor into that layer's score.
4. Sets gradients to zero between batches and, at the end of the probe, sets all floating-point model parameters to requires_grad=False.
5. Selects the top percentage of layers by the accumulated score (default 50%).
6. Builds a PEFT LoraConfig with layers_to_transform=selected_layers, and can assign different ranks to attention modules and MLP modules using rank patterns.

### Code-level observations and cautions

- The code calls model.train() during the probe, so training-mode stochastic layers such as dropout remain active (where the model has them). This is part of the actual implementation and may affect score repeatability.
- The layer score is an accumulation of per-parameter tensor gradient norms over the probe examples; it is not explicitly formed as one concatenated Frobenius norm over all layer gradients. Equal-size probe batches make the summed score comparable within a given run, but its scale is dependent on probe count and model parameterization.
- With 36 layers and chunk size 8, there are five layer chunks. Five probe batches therefore imply up to 25 forward/backward passes through the model (five per chunk), not five total full-model backward passes. The model must be run once per chunk because only the currently selected chunk has gradients enabled.
- The selection rule is a top-k percentage heuristic, not a learned or dynamically updated rank allocator.
- The configuration code applies PEFT to selected layers only; asymmetric attention/MLP ranks alter module rank but do not perform per-layer SVD truncation.

### Relation to GARS

Aletheia answers **where should LoRA adapters be inserted?** GARS asks **which singular directions should be retained within each already-adapted LoRA matrix?** The methods share a gradient-guided motivation but operate at different structural levels. Aletheia is an adjacent layer-selection comparator, not a direct implementation baseline for GARS's energy-retention rule.

## 3. Reproducibility protocol for our project

For each paper with public code:
1. Record repository URL, branch, commit SHA/date and paper version.
2. Identify whether the code implements the paper's stated equations.
3. Run small unit tests for score calculation, rank allocation and exported tensor dimensions before attempting model training.
4. Distinguish the conceptual method in the paper from any deviations or omissions in the current code.
5. Never label a result as a reproduction if the implementation differs materially.

This audit is especially relevant to IGU-LoRA: an old/incomplete public implementation is not an adequate basis for a claimed IGU reproduction.
