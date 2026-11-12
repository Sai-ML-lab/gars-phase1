# Day 2 — student reading queue

This queue is intentionally short enough to complete alongside the research synthesis. The assistant's notes are a navigation aid, not a substitute for reading the papers and verifying claims yourself.

## Read in depth (priority order)

### R1. LoRA — Hu et al. (ICLR 2022)
Primary source: https://openreview.net/forum?id=nZeVKeeFYf9  
Read: Introduction, method, initialization/scaling, inference merging, experiments.  
Be ready to explain:
- Why represent a task update as \(BA\)?
- How are A and B initialized and why?
- What does rank control?
- When can adapters be merged, and what does that imply for inference latency?

### R2. AdaLoRA — Zhang et al. (ICLR 2023)
Primary source: https://openreview.net/forum?id=lq62uWRJjiY  
Read: SVD-style parameterization, importance score, smoothing/uncertainty, global budget schedule, experiments and ablations.  
Be ready to explain:
- What exactly is scored: entries, singular triplets, modules, or layers?
- How do gradients and weights enter its importance score?
- Why use EMA smoothing and an uncertainty term?
- How does its parameterization differ from an SVD performed on ordinary LoRA A/B factors?
- What does its budget schedule do over training?

### R3. GoRA — He et al. (NeurIPS 2025)
Primary source: https://papers.nips.cc/paper_files/paper/2025/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html  
PDF: https://arxiv.org/pdf/2502.12171  
Read: Section 3.2 rank allocation, initialization, experimental settings and ablations.  
Be ready to explain:
- Which gradients are used (pretrained-weight gradients, not LoRA-factor gradients)?
- When are the N-batch gradients measured?
- How does \(I(W)=\operatorname{avg}(|W\odot G|)\) become a rank assignment?
- Why does GoRA complete rank assignment before formal adapter training?
- How does its initialization use compressed gradients?

### R4. FlexLoRA — ICLR 2026
Proceedings: https://proceedings.iclr.cc/paper_files/paper/2026/hash/387936a8cd02bf2ad745b751047b9d49-Abstract-Conference.html  
PDF: https://arxiv.org/pdf/2601.22905  
Read: spectral entropy definition, rank prune/expansion algorithm, global budget, experiments, ablations.  
Be ready to explain:
- Why does FlexLoRA use squared singular values to form an energy distribution?
- What does low versus high spectral entropy mean?
- How are ranks pruned and expanded?
- How does this differ from using spectral flatness only as a predictor of GARS's benefit?

### R5. PARA — 2026 arXiv preprint
Paper: https://arxiv.org/abs/2604.27796  
Read: global threshold, target budget/energy policies, QR-SVD derivation, physical factor reconstruction, latency and parameter results.  
Be ready to explain:
- Why can SVD of \(BA\) be reduced to QR decompositions plus an \(r\times r\) SVD?
- How does PARA choose one global threshold?
- How are variable-rank A/B factors reconstructed?
- How does this differ from GARS's proposed per-layer threshold conditioned on a gradient signal?

## Read selected sections (second priority)

### R6. Activation-Guided Layer Selection for LoRA — Dawadikar et al. (Information, 2026)
DOI: https://doi.org/10.3390/info17030283  
Read: activation-score definition, probing phase, gradient comparison, experimental tables, seeds, latency and limitations.  
Focus: top-k layer selection versus within-layer rank pruning; the paper's comparison of activation and gradient signal stability.

### R7. IGU-LoRA — Cui et al. (ICLR 2026)
Proceedings: https://proceedings.iclr.cc/paper_files/paper/2026/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html  
Read: within-layer Integrated Gradients, uncertainty-aware EMA/deviation score, approximation error theory, matched-budget experiments.  
Focus: why integrated/pathwise gradients differ from instantaneous gradients.

### R8. SoRA — Ding et al. (EMNLP 2023)
PDF: https://aclanthology.org/2023.emnlp-main.252.pdf  
Read: gate formulation, proximal update, sparsifying schedule, physical removal of zeroed components.

### R9. ALoRA — Liu et al. (NAACL 2024)
PDF: https://aclanthology.org/2024.naacl-long.35.pdf  
Read: AB-LoRA ablation-based rank importance, pruning and budget transfer, comparable-budget experiments.

### R10. DyLoRA — Valipour et al. (EACL 2023)
PDF: https://aclanthology.org/2023.eacl-main.239.pdf  
Read: nested/rank-slice training and inference-time rank selection; distinguish this from layer-wise budget redistribution.

### R11. Sawant — ACL Workshop on Insights from Negative Results in NLP (2026)
Paper: https://aclanthology.org/2026.insights-1.14/  
Read: experimental setting, gradient distribution/flatness statistic, allocation rule, single-seed/single-task limitation.  
Focus: GRPO is not the same training regime as our proposed SFT experiments.

## For each paper, submit these notes

1. Two-sentence problem and contribution summary in your own words.
2. The exact mathematical importance/rank-selection equation.
3. When the importance signal is measured.
4. Whether rank changes during training, before training, or after training.
5. The experimental budget-matching protocol.
6. Models, datasets, seeds and metrics.
7. One important limitation or assumption.
8. One sentence explaining how it differs from GARS.

Do not paste an AI-generated summary as your reading notes. Use it to guide your reading, then record your own explanation and questions.


## Newly added prior art — read these before freezing the research gap

### R12. Progressive PEFT — Gebretatios, Ning and Ding (Research Square preprint, May 2026)
Full paper: https://doi.org/10.21203/rs.3.rs-9816617/v1  
Read: gradient score, two-level EMA, median module aggregation, global-budget rank allocator, direct factor resizing/optimizer reset, Tigrinya experiments and limitations.

Be ready to explain:
- Why does it use \(\|\nabla_p L\|/(\|p\|+\epsilon)\), rather than simply \(\|\nabla p\|\)?
- What are the purposes of the parameter-level EMA, module median and module-level EMA?
- How does the softmax rank allocator maintain (or approximate) the global parameter budget after rank clipping/flooring?
- Why does it truncate/zero-pad factors rather than use SVD?
- How does its optimizer reset compare with the GARS implementation and proposal?

**This is a critical comparison**, because it already performs online gradient-derived layer-wise rank scheduling and resets optimizer state when ranks change. Treat it as a preprint, not peer-reviewed evidence.

### R13. Aletheia — Saket (arXiv preprint, April 2026)
Paper: https://arxiv.org/abs/2604.15351  
Code/method card: https://github.com/aletheiaprotocol-ai/aletheia-lora

Read: five-batch gradient probe, top-50% layer selection, asymmetric attention/MLP ranks, paired cross-architecture evaluation and limitations.

Be ready to explain:
- Which gradients are probed (backbone layer weights)?
- Why is this a one-time layer-selection rule rather than dynamic rank pruning?
- What changes in training compute when some layers receive no adapters?
- How does it differ from the GARS objective of retaining singular directions within each LoRA adapter?

### Updated student reading note

After reading Progressive PEFT, write a direct comparison paragraph with LAARA and GARS. Focus on the exact score and mapping:
- Progressive PEFT: relative per-parameter gradient score → EMA/median/EMA → budgeted integer rank → direct factor truncation/zero-padding.
- LAARA: squared A/B gradients / empirical-Fisher proxy → normalization, compression, blending and voting → integer layer rank.
- GARS proposal: normalized layer-gradient statistic → per-layer energy-retention threshold → SVD-derived retained rank.

The comparison should not say that GARS is the first gradient-aware layer-wise rank method. Explain what the energy-retention constraint adds and what evidence would be needed to show that it matters.
