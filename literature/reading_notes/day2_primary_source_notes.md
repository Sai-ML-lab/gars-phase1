# Day 2 — GARS literature: primary-source reading notes (working draft)

**Review date:** 11 November 2026  
**Purpose:** Record what has been verified from primary sources so far, distinguish prior art from the proposed GARS mechanism, and guide the remaining reading.  
**Review type:** Focused structured/narrative literature review for the M.Tech Phase 1 report. This is not a completed PRISMA systematic review.

> Status discipline: “Partial full-text inspected” means the paper's relevant sections were read, not that the complete paper has been read. Do not use a paper's abstract as a substitute for its method and experiment sections.

## 1. What the current primary-source reading establishes

### 1.1 LoRA (Hu et al., ICLR 2022)
Primary source: [OpenReview paper](https://openreview.net/forum?id=nZeVKeeFYf9) · [arXiv full text](https://arxiv.org/abs/2106.09685)

- LoRA freezes pretrained weights and represents a task-specific weight update with two trainable low-rank factors, conventionally written as \(\Delta W = BA\).
- The paper motivates the low-rank update using the intrinsic low-dimensionality observed in pretrained-model adaptation.
- The factors are initialized so the initial update is zero; the adapter may be merged with the base weights for inference, avoiding an extra adapter path at inference.
- The paper establishes the fixed-rank LoRA baseline. It does not establish that every Transformer module needs equal rank; later adaptive-rank work addresses this.
- **Reading status:** Relevant opening/method passages inspected. Full experimental sections and appendices still need student review.

### 1.2 AdaLoRA (Zhang et al., ICLR 2023)
Primary source: [OpenReview paper](https://openreview.net/forum?id=lq62uWRJjiY) · [arXiv full text](https://arxiv.org/abs/2303.10512)

- AdaLoRA uses an SVD-style parameterization of the update, with left singular-vector factors, a diagonal singular-value vector, and right singular-vector factors.
- It assigns importance to singular triplets and masks/prunes low-importance components during training under a global budget schedule.
- The paper describes sensitivity-based importance, combining information from gradients and weights; its method uses an orthogonality regularizer to make the learned factors SVD-like.
- It deliberately starts above the final budget and progressively reduces the budget. Its main distinction from GARS is not simply “uses gradients” or “uses SVD”; it already does both in a different parameterization and importance-scoring scheme.
- A reported advantage of AdaLoRA is retaining singular-vector parameters even when singular values are masked, allowing a masked component to potentially recover. This differs from physically deleting dimensions during training.
- **Implication for GARS:** Compare the exact importance score, granularity (triplet vs layer), score smoothing, budget policy, parameterization, pruning schedule, optimizer-state effects, and whether final tensor shapes shrink.
- **Reading status:** Introduction, SVD-based adaptation and beginning of importance-aware allocation inspected. Need finish exact score equations, scheduler, experiments and ablations.

### 1.3 DyLoRA (Valipour et al., EACL 2023)
Primary source: [ACL Anthology PDF](https://aclanthology.org/2023.eacl-main.239.pdf)

- DyLoRA trains one adapter to work across a range of ranks rather than training separate adapters for each candidate rank.
- During training, it samples rank slices and sorts/reorders the learned representation so different prefixes can be used at inference.
- Its central question is search-free rank selection and rank flexibility, not redistribution of one fixed parameter budget across layers using a live importance signal.
- **Implication for GARS:** It is a required baseline in the approved proposal, but it is not the closest match to the GARS pruning rule.
- **Reading status:** Abstract, introduction and method motivation inspected. Need read the rank-sampling/training algorithm and experimental setup.

### 1.4 SoRA (Ding et al., EMNLP 2023)
Primary source: [ACL Anthology PDF](https://aclanthology.org/2023.emnlp-main.252.pdf)

- SoRA attaches a learnable gate vector to low-rank components and applies an L1/proximal-gradient procedure to encourage sparse gates.
- Components with zero gates are removed from the factor matrices for inference, producing a physically smaller adapter.
- It uses sparse gates and a sparsification schedule rather than GARS's layer-gradient-conditioned energy-retention thresholds.
- **Implication for GARS:** SoRA is important prior art for structural adapter pruning and export. It prevents us from claiming that GARS is the first method to convert pruned rank into smaller factor tensors.
- **Reading status:** Abstract, introduction and relevant related-work/method explanation inspected. Need complete objective/proximal update and experiments.

### 1.5 ALoRA (Liu et al., NAACL 2024)
Primary source: [ACL Anthology PDF](https://aclanthology.org/2024.naacl-long.35.pdf)

- ALoRA initializes equal-rank LoRA modules and uses AB-LoRA to estimate the contribution of each rank.
- It progressively prunes ranks that are redundant or negatively contributing and reallocates the freed budget to other model weights/modules that benefit from additional rank.
- Its contribution is rank-importance estimation and budget reallocation. It is not the same as a gradient-derived energy threshold for each layer.
- **Implication for GARS:** Include a mechanism-level comparison and check the exact AB-LoRA ablation procedure and matched-budget conditions.
- **Reading status:** Abstract, introduction and framework description inspected. Need read AB-LoRA scoring equations, rank-transfer policy, experiments and ablations.

### 1.6 GoRA (He et al., NeurIPS 2025)
Primary source: [NeurIPS 2025 proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html) · [arXiv full text](https://arxiv.org/abs/2502.12171)

- The title and abstract emphasize gradient-driven adaptive rank and initialization. The method's exact timing matters: GoRA computes an **N-batch accumulated gradient of frozen pretrained weight matrices before formal LoRA training**.
- It scores each target pretrained weight matrix using a sensitivity-style statistic \(I(W)=\operatorname{avg}(|W\odot G|)\), normalizes the scores, and assigns per-matrix ranks under an approximately matched total trainable-parameter budget.
- It then uses the compressed gradient to initialize LoRA factors. The paper explicitly completes rank allocation before training to avoid dynamic tensor-shape changes.
- **Important distinction:** GoRA is not the same as the proposal's live per-layer LoRA-factor gradient magnitude driving per-layer SVD energy-retention thresholds during training. But it is direct prior art for using gradients to allocate LoRA rank; GARS must acknowledge it.
- **Implication for GARS:** Compare the gradient source (base-weight gradients vs adapter-factor gradients), timing (pre-training probe vs live/periodic), importance function, granularity, rank-allocation rule, initialization, and training overhead.
- **Reading status:** Primary PDF pages covering introduction and core rank-allocation equations inspected. Need read full initialization, experimental setup, ablations and code.

### 1.7 Activation-Guided Layer Selection for LoRA (Dawadikar et al., Information, 2026)
Primary source: [DOI/publisher record](https://doi.org/10.3390/info17030283)

- The paper uses a frozen-model probing phase with forward hooks to gather layer activation magnitudes, ranks candidate layers, and inserts LoRA only into the top-k layers.
- It explicitly discusses gradient magnitude as an alternative importance signal and notes that raw gradients can be noisy/sensitive to optimization dynamics; AdaLoRA uses smoothing of gradient/weight signals.
- This is **layer selection** (where to place adapters), not necessarily **within-layer rank truncation** of all adapters. The distinction matters.
- Its experiments include multiple seeds and report accuracy as well as compute/latency/throughput and memory; the paper also notes that effective rank masking may not reduce physical adapter tensor dimensions in some implementations.
- **Implication for GARS:** It is direct evidence that activation-based selection and gradient-based importance deserve a careful comparison, but it does not by itself answer whether activations allocate within-layer rank better than gradients under the same adapter budget.
- **Reading status:** Introduction, related work, theoretical motivation, method, experimental setup and a substantial part of results inspected. Need verify exact tables, seed protocol, statistical reporting and limitations.

### 1.8 FlexLoRA (ICLR 2026)
Primary source: [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/387936a8cd02bf2ad745b751047b9d49-Abstract-Conference.html) · [arXiv full text](https://arxiv.org/abs/2601.22905)

- FlexLoRA uses an SVD-style parameterization and defines a normalized matrix-level spectral entropy over squared-singular-value energy:
  \[
  s_i=\frac{\lambda_i^2}{\sum_j\lambda_j^2},\qquad
  I(\Lambda)=-\frac{1}{\log r}\sum_i s_i\log(s_i+\epsilon).
  \]
- It periodically prunes rank from low-entropy/low-importance matrices and expands rank in high-importance matrices under a global rank-adjustment budget. Newly added directions use zero-impact initialization.
- It is therefore direct prior art for spectral-entropy-guided adaptive rank adjustment. It is **peer-reviewed at ICLR 2026**, not merely an arXiv preprint.
- **Implication for GARS:** Do not claim novelty for spectral entropy as a rank-allocation metric. A narrower question may remain: whether a separately defined spectrum-flatness statistic predicts when *gradient-conditioned energy retention* improves over a *budget-matched uniform SVD pruning* baseline.
- **Reading status:** Method equations, pruning/expansion algorithm and initial experiment results inspected. Need full experimental/ablation/appendix review.

### 1.9 PARA: Post-Optimization Adaptive Rank Allocation for LoRA (Kumaravelu et al., 2026 preprint)
Primary source: [arXiv full text](https://arxiv.org/abs/2604.27796)

- PARA is a post-training, data-free compression method. It begins with standard LoRA trained at an adequately high uniform rank; after training it performs SVD of each effective adapter update and applies a **single global threshold** to the singular values across all adapters to meet a target average rank or relative Frobenius-energy retention.
- Crucially, PARA derives the SVD within the low-rank LoRA subspace using QR factorizations of B and Aᵀ and then an SVD of an r×r interaction matrix. This is mathematically the same reduced-QR-SVD construction used in the exploratory GARS notebooks.
- It reconstructs smaller factor tensors by distributing square roots of retained singular values across the two factors. It therefore addresses physical adapter-size reduction through truncated export.
- PARA does not use GARS's gradient-conditioned per-layer energy thresholds. It is a very close baseline for post-hoc spectral compression and a direct prior-art match for the QR-SVD/export component.
- **Implication for GARS:** The QR-SVD factorization and physically truncated factor export cannot be claimed as novel contributions. The comparison should focus on local gradient-conditioned thresholds versus PARA's global singular-value threshold, and whether a training-time signal improves performance at a matched final budget.
- **Reading status:** Primary PDF pages covering motivation, global-threshold method, physical reconstruction and QR-SVD derivation inspected. It remains an arXiv preprint in the reviewed record; check for a later venue version before final bibliography freeze.

### 1.10 Sawant (ACL Workshop on Insights from Negative Results in NLP, 2026)
Primary source: [ACL Anthology record](https://aclanthology.org/2026.insights-1.14/) · DOI: [10.18653/2026.insights-1.14](https://doi.org/10.18653/2026.insights-1.14)

- The work reports a negative result for gradient-proportional LoRA rank allocation under GRPO on a Qwen2.5-1.5B / GSM8K setting.
- The proposal cites it to motivate the possibility that gradient allocation can be harmful when the gradient landscape is relatively flat.
- This result is specific to the reported GRPO/RL setup. It should not be generalized to supervised fine-tuning or all gradient-based allocation.
- **Implication for GARS:** Treat it as motivation for a conditional/reliability question, not proof that gradient signals are generally unreliable.
- **Reading status:** Metadata and abstract-level findings verified. Full paper, exact flatness definition, experimental seed count and limitations still need to be read.

### 1.11 LAARA (Tripathi et al., 2026 preprint)
Primary source: [arXiv](https://arxiv.org/abs/2607.19391)

- The surfaced full-text excerpts describe layer-wise rank allocation from a diagonal Fisher approximation using LoRA gradients.
- This is close prior art for gradient-derived layer importance. Its exact timing, rank mapping, assumptions, matched-budget controls and theoretical claims require full-text verification.
- **Reading status:** Abstract and selected full-text excerpts inspected only; preprint status.

### 1.12 IGU-LoRA (Cui et al., 2026 preprint)
Primary source: [arXiv](https://arxiv.org/abs/2603.13792)

- The surfaced excerpts describe integrated-gradient and uncertainty-aware importance scoring to make rank allocation more robust to instantaneous gradient noise.
- It is important prior art for gradient reliability/stability, but the exact algorithm, optimization overhead and experimental results remain to be verified.
- **Reading status:** Abstract and selected excerpts inspected only; preprint status.

## 2. Preliminary prior-art comparison

| Method | Importance signal | Rank timing / operation | SVD / parameterization | Relation to GARS |
|---|---|---|---|---|
| LoRA | None for rank allocation | Fixed rank | Ordinary A/B factors | Fixed-rank baseline |
| AdaLoRA | Smoothed gradient/weight sensitivity per singular triplet | During training; global budget schedule | Learned SVD-like P/Λ/Q | Direct adaptive-SVD baseline |
| DyLoRA | Nested/rank-slice training | Train one range of ranks; select at inference | Reordered/sliced LoRA factors | Rank-flexibility baseline, not per-layer budget allocation |
| SoRA | Learnable sparse gates with proximal/L1 updates | During training; physically remove zero-gated components at inference | Gated LoRA factors | Structural pruning baseline |
| ALoRA | Ablation-based per-rank contribution | During training; prune and reallocate budget | Gated SVD-like rank components | Rank reallocation baseline |
| GoRA | Precomputed base-weight gradients combined with weight magnitude | Before formal adapter training; allocate rank and initialize factors | Ordinary LoRA-compatible factors | Direct gradient-rank prior art, but not live adapter-gradient thresholding |
| Act-LoRA | Layer activation norms | Probe then choose top-k layers for adapter insertion | Ordinary LoRA in selected layers | Activation comparator; layer selection rather than within-layer rank pruning |
| FlexLoRA | Matrix-level spectral entropy | Periodic prune and expand during training under global budget | SVD-style P/Λ/Q | Direct spectral-allocation prior art |
| PARA | Singular-value magnitudes under a global threshold / target energy | Post-optimization compression | Reduced QR-SVD on ordinary LoRA factors; physically truncated export | Closest baseline for QR-SVD and post-hoc physical compression |
| GARS (proposed) | Live/periodic gradient magnitude of each layer's LoRA factors | Proposal specifies gradient-conditioned per-layer energy-preservation threshold and SVD pruning | QR-SVD on ordinary LoRA A/B factors (in notebooks) | Candidate distinction must be validated against the above |

## 3. What this means for the research gap

The broad gap originally expressed in the proposal—“which signal should drive per-layer capacity, and when is it reliable?”—is too broad as a novelty claim after this prior-art check. The literature already contains:
- gradient-informed rank allocation (GoRA, LAARA, IGU-LoRA and AdaLoRA's sensitivity score),
- activation-based layer selection (Act-LoRA),
- spectral-entropy-based pruning/expansion (FlexLoRA),
- post-hoc QR-SVD global threshold compression and physical export (PARA).

A potentially testable, narrower GARS question is:

> Under a fixed final adapter-parameter budget, does using a live/temporally aggregated gradient statistic from the LoRA factors to set per-layer singular-value energy-retention thresholds improve over a global-threshold or uniform-threshold SVD pruning rule, and does a prespecified measure of each layer's spectral concentration predict the difference in outcome?

This is a *candidate research question*, not yet an established novelty claim. Before making it the report's final claimed contribution:
1. read GoRA, FlexLoRA, PARA, LAARA and IGU-LoRA in full;
2. inspect their code where available;
3. compare exact timing, statistic, threshold/rank mapping and budget controls;
4. verify that the notebook's GARS rule is not already equivalent to a published scheme;
5. discuss material narrowing/refinement with the project mentor.

## 4. Methodology and experimental consequences

1. **Budget matching:** Compare GARS and uniform/global spectral pruning at the same final number of adapter parameters (or a clearly defined nearby budget). Accuracy alone is not a fair comparison if one method retains more parameters.
2. **Signal definitions:** State whether layer importance is computed from \(\|\nabla A_l\|+\|\nabla B_l\|\), a joint Frobenius norm, or a temporally averaged statistic. These are not interchangeable.
3. **Timing:** Specify whether gradient scores are measured per step, at a pruning event, over a moving window, or from a pre-training probe. GoRA uses a pre-training probe; AdaLoRA uses training-time sensitivity; these need different labels.
4. **SVD energy:** The Frobenius-energy retention criterion is based on squared singular values, \(\sum_{i\leq k}\sigma_i^2/\sum_i\sigma_i^2\). Keep this separate from the sum-of-singular-values criterion present in the exploratory notebooks.
5. **Physical compression:** Count tensor-shape parameters after export, not only nonzero/effective rank. Validate that the exported per-layer adapter can load and run.
6. **Mechanism:** A spectral statistic should be tested as a predictor/moderator of the *difference between GARS and matched-budget spectral baselines*, not merely plotted alongside final ranks.
7. **Evaluation:** Report seed-level values, mean and standard deviation, exact hardware/software configuration and the limitations of preliminary scale/domain coverage.

## 5. What still needs reading

Highest priority:
1. Finish AdaLoRA: exact importance equation, smoothing, budget scheduler, ablations and experiments.
2. Finish GoRA: initialization math, experiments, ablations and code.
3. Finish Act-LoRA: full gradient comparison, three-seed protocol, tables and limitations.
4. Finish FlexLoRA: global rank budget, scheduling, five-seed GLUE results, ablations and appendices.
5. Finish PARA: energy policies, benchmark settings, latency/size measurements and limitations.
6. Read full DyLoRA, SoRA and ALoRA methods/results.
7. Read Sawant workshop paper in full.
8. Inspect LAARA and IGU-LoRA preprints critically; do not rely on abstract-only claims.

This file records a preliminary technical reading synthesis. It is not yet the final Chapter 2 literature review and should be refined after the student has personally read and annotated the critical papers.
