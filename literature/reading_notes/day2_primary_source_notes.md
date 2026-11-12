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


## 6. Corrections from additional primary-source verification

### IGU-LoRA is peer-reviewed, not just an arXiv preprint
The ICLR 2026 proceedings list **IGU-LoRA: Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Aware Scoring** by Xuan Cui, HuiYue Li, Run Zeng, Yunfei Zhao, Jinrui Qian, Wei Duan, Bo Liu and Zhanpeng Zhou. The abstract says it aggregates within-layer Integrated Gradients into layer-level scores and uses an uncertainty-aware EMA/deviation scheme to stabilize rank selection. It is direct prior art for gradient-path sensitivity and uncertainty-aware rank allocation.  
Primary record: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html). Full method and experiments still need reading.

### Effective-rank definitions must not be mixed
Roy and Vetterli (EUSIPCO 2007) define effective rank using the entropy of the normalized singular values:
\[
p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
\operatorname{erank}(A)=\exp\left(-\sum_i p_i\log p_i\right).
\]
Source: [The Effective Rank: A Measure of Effective Dimensionality](https://doi.org/10.5281/zenodo.40328).

FlexLoRA instead defines a normalized entropy over the **squared singular-value energy** distribution:
\[
q_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2},\qquad
H_2(A)=-\frac{1}{\log r}\sum_i q_i\log(q_i+\epsilon).
\]
These are related but not identical measures. If GARS uses a spectral-flatness statistic, specify which distribution is used, why it is appropriate, and do not call the two definitions interchangeable. A normalized entropy (H_2\) lies near 0 for concentrated energy and near 1 for a flat squared-energy spectrum (ignoring the small numerical epsilon).

### Important GARS question after the prior-art comparison
The proposal's mechanism currently appears closest to a combination of:
- AdaLoRA / LAARA / GoRA / IGU-LoRA for gradient-derived importance;
- FlexLoRA for singular-spectrum entropy-guided rank allocation;
- PARA for QR-SVD and physically truncated post-hoc LoRA export.

A credible GARS contribution therefore needs to be a carefully isolated, experimentally tested interaction between **adapter-factor gradient statistics**, **layer-specific squared-energy retention thresholds**, and **spectral concentration as a moderator of the relative benefit against matched-budget baselines**. The exact timing and optimizer-state correctness of the live gradients must be specified. Do not claim this interaction is new until the full papers and code of the closest methods have been checked.


## 7. Additional full-text reading: ALoRA, LAARA, IGU-LoRA and recent adjacent methods

### 7.1 ALoRA — Liu et al. (NAACL 2024)
Primary source: [ACL Anthology PDF](https://aclanthology.org/2024.naacl-long.35.pdf) · DOI: 10.18653/v1/2024.naacl-long.35

**Method.** ALoRA treats rank allocation as a budgeted architecture-allocation problem. Each module starts at a uniform rank chosen so the target total budget is already met; it does not first allocate an oversized rank and later compress it. A diagonal gate is inserted between the two LoRA factors. Its architecture variables are relaxed through a sigmoid during the setup, but the actual AB-LoRA workflow freezes those architecture parameters and measures rank importance directly.

For one rank component r, AB-LoRA evaluates three variants on the same validation mini-batch: the full super-network M, the super-network with only r removed, and the network retaining only r. With S(M) set to negative cross-entropy, the importance score is:
\[
IS(r)=S(M)-S(M_{\backslash r})+S(M_r).
\]
A component is considered important if its removal damages the full network and if it performs usefully when acting alone. The workflow trains the super-network for K1 epochs, scores ranks on a validation batch, prunes the nA least important ranks, reallocates the freed ranks to modules that were not pruned (prioritized by average importance), then fine-tunes for K2 epochs to recover performance. This process repeats up to NA rounds.

**Experimental setting.** Main results use Llama-2-7B on SST-2, RTE, QNLI, BoolQ, COPA, ReCoRD and SQuAD; E2E generation and Alpaca instruction tuning are also studied. GPT-2-large and RoBERTa-large are used in backbone ablations. The main setup starts each module at rank 8 with total target budget 8 × number of modules; batch size for validation scoring is 32, K1=1 epoch, K2=0.25 epoch, and up to 8 allocation rounds. Five random seeds are run and median performance is reported. A40 48GB GPUs are used.

**Interpretation.** ALoRA is prior art for adaptive rank pruning and budget reallocation across Transformer modules, including actual rank-wise ablation scoring. It is *not* a gradient-based importance method and does not use singular-value energy retention. It also does not need an oversized initial rank; rank budget is already met at initialization. Its ranking signal is validation-set ablation of individual rank units, so the number of candidate ranks and repeated forward evaluations introduce a measurement cost.

**Limitations stated by the authors.** They do not evaluate 13B/70B backbones due to compute limits and do not study information extraction. This was full-text read from the ACL Anthology PDF, not only the abstract.

### 7.2 LAARA — Tripathi et al. (2026 preprint)
Primary source: [arXiv full text](https://arxiv.org/abs/2607.19391)

**This is one of the closest prior-art papers to GARS.** LAARA calculates empirical Fisher estimates from LoRA A/B gradients during training. It separately tracks gradient-squared estimates for A and B with EMA and bias correction, normalizes scores within each projection type, applies logarithmic compression, blends the A/B scores, then maps the resulting layer importance to bounded integer ranks. A vote-to-change mechanism requires the same proposed target rank for multiple consecutive update intervals before a rank change is committed.

A simplified description of the signal is:
\[
F_{\ell,A}\propto\operatorname{EMA}\left(\|\nabla A_\ell\|_F^2\right),\qquad
F_{\ell,B}\propto\operatorname{EMA}\left(\|\nabla B_\ell\|_F^2\right),
\]
followed by per-projection normalization, log compression and weighted blending. The exact paper equations should be transcribed from its PDF if included in the final report; the text extraction used here loses some displayed formula glyphs.

The authors evaluate DeBERTa-v3-base and Llama-3.2-3B on GLUE and MathInstruct/GSM8K. The appendix states three random runs for GLUE. The stated configuration includes initial rank 4, min rank 2, max rank 8, EMA decay 0.97, warmup 200 steps, rank update interval 200, A/B blend 0.5, log compression 10 and vote patience 2. The paper reports better performance than LoRA/AdaLoRA/DyLoRA at matched budgets on its selected tasks.

**Why this matters for GARS.** LAARA already uses training-time LoRA-factor gradients (via a diagonal empirical-Fisher approximation) to produce per-layer rank allocations. A statement that GARS is the first to use live per-layer LoRA gradients for adaptive rank would therefore be false. The potential distinction is narrower: GARS converts a layer-gradient ratio directly into a layer-specific singular-value *energy-retention threshold*, rather than mapping a Fisher score to an integer rank through normalized/blended score rules.

**Caution about the paper's theory.** LAARA proves monotonic rank profiles under stated assumptions on layer Jacobians, bounded weights and activation normalization. These assumptions are stronger than simply saying “Transformer gradient norms always increase with depth.” The paper's empirical trend is model/task/configuration-dependent; do not generalize its monotonicity statement beyond the theorem's assumptions and tested settings. The empirical-Fisher approximation also discards off-diagonal curvature. Its own limitations mention that diagonal-Fisher approximation may underestimate importance and that the rank interval/update interval/A-B blend are not broadly ablated.

**Status.** The reviewed source is an arXiv preprint. Treat its claims as preprint evidence unless a later peer-reviewed venue version is found.

### 7.3 IGU-LoRA — Cui et al. (ICLR 2026)
Primary source: [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html) · [arXiv full text](https://arxiv.org/abs/2603.13792)

IGU-LoRA also directly addresses the reliability of gradient-derived importance. It extends Integrated Gradients to the parameter space: for a parameter w, it integrates the gradient of loss along a path from a zero baseline to the current parameter value. To make this practical, it discretizes the path into N points and samples one integration point per mini-batch, accumulating an epoch-level estimate. The paper gives a trapezoidal discretization error of order O(N^-2) and a sampling error of order O(M^-1/2), under stated smoothness/boundedness assumptions.

It then applies an exponential moving average to the integrated sensitivity and estimates uncertainty by tracking deviations across mini-batches. The final signal is described as an SNR-like ratio between smoothed sensitivity and uncertainty. Singular-component importance combines this score for the associated SVD factors with singular-value magnitude, after which the top-budget components are retained.

The method prunes during training. Rank pruning begins at epoch 2 and ends at epoch 5, at one-fifth-epoch intervals; after pruning, the model is fine-tuned with early stopping. The paper reports five random seeds and median results. Experiments include RoBERTa-large/GLUE, Qwen2.5-0.5B for reasoning, and additional Llama-2-7B, Llama-3-8B and DeepSeek-R1-Distill-Qwen-2.5-7B studies.

**Implication.** GARS should not claim that gradient noise or uncertainty in rank scores is unstudied. IGU-LoRA proposes a path-integrated, uncertainty-aware alternative. A defensible GARS experiment could compare a simple gradient-norm threshold against temporally averaged or uncertainty-aware alternatives, but only if the extra computation is practical and budgets are matched.

**Limitations/assumptions.** The authors acknowledge non-trivial computational overhead and dependence on integration-path choice and uncertainty estimation. The stability theorem assumes an i.i.d./sub-Gaussian local noise model; the appendix explicitly calls this a stylized local-stationarity approximation after burn-in, rather than an exact description of all LLM training dynamics.

### 7.4 LoRA-Perturb — Jin et al. (Research Square preprint, posted 9 November 2026)
Primary source: [Research Square DOI](https://doi.org/10.21203/rs.3.rs-11155625/v1)

This very recent preprint is adjacent prior art for *layer importance measurement*, although it does not implement dynamic per-step rank pruning. It performs a short LoRA warm-up on a small subset of training data, injects Gaussian perturbations into one layer's LoRA pathway, and measures the resulting change in prediction confidence (or task-specific loss/log-likelihood). It reduces measurement cost using random subset downsampling (100–500 examples), one-pass activation caching, and localized forward reuse. It then uses normalized perturbation scores to generate a static warm-start configuration: higher-importance layers receive higher ranks and inversely scaled lower learning rates. The total rank budget is matched to uniform rank 8, with per-layer ranks bounded [4,16].

The paper studies RoBERTa-base/GLUE and Llama-2-7B/GSM8K/MT-Bench. It reports five seeds for main experiments, but an internal gradient-dynamics stability comparison says three runs and explicitly avoids statistical-significance claims. Its SST-2 ablation uses 2,000 training examples and reports a small absolute accuracy difference; authors describe it as preliminary and do not run formal significance tests. It also states that the layer-importance measurement is a one-time warm-start and that dynamic rank reallocation during training is future work.

**Implication.** This paper does not duplicate GARS, but it demonstrates that activation/perturbation-based layer importance is an active 2026 direction. It can inform the activation/sensitivity comparator and underscores the need to distinguish a one-time warm-start rank schedule from live training-time pruning.

**Status.** Research Square preprint, not peer-reviewed in the reviewed record. The posting date shown in the Scite record is 9 November 2026; re-check its version and status before report submission.

### 7.5 LaRA — Zhou et al. (Information Processing & Management, 2026)
Primary source: [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S0306457325004790) · DOI: 10.1016/j.ipm.2025.104538

LaRA is a layer-wise rank allocator for *recovering already pruned LLMs* with LoRA. It profiles layer features (including information, pruning setting and position), uses a learned LSTM-based Rank Allocator and a Scoring Model trained on roughly 1,000 rank-performance pairs; the allocator is optimized with policy-gradient reinforcement learning. It predicts per-layer rank configurations before fine-tuning rather than using live A/B gradient norms during the adapter training loop. This is relevant background for automated layer-wise rank assignment, but it addresses a different setting: rank allocation for pruned backbones being restored.

### 7.6 Updated method-level comparison

| Method | Signal / evidence | When allocation occurs | What it changes | Key distinction from GARS |
|---|---|---|---|---|
| AdaLoRA | Smoothed gradient/weight sensitivity per singular triplet, with uncertainty | During training | Masks singular values under a global budget | Learned SVD-like P/Λ/Q and triplet scores |
| DyLoRA | Randomly sampled nested rank slice | During training; selects rank at inference | Learns a rank-ordered adapter supporting multiple ranks | Rank flexibility, not layer importance |
| SoRA | Learnable gate with L1/proximal-gradient sparsification | During training; physical deletion after | Removes zero-gated components | Sparse gate regularization, not gradient-conditioned energy threshold |
| ALoRA | Validation-set ablation score of each rank | Periodic prune/reallocation cycles | Prunes individual rank units and reallocates budget | Empirical validation influence, not gradient or spectrum energy |
| GoRA | N-batch gradients of frozen pretrained weights times weights | Before adapter training | Assigns per-matrix rank and gradient-based initialization | Base-weight pre-training probe, not live LoRA-factor score |
| Act-LoRA | Layer activation L2 norm | Pre-training probe | Selects top-k layers to receive LoRA | Selects adapter placement, not within-layer ranks |
| FlexLoRA | Matrix-level squared-energy spectral entropy | During training | Prunes and expands ranks under global budget | Spectral entropy as the allocation signal |
| PARA | Global pool of singular values from QR-SVD; target rank or Frobenius-energy policy | Post-training | Physically truncates each adapter's factors at global threshold | Global spectral threshold, no gradient signal |
| IGU-LoRA | Parameter-space integrated gradients with EMA/uncertainty | During training | Scores singular components and retains top budget | Path-integrated SNR-like score rather than a layer-gradient threshold |
| LAARA | EMA empirical-Fisher traces from squared LoRA A/B gradients; normalization, log compression, A/B blend, voting | During training | Maps layer scores to integer ranks and resizes adapters | Very close gradient-based layer-rank prior art |
| LoRA-Perturb | Prediction-confidence/task-loss drop from noise intervention in LoRA pathway | One-time post-warm-up probe | Static rank + learning-rate warm-start | Perturbation-based layer importance, not live gradient/SVD pruning |
| LaRA | Layer features + learned score/allocator | Before fine-tuning pruned models | Predicts layer ranks | Allocator for restoring pruned backbones |

## 8. Updated assessment of GARS novelty

The initial broad formulation—“gradient-aware per-layer adaptive LoRA rank pruning”—is not a sufficiently differentiated novelty claim after reviewing the 2025–2026 literature. In particular:
- GoRA uses gradients for rank allocation before adapter training.
- LAARA uses squared LoRA A/B gradients as empirical-Fisher proxies to allocate layer ranks during training.
- IGU-LoRA uses integrated gradients and uncertainty-aware scores to prune singular components.
- FlexLoRA uses squared-singular-value spectral entropy to drive per-matrix rank pruning/expansion.
- PARA uses QR-SVD and global singular-value/Frobenius-energy thresholding for post-training physical adapter compression.
- SoRA and ALoRA already establish alternative training-time mechanisms for pruning and reallocating rank.
- LoRA-Perturb provides recent adjacent evidence for perturbation-based layer importance and static differentiated rank/learning-rate warm starts.

The only defensible *candidate* distinction currently visible is the particular policy that maps a live or temporally aggregated layer-level LoRA-factor gradient statistic to a per-layer squared-Frobenius-energy retention threshold, together with an empirical analysis of whether a pre-specified spectral-shape statistic moderates that policy's benefit relative to carefully budget-matched spectral baselines.

This is not yet established as novel. It may be an incremental mapping variation compared with LAARA and IGU-LoRA. Before asserting novelty, we need:
1. Inspect the exact LAARA code and compare its layer score/rank update with the GARS proposal formula.
2. Inspect IGU-LoRA's implementation to see whether its singular-component selection is functionally equivalent to thresholding layer-wise energy.
3. Compare FlexLoRA's rank-level policy and PARA's global/local threshold policies.
4. Search for other 2026 work that directly combines layer gradients, singular-spectrum entropy/flatness and energy-retention thresholding.
5. Discuss the refined contribution with the mentor before claiming algorithmic novelty.

If the mechanism is not sufficiently distinct, reposition Phase 1 as a careful empirical investigation of signal reliability and budget-controlled pruning: which importance signals (raw adapter gradients, smoothed/Fisher, integrated gradients, activations, perturbation) best predict the task loss induced by layer-wise truncation, and under what spectral conditions. The proposed GARS threshold can then be a candidate policy evaluated against prior methods, rather than claiming it is the first adaptive-gradient LoRA algorithm.


### 7.7 Sawant — When Gradient Importance Lies (ACL Workshop, October 2026)
Primary publication record: [ACL Anthology](https://aclanthology.org/2026.insights-1.14/) · DOI 10.18653/2026.insights-1.14 · [arXiv preprint](https://arxiv.org/abs/2605.07366)

The accessible abstract and full-text mirror describe a negative result for proportional gradient-based rank allocation under GRPO. The experiment uses Qwen2.5-1.5B-Instruct, GSM8K, 28 Transformer layers, LoRA on seven projection modules per layer, uniform rank 32, and an equal total rank budget for non-uniform configurations. The reported proportional allocation scores 70.0% vs 74.5% for uniform (a 4.5-point decrease); random allocation scores 67.5%.

The paper measures a relatively flat layer-wise gradient-importance profile under GRPO: max/min ratio 2.17×. After non-uniform rank allocation, the ratio widens to 3.00×. Its random-allocation control shows a high association between assigned rank and subsequent gradient-share changes (reported r=0.972), leading the authors to argue that the allocated rank itself affects the later measured gradient importance. Training reward curves are reported as similar, with performance differences appearing at evaluation.

**Interpretation and limits.** This is an RL/GRPO result, not evidence that gradient-guided capacity allocation fails under SFT. The study is explicitly single-model, single-task and single-seed, with 200 test examples; the authors state that confidence intervals overlap and recommend multi-seed replication. It does not provide a direct same-model/same-data SFT-versus-GRPO comparison; its contrast with SFT gradient concentration relies on prior work. The official ACL Anthology publication is peer-reviewed workshop proceedings, but the body text was consulted through an accessible preprint/mirror because the direct ACL PDF text extraction was unavailable in this session. Use its findings as preliminary, context-specific evidence.

**Relevance to GARS.** The paper makes the timing of gradient measurement important. A gradient profile recorded under one rank allocation may not remain valid after changing the allocation. A robust GARS study should therefore record gradient statistics before and after allocation changes, and should compare static pre-probe scores with online/temporally aggregated scores. It also motivates not generalizing a gradient-based policy across training regimes (SFT vs RL) without evidence.


### 7.8 Progressive PEFT — Gebretatios, Ning and Ding (Research Square preprint, 27 May 2026)
Primary record: [Research Square DOI](https://doi.org/10.21203/rs.3.rs-9816617/v1) · full text read through conclusion and appendices

**This is the closest additional prior-art method found in the expanded search.** It performs dynamic layer-wise LoRA rank allocation using smoothed gradient-derived module importance under global resource constraints.

**Importance signal.** For each trainable LoRA parameter p, Progressive PEFT defines a relative-gradient score:
\[
s_p(t)=\frac{\|\nabla_p L\|_2}{\|p\|_2+\epsilon}.
\]
It maintains a per-parameter EMA (\(\beta_1=0.9\)), takes the median of the smoothed parameter scores within each LoRA module, and then maintains a second module-level EMA (\(\beta_2=0.99\)). Importance is updated each training step, with a 50-step warm-up before it is used for rank changes.

**Rank allocator.** The method updates ranks every \(\tau=200\) steps by default and supports:
- Progressive budget schedules: fixed, linear growth or adaptive growth.
- SAP-PEFT: temperature-controlled softmax over the smoothed module scores, mapping probabilities to bounded integer ranks under a nominal fixed rank budget.

The SAP-PEFT rule uses \(r_{min}=2\), \(r_{max}=32\), default temperature 1.0, then floors/clips each layer rank. The paper says it does not perform a final rescaling because any budget discrepancies are negligible. This is an important implementation detail: despite the fixed-budget motivation, integer rounding and clipping can result in deviation from the exact budget; a controlled reproduction should explicitly count the final physical parameter total.

**Restructuring and optimizer.** On rank decrease it directly truncates A rows and B columns; on increase it zero-pads new dimensions; it updates the LoRA scaling factor from \(\alpha/r\) to \(\alpha/r_{new}\). It performs a full AdamW moment reset after each restructuring, retaining the learning rate. It also suppresses rank changes smaller than a threshold \(\delta=1\). This is directly relevant to the optimizer-state issue in the GARS notebooks.

**Experimental design.** Tigrinya sentiment (49,374 train, 4,000 test with 10% of train held out for validation) and TiDG seven-class news (6,453 train, 807 validation, 807 test). Models: TiroBERTa (124M), Afro-XLM-R (278M), GPT-2 Small (124M). It compares LoRA, DoRA, AdaLoRA, fixed/progressive variants and SAP-PEFT. Five seeds are used; statistical reporting includes bootstrap intervals, Wilcoxon, Holm-Bonferroni and McNemar. It reports a multi-objective efficiency suite (parameters, latency, memory, throughput, energy, CO2). Experiments run in the reported paper on a single RTX 4090 Laptop 16GB, Python 3.12.8, PyTorch 2.9.0 (the exact laptop power draw/carbon estimates require care).

**Results as reported by the authors.** SAP-PEFT is competitive with full fine-tuning on the two encoder architectures and yields very large trainable-parameter reductions; it performs less well on the cross-lingual GPT-2 news task. The paper reports, for example, up to 286×/336× compression in certain model/task settings, and inference latency in the 4.1–5.5 ms range. The authors also report only ~11% post-hoc power to detect a medium effect at five seeds, so non-significant comparisons do not demonstrate equivalence. Treat the outcome as preliminary evidence from this specific low-resource Tigrinya setting, not universal performance evidence.

**Important literature-review correction.** The preprint argues that AdaLoRA does not coordinate a strict global parameter budget across layers. That characterization appears inaccurate: AdaLoRA explicitly introduces a global budget scheduler over singular triplets. We should not repeat Progressive PEFT's critique as an established fact; compare both papers' budget variables and exact post-pruning physical parameter counts instead.

**Relation to GARS.** There is substantial overlap at the problem level and in the training-control approach:
- Both use gradients from LoRA adapter parameters during fine-tuning.
- Both form layer/module-level importance signals.
- Both change adapter ranks during training.
- Both require a budget policy and address optimizer state after rank resizing.
- Both target per-layer adaptation capacity under resource constraints.

The key mechanism-level distinction visible in the papers is:
- **Progressive PEFT:** parameter-wise relative gradient magnitude \(\|\nabla p\|/\|p\|\), per-parameter EMA, module median, module EMA, then global softmax/proportional score-to-integer-rank allocation; direct prefix truncation/zero-padding (no energy-preserving SVD policy).
- **GARS:** the proposal's layer-gradient statistic is normalized across layers and transformed into a layer-specific singular-value energy-retention threshold; retained rank comes from a truncated SVD of the learned LoRA update (the exploratory notebooks use reduced QR-SVD).

This is a real technical difference, but it may still be judged an incremental allocation-rule variation. The Phase 1 report should not claim “the first gradient-aware per-layer rank scheduler.” The potentially defensible contribution is to measure whether using a gradient-conditioned **energy-retention constraint** yields a better quality/physical-size trade-off than gradient-score-to-rank mapping, direct prefix truncation, and global spectral thresholding, and whether spectral concentration predicts when these strategies differ.

**Status:** Research Square preprint; DOI verified, but not peer-reviewed in the reviewed record. The complete paper text was read. The authors' method and claims have been captured above.

### 7.9 Aletheia — Saket (arXiv preprint, 4 April 2026)
Primary source: [arXiv paper](https://arxiv.org/abs/2604.15351) · [public method card](https://huggingface.co/aletheiaprotocol/aletheia-lora)

Aletheia is adjacent prior art for gradient-guided **layer selection**, rather than rank pruning within every layer. It performs a lightweight probe of five forward/backward batches, computes per-layer backbone gradient norms, selects the top 50% of layers, and then applies LoRA only to those layers. The paper also uses asymmetric ranks for attention and MLP modules (e.g. attention r=16 and MLP r=64 in one recipe). It does not update layer ranks dynamically during the main adapter training.

The paper reports cross-architecture tests across 14 successful models from eight families, 0.5B–72B, including an MoE model; three seeds are used. Its reported speedups are 15–28% (mean 23.1%), with quality checks including MMLU and (in a second campaign) GSM8K/HumanEval. The authors report one Pythia-1.4B GPT-NeoX attempt that failed under fp16 NaNs for both Aletheia and standard LoRA. The public method card explicitly cautions that the top-50% selection rule is a heuristic and reported speedups depend on the tested hardware/software.

**Relation to GARS.** Aletheia:
- Uses gradients of backbone layer parameters in a pre-training probe.
- Makes a binary decision about whether a layer receives adapters.
- Uses asymmetric attention/MLP rank configurations but not dynamic within-layer SVD pruning.

It therefore does not duplicate GARS's proposed per-layer gradient-conditioned energy retention, but it adds to the prior art for gradient-driven layer selection and reinforces the need to distinguish *where to attach LoRA*, *which integer ranks to allocate*, and *which singular directions to retain*.

**Status:** arXiv preprint; no peer-reviewed publication record found in the reviewed search. Treat cross-family performance figures as author-reported preprint results.

## 9. Revised novelty and scope assessment after Progressive PEFT

The broader proposal phrase “gradient-aware per-layer adaptive rank scheduling” is already directly represented by Progressive PEFT and LAARA, in addition to GoRA and IGU-LoRA. It is therefore not safe to claim the general concept as novel.

The GARS policy still appears distinguishable at the mathematical/mechanism level:
1. A normalized layer-gradient statistic determines a layer-specific **squared-Frobenius-energy retention threshold**;
2. A per-adapter SVD determines the minimum retained rank satisfying that threshold;
3. The final budget is calibrated across the physical parameter cost of differently shaped adapter matrices.

Progressive PEFT maps a smoothed relative-gradient module score to an integer rank and restructures by direct prefix truncation/zero-padding; LAARA maps an EMA empirical-Fisher score to an integer rank; IGU-LoRA ranks singular components using integrated-gradient sensitivity and uncertainty; PARA uses a global spectral threshold. This difference is technically concrete, but it is not proof of a strong novelty claim.

The strongest Phase 1 contribution may therefore be a **controlled comparative investigation**:
- Does gradient-conditioned per-layer energy retention outperform fixed/local energy retention and PARA-style global spectral thresholding under exactly matched physical adapter budgets?
- Does it offer any benefit beyond a gradient-score-to-rank method such as Progressive PEFT/LAARA?
- Does singular-spectrum concentration predict when each signal/allocation policy is reliable?
- Do physical export and inference measurements confirm the claimed efficiency gains?

This reframing keeps the proposal's core question alive while avoiding an indefensible first-method claim. It should be discussed with the mentor before any formal change to title, objectives or methodology.


## 10. Code-level audit of released implementations (12 November 2026)

The paper and the public code must be treated as separate evidence sources. This audit inspects the currently visible default branches; it is not an independent reproduction.

### IGU-LoRA public repository
Repository: https://github.com/Cuixuans/igulora

Files inspected: iglora/rank_allocator.py, iglora/lora_module.py, iglora/run_pruning_sft.py, iglora/run_final_sft.py, and README.

The visible RankAllocator class docstring explicitly references AdaLoRA. Its per-parameter score is abs(p * p.grad), with one EMA for the sensitivity and a second EMA of the deviation from the smoothed sensitivity. The component-level score aggregates A-row and B-column contributions with the learned diagonal singular-value parameter E. A global threshold is selected to meet a target total-rank schedule. The pruning-training script calls the allocator after optimizer.step(), writes a ranks_to_mask.json file and sets low-importance E entries to zero.

The repository is a two-stage workflow. The final SFT script loads ranks_to_mask.json, writes zeros into the lora_mask vectors, and multiplies E by that mask in the forward pass. It saves all state-dict entries containing lora; the A/B/E tensor dimensions are retained. Thus the observed code masks rank coordinates but does not create physically shortened A/B factor tensors.

**Paper-code caveat:** the inspected public implementation contains no visible parameter-path integration, quadrature nodes or accumulated Integrated Gradients calculation. It uses a p * grad EMA/uncertainty score. The repository has only two February 26, 2026 commits, while the paper appeared on arXiv in March 2026 and was published at ICLR 2026. The safest interpretation is that this is an early or incomplete code snapshot that may not reflect the final paper method. Do not characterize the paper itself as AdaLoRA; state only that the inspected public branch does not demonstrate the published IG mechanism. The detailed code audit is recorded in implementation_audit_igu_aletheia.md.

### Aletheia public repository
Repository: https://github.com/aletheiaprotocol-ai/aletheia-lora

Files inspected: src/aletheia_lora/probe.py, selection.py, config.py, and README.

The implementation samples five batches by default (seeded, shuffled DataLoader with batch size 1) and processes transformer layers in chunks of eight. For each chunk it enables gradients only for parameters in that chunk; for every sampled batch it performs a causal-language-model forward/backward and adds each parameter tensor's gradient norm to the layer's score. It then zeroes gradients, continues to the next chunk, and ultimately sets all floating model parameters to requires_grad=False. The selector keeps the top percentage of layer scores (default 50%); the PEFT config applies LoRA only to selected layers, with optional asymmetric attention/MLP ranks.

Two code-level points should be noted:
1. The probe sets model.train(), so dropout and other training-mode stochastic behavior may affect the probe scores.
2. probe_batches=5 means five examples/batches per layer chunk. For a 36-layer model and chunk size 8, there are five chunks, so the probe can execute up to 25 full-model forward/backward passes, not five total passes.

The score is an accumulation of per-parameter tensor gradient norms, not an explicitly concatenated layer Frobenius norm. Aletheia performs a one-time top-k layer-selection decision before adapter training; it does not dynamically change each selected layer's rank or truncate the singular spectrum of each adapter.

### LAARA code availability
The LAARA arXiv record links a public anonymous code snapshot at https://anonymous.4open.science/r/LAARA-D305/LAARA.py. The host did not permit retrieval through the available browser connector in this session. The full LAARA paper was read, but the code was not independently audited. Do not write that LAARA's implementation has been verified until the file is accessible and inspected.

### Progressive PEFT code availability
The Research Square full text says code/configurations/adapters/analysis scripts will be released publicly, but the reviewed full text and searches did not expose a repository URL. Treat the method as paper-only evidence for now, and do not claim code-level reproduction or verification.

### Implementation-review lesson for GARS
The public implementation of GARS must separately demonstrate:
- that the gradient statistic in code matches the equation in the report;
- that the energy ratio in code is squared singular-value energy if the report calls it Frobenius energy;
- that the layer-wise retained rank is computed correctly;
- that rank resizing/optimizer state handling follows the described timing;
- that exported tensors are physically smaller, not simply masked;
- that the saved adapter reloads and gives equivalent outputs to the corresponding truncated dense update.
