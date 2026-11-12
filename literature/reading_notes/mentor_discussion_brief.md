# Mentor discussion brief — GARS Phase 1 prior-art update

**Prepared:** 12 November 2026  
**Project:** Gradient-Aware Per-Layer Adaptive Rank Scheduling (GARS): When and How Gradient Dynamics Should Guide LoRA Pruning in Large Language Models

## 1. Why a discussion is needed

The focused literature review has identified several 2025–2026 methods that overlap with the broad wording “gradient-aware per-layer adaptive LoRA rank scheduling.” The approved proposal's research question remains relevant, but the novelty statement should be narrowed and supported with a controlled comparative study.

The closest prior art found is:
- **Progressive PEFT** (Research Square preprint, May 2026): online per-parameter relative gradient scores, per-parameter EMA, module median, second module EMA, dynamic layer rank allocation under global budget, direct factor truncation/zero-padding, and AdamW moment reset.
- **LAARA** (arXiv preprint, July 2026): training-time empirical-Fisher estimates from LoRA A/B gradients, projection normalization, log compression, score blending, voting and dynamic layer ranks.
- **IGU-LoRA** (ICLR 2026): parameter-space Integrated Gradients, uncertainty-aware EMA/SNR scoring and singular-component pruning.
- **FlexLoRA** (ICLR 2026): squared-singular-value spectral entropy for rank pruning and expansion.
- **PARA** (arXiv preprint, April 2026): post-hoc reduced QR-SVD, global singular-value threshold/Frobenius-energy policy, physical factor reconstruction.
- **GoRA** (NeurIPS 2025): pre-training gradient probe of frozen pretrained weights for rank allocation and gradient-based initialization.
- **Act-LoRA** (Information, 2026) and **Aletheia** (arXiv preprint, 2026): gradient/activation-driven selection of which layers receive adapters, rather than dynamic within-layer rank pruning.

## 2. The GARS mechanism distinction to test

The proposal uses a normalized layer-gradient statistic to set a layer-specific energy-retention threshold:

\[
\tau_\ell=\operatorname{clamp}
\left(\tau_{base}
\left(\frac{g_\ell}{\bar g}\right)^\beta,
\tau_{min},\tau_{max}\right).
\]

For Frobenius-energy preservation, retain the smallest rank k satisfying:

\[
\frac{\sum_{i=1}^{k}\sigma_{\ell,i}^2}
{\sum_{i=1}^{r}\sigma_{\ell,i}^2}
\geq \tau_\ell.
\]

This differs from Progressive PEFT/LAARA's score-to-integer-rank mapping and PARA's global singular-value threshold. It is an algorithmic policy distinction; it is not yet evidence of a significant research contribution.

Important definitions to freeze:
- Which gradient statistic is primary: sum of A/B Frobenius norms, joint norm, or a squared-gradient/Fisher proxy?
- Is it measured once per step, at pruning events, or through a moving EMA window?
- Is the energy criterion squared singular-value Frobenius energy (not sum of singular values)?
- How is the base threshold calibrated to a fixed *physical* adapter parameter budget?

## 3. Candidate revised research question

> Under an exactly matched physical adapter-parameter budget, does gradient-conditioned per-layer squared-energy retention preserve downstream performance and deployment efficiency better than uniform local energy pruning, PARA-style global spectral thresholding and gradient-score-to-rank policies such as Progressive PEFT/LAARA; and does the layer's singular-spectrum concentration predict when these policies differ?

A potentially stronger scientific contribution is the **signal-reliability analysis**: compare gradient, Fisher/EMA, integrated-gradient, activation and perturbation scores against an independent validation ablation measure, and test whether spectral concentration moderates the signal's predictive value. GARS can be the candidate energy-retention policy within that study.

## 4. Proposed Phase 1 experiment scope

For the first controlled experiment, train one high-rank LoRA parent model and evaluate multiple post-hoc allocation/pruning rules on the *same checkpoint*:
1. Uncompressed LoRA parent.
2. Uniform per-layer squared-energy pruning.
3. Global PARA-style spectral threshold.
4. GARS per-layer gradient-conditioned energy threshold.
5. Simplified Progressive PEFT/LAARA score-to-rank comparator, with its smoothing and mapping stated.

Match the final physical parameter budget, not just average rank:

\[
P_{total}=\sum_\ell k_\ell(d_{out,\ell}+d_{in,\ell}).
\]

For GARS, calibrate \(\tau_{base}\) by a deterministic search. Use at least three parent training seeds if feasible. Report accuracy/F1, physical adapter parameters, tensor shapes, adapter bytes, export/load validity, and latency (merged and unmerged as separate settings).

Then perform a signal-reliability analysis:
- Compare gradient and activation scores with validation-loss increase when a layer's adapter is ablated.
- Measure score/rank-order stability across seeds.
- Calculate squared-energy spectral entropy per layer and test whether it predicts score reliability or the GARS-versus-baseline performance difference.

Keep this separate from the dynamic in-training pruning experiment, which has additional optimizer-state and convergence confounds.

## 5. Questions for the mentor

1. Is it acceptable to position Phase 1 as a comparative empirical study of gradient-signal reliability and energy-preserving rank pruning, rather than claim a broadly novel gradient-driven rank-allocation framework?
2. Should the central deliverable be the GARS energy-threshold policy plus a matched-budget comparison, or should signal reliability/spectral concentration become the primary research contribution?
3. Given the remaining Phase 1 timeline, is one model/task with rigorous matched-budget comparisons and a 3-seed pilot preferable to the originally proposed three scales/three domains?
4. Should the title remain unchanged for Phase 1 while the problem statement/contribution is narrowed, pending mentor guidance?

## 6. Evidence and project records

Detailed primary-source notes and exact paper comparisons:
- literature/reading_notes/day2_primary_source_notes.md
- literature/reading_notes/day2_research_gap_and_next_steps.md
- literature/day2_seed_matrix.csv
- literature/reading_notes/implementation_audit_igu_aletheia.md
- literature/search_log_day2.md

No change to the approved project title/objectives is proposed by this brief. Any material scope change should be agreed with the mentor and recorded through the IITM process.
