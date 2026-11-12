# GARS Phase 1 — research gap, experiment logic and immediate next steps

**Working date:** 12 November 2026  
**Status:** Literature synthesis and candidate research-gap assessment. This is not a final claim of novelty. The student and mentor should agree on the final contribution before the Phase 1 report is frozen.

## 1. What the literature now tells us

The project sits at the intersection of several established strands. A broad claim such as “we introduce gradient-aware per-layer adaptive LoRA pruning” is not defensible without further narrowing.

| Research strand | Established examples | What is already known |
|---|---|---|
| Fixed-rank low-rank adaptation | LoRA (ICLR 2022) | Parameterize a frozen-weight update using two low-rank factors; rank is fixed by the user |
| Training-time sensitivity / rank adaptation | AdaLoRA (ICLR 2023), SoRA (EMNLP 2023), ALoRA (NAACL 2024) | Use sensitivity, gates, proximal sparsity or validation ablation to prune and redistribute rank |
| Dynamic multi-rank adapters | DyLoRA (EACL 2023) | Train one rank-ordered adapter that supports multiple inference-time ranks |
| Gradient-driven allocation / initialization | GoRA (NeurIPS 2025) | Use accumulated gradients of pretrained weights before adapter training to choose ranks and initialize factors |
| Activation-based layer selection | Act-LoRA (Information, 2026) | Probe activation norms and select layers for LoRA insertion; compare activation and gradient rankings |
| Spectral-entropy rank reallocation | FlexLoRA (ICLR 2026) | Use normalized squared-singular-value entropy to prune and expand ranks under a global budget |
| Post-hoc spectral compression | PARA (2026 arXiv preprint) | Use reduced QR-SVD of ordinary LoRA factors, a global singular-value threshold or global Frobenius-energy policy, and physically truncated export |
| Integrated-gradient / uncertainty rank scoring | IGU-LoRA (ICLR 2026) | Use parameter-space Integrated Gradients plus EMA/uncertainty (SNR-like) scores for singular-component pruning |
| Fisher-gradient layer-rank allocation | LAARA (2026 arXiv preprint) | Use EMA squared gradients of LoRA A/B as empirical-Fisher proxies, normalize/blend scores and map them to layer ranks during training |
| Perturbation-based layer importance | LoRA-Perturb (Research Square preprint, 9 Nov 2026) | Warm-up LoRA, perturb one layer's LoRA pathway, measure prediction/task-objective change, then use a static differentiated rank/learning-rate warm start |
| Layer-rank prediction for pruned-backbone recovery | LaRA (Information Processing & Management, 2026) | Predict per-layer ranks for restoring already-pruned LLMs via a learned allocator and score model |

The closest works to GARS are **LAARA, IGU-LoRA, FlexLoRA and PARA**. ALoRA, SoRA and DyLoRA are also important comparisons, but their rank-selection mechanisms differ more substantially.

## 2. The current GARS distinction — precisely stated

The approved proposal maps a layer-gradient ratio to a layer-specific energy-retention threshold:

\[
\tau_\ell =
\operatorname{clamp}\left(
\tau_{base}
\left(\frac{g_\ell}{\bar g}\right)^\beta,
\tau_{min},\tau_{max}
\right).
\]

For Frobenius-energy preservation, the retained rank of an adapter update with singular values \(\sigma_{\ell,i}\) would be the smallest \(k_\ell\) satisfying:

\[
\frac{\sum_{i=1}^{k_\ell}\sigma_{\ell,i}^{2}}
{\sum_{i=1}^{r}\sigma_{\ell,i}^{2}}
\geq \tau_\ell,
\qquad k_\ell\geq k_{min}.
\]

This differs in form from LAARA's direct Fisher-score-to-integer-rank mapping. However, both use LoRA A/B gradients to allocate layer capacity during training. Therefore, GARS may be a new *allocation policy* rather than a new general principle of gradient-guided rank allocation. That is a narrower and more modest claim.

### The gradient signal must be operationally defined

The notebooks currently use:

\[
g_\ell=\|\nabla A_\ell\|_F+\|\nabla B_\ell\|_F.
\]

This is not the same as a joint Frobenius norm:

\[
g^{joint}_\ell=
\sqrt{\|\nabla A_\ell\|_F^2+\|\nabla B_\ell\|_F^2}.
\]

It is also not the same as an empirical-Fisher trace proxy such as:

\[
F_\ell\approx
\operatorname{EMA}\left(
\|\nabla A_\ell\|_F^2+
\|\nabla B_\ell\|_F^2
\right).
\]

Use one defined signal in the primary method and treat alternatives (sum of norms, joint norm, squared-gradient/Fisher proxy, EMA window) as ablations or leave them for future work. A single noisy gradient at a pruning event is not adequate evidence of a stable layer-importance signal.

### The energy metric must be defined consistently

The current exploratory notebooks use a sum-singular-value ratio in at least some cells:
\[
\frac{\sum_{i\leq k}\sigma_i}{\sum_i\sigma_i}.
\]
That is not squared Frobenius energy:
\[
\frac{\sum_{i\leq k}\sigma_i^2}{\sum_i\sigma_i^2}.
\]
The proposal's use of “energy preservation” points toward the squared definition. Freeze the definition before controlled experiments and update the notebooks/methodology accordingly; do not mix the two under the same name.

## 3. A critical fairness issue: GARS does not automatically enforce a fixed total budget

Applying a separate threshold to each layer gives different retained ranks, but it does not guarantee that the sum of physical adapter parameters is the same across methods. This is a serious issue for the planned matched-accuracy / matched-budget comparison.

For a LoRA update on matrix \(W_\ell\in\mathbb R^{d_{out,\ell}\times d_{in,\ell}}\), a rank-\(k_\ell\) adapter has:

\[
P_\ell=k_\ell(d_{out,\ell}+d_{in,\ell})
\]

trainable factor parameters (excluding optional bias/task parameters). Therefore the total adapter budget is:

\[
P_{total}=\sum_\ell k_\ell(d_{out,\ell}+d_{in,\ell}).
\]

Do not match only the arithmetic mean rank if target matrices have different dimensions.

**Proposed budget calibration:** choose the base threshold \(\tau_{base}\) by a deterministic one-dimensional search so that the resulting sum of physical factor parameters is as close as possible to the target budget. Keep the rank floor in force and document cases where the floor prevents an exact match. This makes GARS comparable to PARA's global threshold and to a uniform threshold at the same physical parameter count.

## 4. Candidate research question and hypotheses

### Candidate primary research question

> Under a fixed physical LoRA-adapter parameter budget, how does a per-layer energy-retention policy conditioned on LoRA-factor gradient statistics compare with uniform local and global spectral pruning, and does a layer's singular-spectrum concentration predict the relative benefit of gradient conditioning?

This question is deliberately narrower than “is gradient rank allocation new?” It may still need revision after code-level comparison with LAARA and IGU-LoRA.

### Testable hypotheses

- **H1 — Allocation signal:** Under a matched physical parameter budget, GARS may preserve downstream performance better than a uniform local energy-retention threshold or a global PARA-style spectral threshold.
- **H2 — Signal reliability:** The correlation between a gradient-derived layer score and an independent estimate of layer importance may depend on the layer's squared-energy spectral entropy/flatness.
- **H3 — Export validity:** Physical rank truncation should reduce adapter tensor parameters and serialized adapter bytes, and the exported adapter must load and run correctly.
- **H4 — Stability:** Temporally aggregated gradient scores may produce more stable per-layer ranks than instantaneous event gradients. This needs an explicit measurement (rank changes, score variance, or rank-order correlation), not a narrative claim.

These are hypotheses, not expected results. A null or negative result is scientifically useful if the protocol is controlled.

## 5. A manageable comparison plan

Do not try to implement every method in the literature in ten days. Use the nearest mechanisms and a defensible subset.

### Core controlled pruning comparison

All methods start from the **same high-rank trained LoRA checkpoint**, same model, data split, target modules and seed. Each compressed adapter is calibrated to the same physical parameter budget.

1. **Uncompressed LoRA parent:** high-rank trained adapter; reference performance and parameter count.
2. **Uniform local energy pruning:** same squared-energy retention threshold for every layer; calibrate to target parameter budget.
3. **Global spectral threshold (PARA-style):** collect all adapter singular values and keep a global threshold/target rank; use QR-SVD and physical export.
4. **GARS:** use recorded layer gradient statistics to set per-layer energy thresholds; calibrate the base threshold to the same target physical parameter budget.
5. **Activation-guided comparator (optional):** derive layer scores from activation norms, then use the same per-layer budget-mapping procedure. Be explicit that this isolates the signal from Act-LoRA's top-k layer-selection design.
6. **LAARA/Fisher comparator (optional but high value):** translate EMA squared-gradient/Fisher scores into layer ranks with the same total parameter budget. This tests whether GARS's threshold mapping adds value beyond a direct Fisher-to-rank mapping.

This same-checkpoint comparison isolates allocation/pruning policy from training trajectory. It should be the first rigorous experiment because it is easier to interpret.

### Dynamic training comparison (secondary)

Only after the post-hoc comparison works, test the actual in-training GARS schedule against one training-time comparator (AdaLoRA or LAARA if a reliable implementation is available). Dynamic pruning changes optimization, optimizer states and factor dimensions, so it is a separate experiment rather than being conflated with post-hoc export.

### Required controls

- Fixed model revision, target modules and base checkpoint.
- Identical train/validation/test split; no validation/test data used for gradient probing unless the method explicitly requires validation (e.g., ALoRA).
- Same parent training seed for all post-hoc pruning policies; ideally at least 3 independent parent seeds.
- Same final physical adapter-parameter budget (not merely same nominal rank).
- Same tokenizer, sequence length, batch size, optimizer, number of update steps and precision for parent training.
- Report the uncompressed parent performance to quantify pruning damage.
- For each method, report seed-level score, mean/std, physical parameter count, tensor dimensions, adapter bytes, and export/load validation.

## 6. Testing spectral flatness as a moderator, not as a new rank score

FlexLoRA already uses squared-energy spectral entropy as a rank-allocation signal. GARS should not claim that spectral entropy is new.

A possible use is to test whether spectral shape predicts *when* the gradient-conditioned policy helps or fails.

For each layer after parent training, define:

\[
p_{\ell,i}=\frac{\sigma_{\ell,i}^2}{\sum_j\sigma_{\ell,j}^2},
\qquad
H_\ell=-\frac{1}{\log r_\ell}
\sum_i p_{\ell,i}\log(p_{\ell,i}+\epsilon).
\]

Interpretation: H near 0 means squared energy is concentrated in a few singular directions; H near 1 means energy is more evenly spread. Keep this distinct from Roy–Vetterli effective rank, which uses normalized singular values rather than squared singular-value energy.

A concrete reliability diagnostic is to compare gradient-based layer importance with an independent validation ablation score:
- Score layer \(\ell\) by the validation-loss increase when its LoRA update is removed (or when its trailing directions are truncated).
- Compare this ablation score with (a) instantaneous gradient norm, (b) temporally averaged squared gradient/Fisher score, and (c) activation norm.
- Test whether correlation/rank-order agreement with the ablation score changes with \(H_\ell\).

This directly tests signal reliability. A separate model/task-level analysis can then ask whether average or dispersion of layer entropy predicts the performance difference between GARS and spectral baselines. Avoid an underpowered regression with only a few model/task combinations.

## 7. SVD implementation and optimizer-state safeguards

- Use reduced QR-SVD to decompose LoRA factors without materializing the full \(d_{out}\times d_{in}\) update. This is established prior art (PARA); cite it and treat it as an implementation choice.
- Reconstruct the retained singular components in factor form and verify \(B'A'\) matches the truncated \(\Delta W\) numerically.
- Do not treat zeroed but unchanged rank tensors as physical parameter reduction.
- Save truly smaller factor tensors for each layer and confirm a PEFT-compatible reload plus forward-pass equivalence.
- If pruning during training, perform it at a well-defined optimizer boundary (after optimizer.step and before the next forward/backward cycle), clear stale gradients, and migrate/reset only the affected optimizer-state tensors. Do not silently clear the optimizer state for every parameter.
- Record the chosen policy for newly introduced/reconstructed factor coordinates and Adam moments. A reset of impacted adapter moments may be an acceptable pilot design if stated and tested; an unexamined stale-state carryover is not.

## 8. Recommended scope for the remaining Phase 1 sprint

### Must-have before the Phase 1 report
1. Fully specified GARS gradient statistic, squared-energy retention rule, rank floor and parameter-budget calibration.
2. One working reduced QR-SVD unit-tested against direct dense SVD on small matrices.
3. Physically truncated per-layer export, PEFT reload and forward-output equivalence test.
4. One model + one GLUE task with at least a 3-seed pilot if compute permits.
5. Fixed LoRA parent, uniform local energy pruning, global PARA-style threshold and GARS compared at matched physical budgets.
6. An honest initial spectral analysis, even if it does not support H2.
7. A careful literature gap paragraph that explicitly acknowledges LAARA, IGU-LoRA, FlexLoRA, PARA, GoRA, Act-LoRA, ALoRA and the SoRA/DyLoRA/AdaLoRA family.

### Stretch goals
- Activation-based importance using the same layer-budget mapper.
- LAARA/Fisher-style score-to-rank comparator.
- DyLoRA baseline required by the original proposal, if implementation time permits.
- Second task/domain (MedQuAD) after train/validation splitting is corrected.
- Latency tests on exported adapters; distinguish merged from unmerged deployment.

Do not promise three model scales or three domains in Phase 1 unless the runs are actually completed. Phase 1 can report a rigorous pilot and a feasible remaining plan; do not convert proposed targets into achieved results.

## 9. Immediate next steps

**For the literature work**
- Student: finish LoRA, AdaLoRA and GoRA reading notes in your own words; then read FlexLoRA, PARA and LAARA/IGU-LoRA.
- Joint: resolve the exact GARS difference against LAARA (A/B gradients → rank) and PARA (QR-SVD → global threshold) using code as well as paper text.
- Mentor discussion: present the narrowed candidate question and ask whether an empirical signal-reliability study, with a threshold-based GARS policy as one candidate, is acceptable as the Phase 1 contribution.

**For implementation**
- Freeze the first controlled benchmark (recommend the existing DeBERTa-v3-base/RTE or GPT-2/SST-2 setup; choose one and keep the other as a later replication).
- Add unit tests for rank selection, energy retention, parameter-budget calibration, reduced QR-SVD, optimizer-boundary pruning, physical export and load.
- Do not begin scaling to Gemma/Llama until the small-model method passes these tests.

This is the current evidence-based working plan. The final research gap and scope should be approved by the mentor after reviewing the closest prior art.


## 10. Major prior-art update: Progressive PEFT and Aletheia

The expanded review has now read the full text of two further 2026 papers that materially affect positioning.

### Progressive PEFT (Research Square preprint, May 2026)
This is the closest conceptual overlap located so far. It:
- Computes a relative gradient score for each LoRA parameter: \(\|\nabla_pL\|/(\|p\|+\epsilon)\).
- Applies a per-parameter EMA, module-level median aggregation, and a second EMA at module level.
- Warm-starts for 50 steps and by default updates ranks every 200 steps.
- Allocates layer ranks under a fixed/global budget using softmax or proportional score mappings.
- Directly truncates A/B factors when rank decreases, zero-pads when it increases, and resets AdamW moments after each restructuring.
- Evaluates low-resource Tigrinya sentiment and news classification using TiroBERTa, Afro-XLM-R and GPT-2 Small, five seeds and a multi-objective efficiency suite.

This overlaps with GARS at the level of online gradient-based layer importance, dynamic per-layer rank changes, global budget coordination and optimizer-state reset. The main method-level distinction is the *mapping and pruning operation*: Progressive PEFT maps a smoothed relative-gradient module score to an integer rank and restructures by direct prefix truncation/zero-padding; GARS maps a normalized layer gradient statistic to a squared-energy retention threshold and derives retained rank through SVD of the LoRA update.

This is a substantial overlap, not something to omit. It means we should not claim that GARS is the first gradient-driven per-layer rank scheduler. The proposed energy-threshold policy may still be a different algorithmic choice, but we need to show that it provides more than a cosmetic reparameterization of an existing score-to-rank policy.

Two methodological cautions when discussing Progressive PEFT:
- The preprint's statement that AdaLoRA lacks a global total-budget scheduler is not accurate; AdaLoRA explicitly uses a global budget scheduler over singular triplets. Compare the exact schedules and realized physical parameter counts rather than repeat the criticism.
- SAP-PEFT floors/clips per-layer ranks but states that it does not perform a final rescaling because budget discrepancies are negligible. A reproduction should measure actual total physical parameters; do not assume exact budget preservation merely from the method label.

### Aletheia (arXiv preprint, April 2026)
Aletheia performs a five-batch gradient probe over backbone layer parameters, selects the top 50% of layers, then applies LoRA only in those layers with asymmetric attention/MLP ranks. It is a one-time pre-training **layer-selection** mechanism, not a dynamic rank-pruning method and not within-layer singular-value truncation.

It is therefore adjacent prior art for gradient-based layer selection, but a different technical problem from GARS. Its paper reports a broad cross-architecture evaluation and 15–28% training speedup in its tested environment; treat those as author-reported preprint results, not universal guarantees.

## 11. Revised research-gap position

The evidence now supports three levels of claims:

**Claims we must not make**
- First gradient-aware rank allocation for LoRA.
- First dynamic per-layer rank scheduler.
- First use of LoRA A/B gradients for layer importance.
- First use of squared spectral entropy or energy-based rank allocation.
- First QR-SVD implementation for compact LoRA exports.
- First structural rank pruning or physical adapter truncation.

**A narrower candidate mechanism distinction**
GARS's proposed score-to-threshold map is different from Progressive PEFT/LAARA's score-to-integer-rank mapping:
1. Calculate a per-layer gradient statistic from the LoRA A/B factors.
2. Normalize the statistics across layers.
3. Convert each layer's normalized statistic into its own squared-Frobenius-energy retention threshold.
4. Use reduced QR-SVD of the learned LoRA update to select the minimum rank satisfying that threshold (with a rank floor).
5. Calibrate the base threshold to a fixed total *physical adapter parameter count*.

This differs from Progressive PEFT (relative gradient per parameter → EMA/median/EMA → softmax rank; direct factor truncation) and LAARA (squared A/B gradient empirical Fisher → normalization/log/blend/vote → integer rank). It also differs from PARA (global singular-value threshold) and FlexLoRA (spectral-entropy-based prune/expand).

**However, a method-level difference does not by itself establish research significance.** The Phase 1 contribution should be framed as an empirical question unless a clear advantage can be demonstrated:

> At an exactly matched physical adapter-parameter budget, does gradient-conditioned per-layer squared-energy retention preserve task performance and deployment efficiency better than uniform local energy pruning, PARA-style global spectral thresholding, and gradient-score-to-rank policies such as Progressive PEFT/LAARA; and does a layer's spectral concentration predict when these policies differ?

The more fundamental, potentially stronger research contribution is a **signal-reliability comparison**: how well do different layer-importance signals predict the performance loss caused by removing or truncating that layer's adapter, and how does this relationship change with the layer's singular-spectrum shape? Candidate signals include raw A/B gradient norms, relative gradient/parameter scores, EMA/Fisher statistics, integrated-gradient/SNR scores, activation norms, and perturbation-based sensitivity. GARS's energy-threshold rule can be evaluated as one policy resulting from that investigation.

This framing avoids overstating novelty and is better aligned with the title's question “When and How Gradient Dynamics Should Guide LoRA Pruning.”

## 12. Updated next steps

### Immediate literature work
1. Finish personal reading notes for LoRA, AdaLoRA, GoRA, FlexLoRA, PARA, LAARA and Progressive PEFT. For each, record one paragraph in your own words: its signal, timing, rank-update rule, budget, evidence and limitation.
2. Read the negative-results GRPO paper from the ACL/author PDF if possible; we have verified its official metadata and read a full-text mirror, but did not obtain the official PDF's searchable full text in this session.
3. Decide with the mentor whether the Phase 1 project should present GARS as a candidate energy-threshold algorithm within a comparative study, or make signal reliability the primary research contribution.

### Implementation decision — freeze before more experiments
Use a *single parent checkpoint* for the first allocation-policy experiment. Record gradients and activations from the same training run. Compare:
- uncompressed LoRA parent;
- uniform per-layer squared-energy pruning;
- global PARA-style singular-value threshold;
- GARS per-layer gradient-conditioned energy threshold;
- Progressive PEFT/LAARA-style gradient-score-to-rank allocation (a simplified reproducible comparator, with its exact score and smoothing stated).

All variants must land at the same total physical adapter parameter count:
\[
P_{total}=\sum_\ell k_\ell(d_{out,\ell}+d_{in,\ell}).
\]
Matching average rank alone is insufficient when target matrices have different dimensions. For GARS, determine \(\tau_{base}\) with a deterministic search so that the resulting total physical parameters match the target budget as closely as the rank floor permits.

### Reliability analysis
For each layer, calculate:
- Raw gradient score and smoothed score.
- Activation norm score.
- Validation ablation importance (validation-loss increase when that layer's adapter is removed).
- Squared-energy spectral entropy.
Then quantify Spearman rank correlation / Kendall agreement of each signal with ablation importance, and compare these scores across seeds. Test whether spectral entropy moderates the rank correlation or the performance benefit of GARS. Keep this diagnostic separate from the main final-adapter accuracy comparison.

### Mentor discussion
Bring the mentor:
1. This updated prior-art comparison (especially Progressive PEFT and LAARA).
2. The exact narrowed method distinction (gradient-conditioned energy threshold versus score-to-rank and global threshold).
3. The proposed matched-budget comparison and signal-reliability analysis.
4. A question about whether a comparative empirical Phase 1 study is acceptable if the algorithmic novelty is narrower than first expected.

Do not silently change the approved project objectives or title. Record any agreed refinement in the work log and, if the scope materially changes, follow the IITM process for mentor/coordinator notification.
