# GARS V1 — frozen mathematical and implementation specification

**Status:** V1 for the first controlled Phase 1 experiment  
**Benchmark:** DeBERTa-v3-base on GLUE RTE  
**Scope:** Same-parent-checkpoint, post-hoc adapter allocation/pruning. This is deliberately separate from the later dynamic in-training pruning experiment.

## 1. Unit of allocation

The gradient-importance unit is a **Transformer block**. For block l, the gradient vector is formed by concatenating every LoRA A-factor and B-factor gradient tensor belonging to that block across all configured target projections.

At optimizer step t:

\[
g_l^{(t)} =
\sqrt{
\sum_{m\in\mathcal{M}(l)}
\left(
\|\nabla A_{l,m}^{(t)}\|_F^2+
\|\nabla B_{l,m}^{(t)}\|_F^2
\right)
}.
\]

Here, m indexes adapted projection modules in Transformer block l. This is a joint Frobenius norm across that block's LoRA-factor gradients, not the sum of the individual Frobenius norms.

## 2. Temporal aggregation

To reduce dependence on a single minibatch while staying faithful to the proposal's live-gradient signal, V1 records the arithmetic mean of the per-step joint norms over the final 50 optimizer updates of parent-adapter training:

\[
g_l = \frac{1}{T}\sum_{t\in\mathcal{W}} g_l^{(t)},\qquad |\mathcal{W}|=T=50.
\]

V1 does **not** use an EMA or empirical-Fisher statistic. Those are different candidate signals and belong in later ablations. Capture happens after gradient accumulation and before optimizer.step(), so each recorded score corresponds to the gradient used for that optimizer update.

## 3. Gradient-conditioned threshold

The proposal's rule is retained:

\[
\bar g = \frac{1}{L}\sum_{l=1}^{L}g_l,
\qquad
\tau_l =
\operatorname{clamp}\left[
\tau_{\mathrm{base}}
\left(\frac{g_l}{\bar g}\right)^\beta,
0.80,0.99
\right].
\]

V1 starts with \(\beta=0.5\). The base threshold is not fixed in the final matched-budget comparison: it is calibrated deterministically to the target physical adapter-parameter count. If all layer-gradient scores are zero, all layers receive the same tau_base threshold (neutral fallback).

## 4. Spectral energy and rank selection

For every adapted projection m in block l, obtain the singular values of its LoRA update:

\[
\Delta W_{l,m}=B_{l,m}A_{l,m}
=U_{l,m}\Sigma_{l,m}V_{l,m}^{\top}.
\]

The retained rank k is the smallest integer satisfying:

\[
\frac{\sum_{i=1}^{k}\sigma_{l,m,i}^{2}}
{\sum_{i=1}^{r}\sigma_{l,m,i}^{2}}
\geq \tau_l,
\qquad k\geq r_{\min}=2.
\]

This is squared-singular-value energy, equivalent to retained squared Frobenius norm of the truncated update. It is **not** the sum-singular-value ratio used in some exploratory notebook cells. The V1 implementation must use the squared criterion consistently.

Reduced QR-SVD is used to avoid explicitly materializing the full dense update matrix. With reduced QR:

\[
B=Q_B R_B,\qquad A^\top=Q_A R_A,
\]
\[
R=R_B R_A^\top=U_R\Sigma V_R^\top,
\]
\[
B'=Q_BU_R[:,1:k]\Sigma_k^{1/2},
\qquad
A'=\Sigma_k^{1/2}V_R^\top[1:k,:]Q_A^\top.
\]

The external LoRA scaling coefficient is unchanged. The resulting factors have actual shapes [out_features, k] and [k, in_features].

## 5. Physical adapter-parameter budget

A rank-k adapter for a linear module with dimensions in_features and out_features stores:

\[
P_{l,m}=k_{l,m}(d_{\mathrm{in},l,m}+d_{\mathrm{out},l,m}).
\]

The total physical adapter factor count is:

\[
P_{\mathrm{total}}=\sum_{l,m}P_{l,m}.
\]

V1 uses a target of 60% of the uncompressed parent adapter's A/B factor parameters (a 40% reduction target, consistent with the proposal's expected reduction). Integer ranks and the rank floor can make exact matching impossible. The calibration result must record the achieved count and absolute/relative budget error; never label a non-exact match as exact.

## 6. Comparison policies on the same parent checkpoint

All policies consume the same trained parent LoRA adapter and the same per-module spectra.

1. **Parent LoRA:** no pruning; reference accuracy, trainable parameters and adapter bytes.
2. **Uniform local energy:** one common squared-energy threshold across every adapted projection; calibrate the threshold to the target physical parameter budget.
3. **Global singular-value threshold (PARA-style):** one absolute singular-value cutoff across all adapted projections; calibrate the cutoff to the nearest target physical budget.
4. **GARS:** use block-level gradient importance to set block-specific energy thresholds; calibrate tau_base to the same target physical parameter budget.

These first experiments isolate the allocation/pruning policy. They do not claim to reproduce the entire training algorithms of Progressive PEFT, LAARA or PARA.

## 7. Spectral concentration diagnostic

For each adapter projection:

\[
p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2},
\qquad
H_{\mathrm{sq}}=
-\frac{1}{\log r}\sum_i p_i\log(p_i+\epsilon).
\]

H_sq approaches 0 for energy concentrated in a few directions and 1 for a flat squared-energy spectrum. It is used as a diagnostic/moderator, not as a newly claimed spectral-entropy rank allocator. Report the exact definition to distinguish it from entropy over normalized singular values.

## 8. V1 boundaries

- The first study is **post-hoc**: no factor dimensions change while the parent optimizer is running. Optimizer-state surgery is therefore not part of this first comparison.
- Dynamic pruning during training, Adam-moment reset behavior, adapter merge/unmerge behavior and latency optimization are later work packages.
- The first dataset is GLUE RTE. Do not add MedQuAD or CUAD until this pipeline is correct.
- The first result is a pilot. It is not evidence for the proposal's three-model-scale/three-domain claims.
- The validation split is used for model comparison. Do not tune against the hidden GLUE test labels.

## 9. Required run artifacts

Every seed must save:
- base-model identifier/revision and dataset split fingerprint;
- all training hyperparameters and software versions;
- parent validation accuracy/F1 and training step count;
- the final-window per-block gradient scores and capture count;
- parent adapter configuration and tensors;
- every policy's threshold/cutoff, per-module retained ranks and physical parameter count;
- exported adapter tensor shapes, serialized bytes and load/forward validation;
- the exact Git commit and seed.
