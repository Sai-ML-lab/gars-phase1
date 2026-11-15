# Phase 1 work log

Use this file to record the work actually completed. Add dates, code revisions, observations and blockers after each work session; do not backfill unverified experimental outcomes.

## 15 November 2026 — mentor alignment and V1 implementation

**Mentor alignment:** The student reports that the mentor agreed to proceed with the narrowed comparative-study framing discussed for GARS: evaluate the gradient-conditioned energy-retention policy as a candidate allocation rule against uniform-local-energy and global-spectral-threshold alternatives, with signal reliability and spectral concentration as the mechanism investigation. No claim of first gradient-guided rank scheduling is made.

**Completed in the research repository:**
- Added the frozen GARS V1 mathematical specification.
- Defined the allocation unit as a Transformer block and each target projection as a separate spectral-pruning unit.
- Operationalised the proposal's live gradient magnitude as a joint Frobenius norm over all LoRA A/B factor gradients in a block, averaged over the final 50 optimizer updates.
- Kept the proposal's normalized gradient-to-mean threshold mapping and threshold bounds 0.80–0.99.
- Standardised the pruning criterion as squared-singular-value (Frobenius) energy and retained a rank floor of 2.
- Implemented reduced QR-SVD, energy-based rank selection and normalized squared-spectral entropy.
- Implemented physical A/B parameter counting and deterministic target-budget calibration for GARS, uniform local-energy pruning and a global singular-value cutoff.
- Added unit tests for the mathematical utilities and policy allocation.
- Added a DeBERTa-v3-base / GLUE RTE parent-adapter training runner that records gradient scores over the final 50 optimizer updates and saves a common unpruned parent adapter for post-hoc comparisons.
- Added the V1 config and experiment protocol.

**Not yet completed or verified:**
- Unit tests have been authored but have not yet been executed on the student's Mac environment.
- The parent RTE run has not yet been executed.
- No GARS pruning result, parameter reduction, validation score, adapter export, reload-equivalence result or latency measurement is claimed.
- PEFT rank-pattern export and post-hoc adapter reload validation remain the next implementation stage.

**Next actions:**
1. Pull the updated main branch on the experiment machine.
2. Run Ruff and pytest; fix every failure before training.
3. Run the parent-adapter script for seed 42 only.
4. Inspect and retain the metadata, training log and gradient-importance file.
5. Once the first seed is valid, run seeds 123 and 999 with the same config.
6. Implement and validate the post-hoc allocation/export/reload pipeline before reporting any compression result.

**Primary code revision:** inspect the repository's current main branch and record the exact commit hash used by each run in its metadata.


## 15 November 2026 — first local Ruff feedback

The student ran Ruff on the local Apple Silicon environment and shared six findings. All six were addressed in the repository:
- Removed the shebang from the parent runner because it is invoked as `python experiments/train_parent_rte.py`, not as an executable.
- Removed the unused E402 suppression.
- Removed an unnecessary integer cast around `len(references)`.
- Simplified/reordered standard-library imports in the gradient-importance and policy modules.
- Marked the unused dense-SVD right-singular-vector variable as intentionally unused.

The Ruff check itself has not yet been rerun after the fixes, and pytest has not yet been executed. Record the rerun results here once supplied by the student.


## 15 November 2026 — first local pytest run and test corrections

The student ran `python -m pytest -q` on the experiment Mac. The first run reported 19 passed and 2 failed.

**Failure analysis and corrections committed:**
- The physical-parameter-count test expected 46, but its own documented expression evaluates to (2(4+4)+1(4+4)+3(4+6)=16+8+30=54). The implementation's physical A/B parameter formula, `rank * (in_features + out_features)`, is correct. The test expectation and arithmetic comment were corrected to 54.
- The QR-SVD implementation returns a compact spectrum bounded by the LoRA factor rank (5 in the test). The dense SVD of a 13-by-11 product returns 11 singular values, with six trailing numerical zeros because the product rank is at most 5. The test now compares the five leading singular values, checks that the remaining dense singular values are zero within tolerance, verifies reconstruction and checks the compact factor shapes.
- Clarified the QR-SVD docstring to describe its compact, rank-bounded spectrum and the omission of trailing zeros.

An isolated float64 numerical check of the QR-SVD test case confirmed a maximum leading-spectrum difference of approximately (1.25\times10^{-14}), trailing dense singular values below (3.2\times10^{-15}), and reconstruction error below (8.5\times10^{-15}). This is a targeted numerical sanity check, not a full test-suite rerun.

**Verification still required:** the corrected test suite and Ruff have not yet been rerun on the student's Mac. Do not start the parent-adapter training run until both checks pass.


## 15 November 2026 — local quality gates passed

The student reports that the corrected checks now pass on the experiment Mac:
- `ruff check src/gars tests experiments/train_parent_rte.py`: passed with no findings.
- `python -m pytest -q`: all 21 tests passed.

This clears the implementation quality gate for the first controlled parent-adapter run. No training result is claimed yet.

**Next action:** capture the clean working-tree state and current Git revision, then run only seed 42 using the V1 DeBERTa-v3-base / GLUE RTE config. Inspect the run metadata, all 50 gradient-capture steps, validation metrics, and saved adapter before attempting seeds 123 and 999.


## 15 November 2026 — seed 42 numerical failure and diagnostic hardening

The first seed-42 parent run did not complete and is invalid for research reporting:
- The reported epoch-1 and epoch-2 training-loss means were `NaN`.
- Validation accuracy remained 0.52708 and macro-F1 0.34515 in both reported epochs.
- The run later stopped when the gradient-importance collector detected a non-finite LoRA gradient.
- No completed parent adapter, final metadata or valid gradient profile was produced. Do not use the displayed validation metrics as a trained-parent result.

The user also saw generic Hugging Face `UNEXPECTED` / `MISSING` checkpoint-loading notes, but did not include the actual parameter names. These cannot yet be classified. Task-head parameters may be expected to be newly initialized when loading a base pretraining checkpoint for sequence classification; the exact lists need inspection.

**Debugging changes committed:**
- Added `scripts/diagnose_deberta_rte.py`, a short preflight that reports exact checkpoint loading keys and tests base-model forward, PEFT forward, first backward, gradient finiteness and a small number of optimizer updates on the selected device.
- Hardened the parent runner to fail at the first non-finite logits/loss and to reject non-finite gradients during gradient clipping.
- The parent runner now writes each completed epoch to `training_log.jsonl` immediately, so an interrupted run retains completed-epoch evidence.

**Next action:** run the diagnostic only, not the full training run. Determine whether the first non-finite value appears in the base-model forward, PEFT forward, backward, or after an optimizer update. Inspect the precise missing/unexpected key lists and package versions before changing the learning rate or device. No hyperparameter has been changed yet.


## 15 November 2026 — diagnostic loading-report serialization fix

The first attempt to run `scripts/diagnose_deberta_rte.py` on both MPS and CPU stopped before the base-model forward pass. Transformers returned one or more loading-info collections as sets, and the diagnostic attempted to pass them directly to `json.dumps`. This was a diagnostic-script defect, not a model numerical failure.

The loading report itself showed:
- Unexpected pretraining-head keys under `lm_predictions.lm_head` and `mask_predictions`.
- Missing `classifier.weight`, `classifier.bias`, and `pooler.dense` weights.

These key groups are consistent with loading the DeBERTa-v3-base pretraining checkpoint into a sequence-classification architecture: pretraining-only heads are unused, while downstream classifier/pooler parameters are initialized for the task. The task head still needs to be trained on RTE. No backbone parameter mismatch was shown in the supplied report.

**Correction committed:** diagnostic now converts each loading-info collection to a sorted list of strings before JSON serialization.

**Still unknown:** the numerical root cause of the NaN loss. Neither device diagnostic reached a forward pass, so no conclusion about MPS versus CPU stability can yet be drawn.


## 15 November 2026 — FP16 optimizer instability isolated; FP32 trial prepared

The user successfully ran the numerical preflight through base-model forward, PEFT forward and first backward on MPS:
- All loaded base-model parameters were `torch.float16`.
- Base-model first-batch loss was finite (0.66162109).
- PEFT first-batch loss was finite (0.64697266).
- First backward completed with a finite clipped global gradient norm input (2.55323505).
- The first MPS AdamW update then produced non-finite trainable `pooler` and `classifier` parameters.
- The CPU diagnostic output shared by the user reached a finite base-model and PEFT forward, but did not include the later backward/optimizer result; CPU optimizer behavior therefore remains unknown.

**Working diagnosis:** the checkpoint was implicitly loaded in half precision because the Transformers loading default followed the checkpoint dtype. Running ordinary AdamW directly on FP16 trainable task-head parameters is a plausible cause of the first-step non-finite update: FP16 optimizer states and the default Adam epsilon can lose numerical resolution (1e-8 is below the smallest positive FP16 subnormal). This is strongly consistent with the observed stage of failure, but is not yet confirmed until the same preflight passes with FP32 parameters.

**Corrections committed:**
- Both the numerical diagnostic and parent runner now explicitly request FP32 model loading. The argument is selected by Transformers major version: `dtype=torch.float32` for v5 and `torch_dtype=torch.float32` for v4.
- Both scripts assert that every loaded base-model and PEFT parameter is FP32 before optimization.
- AdamW epsilon is now an explicit config value (`1e-8)) and the diagnostic uses the same epsilon as the parent runner.
- Parent training checks trainable-parameter finiteness immediately after every optimizer update and fails at the first offending update.
- Run metadata records FP32 parameter dtype and AdamW epsilon.

**Next action:** rerun Ruff and all 21 tests, then rerun the five-step MPS numerical preflight only. Do not launch the three-epoch training run until the FP32 diagnostic passes. Do not interpret the incomplete FP16 run as a result.


## 15 November 2026 — FP32 MPS numerical preflight passed

The user reran `scripts/diagnose_deberta_rte.py` on MPS after the explicit-FP32 change. The preflight passed:
- All base-model parameters loaded as FP32.
- Base-model first-batch forward was finite (loss 0.66159737).
- PEFT first-batch forward was finite (loss 0.64707077).
- First backward was finite (pre-clipping global gradient norm 2.55375123).
- Five subsequent optimizer updates completed with finite loss, gradient norm and trainable parameters. Logged losses: 0.60464418, 0.88693833, 0.58858389, 0.79867661, 0.56207579.
- The first update after the initial backward and all five following updates passed parameter-finiteness checks.

This validates the short MPS FP32 numerical preflight only. It is not a parent-training experiment and supplies no reportable accuracy or compression result.

The loading report again showed the pretraining-only MLM/masking heads as unexpected and the sequence-classification classifier/pooler as missing. No mismatched keys or loading errors were reported. This is consistent with transfer from the DeBERTa pretraining checkpoint to the sequence-classification task; the newly initialized downstream head must be trained on RTE.

**Next action:** rerun Ruff and all unit tests on the latest pulled code. If both pass, launch only the complete seed-42 FP32 parent-adapter run. Inspect epoch logs, 50-step gradient coverage, validation results and adapter files before running any additional seeds.
