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


## 15 November 2026 — seed 42 parent-adapter training completed

The student supplied the completed seed-42 training summary. This is the first valid controlled parent-model run after the FP32 stability correction.

**Training execution**
- Base model: `microsoft/deberta-v3-base`; task: GLUE RTE.
- Precision: FP32.
- Optimizer updates: 468 total (156 per epoch).
- Training loss by epoch: 0.66477, 0.39251, 0.29663.
- Validation accuracy by epoch: 0.73285, 0.80505, 0.81949.
- Validation macro-F1 by epoch: 0.71407, 0.80111, 0.81622.
- Final validation loss: 0.45191 on 277 examples.
- Final validation accuracy: 0.81949 (approximately 227/277 examples).
- Final validation macro-F1: 0.81622.

**Gradient capture**
- Exactly 50 optimizer-step captures were recorded.
- All 12 Transformer blocks had a score.
- The reported arithmetic mean joint gradient norms ranged from 0.39952 (block 2) to 1.74784 (block 11); across-block mean of the reported scores was approximately 0.97863.
- Relative-to-mean scores range from approximately 0.408 (block 2) to 1.786 (block 11). This shows measurable layer variation in this one seed, but does not demonstrate reliability or causality.

**Parameter and artifact accounting**
- LoRA A/B factor parameters: 1,179,648.
- Total trainable parameters including modules_to_save: 1,771,778.
- Saved `adapter_model.safetensors`: 7,094,672 bytes.
- Adapter directory also contains PEFT config/README and tokenizer files.
- The student reports the seed-42 model parameter dtype as FP32.

**Interpretation and caveats**
- Training loss decreased monotonically; validation accuracy and macro-F1 increased each epoch. Validation loss fell through epoch 2 and rose modestly in epoch 3 (0.43064 to 0.45191) while accuracy/F1 improved. This is worth noting, but a single validation run is insufficient to diagnose overfitting.
- This is one seed on one model and one task. It is a parent-adapter baseline only; no pruning/compression or GARS benefit has yet been measured.
- The 7,094,672-byte file is the complete saved adapter artifact, not a direct measurement of deployed storage after any truncation. Tokenizer files are separate files in the same directory.
- Detailed run revision, package versions, dataset fingerprints, and saved tensor shapes still need to be inspected from `run_metadata.json` and `adapter_model.safetensors`.

**Next gate:** validate the saved tensor inventory, confirm the stored LoRA A/B factor count and rank-32 module layout, reload the PEFT adapter onto a fresh FP32 DeBERTa sequence-classification base, and reproduce the stored validation metrics. Do not start seeds 123 and 999 until this check passes.


**Adapter-verification utility added:** `scripts/validate_parent_adapter.py` now:
- inventories saved safetensors tensors and checks paired LoRA A/B matrices;
- checks every parent LoRA module has rank 32 and the expected module count (12 blocks × 2 projections = 24);
- verifies saved A/B parameter count against run metadata;
- checks that classifier and pooler `modules_to_save` tensors are present;
- reloads the adapter onto a fresh FP32 sequence-classification base and reruns the GLUE RTE validation split;
- compares accuracy and macro-F1 exactly and validation loss within a configurable tolerance;
- writes `reload_validation.json` only when all checks pass.

This utility has been committed but has not yet been run on the user's machine. Its output is the next gate before starting seeds 123 and 999.


## 15 November 2026 — adapter reload validator label-preservation fix

The first run of `scripts/validate_parent_adapter.py` reached validation evaluation but failed with `KeyError: 'labels'`. The validator tokenization function had removed the original GLUE RTE columns without preserving `label`; unlike the training runner, it had not copied `batch["label"]` into the tokenized batch. Consequently, `DataCollatorWithPadding` had no labels to collate.

**Correction committed:** the validator now preserves `encoded["label"] = batch["label"]`, matching the exact tokenization/collation convention used by the parent training runner. This is a validation-script bug, not an adapter or model failure. Reload equivalence remains untested until the validator is rerun.


## 15 November 2026 — seed 42 parent adapter reload validation passed

The student ran `scripts/validate_parent_adapter.py` on MPS. All checks passed:
- Saved adapter contains 52 tensors, including 24 LoRA A/B module pairs, 2 classifier tensors and 2 pooler tensors.
- Every one of the 24 adapted projections has rank 32: query_proj and value_proj in each of the 12 DeBERTa blocks.
- Saved LoRA A/B factor count is 1,179,648, exactly matching the parent run metadata.
- Saved LoRA weights file is 7,094,672 bytes; saved adapter tensors are FP32.
- A fresh DeBERTa-v3-base FP32 sequence-classification model was wrapped with the saved PEFT adapter and evaluated on the same 277-example GLUE RTE validation split.
- Reloaded accuracy = 0.8194945848375451, macro-F1 = 0.8162154989384289, and loss = 0.45190566916550906; all three absolute differences from the parent run metadata were exactly 0.0.
- The validator wrote `results/v1_parent_rte/seed_42/reload_validation.json`.

**Conclusion:** seed 42's parent adapter serialization, tensor inventory and reload equivalence are validated. This remains a single-seed, single-task parent baseline; no compression result or GARS advantage is claimed.

**Next action:** run parent training for seeds 123 and 999 with the identical V1 config and FP32 runner, then run the same adapter reload validator for each seed. Do not start pruning comparisons until both additional seeds have passed their own reload checks.


## 15 November 2026 — seed 123 parent-adapter reload validation passed

The student supplied the completed validation output for seed 123:
- Reload validation passed on MPS with FP32 base and PEFT model parameters.
- Saved adapter contains 52 tensors: 24 LoRA modules, 2 classifier tensors and 2 pooler tensors.
- Every adapted projection is rank 32; 12 Transformer blocks × 2 target projections = 24 LoRA modules.
- Saved LoRA A/B factor count is 1,179,648, matching seed-123 run metadata.
- Adapter weights file size is 7,094,672 bytes; adapter tensors are FP32.
- Original and reloaded validation results on 277 examples match exactly: accuracy 0.8447653429602888, macro-F1 0.8435911914172783, loss 0.43382309483630316. Absolute differences for all metrics are 0.0.
- The validator wrote `results/v1_parent_rte/seed_123/reload_validation.json`.

**Conclusion:** seed 123's parent adapter save/load pipeline is validated. Along with seed 42, two of the three parent seeds have passed. These are unpruned parent results only; no compression benefit is measured yet.

**Next action:** complete seed 999 with the identical V1 configuration and FP32 settings, then validate its adapter with the same reload utility. Before aggregating across seeds or making a strong reproducibility claim, inspect the recorded model snapshot/revision information because the current reload reports show revision `main` rather than an immutable model commit hash.


## 15 November 2026 — seed 999 adapter reload validation passed; parent baseline complete

The student supplied the seed-999 adapter reload report. All checks passed:
- Reload validation passed on MPS with FP32 base and PEFT model parameters.
- Saved adapter contains 52 tensors: 24 LoRA module pairs, 2 classifier tensors and 2 pooler tensors.
- All 24 target projections have rank 32 across 12 DeBERTa blocks.
- Saved A/B factor count is 1,179,648; adapter weights file is 7,094,672 bytes; saved tensors are FP32.
- Original and reloaded metrics on 277 GLUE RTE validation examples match exactly: accuracy 0.8303249097472925, macro-F1 0.8288167462164544, loss 0.4475828063275133. Absolute differences are 0.0.
- The validator wrote `results/v1_parent_rte/seed_999/reload_validation.json`.

### Three-seed parent baseline (descriptive summary)

Using the three supplied seed-level validation results, the arithmetic mean and sample standard deviation (n=3, denominator n-1) are:
- Accuracy: mean 0.8315283; sample SD 0.0126783 (83.15% ± 1.27 percentage points).
- Macro-F1: mean 0.8295411; sample SD 0.0137022 (82.95% ± 1.37 percentage points).
- Validation loss: mean 0.4444372; sample SD 0.0094428.

Individual results:
- Seed 42: accuracy 0.8194946, macro-F1 0.8162155, loss 0.4519057.
- Seed 123: accuracy 0.8447653, macro-F1 0.8435912, loss 0.4338231.
- Seed 999: accuracy 0.8303249, macro-F1 0.8288167, loss 0.4475828.

Interpret only as a three-seed pilot parent baseline on one model/task; n=3 is too small for strong inferential/significance claims.

### Model revision reproducibility caveat

All three reload reports display base-model revision `main`. The original training metadata did not record an immutable Hub commit SHA (`base_model_revision` was absent/null and the validator fell back to the config's `main`). The three training runs were performed in the same work session against the same model ID, but exact byte-identical pretrained snapshot provenance has not been independently established. Before treating this as fully pinned/reproducible, record the local Hugging Face cache `refs/main` commit SHA and compare the three run timestamps/cache provenance if available. Do not silently rewrite historical metadata to claim a SHA that was not captured at training time.

**Next stage:** construct and validate the post-hoc matched-physical-budget export pipeline. For each seed, compare parent, uniform local squared-energy, global singular-value cutoff and GARS. PEFT's `rank_pattern` and `alpha_pattern` can express per-module rank and alpha; when physically changing rank, `alpha_pattern` must be set to preserve the original external LoRA scale (`alpha/r` for standard LoRA, `alpha/sqrt(r)` for rank-stabilized LoRA). This must be verified by reloading the exported adapter and checking module ranks/scaling before interpreting validation metrics.


## 15 November 2026 — post-hoc physical pruning exporter implemented

**Three-seed parent baseline is complete:** seeds 42, 123 and 999 all passed the adapter save/load validator. The descriptive three-seed parent metrics are accuracy 0.8315283 ± 0.0126783, macro-F1 0.8295411 ± 0.0137022, and validation loss 0.4444372 ± 0.0094428 (sample standard deviation; n=3). These are pilot descriptive statistics, not inferential evidence.

**Implementation committed:**
- Added `src/gars/export.py` with standard-LoRA and rank-stabilized-LoRA alpha adjustment to preserve the original multiplier after rank changes.
- Added utility tests for scaling preservation and exact PEFT module-pattern construction.
- Added `scripts/run_pruning_comparison.py`, which reads one parent adapter and its per-block gradients, computes all module spectra with QR-SVD, calibrates uniform-local-energy/global-singular-value/GARS allocations to the same 60% physical A/B parameter target, and creates physically truncated A/B tensors.
- The exporter updates per-module PEFT `rank_pattern` and `alpha_pattern`, preserves classifier/pooler modules_to_save and tokenizer assets, and writes a per-module allocation report.
- It reloads each export on a fresh FP32 base model, checks all realized A/B ranks and runtime LoRA scaling multipliers, then evaluates the same GLUE RTE validation split.
- Added documentation for scaling preservation and pruning commands. Generated binary adapter directories under `results/v1_pruned` are ignored by Git.

**Critical safeguards:**
- PEFT's rank/alpha pattern names are based on the underlying model's module paths, not the `base_model.model.` state-dict wrapper prefix. Pattern keys are regex-escaped and have no leading `^`, to avoid the optional-prefix matching behavior of PEFT's matcher.
- The exporter refuses a floating `main` revision unless the user explicitly opts in. All three historical run reports show `main` rather than an immutable commit SHA, so exact historical pretrained-snapshot provenance remains unproven. Retrieve and record the cached Hub SHA before pruning; do not silently claim the parent models are pinned.
- Standard LoRA scaling is preserved after rank change with `alpha_k = alpha_0 * k / r_0`; rank-stabilized scaling uses `alpha_k = alpha_0 * sqrt(k / r_0)`. The script verifies the loaded runtime multiplier for every target module.
- The exporter explicitly rejects unsupported DoRA, LoRA-bias, QALoRA and fan-in/fan-out parent configurations.

**Not yet verified:** the new script has not yet been run on the student's machine. Run Ruff and pytest, then use seed 42 first. Do not launch all three pruning comparisons until seed 42 exports load with the expected per-module ranks and preserved scaling.


## 15 November 2026 — package-version test and metadata synchronization

The student ran the expanded test suite after the pruning-export utilities were added. Result: 28 passed and one failure in `test_package_imports`; the implementation package had been bumped to `0.3.0`, while the test still asserted `0.2.0`.

**Correction committed:**
- Updated `tests/test_imports.py` to expect `0.3.0`.
- Added public import checks for `alpha_for_preserved_scaling` and `peft_module_pattern_key`.
- Updated `pyproject.toml` project metadata to `0.3.0` to keep installed package metadata synchronized with `src/gars/__init__.py`.

The previous run's 28 passing tests remain valid; the failing assertion was a stale version expectation. Rerun Ruff and pytest after pulling the version synchronization commit. With the local editable install, run `python -m pip install -e .` once to refresh installed distribution metadata before experiments.

**Workflow note from the student:** the M5 Max is the compute machine, but results/files cannot be transferred out and the student cannot git-push from that machine. Source-code changes will therefore be committed through the connected GitHub integration; the student only needs to pull on the M5 Max, run commands locally, and paste compact terminal/JSON outputs into chat. Keep large adapter/model binaries local and avoid requiring git push or file transfer.
