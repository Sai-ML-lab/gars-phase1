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
