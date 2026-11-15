# GARS Phase 1 Research Workspace

This is a starter workspace for the IIT Madras M.Tech Phase 1 GARS project.
It deliberately preserves exploratory notebooks separately from the reusable
implementation. The starter does not claim that GARS is already validated.

## Environment recommendation

- macOS on Apple Silicon
- Python 3.12 (use a dedicated virtual environment)
- PyTorch with the MPS backend for local GPU experiments
- Hugging Face Transformers + PEFT for the initial LoRA smoke test

Do not install `bitsandbytes` as a required dependency for the local MPS path.
The existing proposal's 4-bit CUDA-oriented workflow must be tested separately;
MPS compatibility and actual memory/performance need to be verified first.

## 1. Create the environment

From this directory:

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

If `python3.12` is unavailable, install Python 3.12 first (for example with
Homebrew: `brew install python@3.12`) and reopen the terminal.

For PyTorch MPS, ensure Apple's Xcode command-line tools are installed:

```bash
xcode-select --install
```

## 2. Run the environment diagnostic

```bash
python scripts/check_environment.py
```

This checks Python/PyTorch versions, MPS availability, matrix multiplication,
QR, SVD, and autograd/backpropagation. Save its output in `results/day1/`.

## 3. Run the tiny LoRA smoke test

```bash
python scripts/smoke_lora.py
```

This downloads a tiny public test model from Hugging Face, wraps its query/value
projections with PEFT LoRA, and performs one forward/backward/optimizer step.
It is an integration test only; it is not a project experiment and its metrics
must not appear as research results.

## 4. Preserve the original notebooks

Copy the three original notebooks into `notebooks/original/` without editing:

- `ars-initial-idea.ipynb`
- `gars-med.ipynb`
- `mtech-major-project-poc-ipynb.ipynb`

Keep future cleaned notebooks in `notebooks/analysis/`.

## Repository structure

```text
configs/       Experiment configurations (YAML/JSON)
experiments/   Reproducible experiment entry points
figures/       Generated report figures
literature/    Search log, paper matrix, notes, PDFs/links
notebooks/    Original and exploratory notebooks
report/        IITM LaTeX report source (add official template here)
results/       Machine-readable results and logs
scripts/       Environment and smoke-test scripts
src/gars/      Reusable GARS implementation
 tests/        Unit tests
```

## Research integrity rules

1. Never overwrite the original notebooks.
2. Every experiment must record config, code revision, seed, dataset split,
   software versions, logs, and output metrics.
3. Separate pilot/smoke tests from controlled experiments.
4. Record failed and incomplete runs; do not silently discard them.
5. Do not claim compression or latency improvements from zeroed-but-unchanged
   adapter tensors. Validate physically truncated export and actual inference.
6. Keep the sum-of-singular-values criterion distinct from the squared
   singular-value (Frobenius-energy) criterion.


## 5. GARS V1 core and unit tests

The first implementation slice now contains:
- reduced QR-SVD for a LoRA update B @ A;
- squared-singular-value energy retention and normalized squared spectral entropy;
- physical A/B factor truncation;
- joint per-Transformer-block LoRA A/B gradient capture;
- GARS, uniform-local-energy and global singular-value budget calibrators;
- unit tests for the above functions.

Install the updated package and run:

```bash
python -m pip install -e ".[dev]"
ruff check src/gars tests
python -m pytest -q
```

## 6. First controlled parent-adapter run

The approved V1 pilot is DeBERTa-v3-base on GLUE RTE. The exact working definitions and protocol are in:
- docs/gars_v1_method_spec.md
- docs/experiment_protocol_v1.md
- configs/v1_deberta_rte.json

The parent runner trains unpruned LoRA and records the final 50 optimizer-step gradient window. It does not prune during training.

Start with one seed after the unit tests pass:

```bash
python experiments/train_parent_rte.py \
  --config configs/v1_deberta_rte.json \
  --seed 42 \
  --device mps
```

Run all three configured seeds only after the first seed completes and its artifacts are inspected:

```bash
python experiments/train_parent_rte.py \
  --config configs/v1_deberta_rte.json \
  --device mps
```

Adapter weight files are ignored by Git; keep them locally or on the experiment machine. Commit the small metadata/JSON results and logs, but never commit model weights or credentials. This runner creates the common parent checkpoint; post-hoc pruning/export and reload validation are the next implementation stage.


## 7. Matched-budget post-hoc pruning comparison

After all three parent adapters are trained and reload-validated, pin the exact DeBERTa Hub commit SHA in `configs/v1_deberta_rte.json` before evaluating pruned adapters. The pruning exporter refuses a floating `main` revision unless the user explicitly opts in after verifying the cached snapshot.

Run one seed at a time, beginning with seed 42:

```bash
python scripts/run_pruning_comparison.py --seed 42 --device mps
```

The script builds uniform-local-energy, global-singular-value and GARS exports at the configured 60% physical LoRA A/B parameter target, reloads every export on a fresh FP32 base, checks the realized per-module ranks and LoRA scaling multiplier, and evaluates the GLUE RTE validation split. Inspect `results/v1_pruned/seed_42/comparison_summary.json` and each method's `allocation.json` before running seeds 123 and 999. Use `--overwrite` only when intentionally rebuilding that seed's policy exports.

Generated adapter binaries/tokenizer copies under `results/v1_pruned/*/*/adapter/` are ignored by Git; commit the small JSON allocation and comparison reports after review, not binary adapter weights.
