"""Benchmark PEFT inference latency with merged and unmerged LoRA adapters.

Uses fixed, representative batches sampled from the configured validation split.
Timing covers model forward execution only: tokenization, collation and host-to-device
transfer are excluded. Each timed forward is synchronized on accelerator devices.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_METHODS = (
    "parent",
    "gars",
    "uniform_local_energy",
    "global_singular_value",
)
METHOD_LABELS = {
    "parent": "Parent LoRA",
    "gars": "GARS",
    "uniform_local_energy": "Uniform local energy",
    "global_singular_value": "Global singular value",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _model_load_dtype_kwargs() -> dict[str, torch.dtype]:
    major_version = int(transformers.__version__.split(".", maxsplit=1)[0])
    if major_version >= 5:
        return {"dtype": torch.float32}
    return {"torch_dtype": torch.float32}


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _adapter_path(
    method: str,
    seed: int,
    parent_root: Path,
    pruned_root: Path,
) -> Path:
    if method == "parent":
        return parent_root / f"seed_{seed}" / "parent_adapter"
    return pruned_root / f"seed_{seed}" / method / "adapter"


def _validate_adapter_artifact(
    method: str,
    seed: int,
    adapter_path: Path,
    config_revision: str,
    parent_root: Path,
    pruned_root: Path,
) -> dict[str, Any]:
    weights_path = adapter_path / "adapter_model.safetensors"
    config_path = adapter_path / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete adapter directory: {adapter_path}")

    if method == "parent":
        metadata = _read_json(parent_root / f"seed_{seed}" / "run_metadata.json")
        reload_report = _read_json(
            parent_root / f"seed_{seed}" / "reload_validation.json"
        )
        if reload_report.get("reload_validation") != "passed":
            raise RuntimeError(f"Parent adapter reload did not pass for seed {seed}.")
        revision = metadata.get("base_model_revision") or config_revision
        if revision != config_revision:
            raise ValueError(f"Parent seed {seed} has unexpected base revision {revision}.")
        return {
            "base_model_revision": revision,
            "adapter_weight_file_bytes": weights_path.stat().st_size,
            "parent_validation": metadata.get("final_validation"),
        }

    allocation = _read_json(
        pruned_root / f"seed_{seed}" / method / "allocation.json"
    )
    if allocation.get("reload_validation") != "passed":
        raise RuntimeError(f"Adapter reload did not pass: seed={seed}, method={method}.")
    revision = allocation.get("base_model_revision")
    if revision != config_revision:
        raise ValueError(
            f"Unexpected base revision for seed={seed}, method={method}: {revision}"
        )
    return {
        "base_model_revision": revision,
        "adapter_weight_file_bytes": weights_path.stat().st_size,
        "achieved_A_B_parameters": allocation.get("achieved_parameters"),
        "parameter_reduction_fraction": allocation.get("parameter_reduction_fraction"),
    }


def _representative_batches(
    config: dict[str, Any],
    tokenizer: Any,
    batch_sizes: list[int],
    batches_per_size: int,
) -> dict[int, list[dict[str, torch.Tensor]]]:
    dataset_cfg = config["dataset"]
    raw = load_dataset(dataset_cfg["name"], dataset_cfg["config"])
    validation = raw[dataset_cfg.get("validation_split", "validation")]

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=int(dataset_cfg.get("max_length", 256)),
        )
        encoded["label"] = batch["label"]
        return encoded

    tokenized = validation.map(
        tokenize,
        batched=True,
        remove_columns=validation.column_names,
        desc="Preparing fixed latency benchmark inputs",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    lengths = [len(tokenized[index]["input_ids"]) for index in range(len(tokenized))]
    length_order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))

    output: dict[int, list[dict[str, torch.Tensor]]] = {}
    quantiles = np.linspace(0.15, 0.85, batches_per_size)
    for batch_size in batch_sizes:
        if batch_size > len(tokenized):
            raise ValueError(
                f"Batch size {batch_size} exceeds validation examples ({len(tokenized)})."
            )
        batches: list[dict[str, torch.Tensor]] = []
        used_indices: set[tuple[int, ...]] = set()
        for quantile in quantiles:
            center = int(round(float(quantile) * (len(length_order) - 1)))
            start = max(0, min(len(length_order) - batch_size, center - batch_size // 2))
            indices = tuple(length_order[start : start + batch_size])
            if indices in used_indices:
                continue
            used_indices.add(indices)
            features = [tokenized[index] for index in indices]
            collated = collator(features)
            inputs = {
                key: value
                for key, value in collated.items()
                if key not in {"labels", "label"}
            }
            batches.append(inputs)
        output[batch_size] = batches
    return output


def _move_batches(
    batches: dict[int, list[dict[str, torch.Tensor]]],
    device: torch.device,
) -> dict[int, list[dict[str, torch.Tensor]]]:
    return {
        batch_size: [
            {key: value.to(device) for key, value in batch.items()}
            for batch in items
        ]
        for batch_size, items in batches.items()
    }


def _forward_logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    validate_finite: bool = True,
) -> torch.Tensor:
    output = model(**batch)
    logits = output.logits
    if validate_finite and not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("Non-finite logits during inference benchmark.")
    return logits


def _benchmark_mode(
    model: torch.nn.Module,
    batches: dict[int, list[dict[str, torch.Tensor]]],
    device: torch.device,
    warmup_steps: int,
    repeats: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    model.eval()
    with torch.inference_mode():
        for batch_size, batch_list in batches.items():
            timings: list[float] = []
            total_nonpad_tokens = 0
            total_padded_tokens = 0
            for batch in batch_list:
                total_nonpad_tokens += int(batch["attention_mask"].sum().item())
                total_padded_tokens += int(batch["attention_mask"].numel())

                for _ in range(warmup_steps):
                    _forward_logits(model, batch, validate_finite=True)
                _synchronize(device)

                for _ in range(repeats):
                    _synchronize(device)
                    start = time.perf_counter()
                    _forward_logits(model, batch, validate_finite=False)
                    _synchronize(device)
                    timings.append(time.perf_counter() - start)

            total_forward_examples = batch_size * len(batch_list) * repeats
            total_forward_nonpad_tokens = total_nonpad_tokens * repeats
            total_forward_padded_tokens = total_padded_tokens * repeats
            total_elapsed = sum(timings)
            results[str(batch_size)] = {
                "batch_size": batch_size,
                "number_of_distinct_batches": len(batch_list),
                "warmup_forwards_per_batch": warmup_steps,
                "timed_forwards_per_batch": repeats,
                "number_of_timed_forwards": len(timings),
                "latency_ms_mean": statistics.mean(timings) * 1000.0,
                "latency_ms_median": statistics.median(timings) * 1000.0,
                "latency_ms_sample_std": (
                    statistics.stdev(timings) * 1000.0 if len(timings) > 1 else None
                ),
                "latency_ms_p95": float(np.percentile(timings, 95)) * 1000.0,
                "latency_ms_min": min(timings) * 1000.0,
                "latency_ms_max": max(timings) * 1000.0,
                "examples_per_second": total_forward_examples / total_elapsed,
                "nonpadding_tokens_per_second": (
                    total_forward_nonpad_tokens / total_elapsed
                ),
                "padded_tokens_per_second": total_forward_padded_tokens / total_elapsed,
                "measurement_scope": (
                    "synchronized model forward only; tokenization, collation, "
                    "host-to-device transfer and model loading excluded"
                ),
            }
    return results


def _load_peft_model(
    config: dict[str, Any],
    adapter_path: Path,
    device: torch.device,
) -> PeftModel:
    base = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        num_labels=int(config["model"].get("num_labels", 2)),
        **_model_load_dtype_kwargs(),
    )
    base.config.problem_type = "single_label_classification"
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.to(device)
    model.eval()
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {torch.float32}:
        raise RuntimeError(f"Expected FP32 model parameters, received {dtypes}.")
    return model


def _benchmark_adapter(
    method: str,
    seed: int,
    config: dict[str, Any],
    adapter_path: Path,
    cpu_batches: dict[int, list[dict[str, torch.Tensor]]],
    device: torch.device,
    warmup_steps: int,
    repeats: int,
) -> dict[str, Any]:
    load_start = time.perf_counter()
    model = _load_peft_model(config, adapter_path, device)
    _synchronize(device)
    model_load_seconds = time.perf_counter() - load_start
    batches = _move_batches(cpu_batches, device)

    unmerged_logits: dict[str, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for batch_size, batch_list in batches.items():
            unmerged_logits[str(batch_size)] = [
                _forward_logits(model, batch).detach().cpu()
                for batch in batch_list
            ]
    _synchronize(device)

    unmerged = _benchmark_mode(
        model, batches, device, warmup_steps=warmup_steps, repeats=repeats
    )

    merge_start = time.perf_counter()
    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.eval()
    _synchronize(device)
    merge_seconds = time.perf_counter() - merge_start
    del model

    merge_errors: dict[str, float] = {}
    with torch.inference_mode():
        for batch_size, batch_list in batches.items():
            errors = []
            for batch, reference in zip(batch_list, unmerged_logits[str(batch_size)]):
                merged_logits = _forward_logits(merged_model, batch).detach().cpu()
                errors.append(float(torch.max(torch.abs(merged_logits - reference)).item()))
            merge_errors[str(batch_size)] = max(errors) if errors else 0.0
    max_merge_error = max(merge_errors.values(), default=0.0)
    if max_merge_error > 5e-4:
        raise RuntimeError(
            f"Merged/unmerged logits differ beyond tolerance for {method}, seed={seed}: "
            f"{max_merge_error:.6g}"
        )

    merged = _benchmark_mode(
        merged_model, batches, device, warmup_steps=warmup_steps, repeats=repeats
    )
    del merged_model
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "adapter_path": str(adapter_path),
        "model_load_seconds_excluded_from_latency": model_load_seconds,
        "merge_seconds_excluded_from_latency": merge_seconds,
        "merged_unmerged_max_abs_logit_error_by_batch_size": merge_errors,
        "max_merged_unmerged_logit_error": max_merge_error,
        "unmerged": unmerged,
        "merged": merged,
    }


def _summarize(
    runs: list[dict[str, Any]],
    seeds: list[int],
    methods: list[str],
    batch_sizes: list[int],
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for mode in ("unmerged", "merged"):
        for method in methods:
            for batch_size in batch_sizes:
                seed_rows = [
                    next(
                        run for run in runs
                        if run["seed"] == seed and run["method"] == method
                    )[mode][str(batch_size)]
                    for seed in seeds
                ]
                median_values = [float(row["latency_ms_median"]) for row in seed_rows]
                parent_values = [
                    float(
                        next(
                            run for run in runs
                            if run["seed"] == seed and run["method"] == "parent"
                        )[mode][str(batch_size)]["latency_ms_median"]
                    )
                    for seed in seeds
                ] if "parent" in methods and method != "parent" else []
                speedups = [
                    parent / current
                    for parent, current in zip(parent_values, median_values)
                ]
                by_condition[f"{mode}:{method}:batch_{batch_size}"] = {
                    "method": method,
                    "mode": mode,
                    "batch_size": batch_size,
                    "median_latency_ms_mean_across_seeds": statistics.mean(median_values),
                    "median_latency_ms_sample_sd_across_seeds": (
                        statistics.stdev(median_values) if len(median_values) > 1 else None
                    ),
                    "mean_examples_per_second_across_seeds": statistics.mean(
                        float(row["examples_per_second"]) for row in seed_rows
                    ),
                    "speedup_vs_parent_by_seed": {
                        str(seed): value for seed, value in zip(seeds, speedups)
                    },
                    "mean_speedup_vs_parent": statistics.mean(speedups) if speedups else None,
                }
    return by_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v1_deberta_rte.json",
    )
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_parent_rte",
    )
    parser.add_argument(
        "--pruned-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_latency",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(DEFAULT_METHODS),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--batches-per-size", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide unique seeds.")
    if not args.methods or len(set(args.methods)) != len(args.methods):
        raise ValueError("Provide unique methods.")
    if any(size < 1 for size in args.batch_sizes) or (
        len(set(args.batch_sizes)) != len(args.batch_sizes)
    ):
        raise ValueError("Batch sizes must be positive and unique.")
    if args.batches_per_size < 1 or args.warmup_steps < 1 or args.repeats < 5:
        raise ValueError("Require batches-per-size >= 1, warmup >= 1 and repeats >= 5.")
    if "parent" not in args.methods and any(
        method != "parent" for method in args.methods
    ):
        print("WARNING: parent baseline omitted; speedups versus parent will not be computed.")

    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_root}. "
                "Choose another output root or pass --overwrite."
            )
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    config = _read_json(args.config)
    revision = str(config["model"]["revision"])
    if len(revision) != 40:
        raise ValueError("Latency benchmark requires a pinned 40-character base revision.")
    device = _device(args.device)
    runs: list[dict[str, Any]] = []
    total_runs = len(args.seeds) * len(args.methods)
    completed = 0

    # Rotate the fixed method order by seed to reduce consistent order effects.
    for seed_index, seed in enumerate(args.seeds):
        parent_adapter = args.parent_root / f"seed_{seed}" / "parent_adapter"
        tokenizer = AutoTokenizer.from_pretrained(parent_adapter, use_fast=True)
        cpu_batches = _representative_batches(
            config, tokenizer, args.batch_sizes, args.batches_per_size
        )
        method_order = list(args.methods)
        if method_order:
            offset = seed_index % len(method_order)
            method_order = method_order[offset:] + method_order[:offset]

        for method in method_order:
            adapter_path = _adapter_path(
                method, seed, args.parent_root, args.pruned_root
            )
            artifact_info = _validate_adapter_artifact(
                method,
                seed,
                adapter_path,
                revision,
                args.parent_root,
                args.pruned_root,
            )
            print(
                f"[{completed + 1}/{total_runs}] Benchmarking seed={seed}, "
                f"method={method}, device={device}",
                flush=True,
            )
            result = _benchmark_adapter(
                method,
                seed,
                config,
                adapter_path,
                cpu_batches,
                device,
                warmup_steps=args.warmup_steps,
                repeats=args.repeats,
            )
            result.update(artifact_info)
            runs.append(result)
            completed += 1
            _write_json(args.output_root / f"seed_{seed}_{method}.json", result)
            _write_json(
                args.output_root / "benchmark_progress.json",
                {
                    "completed_method_seed_runs": completed,
                    "expected_method_seed_runs": total_runs,
                    "completed": [
                        {"seed": item["seed"], "method": item["method"]}
                        for item in runs
                    ],
                },
            )

    summary = {
        "analysis_name": "Merged and unmerged LoRA inference latency benchmark",
        "base_model": config["model"]["id"],
        "base_model_revision": revision,
        "dataset": config["dataset"],
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": _package_version("peft"),
        "seeds": args.seeds,
        "methods": args.methods,
        "batch_sizes": args.batch_sizes,
        "batches_per_size": args.batches_per_size,
        "warmup_steps_per_batch": args.warmup_steps,
        "timed_repeats_per_batch": args.repeats,
        "runs": runs,
        "aggregate": _summarize(runs, args.seeds, args.methods, args.batch_sizes),
        "measurement_scope": (
            "Synchronized model-forward latency on fixed representative tokenized RTE "
            "validation batches. Tokenization, collation, host-to-device transfer, "
            "model loading and adapter merging are excluded from forward latency. "
            "Merged/unmerged output equivalence is checked before timing. "
            "Results are hardware/runtime-specific and do not establish production "
            "serving latency or end-to-end speedup."
        ),
        "run_count": len(runs),
        "all_requested_runs_completed": len(runs) == total_runs,
    }
    report_path = args.output_root / "benchmark_summary.json"
    _write_json(report_path, summary)
    print(f"Saved latency benchmark: {report_path}")
    print("\nMEAN MEDIAN LATENCY BY MODE / METHOD / BATCH SIZE")
    for key, value in summary["aggregate"].items():
        print(
            f"{key}: {value['median_latency_ms_mean_across_seeds']:.3f} ms; "
            f"mean speedup vs parent={value['mean_speedup_vs_parent']}"
        )


if __name__ == "__main__":
    main()
