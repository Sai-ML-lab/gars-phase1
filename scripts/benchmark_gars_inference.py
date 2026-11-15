"""Benchmark parent and pruned LoRA adapters in merged/unmerged inference modes.

Measures synchronized wall-clock forward latency on a fixed prefix of the GLUE
validation split. Model loading and adapter merging are excluded from timings.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover - Python 3.11+ has importlib.metadata
    version = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_VARIANTS = (
    "parent",
    "uniform_local_energy",
    "global_singular_value",
    "gars",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _model_load_dtype_kwargs() -> dict[str, torch.dtype]:
    major_version = int(transformers.__version__.split(".", maxsplit=1)[0])
    if major_version >= 5:
        return {"dtype": torch.float32}
    return {"torch_dtype": torch.float32}


def _package_version(name: str) -> str | None:
    if version is None:
        return None
    try:
        return version(name)
    except Exception:  # Package metadata may not exist in editable environments.
        return None


def _load_validation_batches(
    config: dict[str, Any],
    tokenizer: Any,
    batch_sizes: list[int],
    sample_count: int,
) -> dict[int, list[dict[str, torch.Tensor]]]:
    dataset_config = config["dataset"]
    raw = load_dataset(dataset_config["name"], dataset_config["config"])
    validation = raw[dataset_config.get("validation_split", "validation")]
    required = {"sentence1", "sentence2"}
    if not required.issubset(validation.column_names):
        raise ValueError(
            "This benchmark currently expects a sentence-pair classification "
            "dataset with sentence1 and sentence2 columns."
        )
    sample_count = min(sample_count, len(validation))
    if sample_count < max(batch_sizes):
        raise ValueError("sample_count must be at least the largest batch size.")
    validation = validation.select(range(sample_count))

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=int(dataset_config.get("max_length", 256)),
        )

    tokenized = validation.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation.column_names,
        desc="Tokenizing fixed validation subset for inference benchmark",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    batches: dict[int, list[dict[str, torch.Tensor]]] = {}
    for batch_size in batch_sizes:
        loader = DataLoader(
            tokenized,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collator,
            num_workers=0,
        )
        batches[batch_size] = list(loader)
        if not batches[batch_size]:
            raise RuntimeError(f"No full validation batches for batch_size={batch_size}.")
    return batches


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"labels", "label"}
    }


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        return {
            "mps_current_allocated_bytes": int(current()) if callable(current) else None,
            "mps_driver_allocated_bytes": int(driver()) if callable(driver) else None,
        }
    if device.type == "cuda":
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    return {"device_allocated_bytes": None, "device_reserved_bytes": None}


def _logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    model_inputs = _to_device(batch, device)
    with torch.inference_mode():
        output = model(**model_inputs)
    _synchronize(device)
    if not bool(torch.isfinite(output.logits).all().item()):
        raise RuntimeError("Non-finite logits encountered during inference benchmark.")
    return output.logits.detach().float().cpu()


def _measure_mode(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for index in range(warmup_steps):
            _ = model(**_to_device(batches[index % len(batches)], device))
            _synchronize(device)

        _synchronize(device)
        latencies_seconds: list[float] = []
        total_examples = 0
        start_total = time.perf_counter()
        for index in range(timed_steps):
            batch = batches[index % len(batches)]
            model_inputs = _to_device(batch, device)
            start = time.perf_counter()
            output = model(**model_inputs)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            if not bool(torch.isfinite(output.logits).all().item()):
                raise RuntimeError("Non-finite logits encountered during timed inference.")
            latencies_seconds.append(elapsed)
            total_examples += int(next(iter(model_inputs.values())).shape[0])
        _synchronize(device)
        total_elapsed = time.perf_counter() - start_total

    latency_ms = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    return {
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "timed_examples": total_examples,
        "mean_batch_latency_ms": float(np.mean(latency_ms)),
        "median_batch_latency_ms": float(np.median(latency_ms)),
        "p90_batch_latency_ms": float(np.percentile(latency_ms, 90)),
        "p95_batch_latency_ms": float(np.percentile(latency_ms, 95)),
        "sample_std_batch_latency_ms": (
            float(statistics.stdev(latency_ms.tolist()))
            if latency_ms.size > 1
            else None
        ),
        "mean_latency_ms_per_example": float(np.mean(latency_ms) / batch_size),
        "throughput_examples_per_second": float(total_examples / total_elapsed),
        "total_elapsed_seconds": float(total_elapsed),
    }


def _adapter_directory(
    variant: str,
    seed: int,
    parent_root: Path,
    pruned_root: Path,
) -> Path:
    if variant == "parent":
        return parent_root / f"seed_{seed}" / "parent_adapter"
    return pruned_root / f"seed_{seed}" / variant / "adapter"


def _load_adapter_model(
    adapter_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> PeftModel:
    base = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        num_labels=int(config["model"].get("num_labels", 2)),
        **_model_load_dtype_kwargs(),
    )
    base.config.problem_type = "single_label_classification"
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.to(device)
    model.eval()
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {torch.float32}:
        raise RuntimeError(f"Expected FP32 adapter model; found dtypes {dtypes}.")
    return model


def _run_variant(
    variant: str,
    seed: int,
    adapter_dir: Path,
    config: dict[str, Any],
    batches_by_size: dict[int, list[dict[str, torch.Tensor]]],
    batch_sizes: list[int],
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
    merge_tolerance: float,
) -> dict[str, Any]:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    if device.type == "mps":
        torch.mps.empty_cache()

    model = _load_adapter_model(adapter_dir, config, device)
    _synchronize(device)
    memory_after_load = _memory_snapshot(device)

    unmerged_results = {}
    for batch_size in batch_sizes:
        unmerged_results[str(batch_size)] = _measure_mode(
            model,
            batches_by_size[batch_size],
            batch_size,
            device,
            warmup_steps,
            timed_steps,
        )

    # Check merged/unmerged numerical equivalence on the same fixed batch.
    equivalence_batch = batches_by_size[batch_sizes[0]][0]
    logits_unmerged = _logits(model, equivalence_batch, device)
    if not hasattr(model, "merge_and_unload"):
        raise RuntimeError("The loaded PEFT model does not expose merge_and_unload().")
    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.to(device)
    merged_model.eval()
    logits_merged = _logits(merged_model, equivalence_batch, device)
    difference = (logits_unmerged - logits_merged).abs()
    max_abs_diff = float(difference.max().item())
    mean_abs_diff = float(difference.mean().item())
    if max_abs_diff > merge_tolerance:
        raise RuntimeError(
            f"Merged/unmerged output mismatch for {variant}, seed={seed}: "
            f"max_abs_diff={max_abs_diff:.6g}, tolerance={merge_tolerance:.6g}."
        )

    merged_results = {}
    for batch_size in batch_sizes:
        merged_results[str(batch_size)] = _measure_mode(
            merged_model,
            batches_by_size[batch_size],
            batch_size,
            device,
            warmup_steps,
            timed_steps,
        )

    _synchronize(device)
    memory_after_benchmark = _memory_snapshot(device)
    result = {
        "seed": seed,
        "variant": variant,
        "adapter_directory": str(adapter_dir),
        "base_model_revision": config["model"]["revision"],
        "model_dtype": "float32",
        "device": str(device),
        "merge_equivalence": {
            "status": "passed",
            "comparison_batch_size": int(equivalence_batch["input_ids"].shape[0]),
            "max_absolute_logit_difference": max_abs_diff,
            "mean_absolute_logit_difference": mean_abs_diff,
            "tolerance": merge_tolerance,
        },
        "memory_after_model_load": memory_after_load,
        "memory_after_benchmark": memory_after_benchmark,
        "unmerged": unmerged_results,
        "merged": merged_results,
    }
    del merged_model, model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_latency" / "latency_benchmark.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(DEFAULT_VARIANTS),
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--merge-tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide one or more unique seeds.")
    if not args.variants or len(set(args.variants)) != len(args.variants):
        raise ValueError("Provide one or more unique variants.")
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        raise ValueError("Batch sizes must be positive.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Batch sizes must be unique.")
    if args.sample_count < max(args.batch_sizes):
        raise ValueError("sample-count must be at least the largest batch size.")
    if args.warmup_steps < 0 or args.timed_steps < 2:
        raise ValueError("Require warmup_steps >= 0 and timed_steps >= 2.")
    if args.merge_tolerance <= 0:
        raise ValueError("merge-tolerance must be positive.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    config = _read_json(args.config)
    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.parent_root / f"seed_{args.seeds[0]}" / "parent_adapter",
        use_fast=True,
    )
    batches_by_size = _load_validation_batches(
        config,
        tokenizer,
        args.batch_sizes,
        args.sample_count,
    )

    report: dict[str, Any] = {
        "experiment_name": "GARS V1 merged and unmerged inference benchmark",
        "base_model": config["model"]["id"],
        "base_model_revision": config["model"]["revision"],
        "dataset": config["dataset"],
        "device": str(device),
        "python_version": platform.python_version(),
        "package_versions": {
            package: _package_version(package)
            for package in ("torch", "transformers", "peft", "datasets")
        },
        "design": {
            "model_loading_time_included": False,
            "adapter_merge_time_included": False,
            "input_transfer_included": True,
            "synchronization": "synchronize after every forward on MPS/CUDA",
            "inference_mode": "torch.inference_mode; model.eval",
            "validation_subset": "first sample_count examples in official validation order",
            "drop_last": True,
            "sample_count": min(args.sample_count, len(batches_by_size[args.batch_sizes[0]]) * args.batch_sizes[0]),
            "warmup_steps_per_condition": args.warmup_steps,
            "timed_steps_per_condition": args.timed_steps,
            "merge_equivalence_tolerance": args.merge_tolerance,
            "batch_sizes": args.batch_sizes,
            "latency_statistics": (
                "mean, median, p90, p95 and sample SD of synchronized per-batch wall-clock time"
            ),
        },
        "results": [],
        "completed_conditions": 0,
        "expected_conditions": len(args.seeds) * len(args.variants),
        "all_requested_conditions_completed": False,
    }
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    _write_json(args.output, report)

    for seed in args.seeds:
        for variant in args.variants:
            adapter_dir = _adapter_directory(
                variant,
                seed,
                args.parent_root,
                args.pruned_root,
            )
            print(
                f"Benchmarking seed={seed}, variant={variant}, device={device}",
                flush=True,
            )
            result = _run_variant(
                variant,
                seed,
                adapter_dir,
                config,
                batches_by_size,
                args.batch_sizes,
                device,
                args.warmup_steps,
                args.timed_steps,
                args.merge_tolerance,
            )
            report["results"].append(result)
            report["completed_conditions"] += 1
            report["all_requested_conditions_completed"] = (
                report["completed_conditions"] == report["expected_conditions"]
            )
            _write_json(args.output, report)
            for mode in ("unmerged", "merged"):
                for batch_size in args.batch_sizes:
                    value = result[mode][str(batch_size)]
                    print(
                        f"  {mode:8s} B={batch_size:2d}: "
                        f"median={value['median_batch_latency_ms']:.3f} ms/batch, "
                        f"throughput={value['throughput_examples_per_second']:.1f} examples/s",
                        flush=True,
                    )

    print(f"Saved latency benchmark: {args.output}", flush=True)
    print(
        f"Completed {report['completed_conditions']}/{report['expected_conditions']} conditions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
