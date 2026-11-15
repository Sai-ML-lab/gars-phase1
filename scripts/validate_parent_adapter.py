"""Inspect and reload a trained seed's saved PEFT parent adapter.

This verifies the saved A/B tensor inventory and reproduces validation metrics
from the saved parent checkpoint. It is a serialization check, not a pruning
or compression experiment.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _model_load_dtype_kwargs() -> dict[str, torch.dtype]:
    """Select the explicit FP32 loading argument for Transformers v4 or v5."""
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


def _load_validation(
    config: dict[str, Any],
    tokenizer: Any,
    batch_size: int,
) -> DataLoader:
    dataset_cfg = config["dataset"]
    raw = load_dataset(dataset_cfg["name"], dataset_cfg["config"])
    validation_raw = raw[dataset_cfg.get("validation_split", "validation")]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=int(dataset_cfg.get("max_length", 256)),
        )

    validation = validation_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation_raw.column_names,
        desc="Tokenizing RTE validation split for adapter reload check",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )


def _inspect_adapter_tensors(
    adapter_path: Path,
    expected_lora_parameters: int,
    expected_rank: int,
) -> dict[str, Any]:
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Adapter weights are missing: {weights_path}")

    state = load_file(str(weights_path), device="cpu")
    a_pattern = re.compile(r"\.lora_A(?:\.[^.]+)?\.weight$")
    b_pattern = re.compile(r"\.lora_B(?:\.[^.]+)?\.weight$")
    a_tensors = {key: value for key, value in state.items() if a_pattern.search(key)}
    b_tensors = {key: value for key, value in state.items() if b_pattern.search(key)}

    if not a_tensors or len(a_tensors) != len(b_tensors):
        raise RuntimeError(
            "Saved adapter has missing or unmatched LoRA factors: "
            f"A tensors={len(a_tensors)}, B tensors={len(b_tensors)}."
        )

    def factor_modules(
        factors: dict[str, torch.Tensor],
        pattern: re.Pattern[str],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, tensor in factors.items():
            module_name = pattern.sub("", key)
            if module_name in result:
                raise RuntimeError(f"Duplicate saved factor for {module_name}.")
            result[module_name] = tensor
        return result

    a_by_module = factor_modules(a_tensors, a_pattern)
    b_by_module = factor_modules(b_tensors, b_pattern)
    if set(a_by_module) != set(b_by_module):
        missing_b = sorted(set(a_by_module) - set(b_by_module))
        missing_a = sorted(set(b_by_module) - set(a_by_module))
        raise RuntimeError(
            f"LoRA A/B module mismatch; missing B={missing_b}, missing A={missing_a}."
        )

    module_ranks: dict[str, int] = {}
    factor_parameters = 0
    for module_name in sorted(a_by_module):
        a_tensor = a_by_module[module_name]
        b_tensor = b_by_module[module_name]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise RuntimeError(f"LoRA factors must be matrices: {module_name}.")
        rank_a = int(a_tensor.shape[0])
        rank_b = int(b_tensor.shape[1])
        if rank_a != rank_b:
            raise RuntimeError(
                f"Rank mismatch for {module_name}: A rank={rank_a}, B rank={rank_b}."
            )
        if rank_a != expected_rank:
            raise RuntimeError(
                f"Expected parent rank {expected_rank} for {module_name}, got {rank_a}."
            )
        module_ranks[module_name] = rank_a
        factor_parameters += a_tensor.numel() + b_tensor.numel()

    if factor_parameters != expected_lora_parameters:
        raise RuntimeError(
            "Saved LoRA factor count does not match training metadata: "
            f"file={factor_parameters}, metadata={expected_lora_parameters}."
        )

    classifier_keys = [key for key in state if "classifier" in key]
    pooler_keys = [key for key in state if "pooler" in key]
    if not classifier_keys or not pooler_keys:
        raise RuntimeError(
            "Saved adapter is missing classifier/pooler modules_to_save tensors: "
            f"classifier keys={classifier_keys}, pooler keys={pooler_keys}."
        )

    return {
        "weights_file": str(weights_path),
        "weights_file_bytes": weights_path.stat().st_size,
        "adapter_tensor_count": len(state),
        "lora_module_count": len(module_ranks),
        "lora_factor_parameter_count": factor_parameters,
        "lora_rank_histogram": {
            str(rank): sum(value == rank for value in module_ranks.values())
            for rank in sorted(set(module_ranks.values()))
        },
        "lora_modules": module_ranks,
        "classifier_tensor_count": len(classifier_keys),
        "pooler_tensor_count": len(pooler_keys),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
    }


def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    references: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            outputs = model(**inputs, labels=labels)
            if not bool(torch.isfinite(outputs.logits).all().item()):
                raise RuntimeError("Reloaded adapter produced non-finite validation logits.")
            if not bool(torch.isfinite(outputs.loss).all().item()):
                raise RuntimeError("Reloaded adapter produced a non-finite validation loss.")
            losses.append(float(outputs.loss.detach().float().cpu().item()))
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels.cpu().tolist())

    if not references:
        raise RuntimeError("Validation dataloader produced no examples.")
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(f1_score(references, predictions, average="macro")),
        "num_examples": len(references),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v1_deberta_rte.json",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_parent_rte" / "seed_42",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--loss-tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute difference in validation loss.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata_path = args.seed_dir / "run_metadata.json"
    gradients_path = args.seed_dir / "gradient_importance.json"
    adapter_path = args.seed_dir / "parent_adapter"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gradients = json.loads(gradients_path.read_text(encoding="utf-8"))

    if metadata.get("seed") != gradients.get("seed"):
        raise RuntimeError("Seed mismatch between run metadata and gradient capture.")
    if gradients.get("capture_steps") != int(
        config["training"]["gradient_capture_last_optimizer_steps"]
    ):
        raise RuntimeError("Gradient-capture step count does not match the V1 config.")
    if len(gradients.get("importance_by_transformer_block", {})) != 12:
        raise RuntimeError("Expected gradient scores for all 12 DeBERTa Transformer blocks.")

    tensor_report = _inspect_adapter_tensors(
        adapter_path,
        expected_lora_parameters=int(metadata["lora_A_B_factor_parameter_count"]),
        expected_rank=int(config["lora"]["rank"]),
    )

    device = _device(args.device)
    model_cfg = config["model"]
    model_id = metadata.get("base_model", model_cfg["id"])
    revision = metadata.get("base_model_revision") or model_cfg.get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=int(model_cfg.get("num_labels", 2)),
        **_model_load_dtype_kwargs(),
    )
    base_model.config.problem_type = "single_label_classification"
    base_dtypes = {parameter.dtype for parameter in base_model.parameters()}
    if base_dtypes != {torch.float32}:
        raise RuntimeError(f"Fresh base model is not FP32: {base_dtypes}.")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
    model.to(device)
    model.eval()
    model_dtypes = {parameter.dtype for parameter in model.parameters()}
    if model_dtypes != {torch.float32}:
        raise RuntimeError(f"Reloaded PEFT model is not FP32: {model_dtypes}.")

    validation_loader = _load_validation(
        config,
        tokenizer,
        batch_size=int(config["training"]["per_device_batch_size"]),
    )
    reproduced = _evaluate(model, validation_loader, device)
    reference = metadata["final_validation"]

    differences = {
        key: abs(float(reproduced[key]) - float(reference[key]))
        for key in ("loss", "accuracy", "macro_f1")
    }
    if reproduced["num_examples"] != reference["num_examples"]:
        raise RuntimeError(
            "Validation example-count mismatch: "
            f"reloaded={reproduced['num_examples']}, "
            f"metadata={reference['num_examples']}."
        )
    if differences["accuracy"] > 1e-12 or differences["macro_f1"] > 1e-12:
        raise RuntimeError(
            "Reloaded adapter predictions do not reproduce parent validation metrics: "
            f"absolute differences={differences}."
        )
    if differences["loss"] > args.loss_tolerance:
        raise RuntimeError(
            "Reloaded adapter validation loss differs beyond tolerance: "
            f"difference={differences['loss']}, tolerance={args.loss_tolerance}."
        )

    report = {
        "seed": metadata["seed"],
        "device": str(device),
        "base_model": model_id,
        "base_model_revision": revision,
        "base_model_parameter_dtype": str(next(base_model.parameters()).dtype),
        "reloaded_model_parameter_dtype": str(next(model.parameters()).dtype),
        "tensor_inventory": tensor_report,
        "reference_validation": reference,
        "reloaded_validation": reproduced,
        "absolute_metric_differences": differences,
        "loss_tolerance": args.loss_tolerance,
        "reload_validation": "passed",
    }
    report_path = args.seed_dir / "reload_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved reload report: {report_path}")


if __name__ == "__main__":
    main()
