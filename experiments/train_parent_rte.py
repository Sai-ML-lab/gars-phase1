#!/usr/bin/env python3
"""Train one uncompressed DeBERTa/RTE LoRA parent and record its final-window gradients.

This script creates the common parent checkpoint used by the post-hoc V1
allocation-policy comparison. It does not prune or modify adapter rank during
training.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

# Allow direct execution from a source checkout before package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gars.importance import GradientImportanceAccumulator  # noqa: E402


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        requested = torch.device(preference)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return requested
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"model", "dataset", "training", "lora", "gars", "seeds"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Experiment config is missing keys: {sorted(missing)}")
    return config


def _prepare_datasets(config: dict[str, Any], tokenizer: Any):
    dataset_cfg = config["dataset"]
    raw = load_dataset(dataset_cfg["name"], dataset_cfg["config"])
    train_raw = raw[dataset_cfg.get("train_split", "train")]
    validation_raw = raw[dataset_cfg.get("validation_split", "validation")]

    required_columns = {"sentence1", "sentence2", "label"}
    if not required_columns.issubset(set(train_raw.column_names)):
        raise ValueError(
            "The configured GLUE task is not a sentence-pair classification task "
            "with sentence1, sentence2 and label columns."
        )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=int(dataset_cfg.get("max_length", 256)),
        )
        encoded["label"] = batch["label"]
        return encoded

    train = train_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing RTE train split",
    )
    validation = validation_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation_raw.column_names,
        desc="Tokenizing RTE validation split",
    )
    return train, validation


def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            outputs = model(**model_inputs, labels=labels)
            losses.append(float(outputs.loss.detach().float().cpu().item()))
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels.cpu().tolist())

    if not references:
        raise RuntimeError("Validation dataloader produced no examples.")

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(f1_score(references, predictions, average="macro")),
        "num_examples": int(len(references)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def train_seed(
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    device: torch.device,
) -> Path:
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        use_fast=True,
    )
    train_dataset, validation_dataset = _prepare_datasets(config, tokenizer)

    batch_size = int(training_cfg["per_device_batch_size"])
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    if batch_size < 1 or grad_accum < 1:
        raise ValueError("Batch size and gradient accumulation must be positive.")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        num_labels=int(model_cfg.get("num_labels", 2)),
    )
    base_model.config.problem_type = "single_label_classification"

    target_modules = list(lora_cfg["target_modules"])
    available_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    missing_targets = [name for name in target_modules if name not in available_suffixes]
    if missing_targets:
        raise RuntimeError(
            "Configured LoRA target modules were not found in the loaded model: "
            f"{missing_targets}. Available linear-module suffixes include: "
            f"{sorted(available_suffixes)[:80]}"
        )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=target_modules,
        modules_to_save=list(lora_cfg.get("modules_to_save", [])) or None,
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("PEFT created no trainable parameters.")
    if not any("lora_A" in name or "lora_B" in name for name, p in model.named_parameters()
               if p.requires_grad):
        raise RuntimeError("No trainable LoRA A/B factors were found.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    epochs = int(training_cfg["epochs"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(total_optimizer_steps * float(training_cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    capture_window = int(training_cfg["gradient_capture_last_optimizer_steps"])
    if capture_window < 1 or capture_window > total_optimizer_steps:
        raise ValueError(
            f"Gradient capture window {capture_window} must be between 1 and "
            f"the total optimizer steps ({total_optimizer_steps})."
        )
    capture_start_step = total_optimizer_steps - capture_window + 1
    gradient_accumulator = GradientImportanceAccumulator()

    seed_dir = output_root / f"seed_{seed}"
    adapter_dir = seed_dir / "parent_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "training_log.jsonl"

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    training_log: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        total_batches = len(train_loader)

        for batch_index, batch in enumerate(train_loader):
            # Normalize a final partial accumulation group by its actual size.
            group_start = (batch_index // grad_accum) * grad_accum
            group_size = min(grad_accum, total_batches - group_start)

            labels = batch["labels"].to(device)
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            outputs = model(**model_inputs, labels=labels)
            loss = outputs.loss
            epoch_losses.append(float(loss.detach().float().cpu().item()))
            (loss / group_size).backward()

            is_group_end = (
                (batch_index + 1) % grad_accum == 0
                or (batch_index + 1) == total_batches
            )
            if not is_group_end:
                continue

            global_step += 1
            if global_step >= capture_start_step:
                gradient_accumulator.update(model.named_parameters())

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(training_cfg["max_grad_norm"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        validation_metrics = _evaluate(model, validation_loader, device)
        log_entry = {
            "epoch": epoch + 1,
            "optimizer_step": global_step,
            "mean_train_loss": float(np.mean(epoch_losses)),
            **validation_metrics,
        }
        training_log.append(log_entry)
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} "
            f"step={global_step}/{total_optimizer_steps} "
            f"train_loss={log_entry['mean_train_loss']:.5f} "
            f"val_acc={validation_metrics['accuracy']:.5f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.5f}",
            flush=True,
        )

    if global_step != total_optimizer_steps:
        raise RuntimeError(
            f"Expected {total_optimizer_steps} optimizer steps; observed {global_step}."
        )
    if gradient_accumulator.steps_seen != capture_window:
        raise RuntimeError(
            "Gradient capture count mismatch: expected "
            f"{capture_window}, observed {gradient_accumulator.steps_seen}."
        )

    final_metrics = _evaluate(model, validation_loader, device)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    gradient_scores = gradient_accumulator.mean_scores()
    if not gradient_scores:
        raise RuntimeError("No per-layer gradient scores were accumulated.")

    _write_json(
        seed_dir / "gradient_importance.json",
        {
            "seed": seed,
            "capture_start_optimizer_step": capture_start_step,
            "capture_end_optimizer_step": global_step,
            "capture_steps": gradient_accumulator.steps_seen,
            "aggregation": "arithmetic_mean_of_per_step_joint_lora_A_B_frobenius_norms",
            "importance_by_transformer_block": {
                str(layer_id): score for layer_id, score in gradient_scores.items()
            },
            "accumulator_state": gradient_accumulator.state_dict(),
        },
    )

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    lora_factor_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())

    metadata = {
        "experiment_id": config.get("experiment_id"),
        "seed": seed,
        "base_model": model_cfg["id"],
        "base_model_revision": getattr(model.config, "_commit_hash", None),
        "dataset": dataset_cfg,
        "train_dataset_fingerprint": getattr(train_dataset, "_fingerprint", None),
        "validation_dataset_fingerprint": getattr(validation_dataset, "_fingerprint", None),
        "device": str(device),
        "python_version": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ["torch", "transformers", "peft", "datasets", "numpy", "scikit-learn"]
        },
        "training_config": training_cfg,
        "lora_config": lora_cfg,
        "total_optimizer_steps": total_optimizer_steps,
        "gradient_capture_steps": gradient_accumulator.steps_seen,
        "trainable_parameter_count_including_modules_to_save": trainable_count,
        "lora_A_B_factor_parameter_count": lora_factor_count,
        "total_model_parameter_count": total_count,
        "final_validation": final_metrics,
        "epoch_metrics": training_log,
        "note": (
            "Unpruned parent adapter. This is a controlled research run, not a "
            "pruning result. Policies are applied post hoc to this same checkpoint."
        ),
    }
    _write_json(seed_dir / "run_metadata.json", metadata)

    with log_path.open("w", encoding="utf-8") as handle:
        for entry in training_log:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    print(f"Saved parent adapter and metadata: {seed_dir}", flush=True)
    return seed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v1_deberta_rte.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_parent_rte",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only this seed. Omit to run all seeds in the config.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, mps, cuda, or cpu (default: auto; prefers MPS on Apple Silicon).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    device = _device(args.device)
    seeds = [args.seed] if args.seed is not None else [int(s) for s in config["seeds"]]

    print(f"Experiment: {config.get('experiment_id')}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Output: {args.output_root}")

    for seed in seeds:
        train_seed(config, seed, args.output_root, device)


if __name__ == "__main__":
    main()
