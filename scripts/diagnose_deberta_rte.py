"""Short numerical preflight for the DeBERTa-v3-base / GLUE RTE setup.

This diagnostic separates model-loading/forward issues from PEFT, backward,
and early optimizer-update issues. It does not produce a research result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model_load_dtype_kwargs() -> dict[str, torch.dtype]:
    """Select the explicit FP32 loading argument for Transformers v4 or v5."""
    major_version = int(transformers.__version__.split(".", maxsplit=1)[0])
    if major_version >= 5:
        return {"dtype": torch.float32}
    return {"torch_dtype": torch.float32}


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _string_list(values: Any) -> list[str]:
    """Convert Transformers loading-info collections to JSON-safe strings."""
    if values is None:
        return []
    return sorted(str(value) for value in values)


def _finite(label: str, tensor: torch.Tensor) -> None:
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        finite_values = tensor[finite]
        if finite_values.numel():
            bounds = (
                float(finite_values.min().item()),
                float(finite_values.max().item()),
            )
        else:
            bounds = None
        raise RuntimeError(
            f"{label} contains non-finite values; "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"finite_min_max={bounds}"
        )


def _load_data(config: dict[str, Any], tokenizer: Any, batch_size: int, seed: int):
    dataset_cfg = config["dataset"]
    raw = load_dataset(dataset_cfg["name"], dataset_cfg["config"])
    train_raw = raw[dataset_cfg.get("train_split", "train")]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=int(dataset_cfg.get("max_length", 256)),
        )
        encoded["label"] = batch["label"]
        return encoded

    tokenized = train_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_raw.column_names,
        desc=f"Tokenizing GLUE {dataset_cfg['config']} diagnostic split",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )


def _model_batch(batch: dict[str, torch.Tensor], device: torch.device):
    labels = batch["labels"].to(device)
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels"
    }
    return inputs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v1_deberta_rte.json",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(args.device)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model: {model_cfg['id']}")
    print(f"Configured learning rate: {training_cfg['learning_rate']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        use_fast=True,
    )
    loader = _load_data(
        config,
        tokenizer,
        int(training_cfg["per_device_batch_size"]),
        seed,
    )
    first_batch = next(iter(loader))
    print("First batch labels:", first_batch["labels"].tolist())
    print("First batch input shape:", tuple(first_batch["input_ids"].shape))

    base_model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        num_labels=int(model_cfg.get("num_labels", 2)),
        output_loading_info=True,
        **_model_load_dtype_kwargs(),
    )
    base_model.config.problem_type = "single_label_classification"
    print("Loading report:")
    print(json.dumps({
        "missing_keys": _string_list(loading_info.get("missing_keys", [])),
        "unexpected_keys": _string_list(loading_info.get("unexpected_keys", [])),
        "mismatched_keys": _string_list(loading_info.get("mismatched_keys", [])),
        "error_msgs": _string_list(loading_info.get("error_msgs", [])),
    }, indent=2))
    loaded_dtypes = {parameter.dtype for parameter in base_model.parameters()}
    print("Loaded parameter dtypes:", sorted(str(dtype) for dtype in loaded_dtypes))
    if loaded_dtypes != {torch.float32}:
        raise RuntimeError(
            "Expected all base-model parameters to be float32, "
            f"got {loaded_dtypes}."
        )
    base_model.to(device)
    base_model.train()
    inputs, labels = _model_batch(first_batch, device)

    with torch.no_grad():
        base_outputs = base_model(**inputs, labels=labels)
    _finite("Base-model logits", base_outputs.logits)
    _finite("Base-model loss", base_outputs.loss)
    print(
        "Base-model forward passed:",
        f"loss={float(base_outputs.loss.item()):.8f}",
        f"logit_min={float(base_outputs.logits.min().item()):.6f}",
        f"logit_max={float(base_outputs.logits.max().item()):.6f}",
    )

    target_modules = list(lora_cfg["target_modules"])
    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=target_modules,
            modules_to_save=list(lora_cfg.get("modules_to_save", [])) or None,
            bias="none",
        ),
    )
    peft_model.to(device)
    peft_model.train()
    peft_dtypes = {parameter.dtype for parameter in peft_model.parameters()}
    if peft_dtypes != {torch.float32}:
        raise RuntimeError(f"Expected all PEFT model parameters to be float32, got {peft_dtypes}.")
    trainable = [
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    lora_names = [
        name for name, _ in trainable if "lora_A" in name or "lora_B" in name
    ]
    if not lora_names:
        raise RuntimeError("PEFT created no trainable LoRA A/B parameters.")
    print(f"Trainable parameters: {len(trainable)} tensors")
    print("Example LoRA parameter names:", lora_names[:6])

    outputs = peft_model(**inputs, labels=labels)
    _finite("PEFT initial logits", outputs.logits)
    _finite("PEFT initial loss", outputs.loss)
    print(f"PEFT initial forward passed: loss={float(outputs.loss.item()):.8f}")

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        eps=float(training_cfg.get("adam_epsilon", 1e-8)),
    )
    optimizer.zero_grad(set_to_none=True)
    outputs.loss.backward()

    bad_gradients = [
        name for name, parameter in trainable
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if bad_gradients:
        raise RuntimeError(
            "Non-finite gradients immediately after the first backward pass: "
            + ", ".join(bad_gradients[:10])
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in trainable],
        max_norm=float(training_cfg["max_grad_norm"]),
        error_if_nonfinite=True,
    )
    print(f"First backward passed: clipped global norm input={float(grad_norm.item()):.8f}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    bad_parameters = [
        name for name, parameter in trainable
        if not bool(torch.isfinite(parameter).all().item())
    ]
    if bad_parameters:
        raise RuntimeError(
            "Non-finite trainable parameters after the first optimizer step: "
            + ", ".join(bad_parameters[:10])
        )

    print("Checking subsequent batches after optimizer updates...")
    iterator = iter(loader)
    next(iterator, None)  # The first batch was already used for the initial update.
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_inputs, batch_labels = _model_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        step_outputs = peft_model(**batch_inputs, labels=batch_labels)
        _finite(f"Step {step} logits", step_outputs.logits)
        _finite(f"Step {step} loss", step_outputs.loss)
        step_outputs.loss.backward()
        bad_name = next(
            (
                name for name, parameter in trainable
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all().item())
            ),
            None,
        )
        if bad_name is not None:
            raise RuntimeError(f"Non-finite gradient at diagnostic step {step}: {bad_name}")
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable],
            max_norm=float(training_cfg["max_grad_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        bad_parameters = [
            name for name, parameter in trainable
            if not bool(torch.isfinite(parameter).all().item())
        ]
        if bad_parameters:
            raise RuntimeError(
                f"Non-finite trainable parameter after diagnostic step {step}: "
                + ", ".join(bad_parameters[:10])
            )
        print(
            f"step={step} loss={float(step_outputs.loss.item()):.8f} "
            f"grad_norm={float(norm.item()):.8f}",
            flush=True,
        )

    print("Numerical preflight passed. This is not a research training run.")


if __name__ == "__main__":
    main()
