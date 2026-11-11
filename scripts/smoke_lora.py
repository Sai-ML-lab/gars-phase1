"""Tiny end-to-end Transformers + PEFT LoRA smoke test.

This only verifies package integration and one optimizer step. It is NOT a
scientific experiment and must not be reported as a GARS result.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "hf-internal-testing/tiny-random-bert"


def main() -> int:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading public smoke-test model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
    )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.to(device)
    model.train()

    batch = tokenizer(
        ["A short positive example.", "A different negative example."],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = torch.tensor([1, 0], dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch, labels=labels)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("Smoke-test loss is missing or non-finite")
    loss.backward()

    grad_norms = [
        parameter.grad.detach().norm().item()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not grad_norms:
        raise RuntimeError("No trainable LoRA gradients were produced")
    if not all(torch.isfinite(torch.tensor(value)) for value in grad_norms):
        raise RuntimeError("Non-finite LoRA gradient detected")

    optimizer.step()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if device.type == "mps":
        torch.mps.synchronize()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "device": str(device),
        "loss": float(loss.detach().cpu().item()),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "lora_gradient_tensor_count": len(grad_norms),
        "lora_gradient_norm_sum": float(sum(grad_norms)),
        "optimizer_step": "passed",
        "note": "Integration smoke test only; not a research result.",
    }

    out_dir = Path("results/day1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "lora_smoke_test.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved smoke-test result to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
