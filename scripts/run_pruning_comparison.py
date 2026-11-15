"""Build and evaluate post-hoc physical LoRA rank-pruned adapters.

For one trained seed, this script creates three PEFT-compatible exports:
uniform local squared-energy, global singular-value cutoff, and GARS. It keeps
the original LoRA scaling multiplier constant through per-module alpha_pattern.
Each export is reloaded on a fresh FP32 base model and evaluated on the same
GLUE RTE validation split as the parent run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from gars.export import alpha_for_preserved_scaling, peft_module_pattern_key
from gars.importance import cyclically_shift_importance, transformer_layer_index
from gars.policy import (
    AdapterSpectrum,
    BudgetCalibration,
    calibrate_gars_budget,
    calibrate_global_singular_value_budget,
    calibrate_uniform_energy_budget,
    physical_adapter_parameter_count,
)
from gars.spectral import (
    normalized_squared_spectral_entropy,
    qr_svd_lora_update,
    squared_energy_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_FACTOR_PATTERN = re.compile(
    r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$"
)
_SAVED_MODEL_PREFIX = "base_model.model."


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _factor_inventory(
    state: dict[str, torch.Tensor],
    expected_parent_rank: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, tensor in state.items():
        match = _FACTOR_PATTERN.match(key)
        if match is None:
            continue
        module_name = match.group("module")
        factor = match.group("factor")
        if factor in grouped.setdefault(module_name, {}):
            raise ValueError(f"Duplicate LoRA {factor} factor for {module_name}.")
        grouped[module_name][factor] = (key, tensor)

    if not grouped:
        raise ValueError("No LoRA A/B factors found in the parent adapter.")
    for module_name, factors in grouped.items():
        if set(factors) != {"A", "B"}:
            raise ValueError(
                f"Incomplete LoRA factors for {module_name}: found {sorted(factors)}."
            )
        a_key, a_tensor = factors["A"]
        b_key, b_tensor = factors["B"]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(f"LoRA factors must be matrices for {module_name}.")
        if a_tensor.shape[0] != b_tensor.shape[1]:
            raise ValueError(f"LoRA A/B rank mismatch for {module_name}.")
        if int(a_tensor.shape[0]) != expected_parent_rank:
            raise ValueError(
                f"Expected parent rank {expected_parent_rank} for {module_name}, "
                f"found {a_tensor.shape[0]}."
            )
        if a_tensor.dtype != torch.float32 or b_tensor.dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 parent factors for {module_name}, "
                f"got A={a_tensor.dtype}, B={b_tensor.dtype}."
            )
        factors["A"] = (a_key, a_tensor)
        factors["B"] = (b_key, b_tensor)
    return grouped


def _load_spectra(
    groups: dict[str, dict[str, Any]],
) -> tuple[list[AdapterSpectrum], dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    spectra: list[AdapterSpectrum] = []
    decompositions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for module_name in sorted(groups):
        peft_name = module_name.removeprefix(_SAVED_MODEL_PREFIX)
        if peft_name == module_name:
            raise ValueError(
                f"Saved LoRA key does not have expected PEFT prefix: {module_name}."
            )
        layer_id = transformer_layer_index(peft_name)
        if layer_id is None:
            raise ValueError(f"Could not parse Transformer block from {module_name}.")
        _, a_tensor = groups[module_name]["A"]
        _, b_tensor = groups[module_name]["B"]
        u, singular_values, vh = qr_svd_lora_update(a_tensor, b_tensor)
        spectra.append(
            AdapterSpectrum(
                module_name=module_name,
                layer_id=layer_id,
                singular_values=singular_values,
                in_features=int(a_tensor.shape[1]),
                out_features=int(b_tensor.shape[0]),
            )
        )
        decompositions[module_name] = (u, singular_values, vh)
    return spectra, decompositions


def _load_validation(
    config: dict[str, Any],
    tokenizer: Any,
    batch_size: int,
) -> DataLoader:
    dataset_cfg = config["dataset"]
    raw = load_dataset(dataset_cfg["name"], dataset_cfg["config"])
    validation_raw = raw[dataset_cfg.get("validation_split", "validation")]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=int(dataset_cfg.get("max_length", 256)),
        )
        encoded["label"] = batch["label"]
        return encoded

    validation = validation_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation_raw.column_names,
        desc="Tokenizing RTE validation split for pruning comparison",
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


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    batch_losses: list[float] = []
    predictions: list[int] = []
    references: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            outputs = model(**inputs, labels=labels)
            if not bool(torch.isfinite(outputs.logits).all().item()):
                raise RuntimeError("Non-finite validation logits in pruned adapter.")
            if not bool(torch.isfinite(outputs.loss).all().item()):
                raise RuntimeError("Non-finite validation loss in pruned adapter.")
            batch_losses.append(float(outputs.loss.detach().float().cpu().item()))
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels.cpu().tolist())
    if not references:
        raise RuntimeError("Validation loader produced no examples.")
    return {
        "loss": float(np.mean(batch_losses)),
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(f1_score(references, predictions, average="macro")),
        "num_examples": len(references),
    }


def _make_policy_allocations(
    spectra: list[AdapterSpectrum],
    importance: dict[int, float],
    target_parameters: int,
    gars_cfg: dict[str, Any],
) -> dict[str, BudgetCalibration]:
    min_rank = int(gars_cfg["min_rank"])
    tau_min = float(gars_cfg["tau_min"])
    tau_max = float(gars_cfg["tau_max"])
    beta = float(gars_cfg["beta"])
    return {
        "uniform_local_energy": calibrate_uniform_energy_budget(
            spectra,
            target_parameters=target_parameters,
            min_rank=min_rank,
            tau_min=tau_min,
            tau_max=tau_max,
        ),
        "global_singular_value": calibrate_global_singular_value_budget(
            spectra,
            target_parameters=target_parameters,
            min_rank=min_rank,
        ),
        "gars": calibrate_gars_budget(
            spectra,
            importance_by_layer=importance,
            target_parameters=target_parameters,
            beta=beta,
            min_rank=min_rank,
            tau_min=tau_min,
            tau_max=tau_max,
        ),
    }


def _make_truncated_pair(
    decomposition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    retained_rank: int,
    original_a: torch.Tensor,
    original_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u, singular_values, vh = decomposition
    if not 1 <= retained_rank <= singular_values.numel():
        raise ValueError(f"Invalid retained rank {retained_rank}.")
    roots = torch.sqrt(torch.clamp(singular_values[:retained_rank], min=0.0))
    b_new = (u[:, :retained_rank] * roots.unsqueeze(0)).to(
        dtype=original_b.dtype
    )
    a_new = (roots.unsqueeze(1) * vh[:retained_rank, :]).to(
        dtype=original_a.dtype
    )
    if a_new.shape != (retained_rank, original_a.shape[1]):
        raise RuntimeError("Exported A factor has an unexpected physical shape.")
    if b_new.shape != (original_b.shape[0], retained_rank):
        raise RuntimeError("Exported B factor has an unexpected physical shape.")
    return a_new.contiguous().cpu(), b_new.contiguous().cpu()


def _build_variant(
    parent_adapter_dir: Path,
    output_dir: Path,
    state: dict[str, torch.Tensor],
    groups: dict[str, dict[str, Any]],
    spectra: list[AdapterSpectrum],
    decompositions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    allocation: BudgetCalibration,
    method: str,
    parent_config: dict[str, Any],
    parent_rank: int,
    parent_alpha: float,
    use_rslora: bool,
    parent_factor_parameters: int,
    target_parameters: int,
    target_fraction: float,
    importance: dict[int, float],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Remove it or use --overwrite."
        )
    output_dir.mkdir(parents=True)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir()

    for source in parent_adapter_dir.iterdir():
        if source.is_file() and source.name not in {
            "adapter_model.safetensors",
            "adapter_config.json",
        }:
            shutil.copy2(source, adapter_dir / source.name)

    new_state = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in state.items()
    }
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, float] = {}
    per_module: dict[str, Any] = {}

    for spec in spectra:
        module_name = spec.module_name
        retained_rank = int(allocation.ranks_by_module[module_name])
        a_key, a_original = groups[module_name]["A"]
        b_key, b_original = groups[module_name]["B"]
        a_new, b_new = _make_truncated_pair(
            decompositions[module_name],
            retained_rank,
            a_original,
            b_original,
        )
        new_state[a_key] = a_new
        new_state[b_key] = b_new

        peft_name = module_name.removeprefix(_SAVED_MODEL_PREFIX)
        pattern_key = peft_module_pattern_key(module_name)
        if pattern_key in rank_pattern or pattern_key in alpha_pattern:
            raise RuntimeError(f"Duplicate PEFT module pattern for {module_name}.")
        rank_pattern[pattern_key] = retained_rank
        alpha_pattern[pattern_key] = alpha_for_preserved_scaling(
            parent_rank,
            parent_alpha,
            retained_rank,
            use_rslora=use_rslora,
        )

        if method == "gars":
            threshold = allocation.thresholds_by_layer[spec.layer_id]
        elif method == "uniform_local_energy":
            threshold = allocation.threshold_parameter
        else:
            threshold = None

        per_module[module_name] = {
            "peft_module_name": peft_name,
            "layer_id": spec.layer_id,
            "parent_rank": parent_rank,
            "retained_rank": retained_rank,
            "squared_energy_retained": squared_energy_ratio(
                spec.singular_values, retained_rank
            ),
            "squared_spectral_entropy": normalized_squared_spectral_entropy(
                spec.singular_values
            ),
            "energy_threshold": threshold,
            "global_singular_value_cutoff": (
                allocation.threshold_parameter
                if method == "global_singular_value"
                else None
            ),
            "A_shape": list(a_new.shape),
            "B_shape": list(b_new.shape),
            "physical_factor_parameters": int(a_new.numel() + b_new.numel()),
        }

    config_out = dict(parent_config)
    config_out["rank_pattern"] = rank_pattern
    config_out["alpha_pattern"] = alpha_pattern
    config_out["r"] = parent_rank
    config_out["lora_alpha"] = parent_alpha
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config_out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    weights_path = adapter_dir / "adapter_model.safetensors"
    save_file(new_state, str(weights_path), metadata={"format": "pt"})

    actual_parameters = sum(
        int(tensors["A"][1].numel() + tensors["B"][1].numel())
        for tensors in groups.values()
    )
    exported_parameters = sum(
        int(new_state[tensors["A"][0]].numel() + new_state[tensors["B"][0]].numel())
        for tensors in groups.values()
    )
    if actual_parameters != parent_factor_parameters:
        raise RuntimeError(
            f"Parent factor count mismatch: inventory={actual_parameters}, "
            f"metadata={parent_factor_parameters}."
        )
    if exported_parameters != allocation.achieved_parameters:
        raise RuntimeError(
            f"Exported factor count mismatch: tensors={exported_parameters}, "
            f"allocation={allocation.achieved_parameters}."
        )

    if method == "gars":
        thresholds_by_layer = {
            str(layer_id): float(value)
            for layer_id, value in allocation.thresholds_by_layer.items()
        }
    elif method == "uniform_local_energy":
        thresholds_by_layer = {
            str(layer_id): float(allocation.threshold_parameter)
            for layer_id in sorted(importance)
        }
    else:
        thresholds_by_layer = {}

    parent_weight_path = parent_adapter_dir / "adapter_model.safetensors"
    adapter_directory_bytes = sum(
        path.stat().st_size for path in adapter_dir.iterdir() if path.is_file()
    )
    report = {
        "method": method,
        "parent_rank": parent_rank,
        "parent_alpha": parent_alpha,
        "use_rslora": use_rslora,
        "parent_scaling": (
            parent_alpha / math.sqrt(parent_rank)
            if use_rslora
            else parent_alpha / parent_rank
        ),
        "target_parameter_fraction": target_fraction,
        "target_parameters": target_parameters,
        "target_in_reachable_range": allocation.target_in_reachable_range,
        "achieved_parameters": allocation.achieved_parameters,
        "parameter_reduction_fraction": 1.0
        - allocation.achieved_parameters / parent_factor_parameters,
        "absolute_budget_error": allocation.absolute_error,
        "relative_budget_error": allocation.relative_error,
        "threshold_parameter": allocation.threshold_parameter,
        "thresholds_by_layer": thresholds_by_layer,
        "importance_by_layer": {str(k): v for k, v in importance.items()},
        "rank_histogram": {
            str(rank): sum(
                item["retained_rank"] == rank for item in per_module.values()
            )
            for rank in sorted(
                {item["retained_rank"] for item in per_module.values()}
            )
        },
        "parent_lora_weight_file_bytes": parent_weight_path.stat().st_size,
        "pruned_lora_weight_file_bytes": weights_path.stat().st_size,
        "lora_weight_file_size_reduction_fraction": 1.0
        - weights_path.stat().st_size / parent_weight_path.stat().st_size,
        "adapter_directory_bytes_including_tokenizer": adapter_directory_bytes,
        "per_module": per_module,
        "adapter_directory": str(adapter_dir),
    }
    (output_dir / "allocation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _check_loaded_ranks_and_scaling(
    model: PeftModel,
    expected_ranks: dict[str, int],
    expected_scaling: float,
) -> None:
    actual_modules: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or "default" not in lora_a:
            continue
        peft_name = name.removeprefix(_SAVED_MODEL_PREFIX)
        actual_modules[peft_name] = module

    if set(actual_modules) != set(expected_ranks):
        raise RuntimeError(
            "Reloaded LoRA module names differ from allocation. "
            f"Missing={sorted(set(expected_ranks) - set(actual_modules))}; "
            f"unexpected={sorted(set(actual_modules) - set(expected_ranks))}."
        )

    for name, expected_rank in expected_ranks.items():
        module = actual_modules[name]
        actual_rank = int(module.lora_A["default"].weight.shape[0])
        actual_b_rank = int(module.lora_B["default"].weight.shape[1])
        if actual_rank != expected_rank or actual_b_rank != expected_rank:
            raise RuntimeError(
                f"PEFT did not apply the requested rank for {name}: "
                f"A={actual_rank}, B={actual_b_rank}, expected={expected_rank}."
            )
        actual_scaling = float(module.scaling["default"])
        if not math.isclose(
            actual_scaling,
            expected_scaling,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"PEFT LoRA scaling changed for {name}: "
                f"actual={actual_scaling}, expected={expected_scaling}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v1_deberta_rte.json",
    )
    parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 999])
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_parent_rte",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--importance-cyclic-shift",
        type=int,
        default=0,
        help=(
            "Cyclically reassign observed layer-gradient scores before GARS "
            "allocation; 0 preserves the observed layer-to-score assignment."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["uniform_local_energy", "global_singular_value", "gars"],
        default=None,
        help="Optional subset of policies to export; default runs all three.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-floating-revision",
        action="store_true",
        help="Allow revision='main'; use only after verifying the cached base snapshot.",
    )
    args = parser.parse_args()

    config = _load_json(args.config)
    seed_dir = args.parent_root / f"seed_{args.seed}"
    parent_adapter_dir = seed_dir / "parent_adapter"
    metadata = _load_json(seed_dir / "run_metadata.json")
    gradient_data = _load_json(seed_dir / "gradient_importance.json")
    parent_config = _load_json(parent_adapter_dir / "adapter_config.json")
    if metadata.get("seed") != args.seed or gradient_data.get("seed") != args.seed:
        raise RuntimeError("Requested seed does not match parent metadata/gradient capture.")

    capture_steps = int(config["training"]["gradient_capture_last_optimizer_steps"])
    if int(gradient_data.get("capture_steps", -1)) != capture_steps:
        raise RuntimeError("Parent gradient capture count does not match the V1 config.")
    original_importance = {
        int(key): float(value)
        for key, value in gradient_data["importance_by_transformer_block"].items()
    }
    if len(original_importance) != 12:
        raise RuntimeError(
            f"Expected 12 block-importance scores, got {len(original_importance)}."
        )
    layer_ids = sorted(original_importance)
    if not 0 <= args.importance_cyclic_shift < len(layer_ids):
        raise ValueError(
            "--importance-cyclic-shift must be between zero and the number "
            "of captured layers minus one."
        )
    importance = cyclically_shift_importance(
        original_importance,
        shift=args.importance_cyclic_shift,
    )
    source_layer_by_destination_layer = {
        str(layer_id): str(
            layer_ids[(index - args.importance_cyclic_shift) % len(layer_ids)]
        )
        for index, layer_id in enumerate(layer_ids)
    }
    importance_assignment = {
        "kind": (
            "observed_gradient_layer_alignment"
            if args.importance_cyclic_shift == 0
            else "cyclically_shifted_gradient_layer_alignment"
        ),
        "cyclic_shift": args.importance_cyclic_shift,
        "source_layer_by_destination_layer": source_layer_by_destination_layer,
        "original_importance_by_layer": {
            str(key): value for key, value in original_importance.items()
        },
        "assigned_importance_by_layer": {
            str(key): value for key, value in importance.items()
        },
    }

    parent_revision = metadata.get("base_model_revision")
    config_revision = config["model"].get("revision", "main")
    model_revision = parent_revision or config_revision
    if not re.fullmatch(r"[0-9a-f]{40}", str(model_revision)):
        if not args.allow_floating_revision:
            raise RuntimeError(
                "The parent run does not identify an immutable 40-character model "
                f"revision (resolved value: {model_revision!r}). Retrieve the cached "
                "Hugging Face snapshot SHA, pin it in the config, and rerun. If you "
                "have independently verified that all three parent runs used the same "
                "cached snapshot, you may explicitly pass --allow-floating-revision."
            )
        print(
            "WARNING: proceeding with a floating model revision. "
            "This weakens checkpoint provenance."
        )

    parent_rank = int(parent_config["r"])
    parent_alpha = float(parent_config["lora_alpha"])
    use_rslora = bool(parent_config.get("use_rslora", False))
    unsupported_flags = [
        key for key in ("use_dora", "lora_bias", "use_qalora")
        if bool(parent_config.get(key, False))
    ]
    if unsupported_flags:
        raise ValueError(
            "The V1 physical SVD exporter currently supports standard LoRA only; "
            f"unsupported config flags: {unsupported_flags}."
        )
    if bool(parent_config.get("fan_in_fan_out", False)):
        raise ValueError("fan_in_fan_out=True is not supported by this V1 exporter.")
    if parent_config.get("rank_pattern") or parent_config.get("alpha_pattern"):
        raise ValueError(
            "Expected the parent adapter to use a uniform rank/alpha; "
            "non-empty parent patterns require separate handling."
        )

    parent_weights_path = parent_adapter_dir / "adapter_model.safetensors"
    state = load_file(str(parent_weights_path), device="cpu")
    groups = _factor_inventory(state, expected_parent_rank=parent_rank)
    spectra, decompositions = _load_spectra(groups)

    observed_layers = {spec.layer_id for spec in spectra}
    if observed_layers != set(importance):
        raise RuntimeError(
            "Gradient-importance blocks do not match adapted Transformer blocks: "
            f"spectrum={sorted(observed_layers)}, gradients={sorted(importance)}."
        )
    expected_module_count = (
        len(importance) * len(config["lora"]["target_modules"])
    )
    if len(spectra) != expected_module_count:
        raise RuntimeError(
            f"Expected {expected_module_count} adapted projections, found {len(spectra)}."
        )

    parent_ranks = {spec.module_name: spec.max_rank for spec in spectra}
    parent_factor_parameters = physical_adapter_parameter_count(spectra, parent_ranks)
    if parent_factor_parameters != int(metadata["lora_A_B_factor_parameter_count"]):
        raise RuntimeError(
            "Parent A/B factor count does not match run metadata: "
            f"spectrum={parent_factor_parameters}, "
            f"metadata={metadata['lora_A_B_factor_parameter_count']}."
        )

    target_fraction = float(config["gars"]["target_parameter_fraction"])
    target_parameters = round(parent_factor_parameters * target_fraction)
    allocations = _make_policy_allocations(
        spectra,
        importance,
        target_parameters,
        config["gars"],
    )
    if args.methods is not None:
        allocations = {
            method: allocations[method]
            for method in args.methods
        }
    output_seed_dir = args.output_root / f"seed_{args.seed}"
    if output_seed_dir.exists() and any(output_seed_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_seed_dir}. "
            "Use --overwrite to replace existing policy exports."
        )
    if output_seed_dir.exists() and args.overwrite:
        shutil.rmtree(output_seed_dir)
    output_seed_dir.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(parent_adapter_dir, use_fast=True)
    validation_loader = _load_validation(
        config,
        tokenizer,
        batch_size=int(config["training"]["per_device_batch_size"]),
    )

    summaries: dict[str, Any] = {}
    original_scaling = (
        parent_alpha / math.sqrt(parent_rank)
        if use_rslora
        else parent_alpha / parent_rank
    )

    for method, allocation in allocations.items():
        print(
            f"Building {method}: parameters={allocation.achieved_parameters}/"
            f"{parent_factor_parameters}, threshold={allocation.threshold_parameter}",
            flush=True,
        )
        method_dir = output_seed_dir / method
        report = _build_variant(
            parent_adapter_dir=parent_adapter_dir,
            output_dir=method_dir,
            state=state,
            groups=groups,
            spectra=spectra,
            decompositions=decompositions,
            allocation=allocation,
            method=method,
            parent_config=parent_config,
            parent_rank=parent_rank,
            parent_alpha=parent_alpha,
            use_rslora=use_rslora,
            parent_factor_parameters=parent_factor_parameters,
            target_parameters=target_parameters,
            target_fraction=target_fraction,
            importance=importance,
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            config["model"]["id"],
            revision=model_revision,
            num_labels=int(config["model"].get("num_labels", 2)),
            **_model_load_dtype_kwargs(),
        )
        base_model.config.problem_type = "single_label_classification"
        base_dtypes = {parameter.dtype for parameter in base_model.parameters()}
        if base_dtypes != {torch.float32}:
            raise RuntimeError(f"Fresh base model is not FP32: {base_dtypes}.")
        model = PeftModel.from_pretrained(
            base_model,
            method_dir / "adapter",
            is_trainable=False,
        )
        model.to(device)
        model.eval()
        model_dtypes = {parameter.dtype for parameter in model.parameters()}
        if model_dtypes != {torch.float32}:
            raise RuntimeError(f"Reloaded pruned model is not FP32: {model_dtypes}.")

        expected_ranks = {
            spec.module_name.removeprefix(_SAVED_MODEL_PREFIX):
            int(allocation.ranks_by_module[spec.module_name])
            for spec in spectra
        }
        _check_loaded_ranks_and_scaling(
            model,
            expected_ranks=expected_ranks,
            expected_scaling=original_scaling,
        )
        validation = _evaluate(model, validation_loader, device)
        parent_validation = metadata["final_validation"]
        differences = {
            key: float(validation[key]) - float(parent_validation[key])
            for key in ("accuracy", "macro_f1", "loss")
        }
        report["importance_assignment"] = importance_assignment
        report["validation"] = validation
        report["parent_validation"] = parent_validation
        report["validation_delta_vs_parent"] = differences
        report["reload_validation"] = "passed"
        report["base_model_revision"] = model_revision
        report["device"] = str(device)
        report["model_parameter_dtype"] = "float32"
        (method_dir / "allocation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[method] = {
            "importance_cyclic_shift": args.importance_cyclic_shift,
            "achieved_parameters": report["achieved_parameters"],
            "parameter_reduction_fraction": report["parameter_reduction_fraction"],
            "adapter_weight_file_bytes": report["pruned_lora_weight_file_bytes"],
            "validation": validation,
            "validation_delta_vs_parent": differences,
            "reload_validation": "passed",
            "rank_histogram": report["rank_histogram"],
        }
        print(
            f"{method}: accuracy={validation['accuracy']:.5f}, "
            f"macro_f1={validation['macro_f1']:.5f}, "
            f"loss={validation['loss']:.5f}, "
            f"physical_reduction={report['parameter_reduction_fraction']:.3%}",
            flush=True,
        )

        del model, base_model
        if device.type == "mps":
            torch.mps.empty_cache()

    comparison = {
        "seed": args.seed,
        "base_model": config["model"]["id"],
        "base_model_revision": model_revision,
        "dataset": config["dataset"],
        "target_parameter_fraction": target_fraction,
        "parent_lora_A_B_parameters": parent_factor_parameters,
        "target_parameters": target_parameters,
        "parent_validation": metadata["final_validation"],
        "importance_assignment": importance_assignment,
        "policies": summaries,
        "note": (
            "Post-hoc SVD truncation of the same parent adapter. Physical ranks "
            "are read back from the reloaded PEFT modules and LoRA scaling is "
            "checked against the original parent multiplier."
        ),
    }
    comparison_path = output_seed_dir / "comparison_summary.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison summary: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
