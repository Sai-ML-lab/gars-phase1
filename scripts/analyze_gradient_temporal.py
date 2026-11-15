"""Analyze within-window stability of saved per-step layer gradient scores."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from gars.diagnostics import descriptive_stats, spearman_rank_correlation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 999)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing gradient artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def _rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    return ranks


def _load_step_matrix(path: Path, expected_steps: int | None) -> tuple[list[int], list[int], np.ndarray]:
    artifact = _read_json(path)
    step_records = artifact.get("stepwise_importance_by_optimizer_step")
    if not isinstance(step_records, list) or not step_records:
        raise ValueError(
            f"{path} does not contain stepwise_importance_by_optimizer_step. "
            "Retrain with the updated runner to persist individual capture steps."
        )
    if expected_steps is not None and len(step_records) != expected_steps:
        raise ValueError(
            f"{path} contains {len(step_records)} steps; expected {expected_steps}."
        )

    layer_ids: list[int] | None = None
    optimizer_steps: list[int] = []
    rows: list[list[float]] = []
    for record in step_records:
        step = int(record["optimizer_step"])
        importance = {
            int(layer): float(score)
            for layer, score in record["importance_by_transformer_block"].items()
        }
        current_layers = sorted(importance)
        if layer_ids is None:
            layer_ids = current_layers
        elif current_layers != layer_ids:
            raise ValueError(f"Layer inventory changed within the capture window in {path}.")
        if any(not np.isfinite(score) or score < 0.0 for score in importance.values()):
            raise ValueError(f"Non-finite or negative gradient score in {path}, step {step}.")
        optimizer_steps.append(step)
        rows.append([importance[layer] for layer in layer_ids])

    if optimizer_steps != sorted(optimizer_steps) or len(set(optimizer_steps)) != len(optimizer_steps):
        raise ValueError(f"Optimizer-step IDs are duplicated or unsorted in {path}.")
    if not layer_ids:
        raise ValueError(f"No layer IDs found in {path}.")
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid stepwise gradient matrix in {path}.")
    return optimizer_steps, layer_ids, matrix


def _analyze_seed(
    seed: int,
    path: Path,
    expected_steps: int | None,
) -> dict[str, Any]:
    optimizer_steps, layer_ids, matrix = _load_step_matrix(path, expected_steps)
    step_count, layer_count = matrix.shape
    mean_profile = matrix.mean(axis=0)
    mean_rank = _rank_descending(mean_profile)

    step_mean_rhos: list[float] = []
    adjacent_rhos: list[float] = []
    rank_matrix = np.vstack([_rank_descending(row) for row in matrix])
    for row in matrix:
        rho = spearman_rank_correlation(row.tolist(), mean_profile.tolist())
        if rho is not None:
            step_mean_rhos.append(rho)
    for index in range(step_count - 1):
        rho = spearman_rank_correlation(
            rank_matrix[index].tolist(),
            rank_matrix[index + 1].tolist(),
        )
        if rho is not None:
            adjacent_rhos.append(rho)

    midpoint = step_count // 2
    early_mean = matrix[:midpoint].mean(axis=0)
    late_mean = matrix[midpoint:].mean(axis=0)
    early_late_rho = spearman_rank_correlation(
        early_mean.tolist(),
        late_mean.tolist(),
    )

    per_layer: list[dict[str, Any]] = []
    layer_cv: list[float] = []
    for index, layer_id in enumerate(layer_ids):
        values = matrix[:, index]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if step_count > 1 else 0.0
        cv = std_value / mean_value if mean_value > 0.0 else None
        if cv is not None:
            layer_cv.append(cv)
        rank_values = rank_matrix[:, index]
        per_layer.append(
            {
                "layer_id": layer_id,
                "gradient_mean": mean_value,
                "gradient_sample_std": std_value,
                "gradient_coefficient_of_variation": cv,
                "mean_rank_1_is_highest": float(np.mean(rank_values)),
                "rank_sample_std": (
                    float(np.std(rank_values, ddof=1)) if step_count > 1 else 0.0
                ),
            }
        )

    normalized_matrix = np.divide(
        matrix,
        matrix.mean(axis=1, keepdims=True),
        out=np.ones_like(matrix),
        where=matrix.mean(axis=1, keepdims=True) > 0.0,
    )
    normalized_mean = normalized_matrix.mean(axis=0)
    normalized_profile_correlation = spearman_rank_correlation(
        normalized_mean.tolist(),
        (mean_profile / mean_profile.mean()).tolist()
        if mean_profile.mean() > 0.0
        else mean_profile.tolist(),
    )

    return {
        "seed": seed,
        "capture_optimizer_steps": [optimizer_steps[0], optimizer_steps[-1]],
        "step_count": step_count,
        "layer_count": layer_count,
        "stepwise_gradient_mean_profile": {
            str(layer): float(value)
            for layer, value in zip(layer_ids, mean_profile, strict=True)
        },
        "stepwise_normalized_mean_profile": {
            str(layer): float(value)
            for layer, value in zip(layer_ids, normalized_mean, strict=True)
        },
        "step_profile_vs_window_mean_rank_correlation": descriptive_stats(step_mean_rhos),
        "adjacent_step_rank_correlation": descriptive_stats(adjacent_rhos),
        "early_half_vs_late_half_mean_profile_spearman_rho": early_late_rho,
        "within_layer_gradient_cv": descriptive_stats(layer_cv),
        "normalized_vs_raw_mean_profile_spearman_rho": normalized_profile_correlation,
        "per_layer": per_layer,
        "raw_stepwise_gradient_matrix": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_parent_mrpc",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_parent_mrpc" / "temporal_gradient_analysis.json",
    )
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide one or more unique seeds.")
    if args.expected_steps < 2:
        raise ValueError("expected-steps must be at least 2.")

    per_seed = {
        str(seed): _analyze_seed(
            seed,
            args.parent_root / f"seed_{seed}" / "gradient_importance.json",
            args.expected_steps,
        )
        for seed in args.seeds
    }
    normalized_profiles = {
        seed: values["stepwise_normalized_mean_profile"]
        for seed, values in per_seed.items()
    }
    from gars.diagnostics import pairwise_profile_correlations

    cross_seed = pairwise_profile_correlations(normalized_profiles)
    cross_seed_rhos = [
        float(item["spearman_rho"])
        for item in cross_seed
        if item["spearman_rho"] is not None
    ]
    report = {
        "analysis_name": "Within-window gradient signal temporal stability",
        "seeds": args.seeds,
        "expected_steps": args.expected_steps,
        "design": {
            "capture": "final optimizer updates; joint per-block LoRA A/B gradient Frobenius norm",
            "step_aggregation": "arithmetic mean of step-level scores",
            "adjacent_rank_stability": "Spearman correlation of consecutive optimizer-step layer rankings",
            "within_layer_cv": "sample standard deviation divided by mean over captured updates",
            "temporal_claim_boundary": (
                "Descriptive stability of the captured final window only; does not "
                "establish reliability earlier in training or causal value for allocation."
            ),
        },
        "cross_seed_normalized_mean_profile_rank_stability": {
            "pairwise": cross_seed,
            "rho_mean": statistics.mean(cross_seed_rhos) if cross_seed_rhos else None,
            "rho_sample_std": (
                statistics.stdev(cross_seed_rhos)
                if len(cross_seed_rhos) > 1
                else None
            ),
        },
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved temporal gradient analysis: {args.output}")
    for seed, values in per_seed.items():
        print(
            f"seed={seed}: step-vs-window rho="
            f"{values['step_profile_vs_window_mean_rank_correlation']['mean']}, "
            f"adjacent-step rho={values['adjacent_step_rank_correlation']['mean']}, "
            f"mean layer CV={values['within_layer_gradient_cv']['mean']}, "
            f"early-vs-late rho={values['early_half_vs_late_half_mean_profile_spearman_rho']}"
        )


if __name__ == "__main__":
    main()
