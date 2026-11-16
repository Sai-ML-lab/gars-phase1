"""Analyze final-window gradient temporal reliability without retraining."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from gars.policy import gars_thresholds
from gars.temporal import analyze_stepwise_gradient_scores

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 999)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def _threshold_temporal_summary(
    stepwise: list[dict[str, Any]],
    tau_base: float,
    beta: float,
    tau_min: float,
    tau_max: float,
) -> dict[str, Any]:
    by_layer: dict[int, list[float]] = {}
    lower_clips = 0
    upper_clips = 0
    total = 0
    for row in stepwise:
        scores = {
            int(layer): float(value)
            for layer, value in row["importance_by_transformer_block"].items()
        }
        thresholds = gars_thresholds(
            scores,
            tau_base=tau_base,
            beta=beta,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        for layer, value in thresholds.items():
            by_layer.setdefault(layer, []).append(value)
            lower_clips += abs(value - tau_min) <= 1e-9
            upper_clips += abs(value - tau_max) <= 1e-9
            total += 1

    return {
        "fixed_tau_base_from_observed_v1_calibration": tau_base,
        "beta": beta,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "total_step_layer_thresholds": total,
        "lower_bound_hits": lower_clips,
        "upper_bound_hits": upper_clips,
        "lower_bound_hit_fraction": lower_clips / total if total else None,
        "upper_bound_hit_fraction": upper_clips / total if total else None,
        "per_layer": {
            str(layer): {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
                "lower_bound_hits": sum(abs(v - tau_min) <= 1e-9 for v in values),
                "upper_bound_hits": sum(abs(v - tau_max) <= 1e-9 for v in values),
            }
            for layer, values in sorted(by_layer.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v1_deberta_rte.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned" / "temporal_gradient_analysis.json",
    )
    args = parser.parse_args()

    config = _read_json(args.config)
    gars_cfg = config["gars"]
    seed_reports: dict[str, Any] = {}
    for seed in args.seeds:
        gradient_path = args.parent_root / f"seed_{seed}" / "gradient_importance.json"
        gradient_data = _read_json(gradient_path)
        if int(gradient_data.get("seed", -1)) != seed:
            raise ValueError(f"Gradient artifact seed mismatch for seed {seed}.")
        stepwise = gradient_data.get("stepwise_importance_by_optimizer_step")
        if not isinstance(stepwise, list) or not stepwise:
            raise RuntimeError(
                f"Seed {seed} has no saved stepwise gradient scores. "
                "This historical artifact cannot support temporal analysis; "
                "rerun that parent seed with the current trainer to persist the window."
            )
        if len(stepwise) != int(gradient_data.get("capture_steps", -1)):
            raise ValueError(f"Stepwise score count differs from capture_steps for seed {seed}.")

        observed_means = gradient_data["importance_by_transformer_block"]
        temporal = analyze_stepwise_gradient_scores(stepwise, observed_means)
        allocation = _read_json(
            args.pruned_root / f"seed_{seed}" / "gars" / "allocation.json"
        )
        if allocation.get("reload_validation") != "passed":
            raise RuntimeError(f"Observed GARS reload validation failed for seed {seed}.")
        tau_base = float(allocation["threshold_parameter"])
        thresholds = _threshold_temporal_summary(
            stepwise,
            tau_base=tau_base,
            beta=float(gars_cfg["beta"]),
            tau_min=float(gars_cfg["tau_min"]),
            tau_max=float(gars_cfg["tau_max"]),
        )
        seed_reports[str(seed)] = {
            "capture_window": {
                "capture_steps": gradient_data["capture_steps"],
                "start_optimizer_step": gradient_data.get("capture_start_optimizer_step"),
                "end_optimizer_step": gradient_data.get("capture_end_optimizer_step"),
                "aggregation": gradient_data.get("aggregation"),
            },
            "rank_and_variability": temporal,
            "instantaneous_threshold_dynamics_at_fixed_v1_tau_base": thresholds,
        }

    pairwise_means = [
        seed_reports[str(seed)]["rank_and_variability"][
            "within_window_rank_stability"
        ]["mean_pairwise_spearman_rho"]
        for seed in args.seeds
    ]
    adjacent_means = [
        seed_reports[str(seed)]["rank_and_variability"][
            "within_window_rank_stability"
        ]["adjacent_step_mean_spearman_rho"]
        for seed in args.seeds
    ]
    report = {
        "analysis_name": "GARS final-window temporal gradient reliability",
        "base_model": config["model"]["id"],
        "base_model_revision": config["model"]["revision"],
        "dataset": config["dataset"],
        "seeds": args.seeds,
        "design": {
            "window": "final 50 optimizer updates saved by the parent trainer",
            "rank_reliability": (
                "all pairwise step-level Spearman correlations of layer importance"
            ),
            "instantaneous_thresholds": (
                "recompute tau_l for each captured step using the V1 calibrated "
                "tau_base, beta and clipping bounds; diagnostic only, not a new policy"
            ),
            "aggregation_validation": (
                "stepwise arithmetic means are checked against the saved V1 "
                "importance_by_transformer_block profile"
            ),
        },
        "across_seed_summary": {
            "mean_within_seed_all_pair_rank_rho": (
                statistics.mean(pairwise_means) if pairwise_means else None
            ),
            "sample_sd_within_seed_all_pair_rank_rho": (
                statistics.stdev(pairwise_means) if len(pairwise_means) > 1 else None
            ),
            "mean_adjacent_step_rank_rho": (
                statistics.mean(adjacent_means) if adjacent_means else None
            ),
            "sample_sd_adjacent_step_rank_rho": (
                statistics.stdev(adjacent_means) if len(adjacent_means) > 1 else None
            ),
        },
        "per_seed": seed_reports,
        "inference_boundary": (
            "Descriptive temporal reliability diagnostics from the final capture "
            "window. Updates are sequential and autocorrelated; optimizer steps are "
            "not independent samples. No p-values or significance claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved temporal gradient analysis: {args.output}")
    print(json.dumps(report["across_seed_summary"], indent=2))
    for seed, values in seed_reports.items():
        stability = values["rank_and_variability"]["within_window_rank_stability"]
        clipping = values["instantaneous_threshold_dynamics_at_fixed_v1_tau_base"]
        print(
            f"seed={seed}: all-pair-rho={stability['mean_pairwise_spearman_rho']:.5f}; "
            f"adjacent-rho={stability['adjacent_step_mean_spearman_rho']:.5f}; "
            f"threshold-clipping lower={clipping['lower_bound_hits']}, "
            f"upper={clipping['upper_bound_hits']}"
        )


if __name__ == "__main__":
    main()
