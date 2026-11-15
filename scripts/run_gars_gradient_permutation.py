"""Run a same-parent cyclic gradient-layer alignment ablation.

Each non-identity cyclic shift preserves a seed's exact 12 gradient scores but
reassigns them to different Transformer blocks. Spectra, beta, rank floor,
threshold bounds and physical A/B budget remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from gars.diagnostics import pairwise_profile_correlations
from gars.importance import cyclically_shift_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "run_pruning_comparison.py"
DEFAULT_SEEDS = (42, 123, 999)
DEFAULT_SHIFTS = tuple(range(1, 12))
METRICS = ("accuracy", "macro_f1", "loss")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def _mean_sample_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _format_mean_sample_std(summary: dict[str, float | None]) -> str:
    """Format a mean and sample SD, handling a single observation explicitly."""
    mean = float(summary["mean"])
    sample_std = summary["sample_std"]
    if sample_std is None:
        return f"{mean:.6f} ± n/a (one seed)"
    return f"{mean:.6f} ± {float(sample_std):.6f}"


def _load_actual_gars(
    baseline_root: Path,
    seed: int,
    expected_revision: str,
) -> dict[str, Any]:
    seed_root = baseline_root / f"seed_{seed}"
    summary = _read_json(seed_root / "comparison_summary.json")
    allocation = _read_json(seed_root / "gars" / "allocation.json")
    result = summary.get("policies", {}).get("gars")
    if not isinstance(result, dict):
        raise ValueError(f"Missing original GARS result for seed {seed}.")
    if result.get("reload_validation") != "passed":
        raise RuntimeError(f"Original GARS reload did not pass for seed {seed}.")
    if summary.get("base_model_revision") != expected_revision:
        raise ValueError(f"Unexpected base revision in original seed {seed} result.")
    if allocation.get("base_model_revision") != expected_revision:
        raise ValueError(f"Unexpected base revision in original seed {seed} allocation.")
    ranks = {
        module: int(values["retained_rank"])
        for module, values in allocation["per_module"].items()
    }
    return {
        "seed": seed,
        "validation": result["validation"],
        "achieved_parameters": int(result["achieved_parameters"]),
        "parent_factor_parameters": int(summary["parent_lora_A_B_parameters"]),
        "ranks_by_module": ranks,
        "allocation": allocation,
        "summary": summary,
    }


def _load_shift_result(
    output_root: Path,
    shift: int,
    seed: int,
    expected_revision: str,
    expected_budget: int,
    original_importance: dict[int, float],
) -> dict[str, Any]:
    seed_root = output_root / f"shift_{shift:02d}" / f"seed_{seed}"
    summary = _read_json(seed_root / "comparison_summary.json")
    allocation = _read_json(seed_root / "gars" / "allocation.json")
    result = summary.get("policies", {}).get("gars")
    if not isinstance(result, dict) or set(summary.get("policies", {})) != {"gars"}:
        raise ValueError(f"Expected one GARS export for shift={shift}, seed={seed}.")
    if result.get("reload_validation") != "passed":
        raise RuntimeError(f"Reload validation failed for shift={shift}, seed={seed}.")
    if summary.get("base_model_revision") != expected_revision:
        raise ValueError(f"Unexpected base revision for shift={shift}, seed={seed}.")
    if int(result.get("achieved_parameters", -1)) != expected_budget:
        raise ValueError(f"Physical budget changed for shift={shift}, seed={seed}.")

    expected_importance = cyclically_shift_importance(
        original_importance,
        shift=shift,
    )
    assignment = allocation.get("importance_assignment", {})
    assigned = {
        int(layer): float(value)
        for layer, value in assignment.get("assigned_importance_by_layer", {}).items()
    }
    original_recorded = {
        int(layer): float(value)
        for layer, value in assignment.get("original_importance_by_layer", {}).items()
    }
    if assigned != expected_importance or original_recorded != original_importance:
        raise ValueError(
            f"Importance assignment metadata mismatch for shift={shift}, seed={seed}."
        )
    if int(assignment.get("cyclic_shift", -1)) != shift:
        raise ValueError(f"Wrong cyclic-shift metadata for shift={shift}, seed={seed}.")

    ranks = {
        module: int(values["retained_rank"])
        for module, values in allocation["per_module"].items()
    }
    return {
        "seed": seed,
        "shift": shift,
        "validation": result["validation"],
        "achieved_parameters": int(result["achieved_parameters"]),
        "ranks_by_module": ranks,
        "allocation": allocation,
        "summary": summary,
    }


def _rank_comparison(
    shifted_ranks: dict[str, int],
    original_ranks: dict[str, int],
) -> dict[str, Any]:
    if set(shifted_ranks) != set(original_ranks):
        raise ValueError("Module inventory differs from original GARS.")
    differences = {
        module: shifted_ranks[module] - original_ranks[module]
        for module in original_ranks
    }
    correlations = pairwise_profile_correlations(
        {
            "observed": {key: float(value) for key, value in original_ranks.items()},
            "cyclic_shift": {key: float(value) for key, value in shifted_ranks.items()},
        }
    )
    return {
        "spearman_vs_observed_gars": correlations[0]["spearman_rho"],
        "modules_changed_vs_observed_gars": sum(
            value != 0 for value in differences.values()
        ),
        "mean_absolute_rank_change_vs_observed_gars": statistics.mean(
            abs(value) for value in differences.values()
        ),
        "rank_change_by_module": differences,
    }


def _summarize_shift(
    shift: int,
    results: dict[int, dict[str, Any]],
    observed: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seeds = sorted(results)
    metrics = {
        metric: _mean_sample_std(
            [float(results[seed]["validation"][metric]) for seed in seeds]
        )
        for metric in METRICS
    }
    paired_deltas = {
        metric: _mean_sample_std(
            [
                float(results[seed]["validation"][metric])
                - float(observed[seed]["validation"][metric])
                for seed in seeds
            ]
        )
        for metric in METRICS
    }
    rank_profiles = {
        str(seed): {
            module: float(rank)
            for module, rank in results[seed]["ranks_by_module"].items()
        }
        for seed in seeds
    }
    stability = pairwise_profile_correlations(rank_profiles)
    stability_values = [
        float(row["spearman_rho"])
        for row in stability
        if row["spearman_rho"] is not None
    ]
    return {
        "cyclic_shift": shift,
        "validation_metrics_mean_sample_std_across_seeds": metrics,
        "paired_metric_delta_vs_observed_gars_mean_sample_std": paired_deltas,
        "cross_seed_module_rank_stability": {
            "pairwise": stability,
            "rho_mean": statistics.mean(stability_values) if stability_values else None,
            "rho_sample_std": (
                statistics.stdev(stability_values)
                if len(stability_values) > 1
                else None
            ),
        },
        "per_seed": {
            str(seed): {
                "validation": results[seed]["validation"],
                "metric_delta_vs_observed_gars": {
                    metric: float(results[seed]["validation"][metric])
                    - float(observed[seed]["validation"][metric])
                    for metric in METRICS
                },
                "allocation_change_vs_observed_gars": _rank_comparison(
                    results[seed]["ranks_by_module"],
                    observed[seed]["ranks_by_module"],
                ),
                "reload_validation": results[seed]["allocation"]["reload_validation"],
                "achieved_parameters": results[seed]["achieved_parameters"],
            }
            for seed in seeds
        },
    }


def _summarize_shift_distribution_by_seed(
    shifts: list[int],
    results_by_shift: dict[int, dict[int, dict[str, Any]]],
    observed: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in sorted(observed):
        per_metric: dict[str, Any] = {}
        for metric in METRICS:
            values = [
                float(results_by_shift[shift][seed]["validation"][metric])
                for shift in shifts
            ]
            distribution = _mean_sample_std(values)
            observed_value = float(observed[seed]["validation"][metric])
            per_metric[metric] = {
                **distribution,
                "min": min(values),
                "max": max(values),
                "observed_gars": observed_value,
                "observed_minus_cyclic_mean": observed_value - distribution["mean"],
            }
        output[str(seed)] = per_metric
    return output


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
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_gradient_permutation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--shifts", type=int, nargs="+", default=list(DEFAULT_SHIFTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide one or more unique seeds.")
    if not args.shifts or len(set(args.shifts)) != len(args.shifts):
        raise ValueError("Provide one or more unique cyclic shifts.")
    if any(shift < 1 or shift > 11 for shift in args.shifts):
        raise ValueError("Cyclic shifts must be non-zero values in [1, 11].")

    base_config = _read_json(args.config)
    expected_revision = str(base_config["model"]["revision"])
    if len(expected_revision) != 40:
        raise ValueError("The ablation requires the pinned 40-character base revision.")

    if args.output_root.exists():
        if any(args.output_root.iterdir()):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output root is not empty: {args.output_root}. "
                    "Choose another path or pass --overwrite."
                )
            shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_copy = args.output_root / "config_v1_deberta_rte.json"
    config_copy.write_text(
        json.dumps(base_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observed = {
        seed: _load_actual_gars(args.baseline_root, seed, expected_revision)
        for seed in args.seeds
    }
    budgets = {item["achieved_parameters"] for item in observed.values()}
    if len(budgets) != 1:
        raise ValueError(f"Original GARS budgets differ across seeds: {sorted(budgets)}.")
    expected_budget = next(iter(budgets))

    original_importance_by_seed: dict[int, dict[int, float]] = {}
    for seed in args.seeds:
        gradient_data = _read_json(
            args.parent_root / f"seed_{seed}" / "gradient_importance.json"
        )
        if int(gradient_data.get("seed", -1)) != seed:
            raise ValueError(f"Gradient artifact seed mismatch for seed {seed}.")
        original = {
            int(layer): float(value)
            for layer, value in gradient_data["importance_by_transformer_block"].items()
        }
        if len(original) != 12:
            raise ValueError(f"Expected 12 gradient scores for seed {seed}.")
        original_importance_by_seed[seed] = original

    results_by_shift: dict[int, dict[int, dict[str, Any]]] = {}
    completed = 0
    total_runs = len(args.shifts) * len(args.seeds)
    for shift in args.shifts:
        results_by_shift[shift] = {}
        shift_root = args.output_root / f"shift_{shift:02d}"
        for seed in args.seeds:
            command = [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--config",
                str(config_copy),
                "--seed",
                str(seed),
                "--parent-root",
                str(args.parent_root),
                "--output-root",
                str(shift_root),
                "--device",
                args.device,
                "--methods",
                "gars",
                "--importance-cyclic-shift",
                str(shift),
            ]
            print(
                f"[{completed + 1}/{total_runs}] Running cyclic shift={shift}, seed={seed}",
                flush=True,
            )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            result = _load_shift_result(
                args.output_root,
                shift,
                seed,
                expected_revision,
                expected_budget,
                original_importance_by_seed[seed],
            )
            if int(result["summary"]["parent_lora_A_B_parameters"]) != int(
                observed[seed]["parent_factor_parameters"]
            ):
                raise ValueError(f"Parent factor budget changed for seed {seed}.")
            results_by_shift[shift][seed] = result
            completed += 1

    by_shift = {
        f"{shift:02d}": _summarize_shift(
            shift,
            results_by_shift[shift],
            observed,
        )
        for shift in args.shifts
    }
    per_seed_distribution = _summarize_shift_distribution_by_seed(
        args.shifts,
        results_by_shift,
        observed,
    )
    report = {
        "analysis_name": "GARS cyclic gradient-to-layer alignment ablation",
        "base_model": base_config["model"]["id"],
        "base_model_revision": expected_revision,
        "dataset": base_config["dataset"],
        "seeds": args.seeds,
        "cyclic_shifts": args.shifts,
        "beta": float(base_config["gars"]["beta"]),
        "target_parameter_fraction": float(
            base_config["gars"]["target_parameter_fraction"]
        ),
        "achieved_physical_A_B_parameters": expected_budget,
        "observed_gars_reference": {
            str(seed): {
                "validation": observed[seed]["validation"],
                "ranks_by_module": observed[seed]["ranks_by_module"],
                "reload_validation": observed[seed]["allocation"]["reload_validation"],
            }
            for seed in args.seeds
        },
        "design": {
            "control": (
                "For sorted layer IDs L, destination L[i] receives the original "
                "gradient score from L[(i - shift) mod 12]."
            ),
            "score_distribution": (
                "Every non-zero cyclic shift preserves the exact within-seed "
                "gradient-score multiset and mean."
            ),
            "coverage": (
                "The 11 non-identity cyclic shifts assign every source-layer score "
                "to every other destination layer exactly once across the full sweep."
            ),
            "held_constant": [
                "same three trained parent adapters",
                "same parent spectral decompositions",
                "same beta and tau clipping bounds",
                "same rank floor",
                "same target physical A/B factor budget",
                "same pinned base-model revision and RTE validation examples",
                "same export, scaling-preservation and reload validation path",
            ],
            "changed": "Only the assignment of gradient scores to Transformer layers.",
            "inference_boundary": (
                "This is a deterministic cyclic-alignment control, not a sample "
                "from all possible permutations and not a formal permutation test. "
                "Correlations and validation comparisons are descriptive."
            ),
        },
        "by_cyclic_shift": by_shift,
        "per_seed_distribution_across_cyclic_shifts": per_seed_distribution,
        "run_count": completed,
        "all_requested_runs_completed": completed == total_runs,
    }
    report_path = args.output_root / "gradient_permutation_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved cyclic-alignment report: {report_path}")
    print("\nCYCLIC SHIFT SUMMARY")
    for shift in args.shifts:
        entry = by_shift[f"{shift:02d}"]
        loss = entry["validation_metrics_mean_sample_std_across_seeds"]["loss"]
        delta = entry["paired_metric_delta_vs_observed_gars_mean_sample_std"]["loss"]
        print(
            f"shift={shift:02d}: loss={_format_mean_sample_std(loss)}; "
            "loss_delta_vs_observed_gars="
            f"{_format_mean_sample_std(delta)}"
        )
    print("\nPER-SEED OBSERVED GARS VS CYCLIC-SHIFT LOSS")
    for seed, metrics in per_seed_distribution.items():
        loss = metrics["loss"]
        print(
            f"seed={seed}: observed={loss['observed_gars']:.6f}; "
            f"cyclic_mean={loss['mean']:.6f}; "
            f"cyclic_range=[{loss['min']:.6f}, {loss['max']:.6f}]"
        )


if __name__ == "__main__":
    main()
