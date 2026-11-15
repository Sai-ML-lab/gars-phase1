"""Aggregate full-precision V1 pruning comparison summaries across seeds."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gars", "global_singular_value", "uniform_local_energy")
METRICS = ("accuracy", "macro_f1", "loss")


def _read_summary(path: Path, expected_seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison summary: {path}")
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"{path} records seed={summary.get('seed')}, expected {expected_seed}."
        )
    missing_methods = set(METHODS) - set(summary.get("policies", {}))
    if missing_methods:
        raise ValueError(f"{path} is missing policies: {sorted(missing_methods)}")
    parent = summary.get("parent_validation", {})
    missing_metrics = set(METRICS) - set(parent)
    if missing_metrics:
        raise ValueError(f"{path} parent validation is missing: {sorted(missing_metrics)}")
    for method in METHODS:
        result = summary["policies"][method]
        if result.get("reload_validation") != "passed":
            raise RuntimeError(
                f"{path}: {method} reload_validation is "
                f"{result.get('reload_validation')!r}, not 'passed'."
            )
        missing_metrics = set(METRICS) - set(result.get("validation", {}))
        if missing_metrics:
            raise ValueError(
                f"{path}: {method} validation is missing {sorted(missing_metrics)}."
            )
        if "rank_histogram" not in result:
            raise ValueError(f"{path}: {method} has no rank histogram.")
    return summary


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _metric_values(
    summaries: dict[int, dict[str, Any]],
    method: str,
    metric: str,
    *,
    delta_vs_parent: bool = False,
) -> list[float]:
    values = []
    for seed in summaries:
        policy = summaries[seed]["policies"][method]
        if delta_vs_parent:
            value = float(policy["validation"][metric]) - float(
                summaries[seed]["parent_validation"][metric]
            )
        else:
            value = float(policy["validation"][metric])
        values.append(value)
    return values


def build_report(
    summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seeds = sorted(summaries)
    report: dict[str, Any] = {
        "seeds": seeds,
        "seed_count": len(seeds),
        "methods": {},
        "paired_method_differences": {},
        "budget_checks": {},
        "rank_histograms": {},
        "base_model_revisions": {
            str(seed): summaries[seed].get("base_model_revision") for seed in seeds
        },
    }

    parameter_counts: list[int] = []
    reductions: list[float] = []
    for seed in seeds:
        counts = []
        for method in METHODS:
            result = summaries[seed]["policies"][method]
            counts.append(int(result["achieved_parameters"]))
            reductions.append(float(result["parameter_reduction_fraction"]))
            report["rank_histograms"].setdefault(method, {})[str(seed)] = result[
                "rank_histogram"
            ]
        if len(set(counts)) != 1:
            raise RuntimeError(
                f"Seed {seed} methods do not share a physical parameter budget: {counts}."
            )
        parameter_counts.append(counts[0])

    if len(set(parameter_counts)) != 1:
        raise RuntimeError(
            f"Physical parameter budgets differ across seeds: {parameter_counts}."
        )

    report["budget_checks"] = {
        "parent_factor_parameters": int(
            summaries[seeds[0]]["parent_lora_A_B_parameters"]
        ),
        "achieved_parameters_per_method": parameter_counts[0],
        "parameter_reduction_fraction_mean": statistics.mean(reductions),
        "parameter_reduction_fraction_sample_std": (
            statistics.stdev(reductions) if len(reductions) > 1 else None
        ),
        "same_budget_all_methods_and_seeds": True,
    }

    for method in METHODS:
        report["methods"][method] = {
            metric: _stats(_metric_values(summaries, method, metric))
            for metric in METRICS
        }
        report["methods"][method]["paired_delta_vs_parent"] = {
            metric: _stats(
                _metric_values(summaries, method, metric, delta_vs_parent=True)
            )
            for metric in METRICS
        }
        report["methods"][method]["seed_level"] = {
            str(seed): {
                "validation": summaries[seed]["policies"][method]["validation"],
                "delta_vs_parent": {
                    metric: float(
                        summaries[seed]["policies"][method]["validation"][metric]
                    )
                    - float(summaries[seed]["parent_validation"][metric])
                    for metric in METRICS
                },
                "reload_validation": summaries[seed]["policies"][method][
                    "reload_validation"
                ],
            }
            for seed in seeds
        }

    for baseline in ("global_singular_value", "uniform_local_energy"):
        report["paired_method_differences"][f"gars_minus_{baseline}"] = {
            metric: _stats(
                [
                    float(summaries[seed]["policies"]["gars"]["validation"][metric])
                    - float(
                        summaries[seed]["policies"][baseline]["validation"][metric]
                    )
                    for seed in seeds
                ]
            )
            for metric in METRICS
        }
        report["paired_method_differences"][f"gars_minus_{baseline}"][
            "seed_level"
        ] = {
            str(seed): {
                metric: float(
                    summaries[seed]["policies"]["gars"]["validation"][metric]
                )
                - float(
                    summaries[seed]["policies"][baseline]["validation"][metric]
                )
                for metric in METRICS
            }
            for seed in seeds
        }

    report["parent_validation"] = {
        metric: _stats(
            [float(summaries[seed]["parent_validation"][metric]) for seed in seeds]
        )
        for metric in METRICS
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed list contains duplicates.")

    summaries = {
        seed: _read_summary(
            args.results_root / f"seed_{seed}" / "comparison_summary.json",
            seed,
        )
        for seed in args.seeds
    }
    report = build_report(summaries)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved aggregate report: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
