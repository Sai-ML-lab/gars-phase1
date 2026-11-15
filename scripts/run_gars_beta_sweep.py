"""Run a matched-budget GARS beta sensitivity sweep on saved parent adapters."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from gars.diagnostics import descriptive_stats, pairwise_profile_correlations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (42, 123, 999)
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "run_pruning_comparison.py"
METRICS = ("accuracy", "macro_f1", "loss")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def _beta_label(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def _mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _baseline_ranks(path: Path) -> dict[str, int]:
    allocation = _read_json(path)
    return {
        module: int(values["retained_rank"])
        for module, values in allocation["per_module"].items()
    }


def _load_gars_result(path: Path, expected_seed: int) -> dict[str, Any]:
    summary = _read_json(path)
    if int(summary.get("seed", -1)) != expected_seed:
        raise ValueError(f"Unexpected seed ID in {path}.")
    if set(summary.get("policies", {})) != {"gars"}:
        raise ValueError(f"Expected a GARS-only export summary in {path}.")
    result = summary["policies"]["gars"]
    if result.get("reload_validation") != "passed":
        raise RuntimeError(f"GARS reload validation did not pass: {path}")
    if int(result.get("achieved_parameters", -1)) != 708096:
        raise ValueError(f"Unexpected GARS parameter budget in {path}.")
    return {
        "seed": expected_seed,
        "base_model_revision": summary.get("base_model_revision"),
        "parent_validation": summary["parent_validation"],
        "validation": result["validation"],
        "validation_delta_vs_parent": result["validation_delta_vs_parent"],
        "achieved_parameters": int(result["achieved_parameters"]),
        "parameter_reduction_fraction": float(
            result["parameter_reduction_fraction"]
        ),
        "reload_validation": result["reload_validation"],
        "rank_histogram": result["rank_histogram"],
        "ranks_by_module": _baseline_ranks(
            path.parent / "gars" / "allocation.json"
        ),
        "allocation": _read_json(path.parent / "gars" / "allocation.json"),
    }


def _summarize_beta(
    beta: float,
    results: dict[int, dict[str, Any]],
    baseline_ranks_by_seed: dict[int, dict[str, int]],
) -> dict[str, Any]:
    seeds = sorted(results)
    metrics = {
        metric: _mean_std(
            [float(results[seed]["validation"][metric]) for seed in seeds]
        )
        for metric in METRICS
    }
    paired_deltas = {
        metric: _mean_std(
            [
                float(results[seed]["validation"][metric])
                - float(results[seed]["parent_validation"][metric])
                for seed in seeds
            ]
        )
        for metric in METRICS
    }
    profiles = {
        str(seed): {
            module: float(rank)
            for module, rank in results[seed]["ranks_by_module"].items()
        }
        for seed in seeds
    }
    rank_stability = pairwise_profile_correlations(profiles)
    rank_stability_values = [
        float(row["spearman_rho"])
        for row in rank_stability
        if row["spearman_rho"] is not None
    ]

    allocation_vs_uniform: dict[str, Any] = {}
    for seed in seeds:
        current = results[seed]["ranks_by_module"]
        baseline = baseline_ranks_by_seed[seed]
        if set(current) != set(baseline):
            raise ValueError(f"Seed {seed}: module inventory differs from beta=0 baseline.")
        differences = {
            module: current[module] - baseline[module]
            for module in current
        }
        allocation_vs_uniform[str(seed)] = {
            "spearman_vs_beta_0": pairwise_profile_correlations(
                {
                    "beta_0": {module: float(rank) for module, rank in baseline.items()},
                    "current": {module: float(rank) for module, rank in current.items()},
                }
            )[0]["spearman_rho"],
            "modules_changed_vs_beta_0": sum(value != 0 for value in differences.values()),
            "mean_absolute_rank_change_vs_beta_0": statistics.mean(
                abs(value) for value in differences.values()
            ),
            "rank_change_by_module": differences,
        }

    threshold_values = {
        str(seed): list(
            results[seed]["allocation"].get("thresholds_by_layer", {}).values()
        )
        for seed in seeds
    }
    clipping = {
        str(seed): {
            "at_lower_bound_0_80": sum(
                abs(float(value) - 0.80) <= 1e-9 for value in values
            ),
            "at_upper_bound_0_99": sum(
                abs(float(value) - 0.99) <= 1e-9 for value in values
            ),
            "min": min(map(float, values)),
            "max": max(map(float, values)),
        }
        for seed, values in (
            (int(key), values) for key, values in threshold_values.items()
        )
    }

    return {
        "beta": beta,
        "seed_count": len(seeds),
        "metrics_mean_sample_std": metrics,
        "paired_delta_vs_parent_mean_sample_std": paired_deltas,
        "cross_seed_module_rank_stability": {
            "pairwise": rank_stability,
            "rho_mean": statistics.mean(rank_stability_values)
            if rank_stability_values
            else None,
            "rho_sample_std": statistics.stdev(rank_stability_values)
            if len(rank_stability_values) > 1
            else None,
        },
        "allocation_change_vs_beta_0_uniform": allocation_vs_uniform,
        "threshold_clipping_by_seed": clipping,
        "seed_level": {
            str(seed): {
                "validation": results[seed]["validation"],
                "delta_vs_parent": results[seed]["validation_delta_vs_parent"],
                "rank_histogram": results[seed]["rank_histogram"],
                "reload_validation": results[seed]["reload_validation"],
                "parameter_reduction_fraction": results[seed][
                    "parameter_reduction_fraction"
                ],
            }
            for seed in seeds
        },
    }


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
        "--baseline-pruned-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned",
        help="Existing V1 exports used to validate beta=0 and beta=0.5 allocations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_beta_sweep",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(DEFAULT_BETAS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Provide one or more unique seeds.")
    if len(set(args.betas)) != len(args.betas) or not args.betas:
        raise ValueError("Provide one or more unique beta values.")
    if any(beta < 0.0 or beta > 1.0 for beta in args.betas):
        raise ValueError("This planned sensitivity sweep uses beta values in [0, 1].")
    if 0.0 not in args.betas or 0.5 not in args.betas:
        raise ValueError("Include beta=0.0 and beta=0.5 for baseline equivalence checks.")

    base_config = _read_json(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_root = args.output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    results_by_beta: dict[float, dict[int, dict[str, Any]]] = {}
    baseline_ranks_by_seed = {
        seed: _baseline_ranks(
            args.baseline_pruned_root
            / f"seed_{seed}"
            / "uniform_local_energy"
            / "allocation.json"
        )
        for seed in args.seeds
    }
    original_gars_ranks_by_seed = {
        seed: _baseline_ranks(
            args.baseline_pruned_root / f"seed_{seed}" / "gars" / "allocation.json"
        )
        for seed in args.seeds
    }

    for beta in args.betas:
        label = _beta_label(beta)
        config = json.loads(json.dumps(base_config))
        config["gars"]["beta"] = beta
        config_path = config_root / f"config_beta_{label}.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        beta_output_root = args.output_root / f"beta_{label}"
        results_by_beta[beta] = {}

        for seed in args.seeds:
            command = [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--parent-root",
                str(args.parent_root),
                "--output-root",
                str(beta_output_root),
                "--device",
                args.device,
                "--methods",
                "gars",
            ]
            if args.overwrite:
                command.append("--overwrite")
            print(f"Running beta={beta:.2f}, seed={seed}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

            summary_path = (
                beta_output_root / f"seed_{seed}" / "comparison_summary.json"
            )
            result = _load_gars_result(summary_path, seed)
            if result["base_model_revision"] != base_config["model"]["revision"]:
                raise ValueError(
                    f"Seed {seed}, beta={beta}: unexpected base-model revision."
                )
            results_by_beta[beta][seed] = result

            if math.isclose(beta, 0.0, abs_tol=1e-12):
                if result["ranks_by_module"] != baseline_ranks_by_seed[seed]:
                    raise RuntimeError(
                        f"beta=0 did not reproduce uniform-energy ranks for seed {seed}."
                    )
            if math.isclose(beta, 0.5, abs_tol=1e-12):
                if result["ranks_by_module"] != original_gars_ranks_by_seed[seed]:
                    raise RuntimeError(
                        f"beta=0.5 did not reproduce the V1 GARS ranks for seed {seed}."
                    )

    report = {
        "analysis_name": "GARS beta sensitivity at matched physical LoRA factor budget",
        "seeds": args.seeds,
        "betas": args.betas,
        "design": {
            "beta_0_equivalence": (
                "beta=0 makes all layer thresholds identical and should reproduce "
                "uniform local squared-energy rank allocation."
            ),
            "beta_0_5_equivalence": (
                "beta=0.5 should reproduce the original V1 GARS rank allocation."
            ),
            "budget": "All beta values are calibrated to the same 708096 A/B factor parameters.",
            "selection_warning": (
                "Treat validation curves as sensitivity analysis. Do not select beta "
                "on this validation split and then report the same split as unbiased evaluation."
            ),
        },
        "by_beta": {
            _beta_label(beta): _summarize_beta(
                beta, results_by_beta[beta], baseline_ranks_by_seed
            )
            for beta in args.betas
        },
    }
    report_path = args.output_root / "beta_sweep_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved beta sweep summary: {report_path}")
    for beta in args.betas:
        entry = report["by_beta"][_beta_label(beta)]
        print(
            f"beta={beta:.2f}: "
            f"accuracy={entry['metrics_mean_sample_std']['accuracy']['mean']:.5f}, "
            f"macro_f1={entry['metrics_mean_sample_std']['macro_f1']['mean']:.5f}, "
            f"loss={entry['metrics_mean_sample_std']['loss']['mean']:.5f}, "
            f"rank_stability_rho={entry['cross_seed_module_rank_stability']['rho_mean']:.5f}"
        )


if __name__ == "__main__":
    main()
