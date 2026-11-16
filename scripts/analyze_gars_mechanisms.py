"""Analyze cross-seed GARS gradient, spectral and rank-allocation profiles."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from gars.diagnostics import (
    descriptive_stats,
    pairwise_profile_correlations,
    partial_spearman_rank_correlation,
    spearman_rank_correlation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS_DEFAULT = (42, 123, 999)
METHODS = ("gars", "global_singular_value", "uniform_local_energy")


EXPECTED_BASE_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return value


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


def _rank_order(values: dict[str, float], *, descending: bool = True) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: values[key], reverse=descending)
    return {key: index + 1 for index, key in enumerate(ordered)}


def _layer_means(
    module_values: dict[str, float],
    layer_by_module: dict[str, int],
) -> dict[str, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for module, value in module_values.items():
        grouped[layer_by_module[module]].append(float(value))
    return {str(layer): _mean(values) for layer, values in sorted(grouped.items())}


def _profile_stats(
    correlations: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [
        float(item["spearman_rho"])
        for item in correlations
        if item["spearman_rho"] is not None
    ]
    return {
        "pairwise_correlations": correlations,
        "rho_descriptive_stats": descriptive_stats(values),
    }


def _relative_importance_by_layer(
    seeds: list[int],
    artifacts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    layers = sorted(artifacts[seeds[0]]["normalized_gradient_by_layer"], key=int)
    for layer in layers:
        by_seed = {
            str(seed): float(artifacts[seed]["normalized_gradient_by_layer"][layer])
            for seed in seeds
        }
        stats = descriptive_stats(list(by_seed.values()))
        mean = stats["mean"]
        sample_std = stats["sample_std"]
        output[layer] = {
            "normalized_importance_by_seed": by_seed,
            **stats,
            "coefficient_of_variation": (
                sample_std / mean
                if sample_std is not None and mean not in (None, 0.0)
                else None
            ),
        }
    return output


def _load_seed_artifacts(
    seed: int,
    parent_root: Path,
    pruned_root: Path,
    expected_revision: str = EXPECTED_BASE_REVISION,
    expected_layers: int = 12,
    expected_modules: int = 24,
    expected_capture_steps: int = 50,
    min_rank: int = 2,
    max_rank: int = 32,
) -> dict[str, Any]:
    seed_dir = parent_root / f"seed_{seed}"
    gradient_data = _read_json(seed_dir / "gradient_importance.json")
    metadata = _read_json(seed_dir / "run_metadata.json")
    comparison = _read_json(pruned_root / f"seed_{seed}" / "comparison_summary.json")

    if int(gradient_data.get("seed", -1)) != seed:
        raise ValueError(f"Gradient artifact seed does not match directory seed {seed}.")
    if int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"Parent metadata seed does not match directory seed {seed}.")
    if int(comparison.get("seed", -1)) != seed:
        raise ValueError(f"Comparison summary seed does not match directory seed {seed}.")
    revision = comparison.get("base_model_revision")
    if revision != expected_revision:
        raise ValueError(
            f"Seed {seed} uses base revision {revision!r}; "
            f"expected {expected_revision!r}."
        )

    gradients = {
        str(int(layer)): float(value)
        for layer, value in gradient_data["importance_by_transformer_block"].items()
    }
    if len(gradients) != expected_layers:
        raise ValueError(
            f"Seed {seed}: expected {expected_layers} block scores, got {len(gradients)}."
        )
    if any(not math.isfinite(value) or value < 0 for value in gradients.values()):
        raise ValueError(f"Seed {seed}: gradient scores must be finite and non-negative.")

    capture_steps = int(gradient_data.get("capture_steps", -1))
    if capture_steps != expected_capture_steps:
        raise ValueError(
            f"Seed {seed}: expected {expected_capture_steps} gradient capture steps, "
            f"got {capture_steps}."
        )
    stepwise_records = gradient_data.get("stepwise_importance_by_optimizer_step")
    stepwise_capture_available = (
        isinstance(stepwise_records, list)
        and len(stepwise_records) == expected_capture_steps
    )
    accumulator = gradient_data.get("accumulator_state", {})
    counts = {
        str(int(layer)): int(count)
        for layer, count in accumulator.get("count_by_layer", {}).items()
    }
    if counts and (
        set(counts) != set(gradients)
        or set(counts.values()) != {expected_capture_steps}
    ):
        raise ValueError(
            f"Seed {seed}: accumulator per-layer counts are inconsistent with "
            f"{expected_capture_steps} updates."
        )

    methods: dict[str, Any] = {}
    expected_module_names: set[str] | None = None
    entropy_reference: dict[str, float] | None = None
    layer_reference: dict[str, int] | None = None
    expected_budget: int | None = None

    for method in METHODS:
        allocation = _read_json(
            pruned_root / f"seed_{seed}" / method / "allocation.json"
        )
        if allocation.get("method") != method:
            raise ValueError(
                f"Seed {seed}/{method}: allocation method field does not match."
            )
        if allocation.get("reload_validation") != "passed":
            raise RuntimeError(
                f"Seed {seed}/{method}: reload validation did not pass."
            )
        achieved_parameters = int(allocation.get("achieved_parameters", -1))
        if achieved_parameters <= 0:
            raise ValueError(f"Seed {seed}/{method}: invalid physical parameter count.")
        if expected_budget is None:
            expected_budget = achieved_parameters
        elif achieved_parameters != expected_budget:
            raise ValueError(
                f"Seed {seed}: policies have different physical budgets."
            )

        module_items = allocation.get("per_module", {})
        modules = set(module_items)
        if len(modules) != expected_modules:
            raise ValueError(
                f"Seed {seed}/{method}: expected {expected_modules} adapted projections, "
                f"got {len(modules)}."
            )
        if expected_module_names is None:
            expected_module_names = modules
        elif modules != expected_module_names:
            raise ValueError(f"Seed {seed}: module inventory differs across policies.")

        entropy: dict[str, float] = {}
        layer_by_module: dict[str, int] = {}
        ranks: dict[str, float] = {}
        retained_energy: dict[str, float] = {}

        for module, item in module_items.items():
            layer = str(int(item["layer_id"]))
            value = float(item["squared_spectral_entropy"])
            rank = int(item["retained_rank"])
            retained = float(item["squared_energy_retained"])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{seed}/{method}/{module}: entropy outside [0, 1].")
            if not min_rank <= rank <= max_rank:
                raise ValueError(f"{seed}/{method}/{module}: retained rank out of range.")
            if not 0.0 <= retained <= 1.0 + 1e-8:
                raise ValueError(f"{seed}/{method}/{module}: invalid retained energy.")
            entropy[module] = value
            layer_by_module[module] = int(layer)
            ranks[module] = float(rank)
            retained_energy[module] = retained

        if entropy_reference is None:
            entropy_reference = entropy
            layer_reference = layer_by_module
        else:
            if set(entropy) != set(entropy_reference):
                raise ValueError(f"Seed {seed}: spectral module inventory differs.")
            for module in entropy:
                if not math.isclose(
                    entropy[module], entropy_reference[module], rel_tol=0.0, abs_tol=1e-10
                ):
                    raise ValueError(
                        f"Seed {seed}: spectral entropy differs across policies for {module}."
                    )
                if layer_by_module[module] != layer_reference[module]:
                    raise ValueError(f"Seed {seed}: layer mapping differs for {module}.")

        methods[method] = {
            "ranks_by_module": ranks,
            "entropy_by_module": entropy,
            "layer_by_module": layer_by_module,
            "mean_rank_by_layer": _layer_means(ranks, layer_by_module),
            "mean_retained_energy_by_layer": _layer_means(
                retained_energy, layer_by_module
            ),
            "retained_energy_by_module": retained_energy,
            "thresholds_by_layer": {
                str(key): float(value)
                for key, value in allocation.get("thresholds_by_layer", {}).items()
            },
            "importance_by_layer": {
                str(int(key)): float(value)
                for key, value in allocation.get("importance_by_layer", {}).items()
            },
        }

    if entropy_reference is None or layer_reference is None:
        raise RuntimeError(f"Seed {seed}: no module spectral information found.")
    if set(gradients) != {str(layer) for layer in set(layer_reference.values())}:
        raise ValueError(
            f"Seed {seed}: gradient block IDs do not match module layer IDs."
        )
    gars_importance = methods["gars"]["importance_by_layer"]
    if set(gars_importance) != set(gradients):
        raise ValueError(f"Seed {seed}: GARS allocation lost gradient layers.")
    for layer, score in gradients.items():
        if not math.isclose(gars_importance[layer], score, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"Seed {seed}: saved GARS gradient score does not match parent artifact."
            )

    return {
        "seed": seed,
        "base_model_revision": revision,
        "achieved_parameters": expected_budget,
        "gradient_capture_steps": capture_steps,
        "stepwise_capture_available": stepwise_capture_available,
        "gradient_counts_by_layer": counts,
        "gradient_by_layer": gradients,
        "normalized_gradient_by_layer": {
            layer: (value / _mean(list(gradients.values())))
            if _mean(list(gradients.values())) > 0
            else 1.0
            for layer, value in gradients.items()
        },
        "layer_by_module": layer_reference,
        "entropy_by_module": entropy_reference,
        "entropy_by_layer": _layer_means(entropy_reference, layer_reference),
        "methods": methods,
    }


def build_mechanism_report(
    seeds: list[int],
    artifacts: dict[int, dict[str, Any]],
    base_model_revision: str | None = None,
    expected_capture_steps: int = 50,
) -> dict[str, Any]:
    if base_model_revision is None:
        base_model_revision = artifacts[seeds[0]]["base_model_revision"]
    gradient_profiles = {
        str(seed): artifacts[seed]["gradient_by_layer"] for seed in seeds
    }
    entropy_module_profiles = {
        str(seed): artifacts[seed]["entropy_by_module"] for seed in seeds
    }
    entropy_layer_profiles = {
        str(seed): artifacts[seed]["entropy_by_layer"] for seed in seeds
    }

    method_module_profiles = {
        method: {
            str(seed): artifacts[seed]["methods"][method]["ranks_by_module"]
            for seed in seeds
        }
        for method in METHODS
    }
    method_layer_profiles = {
        method: {
            str(seed): artifacts[seed]["methods"][method]["mean_rank_by_layer"]
            for seed in seeds
        }
        for method in METHODS
    }

    gradient_rank_corr = pairwise_profile_correlations(gradient_profiles)
    entropy_module_corr = pairwise_profile_correlations(entropy_module_profiles)
    entropy_layer_corr = pairwise_profile_correlations(entropy_layer_profiles)
    rank_module_corr = {
        method: pairwise_profile_correlations(method_module_profiles[method])
        for method in METHODS
    }
    rank_layer_corr = {
        method: pairwise_profile_correlations(method_layer_profiles[method])
        for method in METHODS
    }

    per_seed: dict[str, Any] = {}
    gradient_entropy_rhos: list[float] = []
    gradient_rank_rhos: dict[str, list[float]] = {method: [] for method in METHODS}
    entropy_rank_rhos: dict[str, list[float]] = {method: [] for method in METHODS}
    gradient_rank_uplift_rhos: list[float] = []
    entropy_gars_uniform_module_rhos: list[float] = []
    partial_gradient_entropy_rhos: list[float] = []
    partial_gradient_gars_rank_rhos: list[float] = []
    partial_gradient_rank_uplift_rhos: list[float] = []
    partial_entropy_gars_rank_rhos: list[float] = []
    for seed in seeds:
        item = artifacts[seed]
        gradient = item["gradient_by_layer"]
        entropy_layer = item["entropy_by_layer"]
        method_data = item["methods"]
        seed_key = str(seed)

        layer_rows: list[dict[str, Any]] = []
        gars_layer = method_data["gars"]["mean_rank_by_layer"]
        uniform_layer = method_data["uniform_local_energy"]["mean_rank_by_layer"]
        global_layer = method_data["global_singular_value"]["mean_rank_by_layer"]
        rank_uplift = {
            layer: gars_layer[layer] - uniform_layer[layer] for layer in gars_layer
        }
        rank_gars_minus_global = {
            layer: gars_layer[layer] - global_layer[layer] for layer in gars_layer
        }

        ordered_layers = sorted(gradient, key=int)
        layer_depth = [float(int(layer)) for layer in ordered_layers]
        gradient_values = [gradient[layer] for layer in ordered_layers]
        entropy_values = [entropy_layer[layer] for layer in ordered_layers]
        gars_rank_values = [gars_layer[layer] for layer in ordered_layers]
        uniform_rank_values = [uniform_layer[layer] for layer in ordered_layers]
        rank_uplift_values = [rank_uplift[layer] for layer in ordered_layers]

        rho_gradient_entropy = spearman_rank_correlation(
            gradient_values,
            entropy_values,
        )
        if rho_gradient_entropy is not None:
            gradient_entropy_rhos.append(rho_gradient_entropy)

        partial_gradient_entropy = partial_spearman_rank_correlation(
            gradient_values, entropy_values, layer_depth
        )
        partial_gradient_gars_rank = partial_spearman_rank_correlation(
            gradient_values, gars_rank_values, layer_depth
        )
        partial_gradient_uplift = partial_spearman_rank_correlation(
            gradient_values, rank_uplift_values, layer_depth
        )
        partial_entropy_gars_rank = partial_spearman_rank_correlation(
            entropy_values, gars_rank_values, layer_depth
        )
        for value, collection in (
            (partial_gradient_entropy, partial_gradient_entropy_rhos),
            (partial_gradient_gars_rank, partial_gradient_gars_rank_rhos),
            (partial_gradient_uplift, partial_gradient_rank_uplift_rhos),
            (partial_entropy_gars_rank, partial_entropy_gars_rank_rhos),
        ):
            if value is not None:
                collection.append(value)

        for method in METHODS:
            layer_ranks = method_data[method]["mean_rank_by_layer"]
            rho = spearman_rank_correlation(
                [gradient[layer] for layer in sorted(gradient, key=int)],
                [layer_ranks[layer] for layer in sorted(gradient, key=int)],
            )
            if rho is not None:
                gradient_rank_rhos[method].append(rho)
            rho_entropy_rank = spearman_rank_correlation(
                [entropy_layer[layer] for layer in sorted(entropy_layer, key=int)],
                [layer_ranks[layer] for layer in sorted(entropy_layer, key=int)],
            )
            if rho_entropy_rank is not None:
                entropy_rank_rhos[method].append(rho_entropy_rank)

        rho_uplift = spearman_rank_correlation(
            [gradient[layer] for layer in sorted(gradient, key=int)],
            [rank_uplift[layer] for layer in sorted(gradient, key=int)],
        )
        if rho_uplift is not None:
            gradient_rank_uplift_rhos.append(rho_uplift)

        entropy_modules = item["entropy_by_module"]
        gars_ranks = method_data["gars"]["ranks_by_module"]
        uniform_ranks = method_data["uniform_local_energy"]["ranks_by_module"]
        entropy_keys = sorted(entropy_modules)
        module_rank_uplift = {
            module: gars_ranks[module] - uniform_ranks[module]
            for module in entropy_modules
        }
        rho_entropy_uplift = spearman_rank_correlation(
            [entropy_modules[module] for module in entropy_keys],
            [module_rank_uplift[module] for module in entropy_keys],
        )
        if rho_entropy_uplift is not None:
            entropy_gars_uniform_module_rhos.append(rho_entropy_uplift)

        for layer in sorted(gradient, key=int):
            layer_rows.append(
                {
                    "layer_id": int(layer),
                    "gradient_norm_mean_last_50_steps": gradient[layer],
                    "gradient_normalized_to_layer_mean": item[
                        "normalized_gradient_by_layer"
                    ][layer],
                    "gradient_rank_1_is_highest": _rank_order(
                        gradient, descending=True
                    )[layer],
                    "mean_squared_spectral_entropy": entropy_layer[layer],
                    "gars_mean_retained_rank": gars_layer[layer],
                    "uniform_mean_retained_rank": uniform_layer[layer],
                    "global_mean_retained_rank": global_layer[layer],
                    "gars_minus_uniform_mean_rank": rank_uplift[layer],
                    "gars_minus_global_mean_rank": rank_gars_minus_global[layer],
                }
            )

        module_rows = []
        for module in sorted(entropy_modules):
            module_rows.append(
                {
                    "module": module,
                    "layer_id": item["layer_by_module"][module],
                    "squared_spectral_entropy": entropy_modules[module],
                    "gars_retained_rank": int(gars_ranks[module]),
                    "uniform_retained_rank": int(uniform_ranks[module]),
                    "global_retained_rank": int(
                        method_data["global_singular_value"]["ranks_by_module"][module]
                    ),
                    "gars_minus_uniform_rank": int(module_rank_uplift[module]),
                    "gars_retained_energy": method_data["gars"][
                        "retained_energy_by_module"
                    ][module],
                }
            )

        threshold_values = list(
            method_data["gars"]["thresholds_by_layer"].values()
        )
        per_seed[seed_key] = {
            "gradient_vs_spectral_entropy_spearman_rho": rho_gradient_entropy,
            "gradient_vs_spectral_entropy_partial_spearman_controlling_depth": (
                partial_gradient_entropy
            ),
            "gradient_vs_gars_mean_rank_spearman_rho": spearman_rank_correlation(
                gradient_values, gars_rank_values
            ),
            "gradient_vs_gars_mean_rank_partial_spearman_controlling_depth": (
                partial_gradient_gars_rank
            ),
            "gradient_vs_gars_minus_uniform_rank_spearman_rho": rho_uplift,
            "gradient_vs_gars_minus_uniform_rank_partial_spearman_controlling_depth": (
                partial_gradient_uplift
            ),
            "spectral_entropy_vs_gars_mean_rank_partial_spearman_controlling_depth": (
                partial_entropy_gars_rank
            ),
            "layer_depth_trends": {
                "gradient_importance": spearman_rank_correlation(
                    layer_depth, gradient_values
                ),
                "mean_squared_spectral_entropy": spearman_rank_correlation(
                    layer_depth, entropy_values
                ),
                "gars_mean_rank": spearman_rank_correlation(
                    layer_depth, gars_rank_values
                ),
                "uniform_mean_rank": spearman_rank_correlation(
                    layer_depth, uniform_rank_values
                ),
            },
            "gars_threshold_clipping": {
                "number_of_layers": len(threshold_values),
                "at_lower_bound_0_80": sum(
                    math.isclose(value, 0.80, rel_tol=0.0, abs_tol=1e-9)
                    for value in threshold_values
                ),
                "at_upper_bound_0_99": sum(
                    math.isclose(value, 0.99, rel_tol=0.0, abs_tol=1e-9)
                    for value in threshold_values
                ),
                "threshold_mean": descriptive_stats(threshold_values)["mean"],
                "threshold_min": min(threshold_values) if threshold_values else None,
                "threshold_max": max(threshold_values) if threshold_values else None,
            },
            "entropy_vs_gars_minus_uniform_module_rank_spearman_rho": rho_entropy_uplift,
            "mean_spectral_entropy": descriptive_stats(list(entropy_modules.values())),
            "mean_gradient_importance": descriptive_stats(list(gradient.values())),
            "layer_profiles": layer_rows,
            "module_profiles": module_rows,
            "gars_thresholds_by_layer": method_data["gars"]["thresholds_by_layer"],
        }
    within_seed = {
        "gradient_vs_layer_mean_spectral_entropy": descriptive_stats(gradient_entropy_rhos),
        "gradient_vs_layer_mean_rank": {
            method: descriptive_stats(gradient_rank_rhos[method]) for method in METHODS
        },
        "layer_entropy_vs_layer_mean_rank": {
            method: descriptive_stats(entropy_rank_rhos[method]) for method in METHODS
        },
        "gradient_vs_gars_minus_uniform_layer_rank": descriptive_stats(
            gradient_rank_uplift_rhos
        ),
        "module_entropy_vs_gars_minus_uniform_module_rank": descriptive_stats(
            entropy_gars_uniform_module_rhos
        ),
        "depth_adjusted_partial_spearman": {
            "gradient_vs_spectral_entropy": descriptive_stats(
                partial_gradient_entropy_rhos
            ),
            "gradient_vs_gars_mean_rank": descriptive_stats(
                partial_gradient_gars_rank_rhos
            ),
            "gradient_vs_gars_minus_uniform_rank": descriptive_stats(
                partial_gradient_rank_uplift_rhos
            ),
            "spectral_entropy_vs_gars_mean_rank": descriptive_stats(
                partial_entropy_gars_rank_rhos
            ),
        },
    }

    return {
        "analysis_name": "GARS cross-seed gradient reliability and spectral concentration",
        "seeds": seeds,
        "base_model_revision": base_model_revision,
        "design": {
            "gradient_unit": "Transformer block; joint LoRA A/B gradient Frobenius norm",
            "gradient_aggregation": (
                f"arithmetic mean over final {expected_capture_steps} optimizer updates"
            ),
            "spectral_entropy": (
                "normalized entropy of squared singular-value energy; "
                "lower means more concentrated"
            ),
            "allocation_unit": (
                "per LoRA projection module; layer summaries average "
                "the two target projections"
            ),
            "correlation": "Spearman rank correlation; descriptive, not a significance test",
        },
        "data_completeness": {
            "gradient_capture_steps_per_seed": {
                str(seed): artifacts[seed]["gradient_capture_steps"] for seed in seeds
            },
            "temporal_within_window_reliability": {
                "estimable": all(
                    artifacts[seed]["stepwise_capture_available"] for seed in seeds
                ),
                "stepwise_capture_available_by_seed": {
                    str(seed): artifacts[seed]["stepwise_capture_available"]
                    for seed in seeds
                },
                "reason": (
                    "Per-step layer scores are present for every seed; run "
                    "scripts/analyze_gradient_temporal.py for within-window diagnostics."
                    if all(
                        artifacts[seed]["stepwise_capture_available"] for seed in seeds
                    )
                    else (
                        "At least one artifact stores only the final-window mean and "
                        "accumulator sums/counts. Retrain that seed with the updated "
                        "runner to estimate within-window variance/autocorrelation."
                    )
                ),
            },
        },
        "cross_seed_gradient_rank_reliability": {
            "raw_gradient_magnitude_profiles": _profile_stats(gradient_rank_corr),
            "relative_importance_by_layer": _relative_importance_by_layer(
                seeds, artifacts
            ),
        },
        "cross_seed_spectral_profile_stability": {
            "per_module_entropy_profiles": _profile_stats(entropy_module_corr),
            "per_layer_mean_entropy_profiles": _profile_stats(entropy_layer_corr),
        },
        "cross_seed_rank_allocation_stability": {
            "per_module": {
                method: _profile_stats(rank_module_corr[method]) for method in METHODS
            },
            "per_layer_mean_rank": {
                method: _profile_stats(rank_layer_corr[method]) for method in METHODS
            },
        },
        "within_seed_associations": within_seed,
        "per_seed": per_seed,
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
        "--pruned-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS_DEFAULT),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned" / "mechanism_analysis.json",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError("Provide at least two unique seeds for cross-seed analysis.")

    config = _read_json(args.config)
    expected_revision = str(config["model"]["revision"])
    expected_layers = int(
        config["model"].get("num_transformer_layers", 12)
    )
    expected_modules = expected_layers * len(config["lora"]["target_modules"])
    expected_capture_steps = int(
        config["training"]["gradient_capture_last_optimizer_steps"]
    )
    min_rank = int(config["gars"]["min_rank"])
    max_rank = int(config["lora"]["rank"])

    artifacts = {
        seed: _load_seed_artifacts(
            seed,
            args.parent_root,
            args.pruned_root,
            expected_revision,
            expected_layers,
            expected_modules,
            expected_capture_steps,
            min_rank,
            max_rank,
        )
        for seed in args.seeds
    }
    first_modules = set(artifacts[args.seeds[0]]["entropy_by_module"])
    first_layers = set(artifacts[args.seeds[0]]["gradient_by_layer"])
    for seed in args.seeds[1:]:
        if set(artifacts[seed]["entropy_by_module"]) != first_modules:
            raise ValueError("Adapted module names differ across seeds.")
        if set(artifacts[seed]["gradient_by_layer"]) != first_layers:
            raise ValueError("Gradient layer IDs differ across seeds.")

    budgets = {artifacts[seed]["achieved_parameters"] for seed in args.seeds}
    if len(budgets) != 1:
        raise ValueError("Physical A/B parameter budgets differ across seeds.")
    report = build_mechanism_report(
        args.seeds,
        artifacts,
        expected_revision,
        expected_capture_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved mechanism analysis: {args.output}")
    print("\nCROSS-SEED GRADIENT RANK RELIABILITY")
    print(json.dumps(report["cross_seed_gradient_rank_reliability"], indent=2))
    print("\nCROSS-SEED SPECTRAL PROFILE STABILITY")
    print(json.dumps(report["cross_seed_spectral_profile_stability"], indent=2))
    print("\nCROSS-SEED RANK ALLOCATION STABILITY")
    print(json.dumps(report["cross_seed_rank_allocation_stability"], indent=2))
    print("\nWITHIN-SEED ASSOCIATIONS")
    print(json.dumps(report["within_seed_associations"], indent=2))
    print("\nPER-SEED DEPTH AND THRESHOLD DIAGNOSTICS")
    for seed, values in report["per_seed"].items():
        print(
            seed,
            json.dumps(
                {
                    "layer_depth_trends": values["layer_depth_trends"],
                    "gars_threshold_clipping": values["gars_threshold_clipping"],
                    "partial_associations": {
                        key: value
                        for key, value in values.items()
                        if "partial_spearman" in key
                    },
                },
                indent=2,
            ),
        )
    print("\nTEMPORAL RELIABILITY")
    print(json.dumps(report["data_completeness"]["temporal_within_window_reliability"], indent=2))


if __name__ == "__main__":
    main()
