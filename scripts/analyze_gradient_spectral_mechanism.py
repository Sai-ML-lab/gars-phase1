"""Analyze cross-seed gradient reliability, spectral concentration and rank allocation."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from gars.mechanism import (
    correlation_summary,
    descending_average_ranks,
    rank_agreement,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gars", "global_singular_value", "uniform_local_energy")
EXPECTED_LAYERS = tuple(range(12))
EXPECTED_MODULES = 24


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate the mean of an empty sequence.")
    return float(statistics.mean(values))


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _mean_std(values: list[float]) -> dict[str, float | None]:
    return {"mean": _mean(values), "sample_std": _sample_std(values)}


def _layer_means_by_module(
    module_values: dict[str, dict[str, Any]],
    field: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for module_name, item in module_values.items():
        layer = int(item["layer_id"])
        grouped.setdefault(layer, []).append(float(item[field]))
    if set(grouped) != set(EXPECTED_LAYERS):
        raise ValueError(
            f"Expected layers {list(EXPECTED_LAYERS)}, found {sorted(grouped)}."
        )
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("Expected exactly two adapted target projections per block.")
    return {layer: _mean(values) for layer, values in sorted(grouped.items())}


def _read_seed_artifacts(
    seed: int,
    parent_root: Path,
    pruned_root: Path,
) -> dict[str, Any]:
    seed_dir = parent_root / f"seed_{seed}"
    gradient_data = _read_json(seed_dir / "gradient_importance.json")
    if int(gradient_data.get("seed", -1)) != seed:
        raise ValueError(f"Gradient artifact seed mismatch for seed {seed}.")
    if int(gradient_data.get("capture_steps", -1)) != 50:
        raise ValueError(
            f"Seed {seed} has {gradient_data.get('capture_steps')} gradient steps; "
            "V1 expects exactly 50."
        )

    importance = {
        int(layer): float(score)
        for layer, score in gradient_data["importance_by_transformer_block"].items()
    }
    if set(importance) != set(EXPECTED_LAYERS):
        raise ValueError(
            f"Seed {seed}: expected 12 gradient scores, found {sorted(importance)}."
        )
    if any(score < 0 for score in importance.values()):
        raise ValueError(f"Seed {seed}: gradient scores must be non-negative.")

    policies: dict[str, dict[str, Any]] = {}
    module_sets: dict[str, set[str]] = {}
    for method in METHODS:
        allocation_path = pruned_root / f"seed_{seed}" / method / "allocation.json"
        allocation = _read_json(allocation_path)
        if allocation.get("method") != method:
            raise ValueError(
                f"{allocation_path} says method={allocation.get('method')!r}."
            )
        modules = allocation.get("per_module")
        if not isinstance(modules, dict) or len(modules) != EXPECTED_MODULES:
            raise ValueError(
                f"{allocation_path}: expected {EXPECTED_MODULES} module records."
            )
        required_fields = {
            "layer_id",
            "retained_rank",
            "squared_spectral_entropy",
            "squared_energy_retained",
        }
        for module_name, record in modules.items():
            missing = required_fields - set(record)
            if missing:
                raise ValueError(
                    f"{allocation_path}: {module_name} missing {sorted(missing)}."
                )
        module_sets[method] = set(modules)
        policies[method] = {
            "allocation": allocation,
            "modules": modules,
            "rank_by_module": {
                name: int(record["retained_rank"])
                for name, record in modules.items()
            },
            "entropy_by_module": {
                name: float(record["squared_spectral_entropy"])
                for name, record in modules.items()
            },
            "retained_energy_by_module": {
                name: float(record["squared_energy_retained"])
                for name, record in modules.items()
            },
        }

    if len({frozenset(values) for values in module_sets.values()}) != 1:
        raise ValueError(f"Seed {seed}: module inventories differ across policies.")
    shared_modules = module_sets[METHODS[0]]

    # Entropy is calculated from the common, unpruned parent spectrum. It must
    # therefore be identical across allocation policies for the same seed.
    reference_entropy = policies[METHODS[0]]["entropy_by_module"]
    for method in METHODS[1:]:
        for module_name in shared_modules:
            observed = policies[method]["entropy_by_module"][module_name]
            expected = reference_entropy[module_name]
            if abs(observed - expected) > 1e-10:
                raise ValueError(
                    f"Seed {seed}: {method} entropy differs from common parent "
                    f"spectrum for {module_name}."
                )

    layer_entropy = _layer_means_by_module(
        policies["gars"]["modules"],
        "squared_spectral_entropy",
    )
    layer_ranks = {
        method: _layer_means_by_module(policies[method]["modules"], "retained_rank")
        for method in METHODS
    }
    return {
        "seed": seed,
        "gradient_importance": importance,
        "normalized_gradient_importance": {
            layer: score / _mean(list(importance.values()))
            for layer, score in importance.items()
        },
        "gradient_rank": {
            layer: rank
            for layer, rank in zip(
                sorted(importance),
                descending_average_ranks([importance[layer] for layer in sorted(importance)]),
                strict=True,
            )
        },
        "layer_mean_spectral_entropy": layer_entropy,
        "policies": policies,
        "layer_mean_retained_rank": layer_ranks,
        "module_names": sorted(shared_modules),
    }


def _pairwise_seed_records(
    vectors: dict[int, dict[Any, float]],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for left_seed, right_seed in itertools.combinations(sorted(vectors), 2):
        keys = sorted(set(vectors[left_seed]) & set(vectors[right_seed]))
        if len(keys) != len(vectors[left_seed]) or len(keys) != len(vectors[right_seed]):
            raise ValueError("Seed vectors do not have identical keys.")
        left = [float(vectors[left_seed][key]) for key in keys]
        right = [float(vectors[right_seed][key]) for key in keys]
        key = f"{left_seed}_vs_{right_seed}"
        records[key] = {
            "correlation": correlation_summary(left, right),
            "agreement": rank_agreement(
                descending_average_ranks(left),
                descending_average_ranks(right),
            ),
        }
    return records


def _cross_seed_layer_summary(
    seeds: dict[int, dict[str, Any]],
    value_getter,
) -> dict[str, Any]:
    layers = sorted(EXPECTED_LAYERS)
    result: dict[str, Any] = {}
    for layer in layers:
        values = [float(value_getter(seeds[seed], layer)) for seed in sorted(seeds)]
        result[str(layer)] = _mean_std(values)
    return result


def analyze(
    seed_artifacts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seeds = sorted(seed_artifacts)
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for cross-seed reliability.")

    normalized_gradients = {
        seed: item["normalized_gradient_importance"]
        for seed, item in seed_artifacts.items()
    }
    gradient_ranks = {
        seed: item["gradient_rank"] for seed, item in seed_artifacts.items()
    }
    gradient_pairwise = _pairwise_seed_records(normalized_gradients)
    gradient_rank_pairwise = _pairwise_seed_records(gradient_ranks)

    seed_level: dict[str, Any] = {}
    for seed in seeds:
        item = seed_artifacts[seed]
        normalized = item["normalized_gradient_importance"]
        layer_entropy = item["layer_mean_spectral_entropy"]
        layer_gradient = item["gradient_importance"]
        seed_methods: dict[str, Any] = {}
        for method in METHODS:
            layer_ranks = item["layer_mean_retained_rank"][method]
            module_records = item["policies"][method]["modules"]
            module_entropy = item["policies"][method]["entropy_by_module"]
            module_ranks = item["policies"][method]["rank_by_module"]
            module_names = item["module_names"]
            seed_methods[method] = {
                "layer_mean_retained_rank": {
                    str(layer): float(rank)
                    for layer, rank in layer_ranks.items()
                },
                "gradient_vs_layer_mean_retained_rank": correlation_summary(
                    [layer_gradient[layer] for layer in EXPECTED_LAYERS],
                    [layer_ranks[layer] for layer in EXPECTED_LAYERS],
                ),
                "module_entropy_vs_retained_rank": correlation_summary(
                    [module_entropy[name] for name in module_names],
                    [module_ranks[name] for name in module_names],
                ),
                "mean_rank": _mean([float(module_ranks[name]) for name in module_names]),
                "rank_histogram": item["policies"][method]["allocation"]["rank_histogram"],
            }
        gars_thresholds = item["policies"]["gars"]["allocation"].get(
            "thresholds_by_layer", {}
        )
        seed_level[str(seed)] = {
            "gradient_importance_by_layer": {
                str(layer): float(layer_gradient[layer]) for layer in EXPECTED_LAYERS
            },
            "normalized_gradient_importance_by_layer": {
                str(layer): float(normalized[layer]) for layer in EXPECTED_LAYERS
            },
            "gradient_rank_by_layer": {
                str(layer): float(item["gradient_rank"][layer])
                for layer in EXPECTED_LAYERS
            },
            "layer_mean_squared_spectral_entropy": {
                str(layer): float(layer_entropy[layer]) for layer in EXPECTED_LAYERS
            },
            "gradient_vs_layer_mean_spectral_entropy": correlation_summary(
                [layer_gradient[layer] for layer in EXPECTED_LAYERS],
                [layer_entropy[layer] for layer in EXPECTED_LAYERS],
            ),
            "gars_thresholds_by_layer": gars_thresholds,
            "gars_thresholds_at_lower_bound": sum(
                abs(float(value) - 0.80) <= 1e-10
                for value in gars_thresholds.values()
            ),
            "gars_thresholds_at_upper_bound": sum(
                abs(float(value) - 0.99) <= 1e-10
                for value in gars_thresholds.values()
            ),
            "methods": seed_methods,
        }

    gradient_layer_summary = {
        "normalized_importance_mean_and_sample_std": _cross_seed_layer_summary(
            seed_artifacts,
            lambda item, layer: item["normalized_gradient_importance"][layer],
        ),
        "importance_rank_mean_and_sample_std": _cross_seed_layer_summary(
            seed_artifacts,
            lambda item, layer: item["gradient_rank"][layer],
        ),
    }

    allocation_stability: dict[str, Any] = {}
    rank_layer_averages: dict[str, Any] = {}
    spectral_rank_relationships: dict[str, Any] = {}
    for method in METHODS:
        module_rank_vectors = {
            seed: seed_artifacts[seed]["policies"][method]["rank_by_module"]
            for seed in seeds
        }
        layer_rank_vectors = {
            seed: seed_artifacts[seed]["layer_mean_retained_rank"][method]
            for seed in seeds
        }
        allocation_stability[method] = {
            "pairwise_module_rank_stability": _pairwise_seed_records(
                module_rank_vectors
            ),
            "pairwise_layer_mean_rank_stability": _pairwise_seed_records(
                layer_rank_vectors
            ),
        }
        rank_layer_averages[method] = _cross_seed_layer_summary(
            seed_artifacts,
            lambda item, layer, method=method: item["layer_mean_retained_rank"][
                method
            ][layer],
        )
        spectral_rank_relationships[method] = {
            str(seed): seed_level[str(seed)]["methods"][method][
                "module_entropy_vs_retained_rank"
            ]
            for seed in seeds
        }

    layer_entropy_summary = _cross_seed_layer_summary(
        seed_artifacts,
        lambda item, layer: item["layer_mean_spectral_entropy"][layer],
    )
    gradient_entropy_pairwise = {
        str(seed): seed_level[str(seed)]["gradient_vs_layer_mean_spectral_entropy"]
        for seed in seeds
    }

    return {
        "analysis": "GARS V1 gradient-signal reliability and spectral-concentration diagnostics",
        "seeds": seeds,
        "observations_per_seed": {
            "transformer_blocks": len(EXPECTED_LAYERS),
            "adapted_modules": EXPECTED_MODULES,
            "gradient_capture_steps": 50,
        },
        "seed_level": seed_level,
        "gradient_signal_reliability": {
            "pairwise_normalized_importance_correlations": gradient_pairwise,
            "pairwise_importance_rank_correlations": gradient_rank_pairwise,
            "per_layer_across_seed": gradient_layer_summary,
        },
        "spectral_concentration": {
            "per_layer_entropy_mean_and_sample_std_across_seeds": layer_entropy_summary,
            "gradient_importance_vs_layer_mean_entropy_by_seed": gradient_entropy_pairwise,
            "module_entropy_vs_retained_rank_by_method_and_seed": spectral_rank_relationships,
        },
        "rank_allocation_stability": {
            "pairwise_stability": allocation_stability,
            "per_layer_mean_retained_rank_across_seeds": rank_layer_averages,
        },
        "interpretation_notes": [
            "Gradient scores are joint Frobenius norms averaged over the final 50 optimizer updates, as specified for V1.",
            "Gradient scores are normalized by their within-seed layer mean before cross-seed comparison; this removes global scale differences but retains relative layer importance.",
            "Layer spectral entropy is the arithmetic mean of the two target-projection normalized squared-spectrum entropies in that block.",
            "Entropy is computed from the common unpruned parent spectrum and must match across pruning policies within a seed.",
            "Correlations use 12 blocks or 24 modules per seed and are descriptive only; no p-values or significance claims are produced.",
            "The correlation between GARS ranks and gradient importance is policy-coupled by construction, not independent evidence that gradients predict an optimal rank.",
            "Cross-seed rank agreement describes allocation stability under these three training seeds, not generalization to other tasks or model scales.",
        ],
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v1_pruned" / "mechanism_analysis.json",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed list contains duplicates.")

    artifacts = {
        seed: _read_seed_artifacts(seed, args.parent_root, args.pruned_root)
        for seed in args.seeds
    }
    report = analyze(artifacts)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved mechanism analysis: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
