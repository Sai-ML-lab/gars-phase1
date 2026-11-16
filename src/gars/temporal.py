"""Temporal reliability diagnostics for saved optimizer-step gradient profiles."""
from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from scipy.stats import rankdata

from .diagnostics import spearman_rank_correlation


def analyze_stepwise_gradient_scores(
    stepwise_scores: Sequence[Mapping[str, Any]],
    mean_scores_by_layer: Mapping[int | str, float],
) -> dict[str, Any]:
    """Summarize stability, variability and drift in a fixed gradient window."""
    if len(stepwise_scores) < 3:
        raise ValueError("At least three stepwise gradient observations are required.")

    parsed: list[tuple[int, dict[int, float]]] = []
    for row in stepwise_scores:
        step = int(row["optimizer_step"])
        values = {
            int(layer): float(score)
            for layer, score in row["importance_by_transformer_block"].items()
        }
        if not values:
            raise ValueError(f"Optimizer step {step} contains no layer scores.")
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError(f"Optimizer step {step} has invalid gradient scores.")
        parsed.append((step, values))

    parsed.sort(key=lambda item: item[0])
    step_ids = [item[0] for item in parsed]
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("Optimizer step IDs must be unique.")
    layer_ids = sorted(parsed[0][1])
    expected_layers = set(layer_ids)
    for step, values in parsed:
        if set(values) != expected_layers:
            raise ValueError(f"Layer inventory changed at optimizer step {step}.")

    stored_means = {int(layer): float(score) for layer, score in mean_scores_by_layer.items()}
    if set(stored_means) != expected_layers:
        raise ValueError("Stepwise layer inventory differs from stored mean profile.")

    score_matrix = [
        [values[layer] for layer in layer_ids]
        for _, values in parsed
    ]
    num_steps = len(score_matrix)
    num_layers = len(layer_ids)
    calculated_means = {
        layer: statistics.mean(row[index] for row in score_matrix)
        for index, layer in enumerate(layer_ids)
    }
    if any(
        not math.isclose(
            calculated_means[layer],
            stored_means[layer],
            rel_tol=1e-6,
            abs_tol=1e-10,
        )
        for layer in layer_ids
    ):
        raise ValueError(
            "Stepwise arithmetic means do not reproduce importance_by_transformer_block."
        )

    normalized_matrix: list[list[float]] = []
    for row in score_matrix:
        row_mean = statistics.mean(row)
        normalized_matrix.append(
            [value / row_mean for value in row] if row_mean > 0.0 else [1.0] * num_layers
        )

    rank_matrix = [
        {
            layer: float(rank_values[index])
            for index, layer in enumerate(layer_ids)
        }
        for rank_values in (
            rankdata([-value for value in row], method="average")
            for row in score_matrix
        )
    ]

    pairwise_rhos: list[float] = []
    for left, right in itertools.combinations(range(num_steps), 2):
        rho = spearman_rank_correlation(
            [rank_matrix[left][layer] for layer in layer_ids],
            [rank_matrix[right][layer] for layer in layer_ids],
        )
        if rho is not None:
            pairwise_rhos.append(rho)

    lagged: dict[str, Any] = {}
    max_lag = min(10, num_steps - 1)
    for lag in range(1, max_lag + 1):
        values: list[float] = []
        for index in range(num_steps - lag):
            rho = spearman_rank_correlation(
                [rank_matrix[index][layer] for layer in layer_ids],
                [rank_matrix[index + lag][layer] for layer in layer_ids],
            )
            if rho is not None:
                values.append(rho)
        lagged[str(lag)] = {
            "n_pairs": len(values),
            "mean_spearman_rho": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }

    top1_counts = {layer: 0 for layer in layer_ids}
    top3_sets: list[set[int]] = []
    for ranks in rank_matrix:
        top1_counts[min(layer_ids, key=lambda layer: ranks[layer])] += 1
        top3_sets.append(
            {layer for layer in layer_ids if ranks[layer] <= min(3, num_layers)}
        )
    top3_jaccard = [
        len(left & right) / len(left | right)
        for left, right in zip(top3_sets, top3_sets[1:])
    ]

    per_layer: dict[str, Any] = {}
    time_axis = list(range(num_steps))
    for index, layer in enumerate(layer_ids):
        raw_values = [row[index] for row in score_matrix]
        normalized_values = [row[index] for row in normalized_matrix]
        mean_value = statistics.mean(raw_values)
        sd_value = statistics.stdev(raw_values)
        normalized_mean = statistics.mean(normalized_values)
        normalized_sd = statistics.stdev(normalized_values)
        per_layer[str(layer)] = {
            "mean_gradient_norm": mean_value,
            "sample_sd_gradient_norm": sd_value,
            "coefficient_of_variation": sd_value / mean_value if mean_value > 0 else None,
            "mean_normalized_importance": normalized_mean,
            "sample_sd_normalized_importance": normalized_sd,
            "normalized_importance_cv": (
                normalized_sd / normalized_mean if normalized_mean > 0 else None
            ),
            "spearman_trend_vs_capture_step": spearman_rank_correlation(
                time_axis, normalized_values
            ),
            "top1_frequency": top1_counts[layer] / num_steps,
        }

    step_mean_norms = [statistics.mean(row) for row in score_matrix]
    return {
        "n_steps": num_steps,
        "layer_ids": layer_ids,
        "optimizer_step_start": step_ids[0],
        "optimizer_step_end": step_ids[-1],
        "mean_profile_reproduced": True,
        "within_window_rank_stability": {
            "n_step_pairs": len(pairwise_rhos),
            "mean_pairwise_spearman_rho": (
                statistics.mean(pairwise_rhos) if pairwise_rhos else None
            ),
            "sample_sd_pairwise_spearman_rho": (
                statistics.stdev(pairwise_rhos) if len(pairwise_rhos) > 1 else None
            ),
            "minimum_pairwise_spearman_rho": min(pairwise_rhos) if pairwise_rhos else None,
            "maximum_pairwise_spearman_rho": max(pairwise_rhos) if pairwise_rhos else None,
            "adjacent_step_mean_spearman_rho": lagged["1"]["mean_spearman_rho"],
        },
        "lagged_rank_stability": lagged,
        "top_rank_stability": {
            "top1_frequency_by_layer": {
                str(layer): top1_counts[layer] / num_steps for layer in layer_ids
            },
            "mean_adjacent_top3_jaccard": (
                statistics.mean(top3_jaccard) if top3_jaccard else None
            ),
        },
        "global_gradient_magnitude": {
            "mean_across_layers_by_step_mean": statistics.mean(step_mean_norms),
            "sample_sd_across_steps": statistics.stdev(step_mean_norms),
            "coefficient_of_variation_across_steps": (
                statistics.stdev(step_mean_norms) / statistics.mean(step_mean_norms)
                if statistics.mean(step_mean_norms) > 0
                else None
            ),
        },
        "per_layer": per_layer,
    }
