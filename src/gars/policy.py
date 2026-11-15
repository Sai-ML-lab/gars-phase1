"""GARS, uniform-energy and global-singular-value budget allocation policies."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .spectral import rank_for_squared_energy


@dataclass(frozen=True)
class AdapterSpectrum:
    """Spectrum and physical shape of one adapted linear module."""

    module_name: str
    layer_id: int
    singular_values: torch.Tensor
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError("module_name must not be empty.")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative.")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        values = torch.as_tensor(self.singular_values).flatten()
        if values.numel() == 0 or not torch.isfinite(values).all() or torch.any(values < 0):
            raise ValueError("singular_values must be a non-empty finite non-negative vector.")
        if values.numel() > 1 and torch.any(values[1:] > values[:-1] + 1e-7):
            raise ValueError("singular_values must be sorted in descending order.")

    @property
    def max_rank(self) -> int:
        return int(torch.as_tensor(self.singular_values).numel())

    @property
    def parameters_per_rank(self) -> int:
        return self.in_features + self.out_features


@dataclass(frozen=True)
class BudgetCalibration:
    """Result of choosing a pruning threshold for a target physical parameter budget."""

    method: str
    threshold_parameter: float
    thresholds_by_layer: dict[int, float]
    ranks_by_module: dict[str, int]
    achieved_parameters: int
    target_parameters: int
    minimum_parameters: int
    maximum_parameters: int

    @property
    def absolute_error(self) -> int:
        return abs(self.achieved_parameters - self.target_parameters)

    @property
    def relative_error(self) -> float:
        if self.target_parameters == 0:
            return 0.0 if self.achieved_parameters == 0 else float("inf")
        return self.absolute_error / self.target_parameters

    @property
    def target_in_reachable_range(self) -> bool:
        return self.minimum_parameters <= self.target_parameters <= self.maximum_parameters


def gars_thresholds(
    importance_by_layer: Mapping[int, float],
    tau_base: float = 0.90,
    beta: float = 0.5,
    tau_min: float = 0.80,
    tau_max: float = 0.99,
) -> dict[int, float]:
    """Apply the proposal's normalized gradient-to-threshold mapping."""
    if not importance_by_layer:
        raise ValueError("importance_by_layer must not be empty.")
    if tau_base <= 0 or beta < 0:
        raise ValueError("tau_base must be positive and beta must be non-negative.")
    if not 0 < tau_min <= tau_max <= 1:
        raise ValueError("Require 0 < tau_min <= tau_max <= 1.")

    values = {int(k): float(v) for k, v in importance_by_layer.items()}
    if any(k < 0 for k in values):
        raise ValueError("Layer ids must be non-negative.")
    if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in values.values()):
        raise ValueError("Gradient importance values must be finite and non-negative.")

    mean_importance = sum(values.values()) / len(values)
    if mean_importance == 0.0:
        # No layer has a gradient signal: use the same base threshold everywhere.
        ratio_by_layer = {layer_id: 1.0 for layer_id in values}
    else:
        ratio_by_layer = {
            layer_id: score / mean_importance
            for layer_id, score in values.items()
        }

    return {
        layer_id: min(
            tau_max,
            max(tau_min, tau_base * (ratio ** beta)),
        )
        for layer_id, ratio in ratio_by_layer.items()
    }


def _ranks_for_layer_thresholds(
    spectra: list[AdapterSpectrum],
    thresholds_by_layer: Mapping[int, float],
    min_rank: int,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for spec in spectra:
        if spec.layer_id not in thresholds_by_layer:
            raise KeyError(f"No threshold supplied for layer {spec.layer_id}.")
        output[spec.module_name] = rank_for_squared_energy(
            spec.singular_values,
            float(thresholds_by_layer[spec.layer_id]),
            min_rank=min_rank,
        )
    return output


def physical_adapter_parameter_count(
    spectra: list[AdapterSpectrum],
    ranks_by_module: Mapping[str, int],
) -> int:
    """Count stored A/B factor parameters after physical rank truncation."""
    expected_names = {spec.module_name for spec in spectra}
    if set(ranks_by_module) != expected_names:
        raise ValueError("ranks_by_module must contain exactly one rank per spectrum.")
    total = 0
    for spec in spectra:
        rank = int(ranks_by_module[spec.module_name])
        if not 1 <= rank <= spec.max_rank:
            raise ValueError(f"Invalid rank {rank} for module {spec.module_name}.")
        total += rank * spec.parameters_per_rank
    return total


def _budget_for_thresholds(
    spectra: list[AdapterSpectrum],
    thresholds_by_layer: Mapping[int, float],
    min_rank: int,
) -> tuple[dict[str, int], int]:
    ranks = _ranks_for_layer_thresholds(spectra, thresholds_by_layer, min_rank)
    return ranks, physical_adapter_parameter_count(spectra, ranks)


def _budget_result(
    method: str,
    threshold_parameter: float,
    thresholds_by_layer: dict[int, float],
    ranks: dict[str, int],
    spectra: list[AdapterSpectrum],
    target_parameters: int,
    minimum_parameters: int,
    maximum_parameters: int,
) -> BudgetCalibration:
    return BudgetCalibration(
        method=method,
        threshold_parameter=float(threshold_parameter),
        thresholds_by_layer=dict(thresholds_by_layer),
        ranks_by_module=dict(ranks),
        achieved_parameters=physical_adapter_parameter_count(spectra, ranks),
        target_parameters=int(target_parameters),
        minimum_parameters=int(minimum_parameters),
        maximum_parameters=int(maximum_parameters),
    )


def calibrate_gars_budget(
    spectra: list[AdapterSpectrum],
    importance_by_layer: Mapping[int, float],
    target_parameters: int,
    beta: float = 0.5,
    min_rank: int = 2,
    tau_min: float = 0.80,
    tau_max: float = 0.99,
    iterations: int = 64,
) -> BudgetCalibration:
    """Find tau_base whose GARS allocation is nearest to a target physical budget.

    The search is deterministic. Since integer ranks make the budget discrete,
    the requested budget may not be exactly attainable; the returned result
    reports the realized count and error rather than pretending it is exact.
    """
    if not spectra:
        raise ValueError("spectra must not be empty.")
    if target_parameters <= 0 or iterations < 1:
        raise ValueError("target_parameters and iterations must be positive.")
    if not 0 < tau_min <= tau_max <= 1:
        raise ValueError("Require 0 < tau_min <= tau_max <= 1.")

    if beta == 0.0:
        # With beta=0, every layer must receive the same threshold. Delegate to
        # the uniform calibrator so the ablation is exactly rank-equivalent.
        gars_thresholds(
            importance_by_layer,
            tau_base=0.90,
            beta=0.0,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        uniform = calibrate_uniform_energy_budget(
            spectra,
            target_parameters=target_parameters,
            min_rank=min_rank,
            tau_min=tau_min,
            tau_max=tau_max,
            iterations=iterations,
        )
        uniform_thresholds = {
            int(layer_id): uniform.threshold_parameter
            for layer_id in importance_by_layer
        }
        return _budget_result(
            "gars_energy_threshold",
            uniform.threshold_parameter,
            uniform_thresholds,
            uniform.ranks_by_module,
            spectra,
            target_parameters,
            uniform.minimum_parameters,
            uniform.maximum_parameters,
        )

    def evaluate(base: float) -> tuple[dict[int, float], dict[str, int], int]:
        thresholds = gars_thresholds(
            importance_by_layer,
            tau_base=base,
            beta=beta,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        ranks, budget = _budget_for_thresholds(spectra, thresholds, min_rank)
        return thresholds, ranks, budget

    low = 1e-12
    high = 1.0
    low_eval = evaluate(low)
    high_eval = evaluate(high)

    # Expand the upper bracket if required; threshold clipping eventually plateaus.
    while high_eval[2] < target_parameters and high < 1e6:
        high *= 2.0
        high_eval = evaluate(high)

    minimum_budget = low_eval[2]
    maximum_budget = high_eval[2]

    if target_parameters <= minimum_budget:
        return _budget_result(
            "gars_energy_threshold", low, low_eval[0], low_eval[1],
            spectra, target_parameters, minimum_budget, maximum_budget,
        )
    if target_parameters >= maximum_budget:
        return _budget_result(
            "gars_energy_threshold", high, high_eval[0], high_eval[1],
            spectra, target_parameters, minimum_budget, maximum_budget,
        )

    candidates = [(low, low_eval), (high, high_eval)]
    for _ in range(iterations):
        mid = (low + high) / 2.0
        mid_eval = evaluate(mid)
        candidates.append((mid, mid_eval))
        if mid_eval[2] < target_parameters:
            low = mid
        else:
            high = mid

    # Prefer the lower parameter count on exact ties to avoid exceeding the budget.
    best_base, best_eval = min(
        candidates,
        key=lambda item: (
            abs(item[1][2] - target_parameters),
            item[1][2] > target_parameters,
            item[0],
        ),
    )
    return _budget_result(
        "gars_energy_threshold", best_base, best_eval[0], best_eval[1],
        spectra, target_parameters, minimum_budget, maximum_budget,
    )


def calibrate_uniform_energy_budget(
    spectra: list[AdapterSpectrum],
    target_parameters: int,
    min_rank: int = 2,
    tau_min: float = 0.80,
    tau_max: float = 0.99,
    iterations: int = 64,
) -> BudgetCalibration:
    """Calibrate one common squared-energy threshold across all adapter modules."""
    if not spectra:
        raise ValueError("spectra must not be empty.")
    if target_parameters <= 0 or iterations < 1:
        raise ValueError("target_parameters and iterations must be positive.")
    if not 0 < tau_min <= tau_max <= 1:
        raise ValueError("Require 0 < tau_min <= tau_max <= 1.")

    def evaluate(threshold: float) -> tuple[dict[str, int], int]:
        ranks = {
            spec.module_name: rank_for_squared_energy(
                spec.singular_values, threshold, min_rank=min_rank
            )
            for spec in spectra
        }
        return ranks, physical_adapter_parameter_count(spectra, ranks)

    # Higher retention thresholds preserve more rank, hence more parameters.
    low, high = tau_min, tau_max
    low_eval, high_eval = evaluate(low), evaluate(high)
    minimum_budget, maximum_budget = low_eval[1], high_eval[1]

    if target_parameters <= minimum_budget:
        return _budget_result(
            "uniform_local_energy", low, {}, low_eval[0],
            spectra, target_parameters, minimum_budget, maximum_budget,
        )
    if target_parameters >= maximum_budget:
        return _budget_result(
            "uniform_local_energy", high, {}, high_eval[0],
            spectra, target_parameters, minimum_budget, maximum_budget,
        )

    candidates = [(low, low_eval), (high, high_eval)]
    for _ in range(iterations):
        mid = (low + high) / 2.0
        mid_eval = evaluate(mid)
        candidates.append((mid, mid_eval))
        if mid_eval[1] < target_parameters:
            low = mid
        else:
            high = mid

    best_threshold, best_eval = min(
        candidates,
        key=lambda item: (
            abs(item[1][1] - target_parameters),
            item[1][1] > target_parameters,
            item[0],
        ),
    )
    return _budget_result(
        "uniform_local_energy", best_threshold, {}, best_eval[0],
        spectra, target_parameters, minimum_budget, maximum_budget,
    )


def calibrate_global_singular_value_budget(
    spectra: list[AdapterSpectrum],
    target_parameters: int,
    min_rank: int = 2,
) -> BudgetCalibration:
    """Find one global singular-value cutoff nearest to a physical parameter budget."""
    if not spectra:
        raise ValueError("spectra must not be empty.")
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive.")

    all_values = torch.cat(
        [torch.as_tensor(spec.singular_values, dtype=torch.float64, device="cpu").flatten()
         for spec in spectra]
    )
    unique = torch.unique(all_values, sorted=True)
    if unique.numel() == 1:
        delta = max(float(unique[0].item()) * 1e-6, 1e-12)
        candidates = [float(unique[0].item()) - delta, float(unique[0].item()) + delta]
    else:
        values = [float(x) for x in unique.tolist()]
        mids = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
        delta = max(abs(values[0]) * 1e-6, 1e-12)
        candidates = [values[0] - delta, *mids, values[-1] + delta]

    def evaluate(cutoff: float) -> tuple[dict[str, int], int]:
        ranks: dict[str, int] = {}
        for spec in spectra:
            kept = int(torch.sum(torch.as_tensor(spec.singular_values, device="cpu") >= cutoff).item())
            ranks[spec.module_name] = min(max(kept, min_rank), spec.max_rank)
        return ranks, physical_adapter_parameter_count(spectra, ranks)

    evaluations = [(cutoff, evaluate(cutoff)) for cutoff in candidates]
    budgets = [item[1][1] for item in evaluations]
    minimum_budget, maximum_budget = min(budgets), max(budgets)
    best_cutoff, best_eval = min(
        evaluations,
        key=lambda item: (
            abs(item[1][1] - target_parameters),
            item[1][1] > target_parameters,
            item[0],
        ),
    )
    return _budget_result(
        "global_singular_value_cutoff", best_cutoff, {}, best_eval[0],
        spectra, target_parameters, minimum_budget, maximum_budget,
    )
