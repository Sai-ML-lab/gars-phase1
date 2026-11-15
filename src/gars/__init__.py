"""Reusable components for the GARS research project."""

from .importance import GradientImportanceAccumulator, collect_lora_layer_gradient_norms
from .policy import (
    AdapterSpectrum,
    BudgetCalibration,
    calibrate_gars_budget,
    calibrate_global_singular_value_budget,
    calibrate_uniform_energy_budget,
    gars_thresholds,
    physical_adapter_parameter_count,
)
from .spectral import (
    TruncatedFactors,
    normalized_squared_spectral_entropy,
    qr_svd_lora_update,
    rank_for_squared_energy,
    squared_energy_ratio,
    truncate_lora_factors,
)

__version__ = "0.2.0"

__all__ = [
    "AdapterSpectrum",
    "BudgetCalibration",
    "GradientImportanceAccumulator",
    "TruncatedFactors",
    "calibrate_gars_budget",
    "calibrate_global_singular_value_budget",
    "calibrate_uniform_energy_budget",
    "collect_lora_layer_gradient_norms",
    "gars_thresholds",
    "normalized_squared_spectral_entropy",
    "physical_adapter_parameter_count",
    "qr_svd_lora_update",
    "rank_for_squared_energy",
    "squared_energy_ratio",
    "truncate_lora_factors",
]
