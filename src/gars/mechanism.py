"""Small statistical helpers for descriptive GARS mechanism analysis."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


def descending_average_ranks(values: Iterable[float]) -> list[float]:
    """Return one-based ranks, with rank 1 assigned to the largest value."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence.")
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite numbers.")
    return [float(value) for value in rankdata(-array, method="average")]


def _optional_statistic(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def correlation_summary(
    left: Iterable[float],
    right: Iterable[float],
) -> dict[str, int | float | None]:
    """Return descriptive Pearson, Spearman and Kendall tau-b coefficients.

    No p-values are returned: the mechanism analysis uses only 12 blocks or
    24 modules per seed and is intended to be descriptive, not inferential.
    A coefficient is null when either vector is constant.
    """
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("left and right must be equal-length one-dimensional vectors.")
    if x.size < 3:
        raise ValueError("At least three paired observations are required.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation inputs must contain only finite values.")

    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        pearson_value = spearman_value = kendall_value = None
    else:
        pearson_value = _optional_statistic(pearsonr(x, y).statistic)
        spearman_value = _optional_statistic(spearmanr(x, y).statistic)
        kendall_value = _optional_statistic(kendalltau(x, y, variant="b").statistic)

    return {
        "n": int(x.size),
        "pearson_r": pearson_value,
        "spearman_rho": spearman_value,
        "kendall_tau_b": kendall_value,
    }


def rank_agreement(
    left: Iterable[float],
    right: Iterable[float],
) -> dict[str, int | float | None]:
    """Compare paired rank allocations using correlation and absolute error."""
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("left and right must be equal-length one-dimensional vectors.")
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("rank vectors must be non-empty and finite.")
    difference = np.abs(x - y)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        spearman_value = kendall_value = None
    else:
        spearman_value = _optional_statistic(spearmanr(x, y).statistic)
        kendall_value = _optional_statistic(kendalltau(x, y, variant="b").statistic)
    return {
        "n": int(x.size),
        "spearman_rho": spearman_value,
        "kendall_tau_b": kendall_value,
        "exact_rank_match_fraction": float(np.mean(x == y)),
        "mean_absolute_rank_difference": float(np.mean(difference)),
        "max_absolute_rank_difference": float(np.max(difference)),
    }
