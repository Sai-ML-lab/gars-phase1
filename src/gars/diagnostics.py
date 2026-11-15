"""Statistical diagnostics for GARS gradient and spectral profiles."""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata


def spearman_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    """Return Spearman's rho, or None when the inputs are not informative."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Spearman inputs must have equal lengths.")
    if x.size < 2:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs must contain only finite values.")
    rank_x = rankdata(x, method="average")
    rank_y = rankdata(y, method="average")
    if np.ptp(rank_x) == 0.0 or np.ptp(rank_y) == 0.0:
        return None
    value = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return value if math.isfinite(value) else None


def descriptive_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return n, mean and sample standard deviation for finite observations."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"n": 0, "mean": None, "sample_std": None}
    if not np.isfinite(array).all():
        raise ValueError("Descriptive statistics require finite values.")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else None,
    }


def pairwise_profile_correlations(
    profiles: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Compare matching keyed profile values across every pair of profiles."""
    output: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(sorted(profiles), 2):
        left = profiles[left_name]
        right = profiles[right_name]
        if set(left) != set(right):
            raise ValueError(
                f"Profile keys differ for {left_name} and {right_name}."
            )
        keys = sorted(left)
        rho = spearman_rank_correlation(
            [float(left[key]) for key in keys],
            [float(right[key]) for key in keys],
        )
        output.append(
            {
                "left": left_name,
                "right": right_name,
                "n": len(keys),
                "spearman_rho": rho,
            }
        )
    return output


def partial_spearman_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
    control: Sequence[float],
) -> float | None:
    """Correlate ranked variables after linearly residualizing ranked control.

    This is a descriptive partial Spearman correlation. It is useful here for
    checking whether layer-wise associations persist after adjusting for the
    monotonic ordering of Transformer depth. It is not a causal estimate.
    """
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    z = np.asarray(control, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size != z.size:
        raise ValueError("Partial Spearman inputs must have equal lengths.")
    if x.size < 3:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("Partial Spearman inputs must contain only finite values.")

    rank_x = rankdata(x, method="average")
    rank_y = rankdata(y, method="average")
    rank_z = rankdata(z, method="average")
    design = np.column_stack((np.ones(rank_z.size), rank_z))
    residual_x = rank_x - design @ np.linalg.lstsq(design, rank_x, rcond=None)[0]
    residual_y = rank_y - design @ np.linalg.lstsq(design, rank_y, rcond=None)[0]
    if np.ptp(residual_x) <= 1e-12 or np.ptp(residual_y) <= 1e-12:
        return None
    value = float(np.corrcoef(residual_x, residual_y)[0, 1])
    return value if math.isfinite(value) else None
