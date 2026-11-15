"""Tests for GARS profile diagnostics."""
from __future__ import annotations

import pytest

from gars.diagnostics import (
    descriptive_stats,
    pairwise_profile_correlations,
    partial_spearman_rank_correlation,
    spearman_rank_correlation,
)


def test_spearman_perfect_positive_and_negative() -> None:
    assert spearman_rank_correlation([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1, 2, 3], [8, 4, 2]) == pytest.approx(-1.0)


def test_spearman_handles_average_ties() -> None:
    result = spearman_rank_correlation([1, 1, 2, 3], [4, 4, 3, 1])
    assert result is not None
    assert -1.0 <= result <= 1.0


def test_spearman_constant_profile_returns_none() -> None:
    assert spearman_rank_correlation([1, 1, 1], [2, 3, 4]) is None


def test_spearman_rejects_mismatched_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        spearman_rank_correlation([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        spearman_rank_correlation([1, float("nan")], [1, 2])


def test_descriptive_stats_uses_sample_standard_deviation() -> None:
    result = descriptive_stats([1.0, 3.0, 5.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(3.0)
    assert result["sample_std"] == pytest.approx(2.0)


def test_pairwise_profiles_align_by_keys() -> None:
    result = pairwise_profile_correlations(
        {
            "42": {"0": 1.0, "1": 2.0, "2": 3.0},
            "123": {"2": 30.0, "0": 10.0, "1": 20.0},
            "999": {"0": 3.0, "1": 2.0, "2": 1.0},
        }
    )
    assert len(result) == 3
    assert result[0]["spearman_rho"] == pytest.approx(1.0)
    assert result[1]["spearman_rho"] == pytest.approx(-1.0)


def test_pairwise_profiles_reject_different_keys() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        pairwise_profile_correlations(
            {"42": {"0": 1.0, "1": 2.0}, "123": {"0": 2.0, "2": 3.0}}
        )



def test_partial_spearman_removes_perfect_depth_only_association() -> None:
    depth = [0, 1, 2, 3, 4]
    assert (
        partial_spearman_rank_correlation(depth, [10, 20, 30, 40, 50], depth)
        is None
    )


def test_partial_spearman_detects_residual_association_after_depth_control() -> None:
    depth = [0, 1, 2, 3, 4, 5]
    left = [0, 2, 1, 5, 3, 7]
    right = [0, 4, 2, 10, 6, 14]
    result = partial_spearman_rank_correlation(left, right, depth)
    assert result is not None
    assert result == pytest.approx(1.0)


def test_partial_spearman_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        partial_spearman_rank_correlation([1, 2], [1, 2], [1])
