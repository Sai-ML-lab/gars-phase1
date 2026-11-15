import pytest

from gars.mechanism import correlation_summary, descending_average_ranks, rank_agreement


def test_descending_average_ranks_uses_one_for_largest_and_average_for_ties():
    assert descending_average_ranks([4.0, 9.0, 9.0, 1.0]) == [3.0, 1.5, 1.5, 4.0]


def test_correlation_summary_detects_perfect_monotonic_relationships():
    result = correlation_summary([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert result["n"] == 4
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["kendall_tau_b"] == pytest.approx(1.0)


def test_correlation_summary_returns_null_for_constant_vector():
    result = correlation_summary([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert result["pearson_r"] is None
    assert result["spearman_rho"] is None
    assert result["kendall_tau_b"] is None


def test_rank_agreement_reports_exact_matches_and_absolute_error():
    result = rank_agreement([1, 2, 3, 4], [1, 3, 2, 6])
    assert result["exact_rank_match_fraction"] == pytest.approx(0.25)
    assert result["mean_absolute_rank_difference"] == pytest.approx(0.75)
    assert result["max_absolute_rank_difference"] == pytest.approx(2.0)


def test_correlation_summary_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="equal-length"):
        correlation_summary([1.0, 2.0, 3.0], [1.0, 2.0])


def test_rank_agreement_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        rank_agreement([], [])
