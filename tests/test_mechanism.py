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
    assert result["mean_absolute_rank_difference"] == pytest.approx(1.0)
    assert result["max_absolute_rank_difference"] == pytest.approx(2.0)


def test_correlation_summary_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="equal-length"):
        correlation_summary([1.0, 2.0, 3.0], [1.0, 2.0])


def test_rank_agreement_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        rank_agreement([], [])


def test_mechanism_analysis_builds_cross_seed_diagnostics():
    from scripts.analyze_gradient_spectral_mechanism import METHODS, analyze

    artifacts = {}
    for seed_offset, seed in enumerate((42, 123, 999)):
        importance = {
            layer: float((layer + 1) * (1.0 + 0.01 * seed_offset))
            for layer in range(12)
        }
        mean_score = sum(importance.values()) / len(importance)
        normalized = {layer: score / mean_score for layer, score in importance.items()}
        gradient_ranks = descending_average_ranks(
            [importance[layer] for layer in range(12)]
        )
        entropy_by_layer = {
            layer: 0.2 + 0.03 * layer for layer in range(12)
        }
        module_names = [
            f"encoder.layer.{layer}.{projection}"
            for layer in range(12)
            for projection in ("query_proj", "value_proj")
        ]
        policies = {}
        layer_mean_ranks = {}
        for method_index, method in enumerate(METHODS):
            rank_by_module = {
                name: 2 + (index + seed_offset + method_index) % 8
                for index, name in enumerate(module_names)
            }
            entropy_by_module = {
                name: entropy_by_layer[int(name.split(".")[2])]
                for name in module_names
            }
            layer_values = {
                layer: [
                    float(rank_by_module[name])
                    for name in module_names
                    if int(name.split(".")[2]) == layer
                ]
                for layer in range(12)
            }
            layer_mean_ranks[method] = {
                layer: sum(values) / len(values)
                for layer, values in layer_values.items()
            }
            policies[method] = {
                "allocation": {"rank_histogram": {}},
                "modules": {},
                "rank_by_module": rank_by_module,
                "entropy_by_module": entropy_by_module,
            }
        artifacts[seed] = {
            "gradient_importance": importance,
            "normalized_gradient_importance": normalized,
            "gradient_rank": {
                layer: rank
                for layer, rank in enumerate(gradient_ranks)
            },
            "layer_mean_spectral_entropy": entropy_by_layer,
            "policies": policies,
            "layer_mean_retained_rank": layer_mean_ranks,
            "module_names": module_names,
        }

    report = analyze(artifacts)
    assert report["seeds"] == [42, 123, 999]
    assert set(report["rank_allocation_stability"]) == {
        "pairwise_stability",
        "per_layer_mean_retained_rank_across_seeds",
    }
    assert "42_vs_123" in report["gradient_signal_reliability"][
        "pairwise_normalized_importance_correlations"
    ]
    assert report["observations_per_seed"]["adapted_modules"] == 24
