import pytest

from gars.temporal import analyze_stepwise_gradient_scores


def _rows(profiles: list[dict[int, float]]) -> list[dict[str, object]]:
    return [
        {
            "optimizer_step": 100 + index,
            "importance_by_transformer_block": {
                str(layer): value for layer, value in profile.items()
            },
        }
        for index, profile in enumerate(profiles)
    ]


def test_temporal_summary_detects_perfectly_stable_layer_ordering() -> None:
    profiles = [
        {0: 1.0, 1: 2.0, 2: 3.0},
        {0: 2.0, 1: 4.0, 2: 6.0},
        {0: 3.0, 1: 6.0, 2: 9.0},
        {0: 4.0, 1: 8.0, 2: 12.0},
    ]
    means = {0: 2.5, 1: 5.0, 2: 7.5}

    report = analyze_stepwise_gradient_scores(_rows(profiles), means)

    assert report["n_steps"] == 4
    assert report["mean_profile_reproduced"] is True
    stability = report["within_window_rank_stability"]
    assert stability["mean_pairwise_spearman_rho"] == pytest.approx(1.0)
    assert stability["adjacent_step_mean_spearman_rho"] == pytest.approx(1.0)
    assert report["top_rank_stability"]["mean_adjacent_top3_jaccard"] == pytest.approx(1.0)
    assert report["per_layer"]["0"]["coefficient_of_variation"] == pytest.approx(0.51639778)


def test_temporal_summary_detects_rank_reversals() -> None:
    profiles = [
        {0: 3.0, 1: 2.0, 2: 1.0},
        {0: 1.0, 1: 2.0, 2: 3.0},
        {0: 3.0, 1: 2.0, 2: 1.0},
    ]
    means = {0: 7.0 / 3.0, 1: 2.0, 2: 5.0 / 3.0}

    report = analyze_stepwise_gradient_scores(_rows(profiles), means)

    assert report["within_window_rank_stability"]["mean_pairwise_spearman_rho"] < 0.0
    assert report["within_window_rank_stability"]["minimum_pairwise_spearman_rho"] == pytest.approx(-1.0)


def test_temporal_summary_rejects_inconsistent_step_means() -> None:
    profiles = [
        {0: 1.0, 1: 2.0},
        {0: 2.0, 1: 3.0},
        {0: 3.0, 1: 4.0},
    ]

    with pytest.raises(ValueError, match="do not reproduce"):
        analyze_stepwise_gradient_scores(_rows(profiles), {0: 10.0, 1: 3.0})


def test_temporal_summary_rejects_layer_inventory_changes() -> None:
    profiles = _rows([
        {0: 1.0, 1: 2.0},
        {0: 2.0, 1: 3.0},
        {0: 3.0, 2: 4.0},
    ])

    with pytest.raises(ValueError, match="Layer inventory changed"):
        analyze_stepwise_gradient_scores(profiles, {0: 2.0, 1: 2.5})
