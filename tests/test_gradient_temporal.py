import json

import pytest

from scripts.analyze_gradient_temporal import _analyze_seed


def _write_gradient_artifact(path, rows):
    path.write_text(
        json.dumps(
            {
                "stepwise_importance_by_optimizer_step": [
                    {
                        "optimizer_step": 101 + index,
                        "importance_by_transformer_block": {
                            str(layer): value
                            for layer, value in enumerate(row)
                        },
                    }
                    for index, row in enumerate(rows)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_temporal_analysis_reports_rank_stability_and_layer_variability(tmp_path) -> None:
    path = tmp_path / "gradient_importance.json"
    rows = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [1.5, 2.5, 4.5],
        [0.5, 1.5, 2.5],
    ]
    _write_gradient_artifact(path, rows)

    report = _analyze_seed(seed=42, path=path, expected_steps=4)

    assert report["step_count"] == 4
    assert report["layer_count"] == 3
    assert report["capture_optimizer_steps"] == [101, 104]
    assert report["step_profile_vs_window_mean_rank_correlation"]["n"] == 4
    assert report["adjacent_step_rank_correlation"]["mean"] == pytest.approx(1.0)
    assert report["early_half_vs_late_half_mean_profile_spearman_rho"] == pytest.approx(1.0)
    assert report["within_layer_gradient_cv"]["n"] == 3
    assert len(report["per_layer"]) == 3


def test_temporal_analysis_requires_stepwise_gradient_artifact(tmp_path) -> None:
    path = tmp_path / "gradient_importance.json"
    path.write_text(json.dumps({"importance_by_transformer_block": {"0": 1.0}}))

    with pytest.raises(ValueError, match="does not contain stepwise"):
        _analyze_seed(seed=42, path=path, expected_steps=4)


def test_temporal_analysis_rejects_inconsistent_layer_inventory(tmp_path) -> None:
    path = tmp_path / "gradient_importance.json"
    artifact = {
        "stepwise_importance_by_optimizer_step": [
            {
                "optimizer_step": 1,
                "importance_by_transformer_block": {"0": 1.0, "1": 2.0},
            },
            {
                "optimizer_step": 2,
                "importance_by_transformer_block": {"0": 1.0, "2": 2.0},
            },
        ]
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="Layer inventory changed"):
        _analyze_seed(seed=42, path=path, expected_steps=2)
