"""Integration tests for the local GARS mechanism-analysis workflow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_gars_mechanisms import (
    EXPECTED_BASE_REVISION,
    METHODS,
    _load_seed_artifacts,
    build_mechanism_report,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_seed_artifacts(root: Path, seed: int) -> tuple[Path, Path]:
    parent_root = root / "parent"
    pruned_root = root / "pruned"
    gradient = {str(layer): float((layer + 1) * (seed / 100.0)) for layer in range(12)}
    _write_json(
        parent_root / f"seed_{seed}" / "gradient_importance.json",
        {
            "seed": seed,
            "capture_steps": 50,
            "importance_by_transformer_block": gradient,
            "accumulator_state": {
                "count_by_layer": {str(layer): 50 for layer in range(12)}
            },
        },
    )
    _write_json(
        parent_root / f"seed_{seed}" / "run_metadata.json",
        {"seed": seed},
    )
    _write_json(
        pruned_root / f"seed_{seed}" / "comparison_summary.json",
        {
            "seed": seed,
            "base_model_revision": EXPECTED_BASE_REVISION,
        },
    )

    for method in METHODS:
        modules = {}
        for layer in range(12):
            for projection_index, projection in enumerate(("query_proj", "value_proj")):
                name = (
                    "base_model.model.deberta.encoder.layer."
                    f"{layer}.attention.self.{projection}"
                )
                value = 0.2 + layer * 0.01 + projection_index * 0.005
                modules[name] = {
                    "layer_id": layer,
                    "retained_rank": 20 + projection_index,
                    "squared_spectral_entropy": value,
                    "squared_energy_retained": 0.95,
                }
        _write_json(
            pruned_root / f"seed_{seed}" / method / "allocation.json",
            {
                "method": method,
                "reload_validation": "passed",
                "achieved_parameters": 708096,
                "per_module": modules,
                "thresholds_by_layer": {str(layer): 0.9 for layer in range(12)},
                "importance_by_layer": gradient,
            },
        )
    return parent_root, pruned_root


def test_load_seed_artifacts_and_build_cross_seed_report(tmp_path: Path) -> None:
    artifacts = {}
    parent_root = tmp_path / "parent"
    pruned_root = tmp_path / "pruned"
    for seed in (42, 123, 999):
        seed_parent, seed_pruned = _make_seed_artifacts(tmp_path, seed)
        assert seed_parent == parent_root
        assert seed_pruned == pruned_root
        artifacts[seed] = _load_seed_artifacts(seed, parent_root, pruned_root)

    report = build_mechanism_report([42, 123, 999], artifacts)

    assert report["base_model_revision"] == EXPECTED_BASE_REVISION
    assert report["cross_seed_gradient_rank_reliability"][
        "raw_gradient_magnitude_profiles"
    ]["rho_descriptive_stats"]["mean"] == pytest.approx(1.0)
    assert report["cross_seed_rank_allocation_stability"]["per_module"][
        "gars"
    ]["rho_descriptive_stats"]["n"] == 3
    assert report["data_completeness"]["temporal_within_window_reliability"][
        "estimable"
    ] is False
    assert len(report["per_seed"]["42"]["layer_profiles"]) == 12
    assert len(report["per_seed"]["42"]["module_profiles"]) == 24


def test_load_seed_artifacts_rejects_failed_reload(tmp_path: Path) -> None:
    parent_root, pruned_root = _make_seed_artifacts(tmp_path, 42)
    path = pruned_root / "seed_42" / "gars" / "allocation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["reload_validation"] = "failed"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="reload validation"):
        _load_seed_artifacts(42, parent_root, pruned_root)


def test_load_seed_artifacts_rejects_mismatched_revision(tmp_path: Path) -> None:
    parent_root, pruned_root = _make_seed_artifacts(tmp_path, 42)
    path = pruned_root / "seed_42" / "comparison_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["base_model_revision"] = "main"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="base revision"):
        _load_seed_artifacts(42, parent_root, pruned_root)


def test_load_seed_artifacts_enforces_configured_projection_count(tmp_path: Path) -> None:
    parent_root, pruned_root = _make_seed_artifacts(tmp_path, 42)

    with pytest.raises(ValueError, match="expected 23 adapted projections, got 24"):
        _load_seed_artifacts(
            42,
            parent_root,
            pruned_root,
            expected_modules=23,
        )
