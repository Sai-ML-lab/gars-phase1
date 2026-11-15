import pytest
import torch

from gars.policy import (
    AdapterSpectrum,
    calibrate_gars_budget,
    calibrate_global_singular_value_budget,
    calibrate_uniform_energy_budget,
    gars_thresholds,
    physical_adapter_parameter_count,
)


def _spectra() -> list[AdapterSpectrum]:
    return [
        AdapterSpectrum("layer0.q_proj", 0, torch.tensor([5.0, 2.0, 1.0]), 4, 4),
        AdapterSpectrum("layer0.v_proj", 0, torch.tensor([3.0, 1.0, 0.5]), 4, 4),
        AdapterSpectrum("layer1.q_proj", 1, torch.tensor([4.0, 2.0, 0.5]), 4, 6),
    ]


def test_gars_thresholds_match_proposal_mapping_and_clipping() -> None:
    thresholds = gars_thresholds(
        {0: 1.0, 1: 2.0, 2: 3.0},
        tau_base=0.90,
        beta=0.5,
        tau_min=0.80,
        tau_max=0.99,
    )
    assert thresholds[0] == pytest.approx(0.80)
    assert thresholds[1] == pytest.approx(0.90)
    assert thresholds[2] == pytest.approx(0.99)


def test_zero_gradient_profile_uses_uniform_base_threshold() -> None:
    thresholds = gars_thresholds({0: 0.0, 1: 0.0}, tau_base=0.91)
    assert thresholds == {0: pytest.approx(0.91), 1: pytest.approx(0.91)}


def test_physical_parameter_count_uses_actual_module_dimensions() -> None:
    spectra = _spectra()
    ranks = {
        "layer0.q_proj": 2,
        "layer0.v_proj": 1,
        "layer1.q_proj": 3,
    }
    # 2*(4+4) + 1*(4+4) + 3*(4+6)
    assert physical_adapter_parameter_count(spectra, ranks) == 46


@pytest.mark.parametrize(
    "calibrator",
    [
        calibrate_gars_budget,
        calibrate_uniform_energy_budget,
        calibrate_global_singular_value_budget,
    ],
)
def test_budget_calibrators_return_valid_ranks_and_report_realized_budget(calibrator) -> None:
    spectra = _spectra()
    target = 50
    if calibrator is calibrate_gars_budget:
        result = calibrator(
            spectra,
            importance_by_layer={0: 1.0, 1: 1.6},
            target_parameters=target,
            min_rank=1,
        )
    else:
        result = calibrator(spectra, target_parameters=target, min_rank=1)

    assert result.achieved_parameters == physical_adapter_parameter_count(
        spectra, result.ranks_by_module
    )
    assert set(result.ranks_by_module) == {spec.module_name for spec in spectra}
    assert all(
        1 <= result.ranks_by_module[spec.module_name] <= spec.max_rank
        for spec in spectra
    )
    assert result.target_parameters == target
    assert result.minimum_parameters <= result.maximum_parameters
    assert result.absolute_error >= 0


def test_gars_budget_calibration_reports_when_target_is_unreachable() -> None:
    spectra = _spectra()
    result = calibrate_gars_budget(
        spectra,
        importance_by_layer={0: 1.0, 1: 1.0},
        target_parameters=1,
        min_rank=1,
    )
    assert result.achieved_parameters >= result.minimum_parameters
    assert not result.target_in_reachable_range
