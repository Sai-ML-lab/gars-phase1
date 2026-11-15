from gars import (
    AdapterSpectrum,
    GradientImportanceAccumulator,
    __version__,
    alpha_for_preserved_scaling,
    peft_module_pattern_key,
    gars_thresholds,
    rank_for_squared_energy,
)


def test_package_imports() -> None:
    assert __version__ == "0.3.0"
    assert AdapterSpectrum is not None
    assert GradientImportanceAccumulator is not None
    assert gars_thresholds is not None
    assert rank_for_squared_energy is not None
    assert alpha_for_preserved_scaling is not None
    assert peft_module_pattern_key is not None
