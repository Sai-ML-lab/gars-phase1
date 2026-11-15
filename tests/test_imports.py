from gars import (
    AdapterSpectrum,
    GradientImportanceAccumulator,
    __version__,
    gars_thresholds,
    rank_for_squared_energy,
)


def test_package_imports() -> None:
    assert __version__ == "0.2.0"
    assert AdapterSpectrum is not None
    assert GradientImportanceAccumulator is not None
    assert gars_thresholds is not None
    assert rank_for_squared_energy is not None
