from scripts.run_gars_gradient_permutation import _format_mean_sample_std


def test_format_mean_sample_std_shows_na_for_single_seed() -> None:
    assert _format_mean_sample_std({"mean": 0.45328, "sample_std": None}) == (
        "0.453280 ± n/a (one seed)"
    )


def test_format_mean_sample_std_formats_multiple_seeds() -> None:
    assert _format_mean_sample_std(
        {"mean": 0.45328, "sample_std": 0.00125}
    ) == "0.453280 ± 0.001250"
