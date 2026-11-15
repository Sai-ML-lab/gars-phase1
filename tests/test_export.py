import re

import pytest

from gars.export import alpha_for_preserved_scaling, peft_module_pattern_key


def test_standard_lora_alpha_pattern_preserves_alpha_over_rank() -> None:
    parent_alpha = 64.0
    parent_rank = 32
    retained_rank = 12

    adjusted_alpha = alpha_for_preserved_scaling(
        parent_rank,
        parent_alpha,
        retained_rank,
        use_rslora=False,
    )

    assert adjusted_alpha == pytest.approx(24.0)
    assert adjusted_alpha / retained_rank == pytest.approx(parent_alpha / parent_rank)


def test_rslora_alpha_pattern_preserves_alpha_over_sqrt_rank() -> None:
    parent_alpha = 64.0
    parent_rank = 32
    retained_rank = 8

    adjusted_alpha = alpha_for_preserved_scaling(
        parent_rank,
        parent_alpha,
        retained_rank,
        use_rslora=True,
    )

    assert adjusted_alpha == pytest.approx(32.0)
    assert adjusted_alpha / retained_rank**0.5 == pytest.approx(
        parent_alpha / parent_rank**0.5
    )


def test_peft_pattern_key_matches_exact_underlying_module_path() -> None:
    state_dict_name = (
        "base_model.model.deberta.encoder.layer.3.attention.self.query_proj"
    )
    pattern = peft_module_pattern_key(state_dict_name)
    module_path = "deberta.encoder.layer.3.attention.self.query_proj"

    assert re.match(rf"(.*\.)?({pattern})$", module_path)
    assert not re.match(
        rf"(.*\.)?({pattern})$",
        "deberta.encoder.layer.13.attention.self.query_proj",
    )


@pytest.mark.parametrize(
    ("parent_rank", "retained_rank", "parent_alpha"),
    [(0, 1, 8.0), (8, 0, 8.0), (8, 9, 8.0), (8, 4, 0.0)],
)
def test_invalid_alpha_pattern_inputs_are_rejected(
    parent_rank: int,
    retained_rank: int,
    parent_alpha: float,
) -> None:
    with pytest.raises(ValueError):
        alpha_for_preserved_scaling(parent_rank, parent_alpha, retained_rank)


def test_invalid_state_dict_module_prefix_is_rejected() -> None:
    with pytest.raises(ValueError):
        peft_module_pattern_key("deberta.encoder.layer.0.query_proj")
