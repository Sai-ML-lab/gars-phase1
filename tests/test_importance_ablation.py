import math

import pytest

from gars.importance import cyclically_shift_importance


def test_cyclic_shift_reassigns_scores_without_changing_multiset() -> None:
    original = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}

    shifted = cyclically_shift_importance(original, shift=1)

    assert shifted == {0: 4.0, 1: 1.0, 2: 2.0, 3: 3.0}
    assert sorted(shifted.values()) == sorted(original.values())
    assert math.isclose(sum(shifted.values()), sum(original.values()))


def test_zero_cyclic_shift_preserves_original_assignment() -> None:
    original = {2: 0.5, 5: 3.0, 9: 1.5}

    assert cyclically_shift_importance(original, shift=0) == original


def test_all_nonzero_cyclic_shifts_assign_each_source_score_once_per_layer() -> None:
    original = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    assignments = [
        cyclically_shift_importance(original, shift=shift)
        for shift in range(1, len(original))
    ]

    for layer_id in original:
        assigned_values = [assignment[layer_id] for assignment in assignments]
        assert sorted(assigned_values) == sorted(
            value for source_layer, value in original.items()
            if source_layer != layer_id
        )


@pytest.mark.parametrize("shift", [-1, 4])
def test_cyclic_shift_rejects_out_of_range_shift(shift: int) -> None:
    with pytest.raises(ValueError, match="shift must be"):
        cyclically_shift_importance({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, shift)


def test_cyclic_shift_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        cyclically_shift_importance({}, shift=0)

    with pytest.raises(TypeError, match="must be an integer"):
        cyclically_shift_importance({0: 1.0}, shift=True)

    with pytest.raises(ValueError, match="finite and non-negative"):
        cyclically_shift_importance({0: float("nan")}, shift=0)
