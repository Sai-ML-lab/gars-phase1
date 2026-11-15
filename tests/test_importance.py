import math

import pytest
import torch

from gars.importance import (
    GradientImportanceAccumulator,
    collect_lora_layer_gradient_norms,
    transformer_layer_index,
)


def _parameter(name: str, values: list[float]) -> tuple[str, torch.nn.Parameter]:
    parameter = torch.nn.Parameter(torch.zeros(len(values)))
    parameter.grad = torch.tensor(values, dtype=torch.float32)
    return name, parameter


def test_transformer_layer_index_handles_common_hf_names() -> None:
    assert transformer_layer_index(
        "base_model.model.encoder.layer.12.attention.self.query.lora_A.default.weight"
    ) == 12
    assert transformer_layer_index(
        "base_model.model.layers.3.self_attn.q_proj.lora_B.default.weight"
    ) == 3
    assert transformer_layer_index("transformer.h.7.attn.c_attn.lora_A.weight") == 7
    assert transformer_layer_index("classifier.dense.weight") is None


def test_layer_gradient_norm_jointly_aggregates_lora_a_and_b() -> None:
    named = [
        _parameter(
            "base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight",
            [1.0, 1.0, 1.0, 1.0],
        ),
        _parameter(
            "base_model.model.encoder.layer.0.attention.self.query.lora_B.default.weight",
            [2.0, 2.0, 2.0, 2.0],
        ),
        _parameter(
            "base_model.model.encoder.layer.1.attention.self.query.lora_A.default.weight",
            [3.0, 4.0],
        ),
    ]
    scores = collect_lora_layer_gradient_norms(named)
    assert scores[0] == pytest.approx(math.sqrt(4.0 + 16.0))
    assert scores[1] == pytest.approx(5.0)


def test_gradient_accumulator_uses_mean_of_step_norms() -> None:
    accumulator = GradientImportanceAccumulator()
    first = [
        _parameter(
            "model.encoder.layer.0.attention.self.query.lora_A.weight",
            [3.0, 4.0],
        )
    ]
    second = [
        _parameter(
            "model.encoder.layer.0.attention.self.query.lora_A.weight",
            [0.0, 2.0],
        )
    ]
    accumulator.update(first)
    accumulator.update(second)

    assert accumulator.steps_seen == 2
    assert accumulator.mean_scores()[0] == pytest.approx((5.0 + 2.0) / 2.0)
    assert accumulator.state_dict()["aggregation"] == (
        "mean_of_per_step_joint_lora_factor_gradient_norms"
    )


def test_missing_lora_gradients_are_rejected_for_a_capture_step() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):
        GradientImportanceAccumulator().update(
            [("model.encoder.layer.0.attention.query.weight", parameter)]
        )
