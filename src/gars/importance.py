"""Gradient-importance collection for standard PEFT LoRA factors."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import torch


_LAYER_PATTERN = re.compile(
    r"(?:^|\.)(?:layers|layer|h|block)\.(\d+)(?:\.|$)",
    flags=re.IGNORECASE,
)
_LORA_FACTOR_PATTERN = re.compile(r"lora_[AB](?:\.|$|_)", flags=re.IGNORECASE)


def transformer_layer_index(parameter_name: str) -> int | None:
    """Extract a Transformer block index from common HF module-name patterns."""
    match = _LAYER_PATTERN.search(parameter_name)
    return int(match.group(1)) if match else None


def collect_lora_layer_gradient_norms(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[int, float]:
    """Compute a joint Frobenius gradient norm per Transformer block.

    All LoRA A/B gradient tensors attached to a block are treated as one
    concatenated gradient vector. The resulting norm is sqrt(sum_i ||grad_i||_F^2).
    Missing gradients are skipped; this should be called at a consistent optimizer
    boundary with the same model/module set on every captured step.
    """
    squared_norms: dict[int, float] = {}

    for name, parameter in named_parameters:
        if not _LORA_FACTOR_PATTERN.search(name):
            continue
        layer_id = transformer_layer_index(name)
        if layer_id is None or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise ValueError(f"Non-finite LoRA gradient found in {name}.")
        norm = float(torch.linalg.vector_norm(grad.float()).item())
        squared_norms[layer_id] = squared_norms.get(layer_id, 0.0) + norm * norm

    return {
        layer_id: math.sqrt(value)
        for layer_id, value in sorted(squared_norms.items())
    }


class GradientImportanceAccumulator:
    """Accumulate per-step layer gradient norms over a fixed measurement window.

    This deliberately uses the arithmetic mean of per-step joint norms, not an
    EMA or an empirical-Fisher score. It is a transparent operationalisation of
    the proposal's live gradient magnitude and can later be ablated against EMA/Fisher.
    """

    def __init__(self) -> None:
        self._sum_by_layer: dict[int, float] = {}
        self._count_by_layer: dict[int, int] = {}
        self.steps_seen = 0

    def update(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> dict[int, float]:
        """Capture one optimizer-step's gradients and return the step-level scores."""
        step_scores = collect_lora_layer_gradient_norms(named_parameters)
        if not step_scores:
            raise ValueError("No LoRA A/B gradients were found in this capture step.")
        self.steps_seen += 1
        for layer_id, score in step_scores.items():
            self._sum_by_layer[layer_id] = self._sum_by_layer.get(layer_id, 0.0) + score
            self._count_by_layer[layer_id] = self._count_by_layer.get(layer_id, 0) + 1
        return step_scores

    def mean_scores(self) -> dict[int, float]:
        """Return the mean captured per-step norm for each observed layer."""
        return {
            layer_id: self._sum_by_layer[layer_id] / self._count_by_layer[layer_id]
            for layer_id in sorted(self._sum_by_layer)
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps_seen": self.steps_seen,
            "sum_by_layer": dict(self._sum_by_layer),
            "count_by_layer": dict(self._count_by_layer),
            "aggregation": "mean_of_per_step_joint_lora_factor_gradient_norms",
        }
