"""Utilities for exporting physically truncated PEFT LoRA adapters."""
from __future__ import annotations

import math
import re


_STATE_DICT_MODULE_PREFIX = "base_model.model."


def alpha_for_preserved_scaling(
    parent_rank: int,
    parent_alpha: float,
    retained_rank: int,
    use_rslora: bool = False,
) -> float:
    """Return a per-module alpha that preserves the parent's LoRA multiplier.

    Standard LoRA uses alpha / rank. Rank-stabilized LoRA uses alpha / sqrt(rank).
    """
    if parent_rank < 1 or retained_rank < 1 or retained_rank > parent_rank:
        raise ValueError("Require 1 <= retained_rank <= parent_rank.")
    if not math.isfinite(parent_alpha) or parent_alpha <= 0:
        raise ValueError("parent_alpha must be finite and positive.")

    if use_rslora:
        return float(parent_alpha * math.sqrt(retained_rank / parent_rank))
    return float(parent_alpha * retained_rank / parent_rank)


def peft_module_pattern_key(state_dict_module_name: str) -> str:
    """Convert a saved PEFT state-dict module path to an exact rank-pattern key.

    PEFT state dictionaries prefix model modules with 'base_model.model.'.
    LoRA rank/alpha patterns are matched against the underlying base-model
    module name, so the wrapper prefix is removed and dots are regex-escaped.
    The key intentionally has no leading '^': PEFT's matcher wraps patterns
    with an optional module-path prefix before applying the expression.
    """
    if not state_dict_module_name:
        raise ValueError("state_dict_module_name must not be empty.")
    if not state_dict_module_name.startswith(_STATE_DICT_MODULE_PREFIX):
        raise ValueError(
            "Expected a PEFT state-dict module path beginning with "
            f"{_STATE_DICT_MODULE_PREFIX!r}, got {state_dict_module_name!r}."
        )
    base_module_name = state_dict_module_name.removeprefix(_STATE_DICT_MODULE_PREFIX)
    if not base_module_name:
        raise ValueError("The base-model module name must not be empty.")
    return re.escape(base_module_name)
