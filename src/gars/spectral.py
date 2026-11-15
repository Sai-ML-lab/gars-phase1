"""Numerically stable LoRA-factor spectral decomposition and rank selection.

The represented update is DeltaW = B @ A, with A shaped [rank, in_features]
and B shaped [out_features, rank]. LoRA scaling is deliberately external:
it does not affect relative singular-value energy retention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TruncatedFactors:
    """Physically smaller LoRA factors and diagnostics for a truncated update."""

    A: torch.Tensor
    B: torch.Tensor
    singular_values: torch.Tensor
    retained_rank: int
    retained_squared_energy: float


def _validate_factors(A: torch.Tensor, B: torch.Tensor) -> None:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("LoRA factors A and B must both be 2-D tensors.")
    if A.shape[0] != B.shape[1]:
        raise ValueError(
            f"Rank dimensions do not match: A has rank {A.shape[0]}, "
            f"B has rank {B.shape[1]}."
        )
    if A.device != B.device:
        raise ValueError("LoRA factors A and B must be on the same device.")
    if not (A.is_floating_point() and B.is_floating_point()):
        raise TypeError("LoRA factors must use floating-point dtypes.")


def qr_svd_lora_update(
    A: torch.Tensor, B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a thin SVD of B @ A using reduced QR factorisations.

    It avoids materialising the full out_features x in_features update before
    decomposing the rank-sized core. The returned U, S, Vh satisfy
    B @ A ~= U @ diag(S) @ Vh.

    Low-precision factors are promoted to float32 for QR/SVD and the returned
    decomposition stays in the promoted dtype. This avoids unsupported or
    inaccurate low-precision linear-algebra kernels on some devices.
    """
    _validate_factors(A, B)

    work_dtype = (
        torch.float64
        if A.dtype == torch.float64 and B.dtype == torch.float64
        else torch.float32
    )
    with torch.no_grad():
        A_work = A.detach().to(dtype=work_dtype)
        B_work = B.detach().to(dtype=work_dtype)

        # B = Q_B R_B and A.T = Q_A R_A.
        Q_b, R_b = torch.linalg.qr(B_work, mode="reduced")
        Q_a, R_a = torch.linalg.qr(A_work.transpose(0, 1), mode="reduced")

        # The core is at most rank x rank (or smaller if rank exceeds a feature dim).
        core = R_b @ R_a.transpose(0, 1)
        U_core, singular_values, Vh_core = torch.linalg.svd(
            core, full_matrices=False
        )

        U = Q_b @ U_core
        Vh = Vh_core @ Q_a.transpose(0, 1)

    return U, singular_values, Vh


def squared_energy_ratio(
    singular_values: torch.Tensor, retained_rank: int
) -> float:
    """Return retained squared-singular-value energy divided by total energy."""
    values = _validate_singular_values(singular_values)
    if not 0 <= retained_rank <= values.numel():
        raise ValueError("retained_rank must be between zero and the spectrum length.")
    total = torch.sum(values.square())
    if float(total.item()) == 0.0:
        return 1.0 if retained_rank >= 0 else 0.0
    retained = torch.sum(values[:retained_rank].square())
    return float((retained / total).item())


def _validate_singular_values(singular_values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(singular_values).detach().flatten()
    if values.numel() == 0:
        raise ValueError("singular_values must not be empty.")
    if not torch.isfinite(values).all():
        raise ValueError("singular_values must contain only finite values.")
    if torch.any(values < 0):
        raise ValueError("singular_values must be non-negative.")
    if values.numel() > 1 and torch.any(values[1:] > values[:-1] + 1e-7):
        raise ValueError("singular_values must be sorted in descending order.")
    return values


def rank_for_squared_energy(
    singular_values: torch.Tensor,
    threshold: float,
    min_rank: int = 1,
) -> int:
    """Choose the smallest rank meeting a squared-Frobenius-energy threshold.

    The rank floor is applied even when the selected spectrum is all zeros.
    """
    values = _validate_singular_values(singular_values)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1].")
    if not isinstance(min_rank, int) or min_rank < 1:
        raise ValueError("min_rank must be a positive integer.")
    if min_rank > values.numel():
        raise ValueError(
            f"min_rank={min_rank} exceeds available rank {values.numel()}."
        )

    energy = values.to(dtype=torch.float64).square()
    total_energy = energy.sum()
    if float(total_energy.item()) == 0.0:
        return min_rank

    cumulative = torch.cumsum(energy, dim=0) / total_energy
    target = torch.tensor(threshold, dtype=cumulative.dtype, device=cumulative.device)
    rank = int(torch.searchsorted(cumulative, target, right=False).item()) + 1
    return min(max(rank, min_rank), values.numel())


def normalized_squared_spectral_entropy(
    singular_values: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Normalized entropy of squared singular-value mass.

    Values approach 0 for a spectrum concentrated in a few directions and 1
    for a flat spectrum. This is the squared-energy entropy operationalisation;
    it should not be conflated with Roy–Vetterli entropy over normalized sigma_i.
    """
    values = _validate_singular_values(singular_values).to(dtype=torch.float64)
    n = values.numel()
    if n <= 1:
        return 0.0

    energy = values.square()
    total = energy.sum()
    if float(total.item()) <= eps:
        return 0.0

    probabilities = energy / total
    positive = probabilities > 0
    entropy = -torch.sum(
        probabilities[positive] * torch.log(probabilities[positive] + eps)
    )
    normalized = entropy / torch.log(torch.tensor(float(n), dtype=torch.float64))
    return float(torch.clamp(normalized, min=0.0, max=1.0).item())


def truncate_lora_factors(
    A: torch.Tensor,
    B: torch.Tensor,
    retained_rank: int,
) -> TruncatedFactors:
    """Return physically truncated factors approximating the rank-k SVD of B @ A.

    The returned tensors have shapes [k, in_features] and [out_features, k].
    The external LoRA scaling coefficient must be preserved by the caller.
    """
    U, singular_values, Vh = qr_svd_lora_update(A, B)
    max_rank = singular_values.numel()
    if not isinstance(retained_rank, int) or not 1 <= retained_rank <= max_rank:
        raise ValueError(f"retained_rank must be in [1, {max_rank}].")

    kept_s = singular_values[:retained_rank]
    sqrt_s = torch.sqrt(torch.clamp(kept_s, min=0.0))
    B_new = U[:, :retained_rank] * sqrt_s.unsqueeze(0)
    A_new = sqrt_s.unsqueeze(1) * Vh[:retained_rank, :]

    # Retain the original parameter dtype; calculations above are performed in fp32/fp64.
    A_new = A_new.to(dtype=A.dtype)
    B_new = B_new.to(dtype=B.dtype)
    retained_energy = squared_energy_ratio(singular_values, retained_rank)
    return TruncatedFactors(
        A=A_new,
        B=B_new,
        singular_values=singular_values,
        retained_rank=retained_rank,
        retained_squared_energy=retained_energy,
    )
