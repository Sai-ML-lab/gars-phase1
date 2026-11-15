import pytest
import torch

from gars.spectral import (
    normalized_squared_spectral_entropy,
    qr_svd_lora_update,
    rank_for_squared_energy,
    squared_energy_ratio,
    truncate_lora_factors,
)


def test_energy_rank_is_smallest_rank_meeting_squared_energy_threshold() -> None:
    singular_values = torch.tensor([5.0, 2.0, 1.0])
    assert rank_for_squared_energy(singular_values, threshold=0.90, min_rank=1) == 2
    assert squared_energy_ratio(singular_values, retained_rank=2) == pytest.approx(29 / 30)


def test_rank_floor_is_respected() -> None:
    singular_values = torch.tensor([10.0, 1.0, 0.1])
    assert rank_for_squared_energy(singular_values, threshold=0.80, min_rank=2) == 2


def test_zero_spectrum_returns_rank_floor() -> None:
    assert rank_for_squared_energy(torch.zeros(4), threshold=0.99, min_rank=2) == 2


def test_invalid_energy_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        rank_for_squared_energy(torch.tensor([2.0, 1.0]), threshold=0.0)


def test_normalized_squared_entropy_endpoints() -> None:
    concentrated = normalized_squared_spectral_entropy(torch.tensor([1.0, 0.0, 0.0]))
    flat = normalized_squared_spectral_entropy(torch.tensor([1.0, 1.0, 1.0]))
    assert concentrated == pytest.approx(0.0, abs=1e-8)
    assert flat == pytest.approx(1.0, abs=1e-6)


def test_qr_svd_matches_direct_dense_svd() -> None:
    torch.manual_seed(17)
    A = torch.randn(5, 11, dtype=torch.float64)
    B = torch.randn(13, 5, dtype=torch.float64)

    U_qr, S_qr, Vh_qr = qr_svd_lora_update(A, B)
    U_dense, S_dense, _Vh_dense = torch.linalg.svd(B @ A, full_matrices=False)

    assert torch.allclose(S_qr, S_dense, atol=1e-9, rtol=1e-8)
    reconstructed = U_qr @ torch.diag(S_qr) @ Vh_qr
    assert torch.allclose(reconstructed, B @ A, atol=1e-9, rtol=1e-8)

    # Singular vectors can differ by sign within repeated/near-repeated subspaces;
    # compare the represented matrix rather than U/Vh elementwise.
    assert U_dense.shape == U_qr.shape


def test_truncation_returns_physical_rank_k_factors() -> None:
    torch.manual_seed(21)
    A = torch.randn(6, 12)
    B = torch.randn(10, 6)
    result = truncate_lora_factors(A, B, retained_rank=3)

    assert result.A.shape == (3, 12)
    assert result.B.shape == (10, 3)
    assert result.retained_rank == 3
    assert 0.0 <= result.retained_squared_energy <= 1.0

    U, S, Vh = torch.linalg.svd(B @ A, full_matrices=False)
    target = U[:, :3] @ torch.diag(S[:3]) @ Vh[:3, :]
    assert torch.allclose(result.B @ result.A, target, atol=2e-5, rtol=2e-5)


def test_full_rank_truncation_reconstructs_original_update() -> None:
    torch.manual_seed(7)
    A = torch.randn(4, 9)
    B = torch.randn(8, 4)
    result = truncate_lora_factors(A, B, retained_rank=4)
    assert torch.allclose(result.B @ result.A, B @ A, atol=2e-5, rtol=2e-5)


def test_low_precision_factors_are_promoted_for_decomposition_and_cast_back() -> None:
    if not hasattr(torch, "bfloat16"):
        pytest.skip("bfloat16 is unavailable")
    A = torch.randn(3, 8, dtype=torch.bfloat16)
    B = torch.randn(7, 3, dtype=torch.bfloat16)
    result = truncate_lora_factors(A, B, retained_rank=2)
    assert result.A.dtype == torch.bfloat16
    assert result.B.dtype == torch.bfloat16
