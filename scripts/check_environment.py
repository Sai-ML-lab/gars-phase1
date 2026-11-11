"""Check Apple Silicon/PyTorch capabilities needed for the GARS POC.

Run with: python scripts/check_environment.py
This is a diagnostic, not a benchmark.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def run_device_checks(device: torch.device) -> dict[str, Any]:
    """Exercise GEMM, QR, SVD, and autograd on the selected device."""
    results: dict[str, Any] = {"device": str(device), "checks": {}}
    generator = torch.Generator(device="cpu").manual_seed(20261112)
    x_cpu = torch.randn(32, 24, generator=generator, dtype=torch.float32)
    x = x_cpu.to(device).requires_grad_(True)

    checks = results["checks"]
    try:
        product = x @ x.transpose(0, 1)
        checks["matrix_multiplication"] = {
            "status": "passed",
            "shape": list(product.shape),
        }

        q, r = torch.linalg.qr(x, mode="reduced")
        qr_error = torch.linalg.norm(x - q @ r).detach().cpu().item()
        checks["qr"] = {"status": "passed", "reconstruction_error": qr_error}

        u, singular_values, vh = torch.linalg.svd(x, full_matrices=False)
        svd_reconstruction = u @ torch.diag(singular_values) @ vh
        svd_error = torch.linalg.norm(x - svd_reconstruction).detach().cpu().item()
        checks["svd"] = {"status": "passed", "reconstruction_error": svd_error}

        loss = product.square().mean() + singular_values.mean() * 1e-3
        loss.backward()
        grad_norm = x.grad.detach().norm().cpu().item() if x.grad is not None else None
        if grad_norm is None or not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError("Gradient missing or non-finite")
        checks["autograd_backward"] = {"status": "passed", "gradient_norm": grad_norm}

    except Exception as exc:  # Report the failed operation without hiding its traceback.
        checks["device_test"] = {
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass
                
    return results


def main() -> int:
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "package_versions": {},
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": bool(torch.cuda.is_available()),
        "tests": [],
    }

    for package in ["transformers", "peft", "accelerate", "datasets", "numpy", "scipy"]:
        try:
            report["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][package] = "not-installed"

    devices = [torch.device("cpu")]
    if mps_available:
        devices.insert(0, torch.device("mps"))

    for device in devices:
        report["tests"].append(run_device_checks(device))

    out_dir = Path("results/day1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "environment_check.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved diagnostic to: {out_file}")
    if not mps_available:
        print("\nNOTE: MPS is unavailable. CPU diagnostics ran; investigate macOS/PyTorch setup.")
    return 0 if all(
        test["checks"].get("device_test", {}).get("status") != "failed"
        for test in report["tests"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
