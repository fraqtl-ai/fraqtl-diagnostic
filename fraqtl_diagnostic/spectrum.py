"""Per-layer spectral analysis — Hessian capture + eigendecomposition.

Captures the input covariance H = E[x^T x] for a target projection on a fixed
calibration corpus, then eigendecomposes to get the spectrum.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Spectrum:
    """Eigendecomposition of a single layer's input covariance."""
    layer: int
    projection: str
    dim: int
    n_samples: int
    eigvals: np.ndarray      # descending, float64


def capture_single_hessian(
    model, calib_ids: torch.Tensor, target_module, device: str = "cuda"
) -> tuple[torch.Tensor, int]:
    """Single forward pass over calibration, accumulating input covariance in fp32.

    H is accumulated on the compute device (GPU if available) — matmul is
    ~30× faster than CPU for large dims (e.g. 14336×14336 down_proj on 7B).
    Returned to CPU at the end.

    Returns (H_cpu, n_samples). H is [in_dim, in_dim] mean-normalized.
    """
    in_dim = target_module.in_features
    H = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
    count = 0

    def hook(_mod, inp, _out):
        nonlocal count
        x = inp[0].detach().to(torch.float32)
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        H.add_(x.T @ x)
        count += x.size(0)

    h = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(calib_ids.size(0)):
                model(calib_ids[i:i + 1].to(device))
    finally:
        h.remove()
    if count > 0:
        H /= count
    return H.cpu(), count


def eigendecompose(H: torch.Tensor) -> np.ndarray:
    """Return eigenvalues of H, descending, float64. Symmetrizes first."""
    H = 0.5 * (H + H.T)
    try:
        H_gpu = H.to("cuda")
        eig = torch.linalg.eigvalsh(H_gpu).cpu().numpy()
        del H_gpu
        torch.cuda.empty_cache()
    except Exception:
        eig = torch.linalg.eigvalsh(H).numpy()
    return eig[::-1].astype(np.float64).copy()
