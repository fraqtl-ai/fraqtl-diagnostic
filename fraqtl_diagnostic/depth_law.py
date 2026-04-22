"""Depth-law: how γ / knee / k95 evolve across layer depth.

The depth-law is a linear regression of γ_layer vs normalized depth (layer/max_layer).
A high |slope| with high R² means the spectrum shape changes predictably with depth.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .shannon import LayerFingerprint


@dataclass
class DepthLawFit:
    projection: str
    slope: float                 # γ change per unit normalized depth
    intercept: float
    r2: float
    n: int
    gamma_p10: float
    gamma_p50: float
    gamma_p90: float
    consistency_flag: str        # "linear" | "nonlinear" | "noisy"
    mean_minus_integral_sigma: float  # |mean_γ − fit_integral| / σ(γ)


def _consistency_flag(gammas: np.ndarray, slope: float, intercept: float, r2: float) -> tuple[str, float]:
    """Flag non-linearity: if |observed mean − linear-fit integral| / σ_γ > 2σ, trend is nonlinear."""
    observed_mean = float(np.mean(gammas))
    integral = intercept + slope / 2.0   # ∫₀¹ (slope·d + intercept) dd
    sigma = float(np.std(gammas)) or 1e-9
    z = abs(observed_mean - integral) / sigma
    if r2 < 0.3:
        tag = "noisy"
    elif z > 2.0:
        tag = "nonlinear"
    else:
        tag = "linear"
    return tag, float(z)


def fit_depth_law(
    fingerprints: Sequence[LayerFingerprint], projection: str
) -> DepthLawFit | None:
    """Fit γ vs normalized depth for one projection. Returns None if <3 valid γ."""
    rows = [f for f in fingerprints if f.projection == projection and f.gamma is not None]
    if len(rows) < 3:
        return None
    layers = np.array([f.layer for f in rows], dtype=np.float64)
    gammas = np.array([f.gamma for f in rows], dtype=np.float64)
    max_l = max(layers.max(), 1.0)
    depth = layers / max_l
    slope, intercept = np.polyfit(depth, gammas, 1)
    pred = slope * depth + intercept
    ss_res = float(np.sum((gammas - pred) ** 2))
    ss_tot = float(np.sum((gammas - gammas.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    flag, z = _consistency_flag(gammas, float(slope), float(intercept), r2)
    return DepthLawFit(
        projection=projection,
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n=len(rows),
        gamma_p10=float(np.percentile(gammas, 10)),
        gamma_p50=float(np.percentile(gammas, 50)),
        gamma_p90=float(np.percentile(gammas, 90)),
        consistency_flag=flag,
        mean_minus_integral_sigma=z,
    )
