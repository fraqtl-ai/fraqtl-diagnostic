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
    return DepthLawFit(
        projection=projection,
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n=len(rows),
        gamma_p10=float(np.percentile(gammas, 10)),
        gamma_p50=float(np.percentile(gammas, 50)),
        gamma_p90=float(np.percentile(gammas, 90)),
    )
