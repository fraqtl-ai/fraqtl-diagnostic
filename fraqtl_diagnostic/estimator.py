"""Descriptive summary of a DiagnosticReport — measurement-only.

v0.1 scope: aggregate the per-layer fingerprints into a layer-level summary.
No bit-budget predictions, no "headroom" tiers, no "suggested b/w." Those
require atlas-validated regression against actual compression outcomes and
ship in v0.2 alongside Paper 3.

What we DO report:
  - mean γ across stretched_exp regime layers only (suppressed if >10% out)
  - mean k95/dim (effective rank ratio)
  - regime distribution (counts of stretched/near-exp/compressed/etc.)
  - fit-quality distribution
  - Shannon D*(R) table geomean across layers, at R ∈ {2, 3, 4}

What we DO NOT report:
  - Predicted PPL / bit-budget recommendations
  - "Aggressive / balanced / conservative" tiers
  - Headroom score
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .shannon import LayerFingerprint


@dataclass
class DiagnosticSummary:
    mean_gamma: float                # mean KWW γ across stretched_exp layers only
    mean_k95_ratio: float            # mean k95/dim across all layers
    regime_counts: dict              # {regime_name: n_layers}
    fit_quality_counts: dict         # {"good": N, "moderate": M, "poor": K}
    d_star_by_bits: dict             # {"2": geomean D*(2 b/w), "3": ..., "4": ...}
    gamma_summary_suppressed: bool   # True if >10% of layers outside stretched_exp
    n_fingerprints: int
    cta: str = (
        "This is a measurement tool. For actual compression outcomes, "
        "visit https://fraqtl.ai"
    )


# Back-compat alias so any external code that imported the old name keeps working.
CompressionEstimate = DiagnosticSummary


def summarize(fingerprints: Sequence[LayerFingerprint]) -> DiagnosticSummary:
    if not fingerprints:
        raise ValueError("no fingerprints to summarize")

    # k95 mean across ALL layers (measurement metric, always valid)
    k95_ratios = [f.k95 / f.dim for f in fingerprints if f.dim > 0]
    mean_k95 = float(np.mean(k95_ratios)) if k95_ratios else float("nan")

    # mean γ restricted to stretched_exp regime with non-poor fit quality
    inside_kww = [
        f for f in fingerprints
        if f.gamma is not None
        and f.fit_quality != "poor"
        and f.regime == "stretched_exp"
    ]
    gammas_inside = [f.gamma for f in inside_kww]
    mean_gamma = float(np.mean(gammas_inside)) if gammas_inside else float("nan")

    # Suppression rule: >10% layers out of stretched_exp / poor fit → suppress scalar γ
    fitted = [f for f in fingerprints if f.gamma is not None]
    n_fitted = len(fitted)
    n_outside = sum(
        1 for f in fitted
        if f.fit_quality == "poor" or f.regime != "stretched_exp"
    )
    suppress_gamma = n_fitted > 0 and (n_outside / n_fitted) > 0.10

    # Regime + fit-quality distributions
    regime_counts: dict[str, int] = {}
    fit_quality_counts: dict[str, int] = {}
    for f in fingerprints:
        key_r = f.regime or "unfit"
        regime_counts[key_r] = regime_counts.get(key_r, 0) + 1
        key_q = f.fit_quality or "unfit"
        fit_quality_counts[key_q] = fit_quality_counts.get(key_q, 0) + 1

    # D*(R) geomean across layers, at R ∈ {2, 3, 4}
    d_star_by_bits: dict[str, float] = {}
    for bit in ("2", "3", "4"):
        vals = [f.d_star.get(bit) for f in fingerprints if bit in f.d_star]
        vals = [v for v in vals if v and v > 0]
        if vals:
            d_star_by_bits[bit] = float(np.exp(np.mean(np.log(vals))))

    return DiagnosticSummary(
        mean_gamma=mean_gamma,
        mean_k95_ratio=mean_k95,
        regime_counts=regime_counts,
        fit_quality_counts=fit_quality_counts,
        d_star_by_bits=d_star_by_bits,
        gamma_summary_suppressed=bool(suppress_gamma),
        n_fingerprints=len(fingerprints),
    )


# Back-compat alias for older callers that used `estimate_compression`.
estimate_compression = summarize
