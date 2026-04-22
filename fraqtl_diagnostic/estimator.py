"""Compression potential estimate — headline number for the report.

Maps the Shannon ceiling (D*(R) at several bit budgets) into a qualitative
"compression budget" label:

  - `headroom`   : ratio of D*(2 b/w) to D*(4 b/w). High headroom = spectrum
                   concentrates mass on few dims → aggressive compression ok.
  - `budget_bits`: recommended b/w for a given target loss level. This is a
                   pure-math estimate from the Shannon ceiling — does NOT
                   include recipe-specific gains (sign correction, V theorem,
                   per-model calibration). Those are fraQtl's closed engine.

The estimator intentionally does NOT return a "predicted PPL" — predicting PPL
requires calibration to a specific recipe. We return the information-theoretic
ceiling + recommended budget and leave the actual PPL to a downstream
compression run.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .shannon import LayerFingerprint


@dataclass
class CompressionEstimate:
    budget_bits_aggressive: float     # recommended b/w for "aggressive" tier
    budget_bits_balanced: float       # recommended b/w for "balanced" tier
    budget_bits_conservative: float   # recommended b/w for "conservative" tier
    mean_k95_ratio: float             # mean k95/dim — lower = more compressible
    mean_gamma: float                 # mean stretched-exp γ across layers (shape)
    headroom_score: float             # 0–1, higher = more compressible
    headline: str                     # one-line human-readable summary


def estimate_compression(fingerprints: Sequence[LayerFingerprint]) -> CompressionEstimate:
    if not fingerprints:
        raise ValueError("no fingerprints to estimate from")

    k95_ratios = [f.k95 / f.dim for f in fingerprints]
    mean_k95 = float(np.mean(k95_ratios))

    gammas = [f.gamma for f in fingerprints if f.gamma is not None]
    mean_gamma = float(np.mean(gammas)) if gammas else float("nan")

    # headroom_score: low k95/dim + low γ → more compressible
    # k95/dim in [0, 1], γ typically in [0.1, 1.2]. Score = 1 − k95_ratio, penalty
    # if γ > 0.7 (less stretched = closer to exponential = less compressible head).
    gamma_penalty = max(0.0, (mean_gamma - 0.7) / 0.5) if not np.isnan(mean_gamma) else 0.0
    headroom = max(0.0, min(1.0, (1.0 - mean_k95) * (1.0 - 0.5 * gamma_penalty)))

    # Bit budgets scaled by headroom. At full headroom: 2.5 / 3.0 / 3.5.
    # At zero headroom: 3.5 / 4.0 / 4.5.
    def _budget(base_low, base_high):
        return base_high - headroom * (base_high - base_low)

    b_aggr = _budget(2.5, 3.5)
    b_bal = _budget(3.0, 4.0)
    b_cons = _budget(3.5, 4.5)

    # Headline — plain English, no marketing claims
    if headroom >= 0.7:
        tier = "aggressive compression tolerated"
    elif headroom >= 0.4:
        tier = "moderate compression tolerated"
    else:
        tier = "limited compression headroom"
    headline = (
        f"{tier}: suggested {b_bal:.1f} b/w balanced, {b_aggr:.1f} b/w aggressive. "
        f"mean k95/dim = {mean_k95:.2%}, mean γ = {mean_gamma:.3f}"
    )

    return CompressionEstimate(
        budget_bits_aggressive=float(b_aggr),
        budget_bits_balanced=float(b_bal),
        budget_bits_conservative=float(b_cons),
        mean_k95_ratio=mean_k95,
        mean_gamma=mean_gamma,
        headroom_score=float(headroom),
        headline=headline,
    )
