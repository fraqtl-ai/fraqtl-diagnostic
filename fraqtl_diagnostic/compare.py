"""Compare a DiagnosticReport against a reference model.

Typical call:
    from fraqtl_diagnostic.compare import compare_to_reference
    delta = compare_to_reference(report, "mistralai/Mistral-7B-v0.1")
    print(delta.verdict)

Delta fields are per-projection (down_proj, o_proj). Verdict is a single
human-readable string summarizing whether the measured model has preserved
compressibility vs its reference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .shannon import LayerFingerprint
from .references import load_reference, list_reference_models


@dataclass
class ProjectionDelta:
    projection: str
    gamma_this: float
    gamma_ref: float
    gamma_delta: float          # this − ref
    k95_ratio_this: float
    k95_ratio_ref: float
    k95_ratio_delta: float
    depth_law_slope_this: float
    depth_law_slope_ref: float


@dataclass
class ComparisonResult:
    reference_model: str
    reference_available: bool
    this_model: str
    deltas: list[ProjectionDelta] = field(default_factory=list)
    verdict: str = ""
    rationale: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.reference_available:
            return (
                f"no bundled reference for {self.reference_model}. "
                f"Available refs: {', '.join(list_reference_models())}"
            )
        lines = [
            f"Comparing {self.this_model}  vs  {self.reference_model}",
            "",
        ]
        for d in self.deltas:
            lines.append(
                f"  {d.projection:10s}  "
                f"γ: {d.gamma_this:.3f} (this) − {d.gamma_ref:.3f} (ref) = "
                f"{d.gamma_delta:+.3f}    "
                f"k95/dim Δ: {d.k95_ratio_delta:+.2%}"
            )
        lines.append("")
        lines.append(f"  VERDICT: {self.verdict}")
        for r in self.rationale:
            lines.append(f"    · {r}")
        return "\n".join(lines)


def _fingerprint_summary(
    fingerprints: Sequence[LayerFingerprint], projection: str
) -> dict | None:
    rows = [f for f in fingerprints if f.projection == projection and f.gamma is not None]
    if not rows:
        return None
    gammas = np.array([f.gamma for f in rows])
    k95 = np.array([f.k95 / f.dim for f in rows])
    layers = np.array([f.layer for f in rows], dtype=np.float64)
    max_l = max(layers.max(), 1.0)
    depth = layers / max_l
    slope, intercept = np.polyfit(depth, gammas, 1)
    return {
        "gamma_median": float(np.median(gammas)),
        "k95_ratio_median": float(np.median(k95)),
        "depth_law_slope": float(slope),
    }


def _classify(deltas: list[ProjectionDelta]) -> tuple[str, list[str]]:
    """Map per-projection deltas to a single headline verdict + rationale bullets.

    Thresholds are intentionally generous because bundled references were
    measured at different calibration sizes than the default (~0.05–0.1 drift
    is typical even for the *same* model). Conservative = false positives =
    useless verdict.

      preserved : every projection |Δγ| ≤ 0.10 AND |Δk95/dim| ≤ 10pp
      degraded  : any projection Δγ > +0.20 OR Δk95/dim > +20pp (shift toward
                  less compressible)
      broken    : any projection γ > 1.20 OR k95/dim > 0.60 (spectrum collapsed)
    """
    if not deltas:
        return ("no overlap with reference projections", [])

    rationale: list[str] = []
    broken = False
    degraded = False
    preserved_all = True

    for d in deltas:
        if d.gamma_this > 1.20 or d.k95_ratio_this > 0.60:
            broken = True
            rationale.append(
                f"{d.projection}: γ={d.gamma_this:.2f}, k95/dim={d.k95_ratio_this:.2%} — "
                f"spectrum collapsed (γ>1.20 or k95/dim>60%)"
            )
        if d.gamma_delta > 0.20 or d.k95_ratio_delta > 0.20:
            degraded = True
            rationale.append(
                f"{d.projection}: Δγ = {d.gamma_delta:+.3f}, Δk95/dim = "
                f"{d.k95_ratio_delta:+.2%} — materially less compressible than reference"
            )
        if not (abs(d.gamma_delta) <= 0.10 and abs(d.k95_ratio_delta) <= 0.10):
            preserved_all = False

    if broken:
        return ("likely broken — same-recipe compression may crash quality", rationale)
    if degraded:
        return ("degraded — shift is real, tune the recipe before compressing", rationale)
    if preserved_all:
        return ("preserved — safe to apply same recipe as reference", rationale)
    return ("shifted but within safe range — minor recipe tuning expected", rationale)


def compare_to_reference(report, reference_model: str) -> ComparisonResult:
    """Compare a DiagnosticReport to a bundled reference model."""
    ref = load_reference(reference_model)
    if ref is None:
        return ComparisonResult(
            reference_model=reference_model,
            reference_available=False,
            this_model=report.model_id,
        )

    deltas: list[ProjectionDelta] = []
    for proj, ref_proj in ref.items():
        this = _fingerprint_summary(report.fingerprints, proj)
        if this is None or ref_proj.get("gamma_median") is None:
            continue
        deltas.append(ProjectionDelta(
            projection=proj,
            gamma_this=this["gamma_median"],
            gamma_ref=ref_proj["gamma_median"],
            gamma_delta=this["gamma_median"] - ref_proj["gamma_median"],
            k95_ratio_this=this["k95_ratio_median"],
            k95_ratio_ref=ref_proj["k95_ratio_median"],
            k95_ratio_delta=this["k95_ratio_median"] - ref_proj["k95_ratio_median"],
            depth_law_slope_this=this["depth_law_slope"],
            depth_law_slope_ref=ref_proj["depth_law_slope"],
        ))

    verdict, rationale = _classify(deltas)
    return ComparisonResult(
        reference_model=reference_model,
        reference_available=True,
        this_model=report.model_id,
        deltas=deltas,
        verdict=verdict,
        rationale=rationale,
    )
