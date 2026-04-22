"""Shannon ceiling, α / γ / knee fitting, k-for-energy.

Given a descending eigenvalue array λ_i of the input covariance H, fit:
  - α_tail : log-log slope over the tail
  - stretched-exp: λ_i ≈ exp(-β · i^γ + c); γ characterizes decay shape
  - knee   : first index where rolling log-log slope < −5
  - k_for_energy at 95% / 99% / 99.9%
  - Shannon D*(R) = geomean(λ) · 2^(−2R) for bit budgets R
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class LayerFingerprint:
    """Shannon diagnostic for one (layer, projection)."""
    layer: int
    projection: str
    dim: int
    trusted_rank: int
    alpha_head: float
    alpha_mid: float
    alpha_tail: float
    gamma: float | None          # stretched-exp γ (mid region)
    beta: float | None           # stretched-exp β
    best_family: str | None      # "power" | "exp" | "stretched_exp"
    r2_by_family: dict           # {family: R²}
    knee: int | None             # first-hard-slope index, None if no sharp knee
    k95: int                     # directions for 95% energy
    k99: int
    k999: int
    top8_energy: float
    top64_energy: float
    top256_energy: float
    eigval_geomean: float        # geomean over trusted eigvals
    d_star: dict                 # {"2": D*(2 b/w), "3": ..., "4": ...}
    spectrum_decimated: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "layer": self.layer,
            "projection": self.projection,
            "dim": self.dim,
            "gamma": self.gamma,
            "alpha_tail": self.alpha_tail,
            "knee": self.knee,
            "k95": self.k95,
            "k99": self.k99,
            "best_family": self.best_family,
        }


def _safe_r2_aic(pred, log_lam, k):
    resid = log_lam - pred
    rss = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_lam - log_lam.mean()) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    n = log_lam.size
    aic = n * math.log(max(rss / n, 1e-300)) + 2 * k
    return float(r2), float(aic)


def fit_power_exp_stretched(eig: np.ndarray, i_lo: int, i_hi: int) -> dict | None:
    """Fit three candidate decays to log λ over [i_lo, i_hi). Return dict keyed
    by family, each with params + r2 + aic, plus "best" (lowest AIC)."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    i_lo = max(int(i_lo), 1)
    i_hi = min(int(i_hi), int(eig.size))
    if i_hi - i_lo < 10:
        return None
    idx = np.arange(i_lo, i_hi)
    x = (idx + 1).astype(np.float64)
    lam = np.maximum(eig[idx].astype(np.float64), 1e-18)
    log_lam = np.log(lam)

    out: dict = {}

    # power: log λ = -α log x + b
    try:
        slope, b = np.polyfit(np.log(x), log_lam, 1)
        r2, aic = _safe_r2_aic(slope * np.log(x) + b, log_lam, 2)
        out["power"] = {"alpha": float(-slope), "intercept": float(b), "r2": r2, "aic": aic}
    except Exception:
        pass

    # pure exp: log λ = -β x + b
    try:
        slope, b = np.polyfit(x, log_lam, 1)
        r2, aic = _safe_r2_aic(slope * x + b, log_lam, 2)
        out["exp"] = {"beta": float(-slope), "intercept": float(b), "r2": r2, "aic": aic}
    except Exception:
        pass

    # stretched exp: log λ = -β x^γ + c
    try:
        def _f(x_, beta, gamma, c):
            return -beta * np.power(x_, gamma) + c
        p0 = [1e-4, 0.5, float(log_lam.max())]
        popt, _ = curve_fit(_f, x, log_lam, p0=p0, maxfev=5000)
        r2, aic = _safe_r2_aic(_f(x, *popt), log_lam, 3)
        out["stretched_exp"] = {
            "beta": float(popt[0]), "gamma": float(popt[1]),
            "intercept": float(popt[2]), "r2": r2, "aic": aic,
        }
    except Exception:
        pass

    if out:
        out["best"] = min(
            (k for k in out if k != "best"),
            key=lambda k: out[k].get("aic", float("inf")),
        )
    return out or None


def _decimate_spectrum(eig: np.ndarray, n_points: int = 512) -> dict:
    d = eig.size
    if d <= n_points:
        idx = np.arange(d)
    else:
        idx = np.unique(
            np.round(np.logspace(0, math.log10(d - 1), n_points)).astype(int)
        )
        idx = np.clip(idx, 0, d - 1)
    return {"indices": idx.tolist(), "eigvals": eig[idx].astype(np.float64).tolist()}


def fingerprint_layer(
    eig: np.ndarray,
    layer: int,
    projection: str,
    *,
    n_samples: int | None = None,
    bit_budgets: tuple[float, ...] = (2.0, 3.0, 4.0, 4.5),
) -> LayerFingerprint:
    """Full Shannon fingerprint of one layer from its descending eigenvalues."""
    d = eig.size

    # noise floor: min(rank, abs 1e-8 × λ_max)
    lam_max = float(eig[0]) if eig[0] > 0 else 1.0
    rel_floor = 1e-8 * lam_max
    i_floor = int(np.searchsorted(-eig, -rel_floor, side="right"))
    if n_samples is not None and n_samples > 0:
        i_floor = min(i_floor, int(n_samples))
    i_floor = max(min(i_floor, d), 4)

    eig_c = np.maximum(eig, 1e-18)

    def _slope(i_lo, i_hi):
        i_lo = max(int(i_lo), 1)
        i_hi = max(int(i_hi), i_lo + 4)
        i_hi = min(i_hi, d)
        if i_hi - i_lo < 4:
            return float("nan")
        idx = np.arange(i_lo, i_hi)
        slope, _ = np.polyfit(np.log(idx + 1), np.log(eig_c[idx]), 1)
        return float(-slope)

    alpha_head = _slope(1, max(int(d * 0.05), 8))
    alpha_mid = _slope(int(d * 0.02), int(d * 0.80))
    tail_lo, tail_hi = int(d * 0.50), min(int(d * 0.95), i_floor)
    alpha_tail = _slope(tail_lo, tail_hi)

    # Shannon ceiling from geomean of trusted eigvals
    eig_trusted = np.maximum(eig[:i_floor], 1e-18)
    geo_mean = math.exp(float(np.mean(np.log(eig_trusted))))
    d_star = {str(R).rstrip("0").rstrip(".") or "0": geo_mean * (2.0 ** (-2.0 * R))
              for R in bit_budgets}

    # energies + k-for-energy
    cum = np.cumsum(eig_c) / eig_c.sum()
    top8 = float(cum[7]) if d >= 8 else float(cum[-1])
    top64 = float(cum[63]) if d >= 64 else float(cum[-1])
    top256 = float(cum[255]) if d >= 256 else float(cum[-1])

    def _k_for(target):
        hits = np.where(cum >= target)[0]
        return int(hits[0] + 1) if hits.size else int(d)

    k95 = _k_for(0.95)
    k99 = _k_for(0.99)
    k999 = _k_for(0.999)

    # knee detection: first index where smoothed log-log slope < -5
    knee = None
    try:
        log_lam = np.log(eig_c[:i_floor])
        log_i = np.log(np.arange(1, i_floor + 1).astype(np.float64))
        win = max(int(i_floor / 50), 8)
        slope_local = np.diff(log_lam) / np.maximum(np.diff(log_i), 1e-12)
        kernel = np.ones(win) / win
        slope_smooth = np.convolve(slope_local, kernel, mode="valid")
        hits = np.where(slope_smooth < -5.0)[0]
        if hits.size:
            knee = int(hits[0] + win // 2)
    except Exception:
        knee = None

    # stretched-exp / family fit on mid region
    ms = fit_power_exp_stretched(eig, int(d * 0.02), min(int(d * 0.80), i_floor))
    gamma = beta = None
    best_family = None
    r2_by_family: dict = {}
    if ms:
        best_family = ms.get("best")
        for fam in ("power", "exp", "stretched_exp"):
            if fam in ms and isinstance(ms[fam], dict):
                r2_by_family[fam] = ms[fam].get("r2", float("nan"))
        se = ms.get("stretched_exp")
        if isinstance(se, dict):
            gamma = se.get("gamma")
            beta = se.get("beta")

    return LayerFingerprint(
        layer=layer,
        projection=projection,
        dim=int(d),
        trusted_rank=int(i_floor),
        alpha_head=alpha_head,
        alpha_mid=alpha_mid,
        alpha_tail=alpha_tail,
        gamma=float(gamma) if gamma is not None else None,
        beta=float(beta) if beta is not None else None,
        best_family=best_family,
        r2_by_family=r2_by_family,
        knee=knee,
        k95=k95,
        k99=k99,
        k999=k999,
        top8_energy=top8,
        top64_energy=top64,
        top256_energy=top256,
        eigval_geomean=float(geo_mean),
        d_star=d_star,
        spectrum_decimated=_decimate_spectrum(eig, n_points=512),
    )
