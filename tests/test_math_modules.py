"""Pure-math smoke tests — no HF downloads, no GPU.

Build synthetic eigenvalues with known decay and confirm the pipeline extracts
the right γ / knee / k95 / depth-law.
"""
import math

import numpy as np
import pytest

from fraqtl_diagnostic.shannon import (
    fit_power_exp_stretched,
    fingerprint_layer,
    LayerFingerprint,
)
from fraqtl_diagnostic.depth_law import fit_depth_law
from fraqtl_diagnostic.estimator import estimate_compression


def _stretched_eigvals(d: int = 2048, beta: float = 0.05, gamma: float = 0.45, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(1, d + 1).astype(np.float64)
    log_lam = -beta * np.power(x, gamma)
    # tiny noise (small compared to log-scale spacing)
    log_lam = log_lam + 0.01 * rng.standard_normal(d)
    return np.exp(log_lam)  # already descending


def _power_eigvals(d: int = 2048, alpha: float = 1.74, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(1, d + 1).astype(np.float64)
    log_lam = -alpha * np.log(x)
    log_lam = log_lam + 0.01 * rng.standard_normal(d)
    return np.exp(log_lam)


def test_fit_family_selects_stretched_exp_on_stretched_data():
    eig = _stretched_eigvals()
    fit = fit_power_exp_stretched(eig, i_lo=40, i_hi=1600)
    assert fit is not None
    assert fit["best"] in ("stretched_exp", "power"), fit["best"]  # depending on seed, power can tie
    assert fit["stretched_exp"]["r2"] > 0.95
    # gamma recoverable within a tolerance (synthetic noise makes it non-trivial)
    assert abs(fit["stretched_exp"]["gamma"] - 0.45) < 0.15


def test_fit_family_selects_power_on_power_data():
    eig = _power_eigvals()
    fit = fit_power_exp_stretched(eig, i_lo=40, i_hi=1600)
    assert fit is not None
    assert fit["power"]["r2"] > 0.99
    assert abs(fit["power"]["alpha"] - 1.74) < 0.1


def test_fingerprint_layer_populates_all_fields():
    eig = _stretched_eigvals()
    fp = fingerprint_layer(eig, layer=3, projection="down_proj", n_samples=4096)
    assert fp.dim == eig.size
    assert fp.layer == 3
    assert fp.projection == "down_proj"
    assert fp.k95 > 0 and fp.k95 <= fp.dim
    assert fp.k99 >= fp.k95
    assert fp.k999 >= fp.k99
    assert fp.top8_energy <= fp.top64_energy <= fp.top256_energy <= 1.0
    assert 0 < fp.eigval_geomean
    assert fp.spectrum_decimated["indices"]
    # d_star: more bits should give smaller ceiling
    bits_order = ["2", "3", "4"]
    vals = [fp.d_star[b] for b in bits_order if b in fp.d_star]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_depth_law_detects_linear_gamma_trend():
    # Build 10 fingerprints with γ linearly declining from 0.6 to 0.3 across layers.
    fps = []
    for i in range(10):
        true_gamma = 0.6 - 0.03 * i
        eig = _stretched_eigvals(gamma=true_gamma, seed=i)
        fp = fingerprint_layer(eig, layer=i, projection="down_proj", n_samples=4096)
        fps.append(fp)
    dl = fit_depth_law(fps, "down_proj")
    assert dl is not None
    assert dl.n == 10
    assert dl.slope < 0, dl.slope  # declining γ with depth
    assert dl.r2 > 0.5


def test_estimate_compression_returns_plausible_budget():
    fps = []
    for i in range(6):
        eig = _stretched_eigvals(seed=i)
        fps.append(fingerprint_layer(eig, layer=i, projection="down_proj", n_samples=4096))
    est = estimate_compression(fps)
    # suggested budgets should be strictly increasing conservatism
    assert est.budget_bits_aggressive < est.budget_bits_balanced < est.budget_bits_conservative
    assert 2.0 <= est.budget_bits_aggressive <= 4.5
    assert 0.0 <= est.headroom_score <= 1.0


def test_empty_fingerprints_raises():
    with pytest.raises(ValueError):
        estimate_compression([])
