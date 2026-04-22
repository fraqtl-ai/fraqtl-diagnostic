"""Unit tests for the compare / verdict pipeline. No HF downloads."""
from dataclasses import dataclass

import pytest

from fraqtl_diagnostic.compare import compare_to_reference, _classify, ProjectionDelta
from fraqtl_diagnostic.references import list_reference_models, load_reference


@dataclass
class _FakeFingerprint:
    layer: int
    projection: str
    dim: int
    gamma: float
    k95: int


@dataclass
class _FakeReport:
    model_id: str
    fingerprints: list


def _build_fake_report(model_id, down_gamma, down_k95_ratio, o_gamma, o_k95_ratio, n_layers=10):
    fps = []
    for i in range(n_layers):
        fps.append(_FakeFingerprint(layer=i, projection="down_proj",
                                    dim=14336, gamma=down_gamma,
                                    k95=int(14336 * down_k95_ratio)))
        fps.append(_FakeFingerprint(layer=i, projection="o_proj",
                                    dim=4096, gamma=o_gamma,
                                    k95=int(4096 * o_k95_ratio)))
    return _FakeReport(model_id=model_id, fingerprints=fps)


def test_list_reference_models_nonempty():
    refs = list_reference_models()
    assert len(refs) >= 5
    assert "mistralai/Mistral-7B-v0.1" in refs
    assert "Qwen/Qwen2.5-7B" in refs


def test_load_reference_returns_dict_for_known_model():
    ref = load_reference("mistralai/Mistral-7B-v0.1")
    assert ref is not None
    assert "down_proj" in ref or "o_proj" in ref


def test_compare_to_unknown_reference_returns_unavailable():
    r = _build_fake_report("my/custom", 0.6, 0.25, 0.4, 0.3)
    result = compare_to_reference(r, "not-a-real/model")
    assert result.reference_available is False


def test_compare_matched_model_gives_preserved_verdict():
    mistral_ref = load_reference("mistralai/Mistral-7B-v0.1")
    # Build "fake finetune" that matches mistral reference exactly
    r = _build_fake_report(
        "my/fake-mistral-finetune",
        down_gamma=mistral_ref["down_proj"]["gamma_median"],
        down_k95_ratio=mistral_ref["down_proj"]["k95_ratio_median"],
        o_gamma=mistral_ref["o_proj"]["gamma_median"],
        o_k95_ratio=mistral_ref["o_proj"]["k95_ratio_median"],
    )
    result = compare_to_reference(r, "mistralai/Mistral-7B-v0.1")
    assert result.reference_available
    assert len(result.deltas) >= 1
    for d in result.deltas:
        assert abs(d.gamma_delta) < 1e-4
        assert abs(d.k95_ratio_delta) < 1e-3   # integer k95 round-trip introduces ~1/dim precision loss
    assert "preserved" in result.verdict.lower(), result.verdict


def test_classify_broken_when_gamma_exceeds_threshold():
    bad_delta = ProjectionDelta(
        projection="down_proj",
        gamma_this=1.30, gamma_ref=0.6, gamma_delta=0.7,
        k95_ratio_this=0.62, k95_ratio_ref=0.25, k95_ratio_delta=0.37,
        depth_law_slope_this=-0.1, depth_law_slope_ref=-0.3,
    )
    verdict, rationale = _classify([bad_delta])
    assert "broken" in verdict
    assert len(rationale) >= 1


def test_classify_degraded_when_gamma_climbs_moderately():
    mid_delta = ProjectionDelta(
        projection="down_proj",
        gamma_this=0.95, gamma_ref=0.60, gamma_delta=0.35,
        k95_ratio_this=0.50, k95_ratio_ref=0.25, k95_ratio_delta=0.25,
        depth_law_slope_this=-0.1, depth_law_slope_ref=-0.3,
    )
    verdict, _ = _classify([mid_delta])
    assert "degraded" in verdict.lower()
