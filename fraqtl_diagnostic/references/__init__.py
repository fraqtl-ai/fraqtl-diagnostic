"""Bundled reference measurements for stock HF models.

Usage:
    from fraqtl_diagnostic.references import load_reference
    ref = load_reference("mistralai/Mistral-7B-v0.1")
    # ref["down_proj"]["gamma_median"] → 0.585
"""
from __future__ import annotations
import json
from pathlib import Path

_DIR = Path(__file__).parent


def _load_all() -> dict:
    path = _DIR / "stock_models_v1.json"
    with open(path) as f:
        return json.load(f)


def list_reference_models() -> list[str]:
    return sorted(_load_all()["reference_models"].keys())


def load_reference(model_id: str) -> dict | None:
    """Return per-projection reference stats for `model_id`, or None if not bundled."""
    data = _load_all()
    return data["reference_models"].get(model_id)


def calibration_description() -> str:
    return _load_all().get("calibration", "")
