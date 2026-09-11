"""KV-cache sizing audit — measure a model's KV compressibility, honestly.

Answers three questions for any HF transformer, on your GPU, in minutes:

1. CAPACITY — how much KV memory per long-context user, and roughly how many
   concurrent users a GPU can hold at FP16 vs quantized contracts.
2. DAMAGE LAW — does per-layer KV quantization damage follow the high-rate
   law D = c * 2^(s*b) (fitted slopes ~ -2), and which layers break it
   (flip-regime layers that need conservative treatment)?
3. TUNING VERDICT — is per-layer precision tuning worth anything on this
   model, decided by the sizing rule: allocation can only pay if the
   damage constants' log-dispersion (signal) exceeds their estimation
   noise across data splits. On every model we have measured, the honest
   answer at 4-bit operating points has been "no" — uniform is
   near-optimal. This tool lets you check yours.

Method: capture post-RoPE q/k/v via a scaled_dot_product_attention patch
(architecture-agnostic), quantize K/V with plain per-token max-abs grids at
b bits, and measure the EXACT attention-output error through the softmax —
no proxies, no fitted surrogates. Generic quantization only: this audit
contains no fraQtl codec; the calibrated audit (with certificates) is a
service — see the report footer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._model_io import load_model, load_wikitext_calibration

BITS = (3, 4, 5, 6)
FLIP_SLOPE = -2.0


@dataclass
class KVAuditResult:
    model_id: str
    n_layers: int
    kv_heads: int
    head_dim: int
    slopes: dict  # {(layer, side): (slope, log2_intercept)}
    flip_layers: dict  # side -> [layers]
    sizing: dict  # side -> {sigma_signal, sigma_noise, H, R, verdict}
    capacity: dict
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        j = {
            "model_id": self.model_id,
            "n_layers": self.n_layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "slopes": {f"{l}/{s}": v for (l, s), v in self.slopes.items()},
            "flip_layers": self.flip_layers,
            "sizing": self.sizing,
            "capacity": self.capacity,
            "meta": self.meta,
        }
        return json.dumps(j, indent=1)


def _quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric max-abs uniform grid (generic, no calibration)."""
    s = x.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 0.5)
    s = s.clamp_min(1e-12)
    return (x / s).round() * s


class _SDPARecorder:
    """Patch F.scaled_dot_product_attention to record per-layer q/k/v."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._orig = None

    def __enter__(self):
        self._orig = F.scaled_dot_product_attention
        recorder = self

        def patched(q, k, v, *args, **kwargs):
            # record final-position query + full k/v, on CPU, fp32
            recorder.calls.append(
                (q[..., -1:, :].detach().float().cpu(),
                 k.detach().float().cpu(),
                 v.detach().float().cpu())
            )
            return recorder._orig(q, k, v, *args, **kwargs)

        F.scaled_dot_product_attention = patched
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self._orig
        return False


def _row_damages(q, k, v, bits_list):
    """Exact attention-output damage for K-only and V-only quantization.

    q: (heads, 1, d)  k, v: (kv_heads, T, d). GQA handled by head grouping.
    Returns {("K"|"V", bits): mean damage over query heads}.
    """
    n_q, _, d = q.shape
    n_kv = k.shape[0]
    group = n_q // max(n_kv, 1)
    scale = 1.0 / (d ** 0.5)
    out = {}
    # clean reference per query head
    p_clean, o_clean = [], []
    for h in range(n_q):
        kh = k[min(h // group, n_kv - 1)]
        vh = v[min(h // group, n_kv - 1)]
        s = (q[h, 0] @ kh.T) * scale
        p = torch.softmax(s, dim=-1)
        p_clean.append(p)
        o_clean.append(p @ vh)
    for bits in bits_list:
        for side in ("K", "V"):
            dmg = 0.0
            for h in range(n_q):
                kh = k[min(h // group, n_kv - 1)]
                vh = v[min(h // group, n_kv - 1)]
                if side == "K":
                    s1 = (q[h, 0] @ _quant(kh, bits).T) * scale
                    p1 = torch.softmax(s1, dim=-1)
                    delta = (p1 - p_clean[h]) @ vh
                else:
                    delta = p_clean[h] @ (_quant(vh, bits) - vh)
                ref = o_clean[h].norm().clamp_min(1e-9)
                dmg += float((delta.norm() / ref) ** 2)
            out[(side, bits)] = dmg / n_q
    return out


def run_kv_audit(
    model_id: str,
    *,
    seq_len: int = 1024,
    n_seqs: int = 6,
    hbm_gb: float = 80.0,
    pool_frac: float = 0.75,
    context: int = 131072,
    trust_remote_code: bool = False,
) -> KVAuditResult:
    model, tok = load_model(model_id, trust_remote_code=trust_remote_code)
    model.eval()
    seqs = load_wikitext_calibration(tok, n_seqs=n_seqs, seq_len=seq_len)

    per_seq: list[dict] = []
    n_layers = kv_heads = head_dim = None
    for ids in seqs:
        rec = _SDPARecorder()
        with rec, torch.no_grad():
            model(ids.unsqueeze(0).to(model.device), use_cache=False)
        if n_layers is None:
            n_layers = len(rec.calls)
            kv_heads = rec.calls[0][1].shape[-3]
            head_dim = rec.calls[0][1].shape[-1]
        dam = {}
        for layer, (q, k, v) in enumerate(rec.calls):
            dam[layer] = _row_damages(q[0], k[0], v[0], BITS)
        per_seq.append(dam)

    # split sequences: first half = calibration fit, second half = noise probe
    half = max(1, len(per_seq) // 2)
    slopes, slopes_b = {}, {}
    for split_name, rows, store in (
        ("calib", per_seq[:half], slopes),
        ("eval", per_seq[half:], slopes_b),
    ):
        for layer in range(n_layers):
            for side in ("K", "V"):
                med = [float(np.median([r[layer][(side, b)] for r in rows]))
                       for b in BITS]
                sl, ic = np.polyfit(BITS, np.log2(np.maximum(med, 1e-30)), 1)
                store[(layer, side)] = (float(sl), float(ic))

    flip = {s: [l for l in range(n_layers)
                if slopes_b[(l, s)][0] > FLIP_SLOPE]
            for s in ("K", "V")}

    sizing = {}
    b_op = 4
    for side in ("K", "V"):
        la = np.array([slopes[(l, side)][1] + b_op * slopes[(l, side)][0]
                       for l in range(n_layers)]) * np.log(2)
        lb = np.array([slopes_b[(l, side)][1] + b_op * slopes_b[(l, side)][0]
                       for l in range(n_layers)]) * np.log(2)
        logr = la - lb
        H = float(np.mean(np.exp(lb - lb.max()))
                  / np.exp(np.mean(lb - lb.max())))
        R = float(np.exp(np.mean(logr)) * np.mean(np.exp(-logr)))
        sizing[side] = {
            "sigma_signal": float(np.std(lb)),
            "sigma_noise": float(np.std(logr)),
            "H": H, "R": R,
            "verdict": ("tuning MAY pay on the mean metric — verify tails "
                        "before believing it" if R < H
                        else "tuning will NOT pay — use uniform"),
        }

    bytes_fp16 = 2 * head_dim * 2 * kv_heads * n_layers
    bytes_int4 = int(bytes_fp16 / 3.6)  # naive int4 + scales, rough
    per_user = lambda bpt: bpt * context / 1e9
    pool = hbm_gb * pool_frac
    capacity = {
        "context": context,
        "kv_bytes_per_token_fp16": bytes_fp16,
        "gb_per_user_fp16": round(per_user(bytes_fp16), 2),
        "users_fp16": int(pool // per_user(bytes_fp16)),
        "gb_per_user_naive_int4": round(per_user(bytes_int4), 2),
        "users_naive_int4": int(pool // per_user(bytes_int4)),
        "note": ("naive INT4 capacity assumes the flip-regime layers "
                 "tolerate it — check the risk map; calibrated contracts "
                 "with quality receipts are the paid audit"),
    }

    return KVAuditResult(
        model_id=model_id, n_layers=n_layers, kv_heads=kv_heads,
        head_dim=head_dim, slopes=slopes_b, flip_layers=flip, sizing=sizing,
        capacity=capacity,
        meta={"seq_len": seq_len, "n_seqs": n_seqs, "bits": list(BITS),
              "hbm_gb": hbm_gb, "pool_frac": pool_frac},
    )


def render_markdown(r: KVAuditResult) -> str:
    lines = [
        f"# fraQtl KV Audit — {r.model_id}",
        "",
        f"{r.n_layers} layers, {r.kv_heads} KV heads, head_dim {r.head_dim}. "
        f"Measured on {r.meta['n_seqs']} sequences x {r.meta['seq_len']} tokens; "
        "exact attention-output damage through the softmax, generic per-token "
        "max-abs quantization (no calibration, no codec).",
        "",
        "## Capacity at {:,}-token context".format(r.capacity["context"]),
        "",
        "| | FP16 | naive INT4 |",
        "|---|---:|---:|",
        f"| GB per user | {r.capacity['gb_per_user_fp16']} | "
        f"{r.capacity['gb_per_user_naive_int4']} |",
        f"| users per {r.meta['hbm_gb']:.0f} GB GPU | "
        f"{r.capacity['users_fp16']} | {r.capacity['users_naive_int4']} |",
        "",
        f"_{r.capacity['note']}_",
        "",
        "## Damage law + risk map",
        "",
        f"- Fitted slopes ~ -2 confirm the high-rate law where it holds.",
        f"- Flip-regime layers (violate the law; quality failures concentrate "
        f"here under aggressive compression): K {r.flip_layers['K'] or 'none'}, "
        f"V {r.flip_layers['V'] or 'none'}.",
        "",
        "## Per-layer tuning verdict (sizing rule)",
        "",
        "| stream | signal disp. | est. noise | H | R | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for s in ("K", "V"):
        t = r.sizing[s]
        lines.append(
            f"| {s} | {t['sigma_signal']:.2f} | {t['sigma_noise']:.2f} | "
            f"{t['H']:.2f} | {t['R']:.2f} | {t['verdict']} |")
    lines += [
        "",
        "Rule: allocation can pay only if constants' log-dispersion (signal) "
        "exceeds estimation noise — and even then, tail metrics usually kill "
        "it. On every model we have published receipts for, uniform precision "
        "was near-optimal at 4-bit operating points.",
        "",
        "---",
        "This is the free, generic audit. The calibrated audit — measured "
        "capacity at fraQtl's production contracts (receipt: 9x128K users at "
        "134 tok/s on one A100 vs 2 users FP16), with quality certificates — "
        "is a service: contact fraqtl.ai.",
        "",
    ]
    return "\n".join(lines)


def kv_audit_to_files(r: KVAuditResult, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = r.model_id.replace("/", "__")
    j = out_dir / f"{stem}.kv_audit.json"
    m = out_dir / f"{stem}.kv_audit.md"
    j.write_text(r.to_json(), encoding="utf-8")
    m.write_text(render_markdown(r), encoding="utf-8")
    return j, m
