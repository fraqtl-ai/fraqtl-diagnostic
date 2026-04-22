"""Top-level analyze() orchestrator — model → DiagnosticReport."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence
import gc
import json
import time

import numpy as np
import torch

from .version import __version__
from ._model_io import load_model, load_wikitext_calibration, find_target_modules
from .spectrum import capture_single_hessian, eigendecompose
from .shannon import LayerFingerprint, fingerprint_layer, aggregate_per_head
from .depth_law import DepthLawFit, fit_depth_law
from .estimator import CompressionEstimate, estimate_compression


@dataclass
class DiagnosticReport:
    model_id: str
    version: str
    n_layers: int
    projections: list[str]
    fingerprints: list[LayerFingerprint]
    depth_laws: list[DepthLawFit]
    estimate: CompressionEstimate
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        # regime distribution across layers
        regimes: dict[str, int] = {}
        for f in self.fingerprints:
            if f.regime:
                regimes[f.regime] = regimes.get(f.regime, 0) + 1
        regime_str = ", ".join(f"{v} {k}" for k, v in sorted(regimes.items())) if regimes else "—"

        gamma_line = (
            f"  mean γ            : suppressed  [>10% of layers fall outside Kohlrausch "
            f"regime (0,1); see per-layer regime tags]"
            if self.estimate.gamma_summary_suppressed
            else f"  mean γ            : {self.estimate.mean_gamma:.3f}   (across both projections)"
        )
        lines = [
            f"fraQtl Diagnostic v{self.version}",
            f"  model             : {self.model_id}",
            f"  layers            : {self.n_layers}",
            f"  projections       : {', '.join(self.projections)}",
            gamma_line,
            f"  regimes           : {regime_str}",
            f"  mean k95/dim      : {self.estimate.mean_k95_ratio:.2%}",
            f"  headroom score    : {self.estimate.headroom_score:.2f}",
            f"  suggested b/w     : {self.estimate.budget_bits_balanced:.1f} (balanced)"
            f" / {self.estimate.budget_bits_aggressive:.1f} (aggressive)",
        ]
        for dl in self.depth_laws:
            if dl.r2 < 0.3:
                # no trend detectable — honest about it rather than print a meaningless fit
                lines.append(
                    f"  depth-law {dl.projection:10s}  "
                    f"uninformative (R² = {dl.r2:.2f}) — γ is nearly flat across depth, "
                    f"median γ = {dl.gamma_p50:.3f}"
                )
            else:
                lines.append(
                    f"  depth-law {dl.projection:10s}  "
                    f"γ = {dl.slope:+.3f}·depth + {dl.intercept:.3f}   "
                    f"R² = {dl.r2:.2f}   [{dl.consistency_flag}]"
                    f"   (depth ∈ [0,1])"
                )
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> None:
        from .report import report_to_json
        report_to_json(self, Path(path))

    def to_html(self, path: str | Path, *, comparison=None) -> None:
        from .report import report_to_html
        report_to_html(self, Path(path), comparison=comparison)

    def to_png(self, path: str | Path) -> None:
        from .report import report_to_png
        report_to_png(self, Path(path))


def analyze(
    model_id: str,
    *,
    n_seqs: int = 32,
    seq_len: int = 512,
    projections: Sequence[str] = ("down_proj", "o_proj"),
    layer_limit: int | None = None,
    bit_budgets: tuple[float, ...] = (2.0, 3.0, 4.0, 4.5),
    device: str | None = None,
    trust_remote_code: bool = False,
    progress: bool = True,
) -> DiagnosticReport:
    """Fingerprint a transformer. Returns a DiagnosticReport.

    Args:
        model_id: HuggingFace model id or local path.
        n_seqs, seq_len: calibration corpus size (wikitext-2 train split).
        projections: which linear projections to profile per layer.
        layer_limit: if set, only profile layers [0, layer_limit) (for quick smoke).
        bit_budgets: bit budgets to report D*(R) at.
        device: 'cuda' | 'cpu'. Defaults to cuda if available.
        trust_remote_code: passed to AutoModelForCausalLM.
        progress: print per-layer status.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if progress:
        print(f"[fraqtl-diagnostic {__version__}] analyzing {model_id} on {dev}")
        print(f"[0.0s] loading model + tokenizer...", flush=True)
    model, tok = load_model(model_id, trust_remote_code=trust_remote_code)

    if progress:
        print(f"[{time.time()-t0:.1f}s] tokenizing {n_seqs}×{seq_len} calibration seqs...", flush=True)
    calib_ids = load_wikitext_calibration(tok, n_seqs=n_seqs, seq_len=seq_len)

    targets = find_target_modules(model, projections=projections, layer_limit=layer_limit)
    n_layers = len(targets)
    if progress:
        print(f"[{time.time()-t0:.1f}s] found {n_layers} layers × {len(projections)} projections", flush=True)

    # Read attention head structure from model config so o_proj can be fit
    # per-head instead of as one smeared multi-head mixture (R² collapse).
    cfg = getattr(model, "config", None)
    n_heads = getattr(cfg, "num_attention_heads", None) if cfg else None
    head_dim = None
    if cfg and n_heads:
        head_dim = getattr(cfg, "head_dim", None) or (
            getattr(cfg, "hidden_size", 0) // n_heads if getattr(cfg, "hidden_size", None) else None
        )

    fingerprints: list[LayerFingerprint] = []
    for layer_idx in sorted(targets.keys()):
        for proj, mod in targets[layer_idx].items():
            ts = time.time()
            H, count = capture_single_hessian(model, calib_ids, mod, dev)

            if proj == "o_proj" and n_heads and head_dim and (n_heads * head_dim) == H.shape[0]:
                # Per-head o_proj fit: block-diagonal into n_heads head-dim blocks,
                # fit γ per head, aggregate to median γ across heads. This produces
                # a defensible depth-law R² on models where the mixture-of-heads
                # smearing otherwise gives R²≈0.
                head_eigs = []
                for h in range(int(n_heads)):
                    lo, hi = h * head_dim, (h + 1) * head_dim
                    H_h = H[lo:hi, lo:hi].contiguous()
                    head_eigs.append(eigendecompose(H_h))
                fp = aggregate_per_head(
                    head_eigs, layer=layer_idx, projection=proj,
                    n_samples=count, bit_budgets=bit_budgets,
                )
            else:
                eig = eigendecompose(H)
                fp = fingerprint_layer(
                    eig, layer=layer_idx, projection=proj,
                    n_samples=count, bit_budgets=bit_budgets,
                )
            del H
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
            fingerprints.append(fp)
            if progress:
                g = fp.gamma if fp.gamma is not None else float("nan")
                knee = fp.knee if fp.knee is not None else "-"
                print(
                    f"[{time.time()-t0:.1f}s] layer {layer_idx:2d} {proj:10s} "
                    f"dim={fp.dim} γ={g:.3f} α_tail={fp.alpha_tail:.2f} "
                    f"k95={fp.k95} knee={knee} best={fp.best_family}  "
                    f"({time.time()-ts:.1f}s, N={count})",
                    flush=True,
                )

    depth_laws = []
    for proj in projections:
        dl = fit_depth_law(fingerprints, proj)
        if dl is not None:
            depth_laws.append(dl)

    estimate = estimate_compression(fingerprints)

    # free model
    del model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    meta = {
        "elapsed_s": time.time() - t0,
        "device": dev,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "bit_budgets": list(bit_budgets),
    }
    if progress:
        print(f"[{meta['elapsed_s']:.1f}s] done.\n")

    return DiagnosticReport(
        model_id=model_id,
        version=__version__,
        n_layers=n_layers,
        projections=list(projections),
        fingerprints=fingerprints,
        depth_laws=depth_laws,
        estimate=estimate,
        meta=meta,
    )
