"""JSON / HTML / PNG renderers for DiagnosticReport."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np


def _fingerprint_to_dict(fp) -> dict:
    d = asdict(fp)
    # decimated spectrum is already JSON-friendly; just ensure numpy→list
    return d


def report_to_json(report, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model_id": report.model_id,
        "version": report.version,
        "n_layers": report.n_layers,
        "projections": report.projections,
        "meta": report.meta,
        "estimate": asdict(report.estimate),
        "depth_laws": [asdict(dl) for dl in report.depth_laws],
        "fingerprints": [_fingerprint_to_dict(f) for f in report.fingerprints],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _fig_spectrum_overlay(ax, fingerprints, projection):
    rows = [f for f in fingerprints if f.projection == projection]
    if not rows:
        return
    n = len(rows)
    cmap = _colormap(n)
    for i, f in enumerate(rows):
        idx = f.spectrum_decimated.get("indices", [])
        eig = f.spectrum_decimated.get("eigvals", [])
        if idx and eig:
            ax.loglog([i + 1 for i in idx], eig, color=cmap(i / max(n - 1, 1)),
                      alpha=0.6, linewidth=0.8)
    ax.set_title(f"{projection} — per-layer spectrum (log-log)")
    ax.set_xlabel("index")
    ax.set_ylabel("λ_i")


def _colormap(n):
    import matplotlib
    return matplotlib.colormaps["viridis"]


def _fig_gamma_vs_depth(ax, fingerprints, depth_laws, projection):
    rows = [f for f in fingerprints if f.projection == projection and f.gamma is not None]
    if not rows:
        return
    max_l = max(f.layer for f in rows) or 1
    depths = np.array([f.layer / max_l for f in rows])
    gammas = np.array([f.gamma for f in rows])
    ax.scatter(depths, gammas, s=24, alpha=0.7, label=f"{projection}")
    dl = next((d for d in depth_laws if d.projection == projection), None)
    if dl is not None:
        xs = np.linspace(0, 1, 50)
        ys = dl.slope * xs + dl.intercept
        ax.plot(xs, ys, linestyle="--", alpha=0.8,
                label=f"fit: γ = {dl.slope:+.2f}·depth + {dl.intercept:.2f}  R²={dl.r2:.2f}")
    ax.set_xlabel("normalized depth (layer / max_layer)")
    ax.set_ylabel("stretched-exp γ")
    ax.set_title(f"{projection} — depth-law")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)


def _fig_k95(ax, fingerprints, projection):
    rows = sorted(
        [f for f in fingerprints if f.projection == projection],
        key=lambda f: f.layer,
    )
    if not rows:
        return
    layers = [f.layer for f in rows]
    k95 = [f.k95 / f.dim for f in rows]
    ax.plot(layers, k95, marker="o", linestyle="-", alpha=0.8, label="k95/dim")
    ax.set_xlabel("layer")
    ax.set_ylabel("k95 / dim")
    ax.set_title(f"{projection} — compression budget per layer")
    ax.set_ylim(0, max(k95) * 1.2 if k95 else 1.0)


def _fig_summary_panel(ax, s):
    ax.axis("off")
    mean_g = "suppressed" if s.gamma_summary_suppressed else f"{s.mean_gamma:.3f}"
    lines = [
        "Measurement summary",
        "",
        f"  fingerprints  : {s.n_fingerprints}",
        f"  mean γ        : {mean_g}",
        f"  mean k95/dim  : {s.mean_k95_ratio:.2%}",
        "",
        "Regimes:",
    ]
    for k, v in sorted(s.regime_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:18s} {v:3d}")
    lines.append("")
    lines.append("Shannon D*(R):")
    for b, v in sorted(s.d_star_by_bits.items()):
        lines.append(f"  R = {b} b/w : {v:.3e}")
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=10, transform=ax.transAxes)


def report_to_png(report, path: Path, *, figsize=(14, 10)) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj_list = report.projections
    primary = proj_list[0] if proj_list else "down_proj"

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    _fig_spectrum_overlay(axes[0][0], report.fingerprints, primary)
    _fig_gamma_vs_depth(axes[0][1], report.fingerprints, report.depth_laws, primary)
    _fig_k95(axes[1][0], report.fingerprints, primary)
    _fig_summary_panel(axes[1][1], report.summary_stats)
    fig.suptitle(f"fraQtl Diagnostic — {report.model_id}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fraQtl Diagnostic — {model_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 960px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ font-size: 1.5em; border-bottom: 2px solid #333; padding-bottom: 0.2em; }}
  h2 {{ font-size: 1.2em; margin-top: 2em; }}
  table {{ border-collapse: collapse; font-size: 0.9em; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.35em 0.7em; text-align: right; }}
  th {{ background: #f3f3f3; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .headline {{ background: #f7f9fb; border-left: 4px solid #3366cc;
               padding: 0.8em 1.2em; margin: 1em 0; font-family: monospace; }}
  .verdict  {{ background: #fffaf0; border-left: 4px solid #e09020;
               padding: 0.8em 1.2em; margin: 1em 0; }}
  .delta    {{ font-family: monospace; font-weight: 600; }}
  .meta {{ color: #666; font-size: 0.85em; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>

<h1>fraQtl Diagnostic — {model_id}</h1>
<p class="meta">
  v{version} &middot;
  {n_layers} layers &middot;
  projections: {projections} &middot;
  {elapsed:.1f}s on {device}
</p>

<div class="headline">{headline}</div>

<h2>Measurement summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Fingerprints (layer × projection)</td><td>{n_fingerprints}</td></tr>
  <tr><td>Mean γ  (stretched_exp regime, non-poor fits only)</td><td>{mean_gamma_cell}</td></tr>
  <tr><td>Mean k95 / dim</td><td>{mean_k95:.2%}</td></tr>
  <tr><td>Regimes</td><td>{regime_str}</td></tr>
  <tr><td>Fit quality</td><td>{fit_quality_str}</td></tr>
</table>

<h2>Shannon rate-distortion ceiling D*(R)</h2>
<p class="meta">geometric mean over all layers at each bit budget R. <strong>Theoretical floor, not a prediction</strong> —
  the actual PPL a real compressor achieves depends on the recipe (sign correction, rank protection, etc.).</p>
<table>
  <tr><th>R (b/w)</th><th>D*(R) geomean</th></tr>
  {d_star_rows}
</table>

<h2>Depth-law fits</h2>
<table>
  <tr><th>Projection</th><th>Slope</th><th>Intercept</th><th>R²</th><th>γ p10/p50/p90</th><th>n</th></tr>
  {depth_law_rows}
</table>

<h2>Per-layer fingerprint (first 40)</h2>
<table>
  <tr><th>Layer</th><th>Proj</th><th>dim</th><th>γ</th><th>regime</th><th>fit</th>
      <th>k95</th><th>k99</th><th>knee</th></tr>
  {layer_rows}
</table>

<h2>Figures</h2>
<p>(generate the PNG with <code>report.to_png(path)</code> — shown here if embedded)</p>
{png_embed}

{comparison_section}

<h2>About</h2>
<p class="meta">
  This diagnostic measures information-theoretic compressibility from the input covariance
  spectrum of each layer. It does <em>not</em> perform compression — it reports the ceiling and
  shape. The <a href="https://fraqtl.ai">fraQtl compression engine</a> uses this fingerprint
  plus additional engineering (sign correction, per-model calibration) to get closer to the
  ceiling in practice.
</p>

</body>
</html>
"""


def _format_value(v, fmt="{:.3f}"):
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    return fmt.format(v)


def _comparison_html(comparison) -> str:
    if comparison is None or not comparison.reference_available:
        return ""
    rows = []
    for d in comparison.deltas:
        rows.append(
            f"<tr><td>{d.projection}</td>"
            f"<td>{d.gamma_this:.3f}</td>"
            f"<td>{d.gamma_ref:.3f}</td>"
            f"<td class='delta'>{d.gamma_delta:+.3f}</td>"
            f"<td>{d.k95_ratio_this:.2%}</td>"
            f"<td>{d.k95_ratio_ref:.2%}</td>"
            f"<td class='delta'>{d.k95_ratio_delta:+.2%}</td></tr>"
        )
    rationale_html = "".join(f"<li>{r}</li>" for r in comparison.rationale)
    return f"""
<h2>Comparison vs {comparison.reference_model}</h2>
<div class="verdict"><strong>Verdict:</strong> {comparison.verdict}</div>
<table>
  <tr><th>Projection</th>
      <th>γ (this)</th><th>γ (ref)</th><th>Δγ</th>
      <th>k95/dim (this)</th><th>k95/dim (ref)</th><th>Δ k95/dim</th></tr>
  {''.join(rows)}
</table>
{f'<ul>{rationale_html}</ul>' if rationale_html else ''}
"""


def report_to_html(report, path: Path, *, embed_png: bool = True, comparison=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dl_rows = []
    for dl in report.depth_laws:
        dl_rows.append(
            f"<tr><td>{dl.projection}</td>"
            f"<td>{dl.slope:+.3f}</td>"
            f"<td>{dl.intercept:.3f}</td>"
            f"<td>{dl.r2:.3f}</td>"
            f"<td>{dl.gamma_p10:.3f} / {dl.gamma_p50:.3f} / {dl.gamma_p90:.3f}</td>"
            f"<td>{dl.n}</td></tr>"
        )

    layer_rows = []
    for f in report.fingerprints[:40]:
        layer_rows.append(
            f"<tr><td>{f.layer}</td>"
            f"<td>{f.projection}</td>"
            f"<td>{f.dim}</td>"
            f"<td>{_format_value(f.gamma)}</td>"
            f"<td>{f.regime or '—'}</td>"
            f"<td>{f.fit_quality}</td>"
            f"<td>{f.k95}</td>"
            f"<td>{f.k99}</td>"
            f"<td>{'—' if f.knee is None else f.knee}</td></tr>"
        )

    png_embed = ""
    if embed_png:
        png_path = path.with_suffix(".png")
        report_to_png(report, png_path)
        png_embed = f'<p><img src="{png_path.name}" alt="diagnostic figure"></p>'

    s = report.summary_stats
    mean_gamma_cell = (
        "suppressed (&gt;10% layers outside stretched_exp or poor fit)"
        if s.gamma_summary_suppressed else f"{s.mean_gamma:.3f}"
    )
    regime_str = ", ".join(
        f"{v} {k}" for k, v in sorted(s.regime_counts.items(), key=lambda kv: -kv[1])
    ) if s.regime_counts else "—"
    fit_quality_str = (
        f"{s.fit_quality_counts.get('good', 0)} good · "
        f"{s.fit_quality_counts.get('moderate', 0)} moderate · "
        f"{s.fit_quality_counts.get('poor', 0)} poor"
    )
    d_star_rows = "".join(
        f"<tr><td>{b}</td><td>{v:.3e}</td></tr>"
        for b, v in sorted(s.d_star_by_bits.items())
    ) or "<tr><td colspan='2'>—</td></tr>"

    headline = (
        f"{report.n_layers} layers across {', '.join(report.projections)}. "
        f"Measurement complete. {s.cta}"
    )

    html = _HTML_TEMPLATE.format(
        model_id=report.model_id,
        version=report.version,
        n_layers=report.n_layers,
        projections=", ".join(report.projections),
        elapsed=report.meta.get("elapsed_s", float("nan")),
        device=report.meta.get("device", "?"),
        headline=headline,
        n_fingerprints=s.n_fingerprints,
        mean_gamma_cell=mean_gamma_cell,
        mean_k95=s.mean_k95_ratio,
        regime_str=regime_str,
        fit_quality_str=fit_quality_str,
        d_star_rows=d_star_rows,
        depth_law_rows="\n  ".join(dl_rows) if dl_rows else "<tr><td colspan='6'>—</td></tr>",
        layer_rows="\n  ".join(layer_rows),
        png_embed=png_embed,
        comparison_section=_comparison_html(comparison),
    )
    with open(path, "w") as f:
        f.write(html)
