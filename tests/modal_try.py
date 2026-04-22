"""Modal smoke test — runs the full analyze() pipeline on Llama-3.2-1B-Instruct.

Confirms:
  - HF model loads and we find transformer layers
  - Hessian capture + eigendecomposition works end-to-end
  - γ / knee / k95 / depth-law / estimator all populate
  - JSON + HTML + PNG reports write without error

Usage:
    modal run tests/modal_smoke_llama_1b.py
"""
import modal
import pathlib

app = modal.App("fraqtl-diagnostic-smoke")

_PKG = pathlib.Path(__file__).resolve().parents[1] / "fraqtl_diagnostic"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51",
        "accelerate",
        "datasets",
        "numpy",
        "scipy",
        "matplotlib",
        "huggingface_hub>=0.26",
    )
    .add_local_dir(str(_PKG), remote_path="/root/fraqtl_diagnostic")
)
vol = modal.Volume.from_name("fraqtl-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=2400,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke():
    import os
    import sys
    sys.path.insert(0, "/root")

    # HF cache on volume
    os.environ.setdefault("HF_HOME", "/cache/hf")

    from fraqtl_diagnostic import analyze, __version__

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"fraqtl-diagnostic v{__version__} smoke on {model_id}")
    print("-" * 70)

    report = analyze(
        model_id,
        n_seqs=16,          # smaller for smoke
        seq_len=256,
        projections=("down_proj", "o_proj"),
        progress=True,
    )

    print()
    print(report.summary())
    print()

    # Write reports to the volume so we can inspect them later
    out_dir = pathlib.Path("/cache/fraqtl-results/diagnostic-smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "_")
    json_path = out_dir / f"{safe}_fingerprint.json"
    html_path = out_dir / f"{safe}_fingerprint.html"
    png_path = out_dir / f"{safe}_fingerprint.png"

    report.to_json(json_path)
    report.to_png(png_path)
    report.to_html(html_path)

    vol.commit()

    # Sanity checks
    assert report.fingerprints, "no fingerprints produced"
    assert any(f.gamma is not None for f in report.fingerprints), "no γ fit succeeded"
    assert report.depth_laws, "no depth-law fit succeeded"
    assert 2.0 <= report.estimate.budget_bits_aggressive <= 5.0

    print(f"\nJSON : {json_path}  ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"HTML : {html_path}  ({html_path.stat().st_size/1024:.1f} KB)")
    print(f"PNG  : {png_path}  ({png_path.stat().st_size/1024:.1f} KB)")

    return {
        "status": "ok",
        "n_fingerprints": len(report.fingerprints),
        "n_depth_laws": len(report.depth_laws),
        "mean_gamma": report.estimate.mean_gamma,
        "headroom": report.estimate.headroom_score,
        "budget_balanced": report.estimate.budget_bits_balanced,
        "headline": report.estimate.headline,
    }


@app.local_entrypoint()
def main():
    r = smoke.remote()
    print("\n== SMOKE RESULT ==")
    for k, v in r.items():
        print(f"  {k}: {v}")
