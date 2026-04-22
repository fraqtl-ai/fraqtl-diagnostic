"""Modal smoke / try-it runner — runs the full analyze() pipeline on any HF model.

Default: Llama-3.2-1B-Instruct (needs HF auth token in the Modal `huggingface` secret).
Override with any open HF id:

    modal run tests/modal_smoke_llama_1b.py
    modal run tests/modal_smoke_llama_1b.py --model-id mistralai/Mistral-7B-v0.1
    modal run tests/modal_smoke_llama_1b.py --model-id Qwen/Qwen2.5-0.5B --n-seqs 16 --seq-len 256
    modal run tests/modal_smoke_llama_1b.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0

Outputs JSON + HTML + PNG to the fraqtl-hf-cache volume under
/cache/fraqtl-results/diagnostic-smoke/<model>_fingerprint.{json,html,png}.
Use `modal volume get fraqtl-hf-cache fraqtl-results/diagnostic-smoke .` to pull locally.
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
    timeout=3600,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    n_seqs: int = 16,
    seq_len: int = 256,
    projections: str = "down_proj,o_proj",
    layer_limit: int | None = None,
    trust_remote_code: bool = False,
    compare_to: str | None = None,
):
    import os
    import sys
    sys.path.insert(0, "/root")

    os.environ.setdefault("HF_HOME", "/cache/hf")

    from fraqtl_diagnostic import analyze, __version__
    from fraqtl_diagnostic.compare import compare_to_reference

    print(f"fraqtl-diagnostic v{__version__}  on  {model_id}")
    if compare_to:
        print(f"will compare to: {compare_to}")
    print("-" * 70)

    projs = tuple(p.strip() for p in projections.split(",") if p.strip())
    report = analyze(
        model_id,
        n_seqs=n_seqs,
        seq_len=seq_len,
        projections=projs,
        layer_limit=layer_limit,
        trust_remote_code=trust_remote_code,
        progress=True,
    )

    comparison = compare_to_reference(report, compare_to) if compare_to else None

    print()
    print(report.summary())
    if comparison is not None:
        print()
        print(comparison.summary())
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
    report.to_html(html_path, comparison=comparison)

    vol.commit()

    # Sanity checks (measurement-only; no budget_bits since v0.1 is pure measurement)
    assert report.fingerprints, "no fingerprints produced"
    assert any(f.gamma is not None for f in report.fingerprints), "no γ fit succeeded"
    assert report.depth_laws, "no depth-law fit succeeded"
    assert report.summary_stats.n_fingerprints > 0

    print(f"\nJSON : {json_path}  ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"HTML : {html_path}  ({html_path.stat().st_size/1024:.1f} KB)")
    print(f"PNG  : {png_path}  ({png_path.stat().st_size/1024:.1f} KB)")

    s = report.summary_stats
    return {
        "status": "ok",
        "n_fingerprints": s.n_fingerprints,
        "n_depth_laws": len(report.depth_laws),
        "mean_gamma": (None if s.gamma_summary_suppressed else s.mean_gamma),
        "gamma_suppressed": s.gamma_summary_suppressed,
        "mean_k95_ratio": s.mean_k95_ratio,
        "regime_counts": s.regime_counts,
        "fit_quality_counts": s.fit_quality_counts,
        "d_star_by_bits": s.d_star_by_bits,
    }


@app.local_entrypoint()
def main(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    n_seqs: int = 16,
    seq_len: int = 256,
    projections: str = "down_proj,o_proj",
    layer_limit: int = None,
    trust_remote_code: bool = False,
    compare_to: str = None,
):
    r = smoke.remote(
        model_id=model_id,
        n_seqs=n_seqs,
        seq_len=seq_len,
        projections=projections,
        layer_limit=layer_limit,
        trust_remote_code=trust_remote_code,
        compare_to=compare_to,
    )
    print("\n== RESULT ==")
    for k, v in r.items():
        print(f"  {k}: {v}")
    safe = model_id.replace("/", "_")
    print()
    print("To pull reports to this machine:")
    print(f"  modal volume get fraqtl-hf-cache "
          f"fraqtl-results/diagnostic-smoke/{safe}_fingerprint.png ./")
    print(f"  modal volume get fraqtl-hf-cache "
          f"fraqtl-results/diagnostic-smoke/{safe}_fingerprint.html ./")
