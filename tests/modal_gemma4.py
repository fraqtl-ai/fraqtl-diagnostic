"""One-off Modal run for Gemma-4 (bleeding-edge; not yet in stable transformers).

Uses A100-80GB (Gemma-4-31B fp16 = 62 GB) and installs transformers from git
so the `gemma4` model_type is recognized.

    modal run --detach tests/modal_gemma4.py --model-id google/gemma-4-31B-it
"""
import modal
import pathlib

app = modal.App("fraqtl-gemma4")

_PKG = pathlib.Path(__file__).resolve().parents[1] / "fraqtl_diagnostic"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "accelerate",
        "datasets",
        "numpy",
        "scipy",
        "matplotlib",
        "huggingface_hub>=0.26",
        # install transformers from git for Gemma-4 / Qwen3.6 / etc. support
        "git+https://github.com/huggingface/transformers.git",
        extra_options="--no-cache-dir",
    )
    .add_local_dir(str(_PKG), remote_path="/root/fraqtl_diagnostic")
)
vol = modal.Volume.from_name("fraqtl-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(model_id: str = "google/gemma-4-31B-it", n_seqs: int = 32, seq_len: int = 512):
    import os, sys, pathlib
    sys.path.insert(0, "/root")
    os.environ.setdefault("HF_HOME", "/cache/hf")

    from fraqtl_diagnostic import analyze
    from fraqtl_diagnostic.compare import compare_to_reference

    print(f"Analyzing {model_id} on A100-80GB (bleeding-edge transformers)")
    report = analyze(
        model_id,
        n_seqs=n_seqs,
        seq_len=seq_len,
        projections=("down_proj", "o_proj"),
        progress=True,
    )

    print()
    print(report.summary())

    out = pathlib.Path("/cache/fraqtl-results/diagnostic-smoke")
    out.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "_")
    report.to_json(out / f"{safe}_fingerprint.json")
    report.to_png(out / f"{safe}_fingerprint.png")
    report.to_html(out / f"{safe}_fingerprint.html")
    vol.commit()
    s = report.summary_stats
    return {
        "status": "ok",
        "n_fingerprints": s.n_fingerprints,
        "mean_gamma": None if s.gamma_summary_suppressed else s.mean_gamma,
        "gamma_suppressed": s.gamma_summary_suppressed,
        "regime_counts": s.regime_counts,
    }


@app.local_entrypoint()
def main(model_id: str = "google/gemma-4-31B-it", n_seqs: int = 32, seq_len: int = 512):
    r = run.remote(model_id=model_id, n_seqs=n_seqs, seq_len=seq_len)
    print("\n== RESULT ==")
    for k, v in r.items():
        print(f"  {k}: {v}")
