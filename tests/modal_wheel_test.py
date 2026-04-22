"""Verify the built wheel installs cleanly and the CLI works end-to-end.

Fires on a FRESH debian image — no pre-installed fraqtl_diagnostic —
then `pip install /wheel` and runs `fraqtl analyze Qwen/Qwen2.5-0.5B`.

If this passes, the wheel is ready to upload to PyPI.

    modal run tests/modal_wheel_test.py
"""
import pathlib
import modal

app = modal.App("fraqtl-diagnostic-wheel-test")

_WHEEL = pathlib.Path(__file__).resolve().parents[1] / "dist" / "fraqtl_diagnostic-0.1.0-py3-none-any.whl"
assert _WHEEL.exists(), f"build the wheel first: python -m build  ({_WHEEL})"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(str(_WHEEL), remote_path="/root/fraqtl_diagnostic-0.1.0-py3-none-any.whl",
                    copy=True)
    .run_commands(
        "pip install --no-cache-dir /root/fraqtl_diagnostic-0.1.0-py3-none-any.whl",
    )
)

vol = modal.Volume.from_name("fraqtl-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_wheel():
    import os
    import subprocess

    os.environ.setdefault("HF_HOME", "/cache/hf")

    # CLI should be on PATH after `pip install ./wheel`
    print(">>> fraqtl --version")
    subprocess.run(["fraqtl", "--version"], check=True)

    # Python import smoke
    print("\n>>> import fraqtl_diagnostic")
    import fraqtl_diagnostic
    print(f"   version: {fraqtl_diagnostic.__version__}")
    print(f"   module path: {fraqtl_diagnostic.__file__}")

    # Full CLI run on a small open model
    print("\n>>> fraqtl analyze Qwen/Qwen2.5-0.5B --layer-limit 4 --n-seqs 4 --seq-len 128")
    out_dir = "/cache/fraqtl-results/wheel-test"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [
            "fraqtl", "analyze", "Qwen/Qwen2.5-0.5B",
            "--layer-limit", "4",
            "--n-seqs", "4",
            "--seq-len", "128",
            "--out-dir", out_dir,
        ],
        check=True,
    )

    # Confirm output files exist
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        size = os.path.getsize(path)
        print(f"   {name:60s}  {size/1024:.1f} KB")

    vol.commit()
    return {"status": "ok"}


@app.local_entrypoint()
def main():
    r = test_wheel.remote()
    print("\n== WHEEL-TEST RESULT ==")
    for k, v in r.items():
        print(f"  {k}: {v}")
