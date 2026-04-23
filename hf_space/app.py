"""Gradio app for HuggingFace Spaces — zero-install fraqtl-diagnostic demo.

Deploy:
  1. Create an HF Space at https://huggingface.co/new-space
     - Owner: fraQtl
     - Name: diagnostic-demo
     - SDK: Gradio
     - Hardware: T4 small (free tier)
  2. Copy this file + requirements.txt + README.md from hf_space/ to the Space repo root
  3. git push to Space; it auto-builds

The Space shows a text box for HF model id, a Run button, and renders the
TL;DR + 4-panel figure inline when finished.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

import gradio as gr
from fraqtl_diagnostic import analyze


def run_diagnostic(model_id: str, n_seqs: int = 16, seq_len: int = 256):
    """Run fraqtl on a model id and return (summary_text, png_path)."""
    if not model_id or "/" not in model_id:
        return (
            "Please provide a valid HuggingFace model id (e.g. `Qwen/Qwen2.5-0.5B`).",
            None,
        )

    try:
        report = analyze(
            model_id,
            n_seqs=int(n_seqs),
            seq_len=int(seq_len),
            projections=("down_proj", "o_proj"),
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error while analyzing {model_id}:\n\n{exc}", None

    out_dir = Path(tempfile.mkdtemp(prefix="fraqtl_space_"))
    png_path = out_dir / "fingerprint.png"
    report.to_png(png_path)

    return report.summary(), str(png_path)


EXAMPLES = [
    ["Qwen/Qwen2.5-0.5B", 16, 256],
    ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", 16, 256],
    ["microsoft/Phi-3-mini-4k-instruct", 16, 256],
]


with gr.Blocks(title="fraQtl Diagnostic") as demo:
    gr.Markdown(
        """
        # fraQtl Diagnostic

        Fingerprint any HuggingFace transformer's compression potential. Reports per-layer
        γ, k95, regime tags, and the Shannon D*(R) ceiling. Measurement-only — this tool
        tells you what your model *could* compress to; actual compression is the closed
        engine at [fraqtl.ai](https://fraqtl.ai).

        **Local install:** `pip install fraqtl-diagnostic`
        **Source:** [github.com/fraqtl-ai/fraqtl-diagnostic](https://github.com/fraqtl-ai/fraqtl-diagnostic)

        Runs on the free T4 this Space provides. ~3–5 min for a 0.5–1 B model.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            model_id = gr.Textbox(
                label="HuggingFace model id",
                placeholder="e.g. Qwen/Qwen2.5-0.5B",
                value="Qwen/Qwen2.5-0.5B",
            )
            n_seqs = gr.Slider(
                label="calibration sequences (smaller = faster)",
                minimum=4, maximum=64, step=4, value=16,
            )
            seq_len = gr.Slider(
                label="sequence length",
                minimum=64, maximum=1024, step=64, value=256,
            )
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[model_id, n_seqs, seq_len],
            )
        with gr.Column(scale=2):
            summary = gr.Textbox(label="Summary", lines=25, show_copy_button=True)
            figure = gr.Image(label="Fingerprint", interactive=False)

    btn.click(run_diagnostic, inputs=[model_id, n_seqs, seq_len], outputs=[summary, figure])


if __name__ == "__main__":
    demo.launch()
