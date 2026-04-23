---
title: fraQtl Diagnostic
emoji: "\U0001F52E"
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Fingerprint any transformer's compression potential in 5 min.
tags:
  - transformer
  - compression
  - quantization
  - shannon
  - diagnostic
  - kolmogorov-williams-watts
  - kww
models:
  - mistralai/Mistral-7B-v0.1
  - Qwen/Qwen2.5-3B
  - meta-llama/Llama-3.2-3B
  - microsoft/Phi-3-mini-4k-instruct
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0
---

<p align="center">
  <img src="https://raw.githubusercontent.com/fraqtl-ai/fraqtl-diagnostic/main/docs/logo.png" alt="fraQtl" width="140"/>
</p>

<h1 align="center">fraQtl Diagnostic — live demo</h1>

<p align="center">
  <b>Fingerprint any HuggingFace transformer's compression potential — in ~5 min on this free T4.</b>
</p>

---

## What this demo does

Drop any HF model id into the box on the left. The tool:

1. Loads the model (fp16)
2. Captures the per-layer input-covariance Hessian on wikitext-2 calibration
3. Eigendecomposes + fits a Kohlrausch-Williams-Watts γ (spectrum shape)
4. Reports regime tags, effective rank (k95), and the Shannon D*(R) rate-distortion floor
5. Renders a 4-panel figure + plain-English TL;DR

It does **not** compress the model. It tells you whether compression will
work, before you spend GPU-hours finding out.

---

## The paired hero

Same diagnostic. Two models. Opposite compression outcomes.

<table>
<tr>
<td width="50%">
<img src="https://raw.githubusercontent.com/fraqtl-ai/fraqtl-diagnostic/main/examples/launch_screenshots/tweet_mistral_clean.png" alt="Mistral-7B clean"/>
<p align="center"><b>Mistral-7B</b> — 0/64 layers outside universality class<br/>→ INT4 compresses cleanly (+0.46 PPL)</p>
</td>
<td width="50%">
<img src="https://raw.githubusercontent.com/fraqtl-ai/fraqtl-diagnostic/main/examples/launch_screenshots/tweet_qwen3b_collapse.png" alt="Qwen2.5-3B collapse"/>
<p align="center"><b>Qwen-2.5-3B</b> — 4/72 layers outside universality class<br/>→ INT3 collapses (+23394 PPL) 💥</p>
</td>
</tr>
</table>

---

## Try these models in the demo

- `Qwen/Qwen2.5-0.5B` — fastest (~3 min on T4)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — small dense transformer
- `microsoft/Phi-3-mini-4k-instruct` — mid-size dense
- `Qwen/Qwen2.5-3B` — the anomaly case above
- `mistralai/Mistral-7B-v0.1` — the clean case above (needs HF auth token)

Gated models (Llama, Mistral) require a HF token at the top-right avatar → Settings → API Keys.

---

## What you get (from the live run)

- **TL;DR** — plain-English verdict
- **Technical details** — mean γ, k95, regime counts, fit-quality breakdown
- **Shannon D*(R)** — theoretical rate-distortion floor at R ∈ {2, 3, 4}
- **Depth-law fits** — γ vs layer depth, per-projection, with consistency flag
- **4-panel figure** — spectrum overlay, γ depth-law, k95 per layer, summary

---

## Local install

Everything here works identically on your own machine:

```bash
pip install fraqtl-diagnostic
fraqtl analyze meta-llama/Llama-3.2-1B-Instruct
```

- **PyPI:** [pypi.org/project/fraqtl-diagnostic](https://pypi.org/project/fraqtl-diagnostic/)
- **GitHub:** [github.com/fraqtl-ai/fraqtl-diagnostic](https://github.com/fraqtl-ai/fraqtl-diagnostic)
- **Docs:** README in the repo has a full interpretation guide

---

## What's open vs closed

**Apache 2.0 (this Space + the PyPI package):**
- Spectrum measurement, KWW γ fit, regime tags, depth-law
- Shannon D*(R) ceiling
- Per-head attention fit, fit-quality flag
- CLI, HTML/JSON/PNG output
- 8 reference models for `--compare-to`

**Closed (the fraQtl product, [fraqtl.ai](https://fraqtl.ai)):**
- Compression engine itself — sign correction, per-model calibration,
  fused-MoE expert packing, quantizer, packed-safetensors loader

Measurement is free. Compression is the product.

---

## Honest caveats

- Depth-law R² varies by model. This is a per-model fingerprint, not a
  universal law. Reported explicitly (linear / nonlinear / noisy).
- γ > 1 (compressed-exp regime) happens on some small or pathological
  layers — explicitly marked in the regime tags.
- GPT-2 arch not yet supported (non-standard projection naming). GGUF,
  raw `.pt`, ONNX require conversion. See the GitHub README for roadmap.
- v0.1 does **not** predict PPL loss. That's atlas-validated and ships
  in v0.2 with Paper 3 (~4 weeks).

---

## Feedback

"Breaks on my model" reports welcome — [open an issue](https://github.com/fraqtl-ai/fraqtl-diagnostic/issues) with the HF model id.
