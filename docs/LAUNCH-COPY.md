# Launch copy — fraqtl-diagnostic v0.1

Honest, defensible, no claims the tool can't back up. Fire whenever.

---

## Twitter / X thread (5 tweets)

**1/**
shipped `fraqtl-diagnostic` — the first measurement step before compressing a transformer.

`pip install fraqtl-diagnostic`
`fraqtl analyze <any-HF-model>`

reports per-layer spectrum shape (γ), effective rank (k95), Shannon-ceiling bit budget. Apache 2.0.

**2/**
example — Mistral-7B on A100 (~10 min):

    layers        : 32
    mean γ        : 0.66
    mean k95/dim  : 37%
    headroom      : 0.63
    suggested b/w : 3.4 balanced / 2.9 aggressive

numbers before you compress. not vibes.

**3/**
main use case: fine-tuned a model and want to know if your fine-tune preserved compressibility?

`fraqtl analyze my-org/my-finetune --compare-to mistralai/Mistral-7B-v0.1`

ends with one line: preserved / shifted / degraded / broken.

**4/**
what's open (Apache 2.0):
— per-layer Hessian spectrum + KWW γ fit
— k95 / knee / Shannon D*(R)
— comparison to 8 bundled reference models

closed: the compression engine itself (sign correction, per-model calibration, fused-MoE packing). that's the product.

**5/**
try in 30 sec on free T4:
colab.research.google.com/github/fraqtl-ai/fraqtl-diagnostic/blob/main/examples/quickstart.ipynb

source: github.com/fraqtl-ai/fraqtl-diagnostic
feedback welcome, especially "it breaks on <model>" reports.

---

## Reddit /r/LocalLLaMA

**Title:** `[Open source] fraqtl-diagnostic — measure any transformer's spectrum shape and bit-budget ceiling in ~5 min`

**Body:**

`pip install fraqtl-diagnostic`. Apache 2.0. A measurement tool — it does NOT compress your model, it tells you what the information-theoretic ceiling is and reports per-layer shape so you can decide whether compression is worth attempting.

**Per layer, per projection:**
- γ — KWW (Kohlrausch) shape parameter fit on the Hessian input-covariance eigenvalue spectrum
- k95 — fraction of eigendirections holding 95% of eigenvalue mass
- knee — index where log-log slope crashes (if any)
- Shannon D*(R) — Cover & Thomas rate-distortion lower bound at bit budgets R ∈ {2, 3, 4, 4.5}

**Example — Mistral-7B on A100 (~10 min):**
```
layers        : 32
mean γ        : 0.66
mean k95/dim  : 37%
headroom      : 0.63
suggested b/w : 3.4 (balanced) / 2.9 (aggressive)
```

**`--compare-to` flag** against 8 bundled reference models (Llama-3.2-3B, Mistral-7B, Qwen2.5-3B/7B/14B, Phi-3-mini, TinyLlama-1.1B, OLMoE-1B-7B). Emits a one-line verdict vs the reference (preserved / shifted / degraded / broken). Meant for fine-tune sanity checks.

**Try in 30 sec** (free T4 on Colab):
colab.research.google.com/github/fraqtl-ai/fraqtl-diagnostic/blob/main/examples/quickstart.ipynb

**Install:** `pip install fraqtl-diagnostic`
**Source:** github.com/fraqtl-ai/fraqtl-diagnostic
**PyPI:** pypi.org/project/fraqtl-diagnostic/

Full disclosure — I sell a compression engine as a product. The diagnostic is open because measurement shouldn't be vendor-locked. The compression engine (sign correction, per-model calibration, fused-MoE expert packing) stays closed.

Known soft spots I'll call out before you do:
- depth-law R² varies by model — some models show a tight per-layer γ trend, others are nearly flat (well-regularized bases). Trend is a per-model fingerprint, not a universal law.
- γ > 1 on some layers = compressed-exp regime (not stretched), marked explicitly in the output.
- GPT-2 arch-family not yet supported (non-standard projection names). Llama/Qwen/Mistral/Mixtral/TinyLlama/MoE work out of the box.

Feedback + "breaks on X model" reports welcome.

---

## Hacker News "Show HN"

**Title:** `Show HN: Fraqtl-diagnostic – fingerprint any transformer's compression potential`

**Body:**

`pip install fraqtl-diagnostic` → `fraqtl analyze <HF model>` → per-layer spectrum report + Shannon-ceiling bit budget, in 3–10 minutes on a single GPU.

Motivation: I sell a transformer compression engine. The first question from every prospective customer is "will it work on *my* model?" The honest answer requires measuring the model's Hessian spectrum, fitting a shape parameter, and computing the rate-distortion floor. This tool makes that measurement a one-liner.

**What's measured, per layer per projection:**
- KWW γ (Kohlrausch shape) fit on the eigenvalue spectrum
- k95 — effective rank (directions for 95% energy)
- knee — sharp-cutoff index
- Shannon D*(R) at bit budgets 2 / 3 / 4 / 4.5 b/w

**Example — Mistral-7B** (~10 min on A100): mean γ = 0.66, mean k95/dim = 37%, headroom = 0.63, suggested bit budget = 3.4 b/w balanced.

**Fine-tune comparison flag:** `--compare-to <reference_model>` emits a per-projection delta table and a one-line verdict (preserved / shifted / degraded / broken). 8 reference models bundled.

**Open (Apache 2.0):** spectrum measurement, γ fitting, D*(R), reference-table comparison, CLI, HTML/JSON/PNG output, unit tests.

**Closed (product):** the compression engine — sign correction, per-model calibration, fused-MoE expert packing, quantizer.

Honest caveats:
- This is measurement, not theory. The tool reports; it does not claim "universal law." Depth-trend R² varies a lot by model.
- γ fit sometimes exceeds 1 (compressed-exp regime); that's marked explicitly in the output.
- GPT-2's non-standard projection naming is not yet supported.

Links:
- GitHub: https://github.com/fraqtl-ai/fraqtl-diagnostic
- PyPI: https://pypi.org/project/fraqtl-diagnostic/
- Colab: https://colab.research.google.com/github/fraqtl-ai/fraqtl-diagnostic/blob/main/examples/quickstart.ipynb

Happy to answer questions about the measurement methodology, where it breaks, and the open/closed split rationale.

---

## Suggested posting order

1. **Tweet thread** — fires the warm audience first
2. **HN Show HN** — 30–60 min later, early morning US Pacific (best TZ for front page)
3. **Reddit /r/LocalLLaMA** — same day or next, once Twitter has early engagement

---

## Do NOT say

- "universality proven" / "depth-law proven" — R² is not tight enough
- "compressed in 3 minutes" — the DIAGNOSTIC runs in ~3 min, compression is a separate product
- "stretched-exponential fit" — use "KWW γ" or just "shape parameter γ"; γ > 1 is compressed-exp, not stretched
- Specific PPL-delta numbers for compression — that's product-side, not diagnostic-side
- "free compression tool" — the tool is free; compression is not
