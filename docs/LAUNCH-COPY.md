# Launch copy — fraqtl-diagnostic v0.1

Drafts for announcement across channels. Tune + fire after Day 5 film lands.

## Tweet / X thread (5 tweets)

**1/**
we open-sourced the tool we use to decide if a model is worth compressing.

`pip install fraqtl-diagnostic`
`fraqtl analyze <your-model>`

30 min on an A100. spits out γ, k95, depth-law, and a Shannon-based bit budget.
no compression runs, no guessing.

**2/**
example: Qwen2.5-0.5B in 2:44 on A100

    mean γ           : 0.807
    mean k95/dim     : 25.9%
    headroom         : 0.66
    suggested b/w    : 3.3 (balanced) / 2.8 (aggressive)
    depth-law        : γ = -0.25·depth + 1.22,  R² = 0.52

decide in minutes whether your model can survive 3-bit.

**3/**
what it measures (per layer, per projection):

- γ : stretched-exp shape of the Hessian spectrum
- k95: # of directions holding 95% of eigenvalue energy
- knee: sharp-cutoff index if any
- depth-law: how γ evolves layer-by-layer
- Shannon D*(R): info-theoretic floor at any bit budget

everything else is research. this is the measurement tool.

**4/**
works on any HuggingFace transformer. tested on llama / qwen / mistral /
phi / tinyllama / olmoe families. one CLI, one HTML report.

github: github.com/fraqtl-ai/fraqtl-diagnostic
paper 3 with the depth-law universality proof is up next.

**5/**
we kept the compression engine — sign correction, per-model calibration,
fused MoE experts — closed. the diagnostic is free and Apache 2.0, because
measurement shouldn't be a black box.

try it on your favorite model. DM the γ.

---

## Reddit /r/LocalLLaMA post

**Title:** We open-sourced the tool we use to predict how much you can compress any model

**Body:**

After getting tired of "will this model survive 3-bit quant?" guesses, we
packaged the diagnostic we use internally and shipped it as `pip install
fraqtl-diagnostic`.

What it does:

- Loads any HF-compatible transformer
- Measures per-layer Hessian spectrum on wikitext-2 calibration
- Fits a stretched-exponential γ + depth-law across layers
- Reports Shannon rate-distortion ceiling at 2 / 3 / 4 / 4.5 bits
- Tells you a "compression potential" score and suggested b/w

No compression runs. Takes 3 min on a 0.5B model, ~10 min on 7B.

Example on Qwen2.5-0.5B (A100, 2:44 wall):

```
layers: 24  projections: down_proj, o_proj
mean γ       : 0.807
mean k95/dim : 25.9%
headroom     : 0.66
suggested    : 3.3 b/w balanced / 2.8 aggressive
depth-law    : γ = -0.25·depth + 1.22   R² = 0.52
```

HTML report has a 4-panel figure (spectrum overlay, γ per layer, k95 per
layer, summary).

Install:
```
pip install fraqtl-diagnostic
fraqtl analyze meta-llama/Llama-3.2-1B-Instruct
```

Or one-liner via Modal if you don't have a GPU handy:
```
modal run tests/modal_try.py --model-id mistralai/Mistral-7B-v0.1
```

Source: github.com/fraqtl-ai/fraqtl-diagnostic  (Apache 2.0)

The compression engine (sign correction, recipe calibration, fused-expert
MoE) stays closed — it's how we make money. Diagnostic is open because
measurement shouldn't be vendor-locked.

Feedback / architecture requests welcome.

---

## Hacker News "Show HN"

**Title:** Show HN: fraQtl Diagnostic — fingerprint any transformer's compression potential

**Body:**

`pip install fraqtl-diagnostic` → `fraqtl analyze <your HF model>` →
30-minute report of per-layer spectrum shape, Shannon rate-distortion
ceiling, and suggested bit budget.

Motivation: we ship a compression engine as a product, and the recurring
question from prospective customers is "will it work on *my* model?" The
honest answer requires measuring the model's Hessian spectrum first —
this tool makes that measurement a one-liner.

What's open (Apache 2.0): spectral measurement, γ / knee / k95 fitting,
Shannon D*(R) computation, depth-law regression, report generation.

What's closed: the actual compression recipe (sign correction, rank
protection, per-model calibration). That's the product.

Data from 8 model families (Llama 3.2, Qwen2.5, Mistral, Phi-3, TinyLlama,
OLMoE, ...) is included as reference JSON, so you can eyeball where your
model sits vs peers without having to re-run on 10 GB of weights you
don't have.

Paper 3 with the universality proof (depth-law is stable across
architectures at matched parameter count) is coming in ~4 weeks. Next
release (v1.0) adds a Shannon-efficiency *grade* — "your model is at
X% of the ceiling vs competitor Y%."

github: https://github.com/fraqtl-ai/fraqtl-diagnostic

Happy to answer questions about the measurement methodology or why we
left the compression engine closed.
