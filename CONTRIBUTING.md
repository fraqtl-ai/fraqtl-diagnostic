# Contributing to fraQtl Diagnostic

Thanks for your interest. This tool exists so more people can inspect their
models' compression potential — contributions that extend it to new model
families, improve reporting, or tighten the math are welcome.

## What's in scope

- Support for new architectures (non-standard attention, MoE variants,
  MLP activations beyond SwiGLU / GeGLU)
- Better spectrum fitting (alternative families, robustness to noise floor)
- Faster Hessian capture (flash-attention hooks, fp8 accumulators)
- Clearer reports (HTML, PDF, plots)
- Additional calibration corpora beyond wikitext-2
- Tests (especially model-family regressions)

## What's out of scope

- Compression recipes (sign correction, rank protection, per-model
  calibration) — the diagnostic tells you the ceiling; the engine that
  pushes toward it is the closed half of the product.
- Anything that would expose closed IP (quantizer internals, recipe
  tables, customer data).

## Dev workflow

```bash
git clone https://github.com/fraqtl-ai/fraqtl-diagnostic
cd fraqtl-diagnostic
pip install -e '.[dev]'
pytest tests/                   # pure-math smokes, no HF downloads
```

## PRs

- Keep PRs focused. One feature or one bug per PR.
- Add or update a test when behavior changes.
- If you add a new architecture, include a short smoke run in
  `tests/modal_try.py` so we can reproduce.
- Use a feature branch from `main`. Do not push to `main`.

## Issues

Bug reports — include:
- model id (HuggingFace)
- `fraqtl --version`
- Python / torch / transformers versions
- full stack trace

Feature requests — include:
- what you're trying to measure
- why the current metrics don't cover it
- reference papers or prior art if any
