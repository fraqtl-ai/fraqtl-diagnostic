# fraQtl Diagnostic

Fingerprint any transformer's compression potential — fast.

Measures per-layer:
- **γ** (stretched-exponential decay shape)
- **knee** (spectrum cutoff index)
- **k95** (directions needed for 95% of eigenvalue energy)
- **depth-law** (how shape evolves across layers)
- **compression potential** + suggested bit budgets (Shannon-based)

Works on any HuggingFace-compatible transformer. ~3–5 min on an A100 for a 1B model, longer on CPU.

## Install

```bash
pip install fraqtl-diagnostic
```

## Use

```bash
fraqtl analyze meta-llama/Llama-3.2-1B-Instruct
```

```python
from fraqtl_diagnostic import analyze
report = analyze("meta-llama/Llama-3.2-1B-Instruct")
print(report.summary())
report.to_html("llama-1b_fingerprint.html")
```

## What you get

- `*.json` — machine-readable per-layer fingerprint
- `*.html` — readable report with tables and embedded figure
- `*.png` — 4-panel figure: spectrum overlay, γ depth-law, k95/layer, compression potential

## Status

**v0.1** (current): diagnostic metrics + suggested bit budgets.
**v1.0** (coming with Paper 3, ~4 weeks): adds Shannon-efficiency grading — "your model is at X% of the theoretical ceiling vs competitors at Y%."

Same `pip install fraqtl-diagnostic` — grading is a layer on top of diagnostic v0.1, not a separate tool.

## Research

Shannon-universality research that drives these measurements is part of the fraQtl project.
Paper 3 draft forthcoming.

## Want to actually compress your model?

Diagnostic tells you the ceiling. Compression is closed-source:
[fraqtl.ai/compress](https://fraqtl.ai)

## License

Apache 2.0.
