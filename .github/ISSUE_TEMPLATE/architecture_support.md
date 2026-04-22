---
name: New architecture / model family support
about: Diagnostic should work on a model family it currently doesn't
labels: enhancement, architecture
---

**Model family / HF ids affected:**

**Where it fails:**
(e.g. `find_target_modules` can't locate layers, or eigendecomposition blows up,
or custom attention layer with no `self_attn.o_proj`)

**Expected projection names** for this family:
- MLP projections:
- Attention projections:

**Reference code** (HF modeling file path, if known):

**Calibration notes** (needs special tokenizer, trust_remote_code, etc.):
