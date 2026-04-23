# fraqtl-diagnostic — full integration gameplan

**Last updated:** 2026-04-22
**Current version:** v0.1.1 (live on PyPI, GitHub public)

This document maps every known outstanding item to a target release, an
owner, and an estimate. It's the source of truth for "what ships when."

---

## Release tiers

| Tier | Target date | Theme | Ship criteria |
|---|---|---|---|
| v0.1.1 | ✅ LIVE (2026-04-22) | Measurement-only launch | PyPI up, repo public, paired hero demo |
| v0.1.2 | 1 week | Polish from community feedback | Issues from launch addressed |
| v0.1.3 | 2 weeks | GGUF support | r/LocalLLaMA users can install & run on GGUF |
| v0.2 | 4 weeks | Paper 3 + calibrated PPL prediction | Atlas n≥15, 3-seed variance, LOO-CV positive |
| v1.0 | 4-6 weeks | Shannon-efficiency grader | Competitor comparison (bnb/AWQ/GPTQ), efficiency % |
| v1.1+ | 2-3 months | Format expansion, community | ONNX, raw .pt, expanded arch coverage |

---

## v0.1.2 — polish (1 week)

Non-blocker fixes landed from launch feedback. Fire issues as community
requests and close them fast.

| # | Item | Owner | Est | Issue? |
|---|---|---|---|---|
| 1 | Near-exp branch in TL;DR label | infra | 30 min | file during week |
| 2 | Tweak `k95<0.20 = "highly compressible"` threshold based on atlas walk | math + infra | 1 hr | depends on §v0.1.2.2 below |
| 3 | `Qwen/Qwen2.5-3B` raises "anomaly detected — 4 layers outside" at TL;DR level | infra | 45 min | currently buried as regime count |
| 4 | GPT-2 arch-family support (c_attn / c_fc / c_proj) | infra | 4 hr | launch issue #1 predictable |
| 5 | Phi-3 fused `qkv_proj` + `gate_up_proj` handling | infra | 3 hr | graceful skip works today; full unpack for v0.1.2 |
| 6 | Sharded safetensors smoke (70B class) | infra | 1 hr | verify device_map="auto" actually works |
| 7 | Fix DejaVu Sans Mono missing `💥` emoji glyph in tweet image templates | infra | 15 min | launch artifact only |

**Gate:** every item lands as a patch release. No breaking changes.

### v0.1.2.2 — Atlas walk sanity check

**Owner:** math agent (or infra if math is heads-down on Paper 3).
**Est:** 30 min wall.

Run v0.1.1 on 8 reference atlas models, confirm TL;DR labels are
intuitive. Flag any mismatch. This is the pre-work for item #2 above.

```bash
for m in TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/Phi-3-mini-4k-instruct \
         meta-llama/Llama-3.2-3B mistralai/Mistral-7B-v0.1 \
         allenai/OLMoE-1B-7B-0924 Qwen/Qwen2.5-3B \
         Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B; do
  modal run --detach tests/modal_try.py --model-id "$m"
done
```

---

## v0.1.3 — GGUF support (2 weeks)

**Why first format:** r/LocalLLaMA's adoption gate. Launching without
GGUF leaves a huge audience stuck.

**Approach:**
1. Detect `.gguf` extension / GGUF magic bytes in model path
2. Load via `llama-cpp-python` or `gguf-py` → extract fp16 tensors
3. Build synthetic HF-style config on the fly (mapping GGUF metadata →
   `AutoConfig`)
4. Feed into existing `capture_single_hessian` pipeline unchanged

**Out of scope for v0.1.3:** running the forward pass via llama.cpp's
kernel. We rebuild the model on the HF side using the loaded tensors so
the Hessian-capture hook logic doesn't change.

**Est:** 2 days infra work + 1 day smoke testing on popular GGUF models.

**Owner:** infra.

**Gate:** `fraqtl analyze TheBloke/Mistral-7B-Instruct-v0.2-GGUF` produces
the same numerical fingerprint as `fraqtl analyze mistralai/Mistral-7B-Instruct-v0.2`
(within fp16 precision), confirming conversion is lossless on the spectrum
side.

---

## v0.2 — calibrated PPL prediction (4 weeks, with Paper 3)

Critical path for the compression-product conversion story. Shannon
efficiency grade depends on this landing.

### v0.2.1 — Atlas expansion (math agent)

**Why:** n=8 atlas fails LOO-CV at all bit rates (see
`docs/research/shannon-universality/results-2026-04-22/GAMMA-CALIBRATION.md`).

**Ask:** add 7 models to reach n=15:
- 2 more MoE (currently: OLMoE only). Add Mixtral-8x7B + DBRX
- 2 more 14B+ (currently: Qwen2.5-14B only). Add Llama-3.1-70B + Qwen2.5-32B
- 2 more 500M–2B (currently: TinyLlama only). Add Qwen2.5-1.5B + SmolLM2-1.7B
- 1 non-Llama-family dense: add Falcon-7B or Gemma-7B

**Est:** 2 days of Modal A100 time. Math agent runs `exp_GAMMA_CALIBRATION.py`
on each, pushes JSONs to the research data dir.

**Owner:** math agent.

**Gate:** LOO-CV R² > 0 on at least one of the three predictors (γ-only,
size-only, γ+size) at INT4.

### v0.2.2 — 3-seed variance baseline (math agent)

**Why:** tells us whether the remaining fit error is noise-limited or
fit-limited.

**Ask:** re-run all n≥15 atlas models with 3 different calibration seeds,
report seed-vs-model variance.

**Est:** 3 days of Modal time (3× the atlas run).

**Owner:** math agent.

**Gate:** if seed variance < fit variance, push harder on §v0.2.3; if
seed variance ≈ fit variance, we're signal-limited and need more models.

### v0.2.3 — Γ-function physical form (math agent)

**Why:** reviewer flagged log-linear OLS as structurally wrong. The real
form involves `Γ(1 + 1/γ)` from rate-distortion on the stretched-exp
spectrum.

**Ask:** derive from `math.md §3` and verify on the expanded atlas.

**Est:** 1 week of math agent time, pure analysis.

**Owner:** math agent.

**Gate:** Γ-function fit has lower LOO-CV error than log-linear baseline
on the n=15 atlas.

### v0.2.4 — Predicted PPL output in diagnostic (infra)

**Why:** once v0.2.3 lands, surface the prediction in the tool output.

**Ask:** new section in TL;DR:
```
━━ Predicted INT-N outcomes (from γ, ± atlas σ) ━━
  INT4  : +0.18 ± 0.07 PPL  (competitive)
  INT3  : +1.2  ± 0.4  PPL  (tight)
  INT2  : broken (γ-class predicts collapse)
```

**Est:** 1 day infra work + HTML/JSON schema additions.

**Owner:** infra, blocked on math agent delivery.

**Gate:** predictions within atlas σ of actual compression runs on 3+
held-out models.

---

## v1.0 — Shannon efficiency grader (with Paper 3 launch)

**Scope:** the "your model is at X% of the ceiling vs competitor Y%"
headline feature.

### v1.0.1 — Competitor baseline matrix (BD + math)

**Why:** efficiency % is relative. Needs bnb / AWQ / GPTQ baselines at
matched bit budgets.

**Ask:** run bnb NF4, AWQ-4bit, GPTQ-4bit on each atlas model; record
PPL delta. Produce a per-model reference table.

**Est:** 1 week, Modal-heavy.

**Owner:** BD agent for run coordination; math agent for eval methodology.

### v1.0.2 — `fraqtl_grade` module (infra)

**Why:** the grader is the CTA — "you're at 40% of ceiling, our engine
gets you to 75%."

**Output format** (from `LAUNCH-COPY.md` v0.2 section):
```
COMPRESSION OUTCOMES (predicted from γ, ± σ from atlas):
  Target 4 b/w:
    bnb NF4:        +0.15 PPL ± 0.08    (20% of ceiling)
    AWQ 4-bit:      +0.12 PPL ± 0.09    (25%)
    fraQtl MLP:     +0.04 PPL ± 0.02    (75%)
    Shannon floor:  +0.03 PPL
```

**Est:** 2 days infra (code), plus gated on v1.0.1.

**Owner:** infra, blocked on competitor baselines.

**Gate:** `--compare-to <competitor>` flag returns sensible efficiency %
on the atlas.

### v1.0.3 — Paper 3 draft + submission (math agent, BD)

**Why:** v1.0 ships WITH Paper 3. Paper provides academic credibility;
tool is the reference implementation.

**Est:** 2 weeks of math agent writing, BD for cover letter + submission
logistics.

**Gate:** Paper draft on arXiv + submission to NeurIPS/ICML.

---

## v1.1+ — Format expansion (2-3 months)

After v1.0 lands. Not gating launches.

### v1.1.1 — Raw PyTorch `.pt` + `--arch-hint`

**Ask:** user passes `--arch-hint llama3` to tell us the architecture
when there's no `config.json`. We construct the config from a bundled
template and load weights.

**Est:** 1 day.

**Owner:** infra.

### v1.1.2 — ONNX support

**Why:** enterprise users (inference-runtime-first shops).

**Approach:** walk ONNX graph, identify projection nodes by pattern
match, extract their weight initializers, reconstruct per-layer Hessian
capture on the PyTorch side.

**Est:** 3-5 days. Non-trivial, needs ONNX expertise.

**Owner:** infra + community contributor (good first issue).

### v1.1.3 — Additional arch families

- **GLM-4 / ChatGLM:** non-standard layer structure, custom attention
- **Mamba / state-space models:** no attention to analyze; different diagnostic entirely
- **Phi-3.5 moe / Mixtral-style MoE beyond OLMoE:** fused expert variants

Each is ~1-2 days, prioritized by community request count.

---

## Ongoing — community + BD

### Community / OSS hygiene

| Item | Target | Owner |
|---|---|---|
| GitHub issue triage rotation | weekly | infra |
| CONTRIBUTING.md keeps current | as needed | infra |
| "Try on your model" showcase page on fraqtl.ai | v0.2 launch | BD |
| Reference-model atlas page (public γ / k95 / regime per model) | v0.2 launch | BD + math |
| Discord / Slack community | v0.2 launch | BD |

### BD / conversion hooks

| Item | Target | Owner |
|---|---|---|
| fraqtl.ai landing page with live diagnostic demo | v0.2 launch | BD |
| "Book a compression eval" form fed from diagnostic output | v1.0 launch | BD |
| Enterprise SLA for sharded repo scale | post-v1.0 | BD |
| Pilot partnership pipeline (2-3 customers) | v1.0–v1.1 | BD |

---

## Dependency graph (critical path for v1.0)

```
v0.1.1 (live)
  ↓
v0.1.2 (polish, 1 week)
  ↓
v0.1.3 (GGUF, 2 weeks)            ← parallel with math work
  │
  │     v0.2.1 atlas expansion ────┐
  │                                 │
  │     v0.2.2 3-seed variance ────┼──→ v0.2.3 Γ-function fit ──→ v0.2.4 prediction output
  │                                 │
  │     v1.0.1 competitor matrix ──┘
  ↓
  v0.2 (predictor ships, 4 weeks)
  ↓
  v1.0 (grader + Paper 3, 6 weeks total)
  ↓
  v1.1+ (format expansion, community growth)
```

**Critical path:** math agent's §v0.2.1 → §v0.2.3 is the bottleneck. 4
weeks if they have bandwidth; more if Paper 3 writing steals their
cycles.

**Parallelizable:** infra's GGUF work (§v0.1.3) runs in parallel with
math's atlas expansion.

---

## One-week look-ahead (this week)

Priority order if capacity is limited:

1. **(Day 1)** Monitor launch feedback, triage issues
2. **(Day 1)** File the v0.1.2 polish items as GitHub issues
3. **(Day 1-2)** Atlas walk sanity check on 8 reference models (§v0.1.2.2)
4. **(Day 2-3)** Start v0.1.3 GGUF scaffolding (loader detection + gguf-py integration)
5. **(Day 3-5)** Math agent kicks off atlas expansion (§v0.2.1) — 7 new models
6. **(Day 5-7)** Ship v0.1.2 polish release

Anything that lands before v0.1.3 is a bonus. v0.1.3 (GGUF) is the next
real release milestone.

---

## Open questions / decisions needed

| # | Question | Needs decision from | Blocker? |
|---|---|---|---|
| 1 | Priority: GGUF first vs. `--compare-to` enhancements first | user | v0.1.2 timing |
| 2 | Atlas expansion budget (7 models × 3 seeds = ~9 hours A100) | user | v0.2.1 fire date |
| 3 | Competitor baseline scope: 3 methods (bnb/AWQ/GPTQ) or 5? | BD | v1.0 timeline |
| 4 | Paper 3 venue: arXiv-only vs NeurIPS/ICML submission | math + user | v1.0 marketing |
| 5 | Community channel: Discord vs Slack vs GitHub Discussions | BD | v0.2 launch |

---

## File/commit state at time of this doc

- Public repo: `github.com/fraqtl-ai/fraqtl-diagnostic`
- Private monorepo: `fraqtl/diagnostic-public/`
- HEAD: `51ca264` (quickstart notebook using PyPI install)
- PyPI: `fraqtl-diagnostic 0.1.1` live
- Outstanding local change: this file
