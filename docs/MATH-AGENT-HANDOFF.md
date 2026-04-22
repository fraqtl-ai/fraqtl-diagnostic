# Math-agent handoff — fraqtl-diagnostic v0.1 → v0.2

**Context at time of writing:** v0.1 shipped (or shipping today) as a
measurement-only tool. Your round-1 and round-2 review comments are all
implemented — see the table in §1. This doc is your hand-off for (a) a
small optional pre-launch eyeball task and (b) the v0.2 work (~4 weeks)
that ships with Paper 3.

Read this whole file first, then skim the referenced source. Everything
lives in `fraqtl/diagnostic-public/` inside the private `fraqtl` monorepo.

---

## 1. What v0.1 contains (math-layer only)

All past review points implemented — confirm by reading the file cited.

| Your review ask | Status | File / symbol |
|---|---|---|
| γ > 1 regime tag | ✅ | `fraqtl_diagnostic/shannon.py::_classify_regime` — tags `stretched_exp` / `near_exponential` / `compressed_exp` / `super_gaussian`; falls back to `None` outside [0.05, 3] |
| Mean-γ vs depth-law consistency flag | ✅ | `fraqtl_diagnostic/depth_law.py::_consistency_flag` — computes z = `|observed_mean − (intercept + slope/2)| / σ_γ`; tags as `linear` / `nonlinear` / `noisy` |
| Per-head o_proj fit | ✅ | `fraqtl_diagnostic/api.py` — block-diagonal extraction under `if proj == "o_proj" and n_heads`; `fraqtl_diagnostic/shannon.py::aggregate_per_head` does median γ aggregation across heads |
| curve_fit bounds (γ ∈ [0.05, 3], β > 1e-6) | ✅ | `fraqtl_diagnostic/shannon.py::fit_power_exp_stretched` — `bounds=([1e-6, 0.05, -inf], [inf, 3.0, inf])` |
| Pre-fit initial guess | ✅ | same function — polyfit on x vs log_lam first, then seed curve_fit with γ=1.0 |
| R² fit_quality flag (good/moderate/poor) | ✅ | `fraqtl_diagnostic/shannon.py::_fit_quality` — poor fits (R²<0.80) have γ refused at source and are excluded from `aggregate_per_head` aggregation |
| >10% suppression rule | ✅ | `fraqtl_diagnostic/estimator.py::summarize` — fires the `gamma_summary_suppressed` flag + surfaces it in CLI/HTML |
| Measurement-only positioning (no bit-budget predictions) | ✅ | `CompressionEstimate` → `DiagnosticSummary`. `budget_bits_*` and `headroom_score` fields removed. D*(R) reported as "theoretical floor" with that caveat in the HTML |
| Plain-English TL;DR | ✅ | `fraqtl_diagnostic/estimator.py::_build_tldr` — 3-4 bullets generated from regime counts + k95 + poor-fit fraction |

## 2. Optional pre-launch: TL;DR threshold eyeball

**Time: ~10 min if you want to engage pre-launch. Not a blocker.**

`_build_tldr` in `estimator.py` picks three word-labels from numeric thresholds I (infra agent) guessed. Atlas of 8 models hasn't been run through this yet. Thresholds:

| Label | Threshold | Current heuristic |
|---|---|---|
| "Normal transformer" | `stretched_frac ≥ 0.85` AND `poor_frac < 0.15` | |
| "Mostly normal (some layers unusual)" | `stretched_frac ≥ 0.60` | |
| "Unusual" / "Mixed" | otherwise | |
| "highly compressible" | `mean_k95 < 0.20` | |
| "moderately compressible" | `mean_k95 < 0.40` | |
| "limited compression headroom" | `mean_k95 ≥ 0.40` | |

Atlas reference numbers for calibration (from `docs/research/shannon-universality/data/shannon_ceiling_*.json`):

| Model | stretched_frac | mean k95/dim | Current TL;DR label |
|---|---|---|---|
| TinyLlama-1.1B | ~1.00 | 0.13 | "Normal transformer, highly compressible" ✓ |
| Phi-3-mini (3.8B) | high | 0.22 | "Normal, moderately compressible" |
| Llama-3.2-3B | high | 0.25 | same |
| Mistral-7B | high (atlas recorded γ=0.60) | 0.37 | "Normal, moderately compressible" ✓ |
| Qwen-2.5-3B | mixed (anomaly flag in fraqtl_score) | higher | "Mixed / Unusual" ← desired |
| Qwen-2.5-7B | γ=0.931 (near-exp) | ~0.45 | might land "limited compression headroom" |
| Qwen-2.5-14B | γ=0.86 | | |
| Qwen3-8B | 0.93 stretched_frac, 0.50 k95 | | "Normal transformer, limited compression headroom" (verified v0.1.1 smoke) |

**Ask:** run v0.1.1 on those 8 atlas models, sanity-check the TL;DR label each one gets, and propose 3 threshold changes if needed. If you think the labels are fine, ship is clean.

Run command (Modal, free tier has A100):
```bash
cd fraqtl/diagnostic-public
modal run --detach tests/modal_try.py --model-id <hf-id>
```

Results auto-land on the `fraqtl-hf-cache` Modal volume under
`/cache/fraqtl-results/diagnostic-smoke/*_fingerprint.{json,html,png}`.

---

## 3. v0.2 — Paper 3 launch (~4 weeks, post-ship)

This is the real ask. v0.1 ships today as measurement; v0.2 adds
**calibrated γ → ΔPPL prediction + Shannon efficiency grade** gated on
Paper 3. Your critical-path items:

### 3.1 Expand the γ-calibration atlas to n ≥ 15

**Why:** the n=8 atlas fails LOO-CV at all bit rates (see
`docs/research/shannon-universality/results-2026-04-22/GAMMA-CALIBRATION.md`).
Training R²=0.53 at INT4 is 3-param overfit on 8 points; γ-coef CI
includes zero.

**What to add:** 2 more MoE (already have OLMoE — need 2 more), 2 more
14B+ (already Qwen2.5-14B — add Llama-3.1-70B, Mixtral-8x7B), 2 more
500M–2B (already TinyLlama 1.1B — add Qwen2.5-1.5B, SmolLM2-1.7B).
Target: 15 models, 3 seeds each, uniform INT4/INT3/INT2 per the existing
script.

**Reproduction path:**
```
experiments/exp_GAMMA_CALIBRATION.py   ← existing
data/gamma_cal_<model>.json            ← per-model output
experiments/analysis_gamma_calibration.py  ← LOO-CV + bootstrap
```

### 3.2 Replace log-linear fit with physically-motivated form

**Why:** `GAMMA-CALIBRATION.md §6 long` — rate-distortion on a stretched-
exp spectrum gives a closed form involving `Γ(1 + 1/γ)` (gamma-function,
not γ-linear). The current OLS fit ignores this structure.

**Ask:** derive the analytical ΔPPL(γ, bits) form from
`math.md §3` and verify it on the expanded atlas (3.1).

### 3.3 3-seed variance per measurement

**Why:** establishes whether the fit error is fit-limited or noise-limited.
If fit variance < seed variance, we're signal-limited regardless of
atlas size.

**Ask:** re-run all n≥15 atlas models with 3 seeds each, report seed-vs-
model variance. Ship the variance floor alongside the calibration.

### 3.4 Shannon efficiency grader

**Why:** v1.0 headline feature = "your model is at X% of the Shannon
ceiling vs competitor Y%." Needs calibrated `fraQtl achievable ΔPPL at
budget R` number and a baseline from bnb/AWQ/GPTQ at the same budget on
the same model.

**Ask:** define the grading formula. Current stub:
```
fraqtl_diagnostic/grader.py  ← raises NotImplementedError in v0.1
```

Target output (also in `diagnostic-public/docs/LAUNCH-COPY.md`):
```
COMPRESSION OUTCOMES (predicted from γ fingerprint, ±σ from atlas validation):
  Target 4 b/w:
    bnb NF4:        +0.15 PPL ± 0.08    (20% of ceiling)
    AWQ 4-bit:      +0.12 PPL ± 0.09    (25%)
    fraQtl MLP:     +0.04 PPL ± 0.02    (75%) ← ours
    Shannon floor:  +0.03 PPL (theoretical minimum)
```

---

## 4. File map

All paths are relative to `fraqtl/diagnostic-public/`.

### Source (what you own / need to review)
```
fraqtl_diagnostic/
├── __init__.py          public API exports
├── version.py           0.1.1
├── shannon.py           γ fit, regime tag, fit_quality, aggregate_per_head  ← your math
├── depth_law.py         linear γ(depth) fit + consistency flag             ← your math
├── estimator.py         DiagnosticSummary + TL;DR builder                  ← review §2
├── spectrum.py          Hessian capture + eigendecompose                   ← infra, not math
├── _model_io.py         HF load + target-module resolution                 ← infra
├── api.py               orchestrator; per-head o_proj dispatch lives here  ← your math
├── report.py            JSON / HTML / PNG renderers                        ← infra
├── cli.py               `fraqtl analyze` entry                             ← infra
├── compare.py           --compare-to reference delta + verdict             ← infra (thresholds)
├── grader.py            v1.0 stub                                          ← your v0.2 target
└── references/
    ├── __init__.py
    └── stock_models_v1.json  ← 8-model atlas bundled for --compare-to
```

### Tests
```
tests/
├── test_math_modules.py  pure-math unit tests (synthetic eigenvalues, no HF)
├── test_compare.py       compare/verdict unit tests
└── modal_try.py          one-liner HF smoke on any model (used for atlas runs)
```

### Research / calibration data (NOT in public repo — in `Compression/`)
```
~/Documents/Projects/Zenalyze/Compression/zenalyze-compression-/
├── docs/research/shannon-universality/
│   ├── data/shannon_ceiling_<model>.json  ← per-model full diagnostic dump (atlas)
│   ├── data/gamma_cal_<model>.json         ← INT4/INT3/INT2 ΔPPL per model
│   ├── results-2026-04-22/GAMMA-CALIBRATION.md  ← the +23394 write-up
│   └── experiments/analysis_shannon.py     ← original (pre-public) analysis
```

### Key reference documents to read before v0.2
- `docs/research/shannon-universality/results-2026-04-22/GAMMA-CALIBRATION.md` — the n=8 LOO failure + path forward
- `docs/MATH-HOLDUP-ANALYSIS-2026-04-14.md` (if still there) — Shannon-ceiling derivation context
- `docs/research/SCIENCE-DIAGNOSTIC-2026-04-19.md` — original Hurwitz-zeta framing

---

## 5. Your kickoff prompt

If you're reading this fresh:

```
You are the math agent for fraQtl. v0.1 of fraqtl-diagnostic just shipped
as a measurement-only tool (pip install fraqtl-diagnostic). All of your
round-1 and round-2 review comments are implemented — see
diagnostic-public/docs/MATH-AGENT-HANDOFF.md §1 for the status table.

Your optional pre-launch task: §2 (atlas-walk the TL;DR thresholds,
~10 min).

Your critical-path work: §3, leading to v0.2 (4 weeks) which ships with
Paper 3 and adds calibrated γ → ΔPPL prediction + Shannon efficiency
grader. The n=8 atlas fails LOO-CV (see GAMMA-CALIBRATION.md); target
n ≥ 15 with 3 seeds per model.

All files live in ~/Documents/Projects/Zenalyze/fraqtl/diagnostic-public/
(and the calibration data in ~/Documents/Projects/Zenalyze/Compression/).
Read MATH-AGENT-HANDOFF.md §4 for the full map.

Start by confirming the v0.1 math in §1 is to your satisfaction — if
anything's wrong, flag before launch. Otherwise move to §2 or §3
depending on bandwidth.
```
