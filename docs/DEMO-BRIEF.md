# Demo brief — fraqtl-diagnostic v0.1

Self-contained handoff for the demo-film agent. Everything needed to produce
a film-grade recording without asking follow-up questions.

---

## 1. What shipped (1-paragraph pitch)

`fraqtl-diagnostic` is an open-source CLI + Python library that measures any
HuggingFace transformer's compression potential in ~3 minutes. It reports
per-layer γ (stretched-exp spectrum shape), k95 (effective rank), knee,
depth-law regression, and a Shannon rate-distortion bit budget. It does NOT
compress the model — it tells you whether compression will work. The
companion paid engine (sign correction, per-model calibration, fused-MoE
packing) is closed IP.

---

## 2. Channels that are live

| Channel | URL | Status |
|---|---|---|
| PyPI | https://pypi.org/project/fraqtl-diagnostic/ | ✅ v0.1.0 published |
| GitHub | https://github.com/fraqtl-ai/fraqtl-diagnostic | ✅ public, Apache-2.0 |
| Colab quickstart | https://colab.research.google.com/github/fraqtl-ai/fraqtl-diagnostic/blob/main/examples/quickstart.ipynb | ✅ works on free T4 |
| Modal runner | `modal run tests/modal_try.py --model-id <hf-id>` | ✅ works on A100-40GB |

---

## 3. The hero commands (for terminal-recording shots)

### Install
```bash
pip install fraqtl-diagnostic
```

### Smallest run (A100, ~3 min) — use this if you have GPU access
```bash
fraqtl analyze Qwen/Qwen2.5-0.5B
```

### Larger run (A100, ~10 min) — the money shot
```bash
fraqtl analyze mistralai/Mistral-7B-v0.1 --n-seqs 32 --seq-len 512
```

### Money feature — fine-tune comparison
```bash
fraqtl analyze <custom-model> --compare-to mistralai/Mistral-7B-v0.1
```
Ends with a one-line verdict: `preserved / shifted / degraded / broken`.

### Modal one-liner (no local GPU needed)
```bash
modal run --detach tests/modal_try.py --model-id mistralai/Mistral-7B-v0.1 --n-seqs 32 --seq-len 512
```

---

## 4. Verified reference numbers (use these in narration, not fake ones)

All measured on Modal A100, published on the PyPI package reference table.

**Qwen/Qwen2.5-0.5B** — 24 layers, 2:44 wall:
```
mean γ           : 0.807
mean k95/dim     : 25.9%
headroom         : 0.66
suggested b/w    : 3.3 balanced / 2.8 aggressive
depth-law        : γ = −0.25·depth + 1.22,  R² = 0.52
```

**mistralai/Mistral-7B-v0.1** — 32 layers, ~10 min wall:
```
mean γ           : 0.660
mean k95/dim     : 36.8%
headroom         : 0.63
suggested b/w    : 3.4 balanced / 2.9 aggressive
```

**meta-llama/Llama-3.2-1B-Instruct** — 16 layers, 4:37 wall:
```
mean γ           : 0.818
mean k95/dim     : 21.5%
headroom         : 0.69
suggested b/w    : 3.3 balanced / 2.8 aggressive
```

---

## 5. Screens to capture

1. **Terminal install** — `pip install fraqtl-diagnostic` showing the clean install
2. **CLI help** — `fraqtl analyze --help` (shows the flag surface in one screen)
3. **Reference list** — `fraqtl list-refs` (8 bundled models shown)
4. **Live run stream** — `fraqtl analyze Qwen/Qwen2.5-0.5B` — capture the
   per-layer γ lines streaming (best visual moment; make sure it's colorful
   enough on the target terminal)
5. **Final summary block** — the `fraQtl Diagnostic vX.X.X` box with γ /
   k95 / headroom / b/w
6. **HTML report** — `open Qwen_Qwen2.5-0.5B_fingerprint.html` →
   screenshot / screen-capture the rendered page (tables + embedded PNG)
7. **The 4-panel PNG** — full-screen the saved `*_fingerprint.png`. This
   is the one for Twitter's image attachment.

---

## 6. Hero narration beats (~45 sec cut)

0:00 — "Before compressing a model, can you tell how much it'll survive? We
     open-sourced the measurement we use. Three minutes, any transformer."

0:10 — `pip install fraqtl-diagnostic && fraqtl analyze Qwen/Qwen2.5-0.5B`

0:15 — show live streaming per-layer γ lines (speed up 3×)

0:25 — pause on final summary block: "mean γ 0.81, headroom 0.69,
     suggested 3.3 bits/weight."

0:30 — open HTML report in browser, pan across the 4-panel PNG

0:35 — cut to fine-tune use case: `fraqtl analyze my/finetune --compare-to
     mistralai/Mistral-7B-v0.1` → verdict line "preserved — safe to apply
     same recipe as reference"

0:42 — "One number for 'can this ship.' pip install fraqtl-diagnostic.
     Apache 2.0."

0:45 — end card: logo + github.com/fraqtl-ai/fraqtl-diagnostic + fraqtl.ai

---

## 7. Brand assets

- **Logo:** `diagnostic-public/docs/logo.png` — dark background, purple brush circle with white Q, "fraQtl" wordmark below
- **OG card:** `diagnostic-public/docs/og-image.png` — 1280×640, used as GitHub social preview and Twitter image
- **Primary color:** `#3366cc` (accent blue)
- **Dark background:** `#0e1116`
- **Muted text:** `#808a94`
- **Callout green:** `#90b090` (used for "Open diagnostic. Closed engine." line)

---

## 8. Launch copy (drafts, ready to fire)

All three in: `diagnostic-public/docs/LAUNCH-COPY.md`
- Tweet thread (5 tweets)
- Reddit /r/LocalLLaMA post
- Hacker News "Show HN" post

Suggested firing order:
1. Twitter thread
2. Hacker News Show HN (30–60 min later, early morning Pacific)
3. Reddit /r/LocalLLaMA (same day or next, once Twitter has a few interactions)

---

## 9. Known gotchas (so the demo agent doesn't hit them on camera)

- **pyenv + lzma**: pyenv Python built without `xz` lacks `_lzma` → transformers
  import chain breaks. Fix: `brew install xz && pyenv uninstall <ver> && pyenv install <ver>`.
  Recommend recording in a **fresh Python env** (`python3.12 -m venv ~/demo`) OR
  in Colab OR in a Docker container to avoid this.
- **Python 3.14** lacks torch wheels at the moment. Stick to Python 3.10 / 3.11 / 3.12.
- **Modal non-detached**: long runs (Mistral-7B+) need `modal run --detach`, otherwise
  the local client's heartbeat dies mid-run → app stops → remote error.
- **α_tail values in output** can look like garbage (e.g. 325.78) on shallow layers where
  the spectrum crashes into the noise floor fast — this is real data, not a bug.
  γ is the headline number; α_tail is supplementary.
- **depth-law R²** is often low (0.02–0.5) on models with uniform-per-layer spectrum
  (Mistral-7B base), high (0.6+) on models with clear shallow→deep shift (fine-tunes).
  Don't say "R² = 0.8" unless that's what your recording actually shows.

---

## 10. Post-demo handoff (what to produce)

The demo agent should leave the following in the repo after filming:
- `diagnostic-public/docs/demo/` directory with:
  - `terminal_full.mp4` — un-edited terminal recording (for later re-cuts)
  - `hero_45sec.mp4` — edited 45-sec cut
  - `still_fingerprint.png` — framegrab of the 4-panel PNG on screen
  - `still_terminal_box.png` — framegrab of the summary box
- Tweet-ready MP4 + PNG, 1:1 and 16:9 variants
- Subtitle SRT file for accessibility

Upload the video files somewhere the repo can reference (GitHub Releases,
Loom, or a CDN); do NOT commit large MP4s into git.
