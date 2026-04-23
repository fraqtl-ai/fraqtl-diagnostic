"""Model + calibration data loading."""
from __future__ import annotations
from typing import Iterable, Sequence

import sys

import torch


_LZMA_HELP = """
Your Python installation is missing the _lzma C extension. This is a
common pyenv-on-macOS issue: pyenv built Python without the xz library.

Fix:
  brew install xz
  pyenv uninstall <your version>
  pyenv install <your version>
  # then reopen your terminal

Alternatively, use a different Python (brew's python@3.12, conda, or Docker):
  brew install python@3.12
  /opt/homebrew/bin/python3.12 -m venv ~/fraqtl-try
  source ~/fraqtl-try/bin/activate
  pip install fraqtl-diagnostic
"""


def _import_transformers():
    """Import transformers with a helpful error for the pyenv-lzma case."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as e:
        if "_lzma" in str(e):
            print(_LZMA_HELP, file=sys.stderr)
            raise ModuleNotFoundError(
                "Missing _lzma C extension in this Python. See stderr for fix."
            ) from e
        raise


def load_model(model_id: str, *, trust_remote_code: bool = False):
    """Load HF model + tokenizer in fp16. Returns (model, tokenizer)."""
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tok


def load_wikitext_calibration(
    tokenizer, n_seqs: int = 32, seq_len: int = 512
) -> torch.Tensor:
    """Return [N, S] token tensor from wikitext-2-raw-v1 train split.

    Picks the first N sequences with >=seq_len tokens and >=200 chars.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    chosen = []
    for t in ds["text"]:
        if len(t) < 200:
            continue
        ids = tokenizer(
            t, return_tensors="pt", truncation=True, max_length=seq_len
        ).input_ids[0]
        if ids.size(0) >= seq_len:
            chosen.append(ids[:seq_len])
        if len(chosen) >= n_seqs:
            break
    return torch.stack(chosen)


def find_target_modules(
    model, projections: Sequence[str] = ("down_proj", "o_proj"),
    layer_limit: int | None = None,
) -> dict:
    """Locate target projection modules across transformer layers.

    Returns {layer_idx: {proj_name: module}}. Supports both standard
    (model.model.layers) and wrapped (model.model.language_model.layers) variants.
    """
    for attr_path in (
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
    ):
        try:
            layers = attr_path(model)
            break
        except AttributeError:
            layers = None
    if layers is None:
        raise RuntimeError(
            "Could not locate transformer layers — tried model.model.layers and "
            "model.model.language_model.layers. File an issue with your model's class name."
        )

    out: dict = {}
    missing_per_proj: dict[str, int] = {p: 0 for p in projections}
    for i, block in enumerate(layers):
        out[i] = {}
        for proj in projections:
            proj = proj.strip()
            mod = None
            if proj in ("down_proj", "gate_proj", "up_proj", "gate_up_proj"):
                mlp = getattr(block, "mlp", None)
                if mlp is not None:
                    mod = getattr(mlp, proj, None)
            elif proj in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"):
                attn = getattr(block, "self_attn", None)
                if attn is not None:
                    mod = getattr(attn, proj, None)
            if mod is not None:
                out[i][proj] = mod
            else:
                missing_per_proj[proj] += 1
    if layer_limit:
        out = {i: v for i, v in out.items() if i < layer_limit}

    # Fail fast if NONE of the requested projections exist — common when a user
    # points at a model with a non-standard architecture (e.g. GPT-2 uses c_attn).
    covered = {p for layer in out.values() for p in layer}
    if not covered:
        raise RuntimeError(
            f"None of the requested projections {tuple(projections)} were found on "
            f"any layer. Check the model class — this tool currently supports Llama / "
            f"Qwen / Mistral / Mixtral / TinyLlama / OLMoE / Phi-3-style naming. "
            f"GPT-2 (c_attn / c_fc / c_proj) not yet supported."
        )
    # Soft note for partial coverage (e.g. Phi-3 has no separate q/k/v_proj).
    skipped = [p for p, n in missing_per_proj.items() if n == len(layers)]
    if skipped:
        import warnings
        warnings.warn(
            f"projections {skipped} not present on this model — skipped. "
            f"Diagnostic continues on: {sorted(covered)}.",
            stacklevel=2,
        )
    return out
