"""Command-line entry: `fraqtl analyze <model>`."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .api import analyze
from .version import __version__


_EPILOG = """
examples:

  # smallest / fastest smoke (~3 min on A100, ~5 min on free Colab T4)
  fraqtl analyze Qwen/Qwen2.5-0.5B

  # real run on a production-size model
  fraqtl analyze mistralai/Mistral-7B-v0.1 --n-seqs 32 --seq-len 512

  # fine-tune vs base model verdict (preserved / shifted / degraded / broken)
  fraqtl analyze my-org/my-finetune --compare-to mistralai/Mistral-7B-v0.1

  # list bundled reference models
  fraqtl list-refs

outputs:
  *_fingerprint.json  machine-readable per-layer data
  *_fingerprint.html  readable report with tables + embedded figure
  *_fingerprint.png   4-panel figure (spectrum, γ depth-law, k95, summary)

supported inputs:
  HuggingFace model ids (e.g. "mistralai/Mistral-7B-v0.1") or local paths to
  HF-format checkpoints (containing config.json + safetensors).

  Not supported: GGUF, ONNX, raw .pt files (yet — see gameplan).

troubleshooting:
  - ModuleNotFoundError: _lzma → pyenv built Python without xz.
    Fix: brew install xz && pyenv uninstall <ver> && pyenv install <ver>
  - Out of memory → lower --n-seqs (default 32) or --seq-len (default 512),
    or pick a smaller model.

docs:
  github.com/fraqtl-ai/fraqtl-diagnostic
"""


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fraqtl",
        description="fraQtl Diagnostic — fingerprint any transformer's compression potential.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"fraqtl-diagnostic {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Analyze a model and write JSON + HTML + PNG reports.")
    a.add_argument("model_id", help="HuggingFace model id or local path")
    a.add_argument("--out-dir", default=".", help="Output directory (default: .)")
    a.add_argument("--n-seqs", type=int, default=32, help="Calibration sequences (default: 32)")
    a.add_argument("--seq-len", type=int, default=512, help="Tokens per sequence (default: 512)")
    a.add_argument("--projections", default="down_proj,o_proj",
                   help="Comma-separated projections (default: down_proj,o_proj)")
    a.add_argument("--layer-limit", type=int, default=None,
                   help="Only profile first N layers (for smoke runs)")
    a.add_argument("--trust-remote-code", action="store_true")
    a.add_argument("--quiet", action="store_true", help="Suppress per-layer progress")
    a.add_argument("--compare-to", default=None, metavar="REFERENCE_ID",
                   help="Compare against a bundled reference model and emit a verdict. "
                        "Run `fraqtl list-refs` to see bundled references.")

    kv = sub.add_parser("kv-audit", help="Measure KV-cache compressibility: damage law, risk map, tuning verdict, capacity.")
    kv.add_argument("model_id", help="HuggingFace model id or local path")
    kv.add_argument("--seq-len", type=int, default=1024)
    kv.add_argument("--n-seqs", type=int, default=6)
    kv.add_argument("--hbm-gb", type=float, default=80.0)
    kv.add_argument("--context", type=int, default=131072)
    kv.add_argument("--out-dir", default="reports")
    kv.add_argument("--trust-remote-code", action="store_true")
    kv.add_argument("--cleanup-cache", action="store_true", help="Delete the downloaded model from the HF cache after the audit (reclaims disk space).")

    sub.add_parser("list-refs", help="List bundled reference models.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    if args.command == "list-refs":
        from .references import list_reference_models, calibration_description
        print("Bundled reference models:")
        for m in list_reference_models():
            print(f"  {m}")
        print()
        print(f"Calibration: {calibration_description()}")
        return 0

    if args.command == "kv-audit":
        from pathlib import Path as _P
        from .kv_audit import kv_audit_to_files, run_kv_audit
        result = run_kv_audit(
            args.model_id, seq_len=args.seq_len, n_seqs=args.n_seqs,
            hbm_gb=args.hbm_gb, context=args.context,
            trust_remote_code=args.trust_remote_code,
        )
        j, md = kv_audit_to_files(result, _P(args.out_dir))
        print(f"wrote {j}")
        print(f"wrote {md}")
        if args.cleanup_cache:
            from .kv_audit import evict_model_cache
            gone = evict_model_cache(args.model_id)
            print(f"cleaned model cache: {gone}" if gone else "cache dir not found (nothing deleted)")
        return 0

    if args.command != "analyze":
        return 2

    projections = tuple(p.strip() for p in args.projections.split(",") if p.strip())
    report = analyze(
        args.model_id,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        projections=projections,
        layer_limit=args.layer_limit,
        trust_remote_code=args.trust_remote_code,
        progress=not args.quiet,
    )

    comparison = None
    if args.compare_to:
        from .compare import compare_to_reference
        comparison = compare_to_reference(report, args.compare_to)

    # write reports
    safe = args.model_id.replace("/", "_")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{safe}_fingerprint.json"
    html_path = out / f"{safe}_fingerprint.html"
    png_path = out / f"{safe}_fingerprint.png"

    report.to_json(json_path)
    report.to_png(png_path)
    report.to_html(html_path, comparison=comparison)

    print()
    print(report.summary())
    if comparison is not None:
        print()
        print(comparison.summary())
    print()
    print(f"JSON : {json_path}")
    print(f"HTML : {html_path}")
    print(f"PNG  : {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
