"""Command-line entry: `fraqtl analyze <model>`."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .api import analyze
from .version import __version__


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fraqtl",
        description="fraQtl Diagnostic — fingerprint any transformer's compression potential.",
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

    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
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

    # write reports
    safe = args.model_id.replace("/", "_")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{safe}_fingerprint.json"
    html_path = out / f"{safe}_fingerprint.html"
    png_path = out / f"{safe}_fingerprint.png"

    report.to_json(json_path)
    report.to_png(png_path)
    report.to_html(html_path)

    print()
    print(report.summary())
    print()
    print(f"JSON : {json_path}")
    print(f"HTML : {html_path}")
    print(f"PNG  : {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
