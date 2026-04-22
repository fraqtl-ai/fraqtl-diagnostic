"""fraQtl Diagnostic — fingerprint any transformer's compression potential.

Public API:
    analyze(model_id, *, n_seqs=32, seq_len=512, projections=('down_proj', 'o_proj'),
            out_dir=None, device=None) -> DiagnosticReport
    DiagnosticReport.to_json(path)
    DiagnosticReport.to_html(path)
    DiagnosticReport.to_png(path)

Typical use:
    >>> from fraqtl_diagnostic import analyze
    >>> report = analyze('meta-llama/Llama-3.2-1B-Instruct')
    >>> report.summary()
    >>> report.to_html('llama-3.2-1b_fingerprint.html')
"""
from .api import analyze, DiagnosticReport
from .compare import compare_to_reference, ComparisonResult
from .estimator import DiagnosticSummary, summarize
from .references import list_reference_models
from .version import __version__

__all__ = [
    "analyze",
    "DiagnosticReport",
    "DiagnosticSummary",
    "summarize",
    "compare_to_reference",
    "ComparisonResult",
    "list_reference_models",
    "__version__",
]
