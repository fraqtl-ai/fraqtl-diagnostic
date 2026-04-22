"""Shannon-efficiency grader — v1.0 (4-week ETA).

v0.1 stub: grading requires the depth-law math formalized in Paper 3, which
has not yet been publicly released. Calling `grade(...)` raises NotImplementedError
with an ETA and a link to the paper once available.

When v1.0 ships, grade() will return a ShannonGrade with:
  - efficiency_pct : % of the Shannon ceiling achieved by the model's weights
                     vs a universal depth-law reference
  - peer_comparison: efficiency_pct of 5–10 reference models at similar parameter
                     count, so you can see "where this model stands"
"""
from __future__ import annotations


class ShannonGrade:
    """Placeholder — populated in v1.0 alongside Paper 3."""


def grade(*args, **kwargs):
    raise NotImplementedError(
        "Shannon efficiency grading is part of fraqtl-diagnostic v1.0, shipping "
        "with Paper 3 (~4 weeks). v0.1 reports raw fingerprint metrics (γ, knee, "
        "k95, depth-law, compression potential) which you can interpret directly "
        "via the HTML report."
    )
