"""
Stage 6 — Score scaling.  [Built in Phase 6]

Maps model log-odds to a human-readable score via the standard PDO / factor /
offset transform, producing the per-bin points table — which *is* the scorecard.
See docs/methodology.md section 6.
"""
from __future__ import annotations


def build_scorecard(model, config=None):
    """Turn the fitted PD model into a scaled points table.

    Not yet implemented — arrives in Phase 6.

    Parameters
    ----------
    model : object
        The fitted PD model (coefficients + WOE tables).
    config : dict | None
        Run configuration (scaling.pdo, base_score, base_odds).
    """
    raise NotImplementedError("scaling.build_scorecard is scheduled for Phase 6.")
