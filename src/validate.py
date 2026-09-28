"""
Stage 5 — Validation.  [Built in Phase 5]

The credibility layer: KS, Gini/AUC, ROC, calibration, PSI (stability), and
rank-ordering by score band — all on held-out data. See docs/methodology.md
section 5.
"""
from __future__ import annotations


def evaluate(model, config=None):
    """Compute the validation metrics for a fitted PD model.

    Not yet implemented — arrives in Phase 5.

    Parameters
    ----------
    model : object
        The fitted PD model plus the data needed to score held-out accounts.
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("validate.evaluate is scheduled for Phase 5.")
