"""
Stage 3 — Weight of Evidence & Information Value.  [Built in Phase 3]

Recodes each bin to its WOE (log-odds of good vs. bad) and computes per-feature
Information Value for selection: drop the weak (IV < 0.02), investigate the
suspiciously strong (IV > 0.5, possible leakage). See docs/methodology.md
section 3.
"""
from __future__ import annotations


def compute_woe_iv(df, config=None):
    """Compute WOE tables and IV per feature, and return the selected set.

    Not yet implemented — arrives in Phase 3.

    Parameters
    ----------
    df : pandas.DataFrame
        Binned training data (from Stage 2).
    config : dict | None
        Run configuration (woe.iv_min, woe.iv_leakage_flag).
    """
    raise NotImplementedError("woe.compute_woe_iv is scheduled for Phase 3.")
