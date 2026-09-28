"""
Stage 8 — Benchmark & the trade-off.  [Built in Phase 8]

Trains a gradient-boosted model on the same data and compares it to the scorecard
on KS / Gini / AUC, then discusses the accuracy-vs-interpretability trade-off —
the core interview talking point. See docs/methodology.md section 8.
"""
from __future__ import annotations


def run_benchmark(df, config=None):
    """Train the gradient-boosted benchmark and compare it to the scorecard.

    Not yet implemented — arrives in Phase 8.

    Parameters
    ----------
    df : pandas.DataFrame
        The modelling data.
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("benchmark.run_benchmark is scheduled for Phase 8.")
