"""
Stage 2 — Binning / coarse classing.  [Built in Phase 2]

Fine-classes then coarse-classes each feature into a small number of bins with a
monotonic risk trend (special/missing bins handled explicitly). Monotonicity is
what makes the final points table stable and defensible. See
docs/methodology.md section 2.
"""
from __future__ import annotations


def fit_bins(df, config=None):
    """Learn bin edges / groupings for every feature.

    Not yet implemented — arrives in Phase 2. The signature is fixed now so the
    pipeline runner and tests can wire against it.

    Parameters
    ----------
    df : pandas.DataFrame
        The training data (features + target).
    config : dict | None
        Run configuration (binning.max_bins, min_bin_fraction, enforce_monotonic).
    """
    raise NotImplementedError("binning.fit_bins is scheduled for Phase 2.")
