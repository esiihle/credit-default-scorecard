"""
Stage 4 — PD model.  [Built in Phase 4]

Logistic regression on the WOE-transformed features. Because inputs are WOE, the
model is linear in the log-odds of default — the property that makes the points
table possible. Diagnostics: coefficient signs, VIF, significance. See
docs/methodology.md section 4.
"""
from __future__ import annotations


def fit_pd_model(woe_features, config=None):
    """Fit the logistic PD model on WOE features.

    Not yet implemented — arrives in Phase 4.

    Parameters
    ----------
    woe_features : object
        WOE-encoded training features + target.
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("model.fit_pd_model is scheduled for Phase 4.")
