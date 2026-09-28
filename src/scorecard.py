"""
Stage 7 — Applicant scorer & reason codes.  [Built in Phase 7]

Given an applicant, returns their PD, their scaled score, and reason codes — the
bins contributing the largest negative points (the adverse-action / explainability
output a bank must provide). See docs/methodology.md section 7.
"""
from __future__ import annotations


def score_applicant(applicant, scorecard=None):
    """Score a single applicant and return PD, score, and reason codes.

    Not yet implemented — arrives in Phase 7.

    Parameters
    ----------
    applicant : dict
        Attribute values for one applicant.
    scorecard : object | None
        The points table from Stage 6.
    """
    raise NotImplementedError("scorecard.score_applicant is scheduled for Phase 7.")
