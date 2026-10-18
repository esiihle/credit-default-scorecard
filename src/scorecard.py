r"""
Stage 7 — Applicant scorer & reason codes.  [Built in Phase 7, 18 Oct 2026]

Given one applicant, return their score, their probability of default, and
**reason codes** — the characteristics where they lost the most points.

Why reason codes are a legal artefact, not a nice-to-have
---------------------------------------------------------
A declined applicant is entitled to be told *why*. In the United States that is
the adverse-action notice required by ECOA / Regulation B; in South Africa the
National Credit Act obliges a lender to give reasons on request. "The model said
no" is not an answer in either jurisdiction.

This is the clearest practical reason the whole project uses a points table
rather than a gradient-boosted model. Reason codes here are not an
approximation, a surrogate model, or a SHAP value — they are **read directly off
the arithmetic that produced the decision**. The sum of the points *is* the
score, so the characteristic that cost the most points *is* the reason.

The reference point: lost points, not low points
------------------------------------------------
A reason code must answer "why did this applicant score lower than they could
have", so each characteristic is measured against the **best bin available in
that characteristic**:

    points_lost = best_points_for_this_characteristic - applicant's_points

Measuring against the mean would be wrong in a specific and unfair way: an
applicant in a slightly-below-average bin of a strongly predictive
characteristic would be told about that instead of the characteristic where they
actually lost 40 points. The regulatory question is where the points went, and
the best attainable bin is the only reference that answers it.

A characteristic where the applicant already holds the best bin has lost nothing
and can never be a reason code. That falls out of the arithmetic rather than
needing a special case — which is the sort of thing worth checking, so
`test_best_possible_applicant_has_no_reason_codes` does.

See docs/methodology.md section 7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import binning, scaling


class ScorerError(Exception):
    """Raised when an applicant cannot be scored."""


class FairLendingError(Exception):
    """Raised when an explanation would cite a protected characteristic.

    A distinct exception type because this is not a bug in the data or the
    arithmetic — the code is working correctly. It means the MODEL contains a
    characteristic that may not lawfully drive a credit decision, and the only
    real fix is upstream: remove it from the feature set and refit.
    """


# ===========================================================================
# Scoring one applicant
# ===========================================================================

def applicant_to_frame(applicant, config=None):
    """Turn a dict of raw attribute values into a one-row frame.

    Missing attributes are left as NaN rather than guessed. Phase 2's binning
    sends them to the ``MISSING`` bin and Phase 3 gives that bin its own WOE, so
    an incomplete application is scoreable — which it has to be, because
    incomplete applications are the normal case.
    """
    config = config or {}
    schema = config.get("schema", {})
    expected = (list(schema.get("numeric_columns", []))
                + list(schema.get("categorical_columns", [])))

    row = {column: applicant.get(column, np.nan) for column in expected}
    unknown = set(applicant) - set(expected) - {schema.get("target_column")}
    if unknown:
        raise ScorerError(
            f"Unknown attributes for this scorecard: {sorted(unknown)}. "
            f"Expected any of: {sorted(expected)}"
        )
    return pd.DataFrame([row])


def score_applicant(applicant, scorecard, binning_spec, config=None,
                    top_reasons=4):
    """Score one applicant and explain the result.

    Parameters
    ----------
    applicant : dict or pandas.DataFrame
        Raw attribute values (not binned). A one-row frame is also accepted.
    scorecard : dict
        The points table from `scaling.build_points_table`.
    binning_spec : dict
        The frozen bin definitions from `binning.fit_bins`.
    config : dict | None
    top_reasons : int
        How many reason codes to return. Four is the common regulatory practice.

    Returns
    -------
    dict
        ``score``, ``probability_of_default``, ``odds``, the per-characteristic
        breakdown, and the ranked reason codes.
    """
    frame = (applicant.copy() if isinstance(applicant, pd.DataFrame)
             else applicant_to_frame(applicant, config))
    if len(frame) != 1:
        raise ScorerError(f"Expected one applicant, got {len(frame)} rows.")

    binned = binning.apply_bins(frame, binning_spec)
    total, contributions = scaling.score_applicants(binned, scorecard)

    score = float(total.iloc[0])
    constants = scorecard["constants"]

    breakdown = []
    table = scorecard["table"]
    for characteristic in contributions.columns:
        rows = table[table["feature"] == characteristic]
        awarded = float(contributions.iloc[0][characteristic])
        best = float(rows["points"].max())
        worst = float(rows["points"].min())
        bin_label = str(binned.iloc[0][f"{characteristic}_bin"])

        breakdown.append({
            "characteristic": characteristic,
            "applicant_bin": bin_label,
            "points": awarded,
            "best_available": best,
            "worst_available": worst,
            # The reason-code quantity: how far short of the best this is.
            "points_lost": best - awarded,
            "best_bin": str(rows.loc[rows["points"].idxmax(), "bin"]),
        })

    breakdown = pd.DataFrame(breakdown).sort_values(
        "points_lost", ascending=False).reset_index(drop=True)

    return {
        "score": score,
        "probability_of_default": float(scaling.score_to_pd(score, constants)),
        "good_bad_odds": float(scaling.score_to_odds(score, constants)),
        "score_range": scorecard["score_range"],
        "breakdown": breakdown,
        "reason_codes": reason_codes(breakdown, top_reasons, config=config),
        "protected_in_model": sorted(
            set((config or {}).get("fair_lending", {}).get(
                "protected_characteristics", []))
            & set(breakdown["characteristic"])),
    }


def reason_codes(breakdown, top_n=4, minimum_points_lost=1.0, config=None):
    """The ranked reasons an applicant did not score higher.

    Only characteristics that actually cost points qualify. A characteristic
    where the applicant already holds the best bin has lost nothing and is
    excluded — a reason code saying "your perfect credit history counted against
    you" would be worse than no notice at all.

    ``minimum_points_lost`` suppresses trivia: being one point short of the best
    bin is not a reason anyone was declined, and listing it would bury the real
    cause.

    **The fair-lending guard.** A characteristic listed under
    ``fair_lending.protected_characteristics`` may not lawfully be cited as a
    reason for a credit decision. If one would appear, the behaviour is set by
    ``fair_lending.on_protected_reason_code``:

    * ``refuse`` (the default) — raise `FairLendingError`. An explanation that
      cannot lawfully be given should not be produced at all.
    * ``warn`` — suppress the code and attach a warning.

    Suppressing the *notice* does not make the model lawful: the characteristic
    is still driving the score. The only real fix is upstream — remove it from
    the feature set and refit. The error says so.
    """
    cfg = (config or {}).get("fair_lending", {})
    protected = set(cfg.get("protected_characteristics", []))
    policy = cfg.get("on_protected_reason_code", "refuse")

    qualifying = breakdown[breakdown["points_lost"] >= minimum_points_lost]

    offending = [c for c in qualifying["characteristic"] if c in protected]
    if offending:
        if policy == "refuse":
            raise FairLendingError(
                f"The explanation would cite protected "
                f"characteristic(s) {sorted(set(offending))}, which may not "
                "lawfully drive a credit decision (ECOA / Regulation B in the "
                "US; the National Credit Act and Equality Act in South "
                "Africa).\n"
                "  Suppressing the notice would NOT make the model lawful — the "
                "characteristic is still driving the score.\n"
                "  Fix it upstream: drop it from schema.categorical_columns (or "
                "numeric_columns), refit from Phase 2, and re-validate."
            )
        qualifying = qualifying[~qualifying["characteristic"].isin(protected)]

    codes = []
    for rank, (_, row) in enumerate(qualifying.head(top_n).iterrows(), start=1):
        codes.append({
            "rank": rank,
            "characteristic": row["characteristic"],
            "your_value": row["applicant_bin"],
            "points_lost": float(row["points_lost"]),
            "best_value": row["best_bin"],
        })
    return codes


def decision(result, cutoff_score):
    """Approve or decline against a cut-off, with the reasons attached.

    The cut-off is a *policy* input, chosen from the Phase 6 cut-off table by
    whoever owns the credit policy. It is not a model parameter and nothing here
    picks it.
    """
    approved = result["score"] >= cutoff_score
    return {
        "approved": bool(approved),
        "score": result["score"],
        "cutoff": float(cutoff_score),
        "margin": float(result["score"] - cutoff_score),
        "probability_of_default": result["probability_of_default"],
        # Reasons are attached on a decline, which is when they are required.
        "reason_codes": [] if approved else result["reason_codes"],
    }


# ===========================================================================
# Explaining it in words
# ===========================================================================

def explain(result, cutoff_score=None):
    """A plain-language explanation of one applicant's score.

    Written to be read by someone who has never seen a logistic regression,
    because that is who an adverse-action notice is for.
    """
    lines = []
    lines.append(f"Score: {result['score']:.0f}"
                 f"   (range {result['score_range']['minimum']}"
                 f"-{result['score_range']['maximum']})")
    lines.append(f"Probability of default: "
                 f"{result['probability_of_default']:.1%}")
    lines.append(f"Good:bad odds: {result['good_bad_odds']:.1f} : 1")

    if cutoff_score is not None:
        verdict = decision(result, cutoff_score)
        lines.append("")
        lines.append(f"Decision at a cut-off of {cutoff_score:.0f}: "
                     f"{'APPROVE' if verdict['approved'] else 'DECLINE'}"
                     f"   ({verdict['margin']:+.0f} points)")

    lines.append("")
    lines.append("How the score was built:")
    for _, row in result["breakdown"].iterrows():
        marker = "  " if row["points_lost"] < 1 else "* "
        lines.append(f"  {marker}{row['characteristic']:<26} "
                     f"{row['applicant_bin'][:34]:<36} "
                     f"{row['points']:>4.0f} pts"
                     f"   (best available {row['best_available']:.0f})")

    if result.get("protected_in_model"):
        lines.append("")
        lines.append("WARNING: this model contains protected "
                     f"characteristic(s): {result['protected_in_model']}.")
        lines.append("They are excluded from the reasons below, but they are "
                     "still driving the")
        lines.append("score. The model is not fit for lending use until they "
                     "are removed and it")
        lines.append("is refitted.")

    if result["reason_codes"]:
        lines.append("")
        lines.append("Main reasons the score is not higher:")
        for code in result["reason_codes"]:
            lines.append(f"  {code['rank']}. {code['characteristic']}: "
                         f"you are in '{code['your_value']}', which costs "
                         f"{code['points_lost']:.0f} points against the best "
                         f"available ('{code['best_value']}').")
    else:
        lines.append("")
        lines.append("No reason codes: this applicant holds the best available "
                     "bin in every characteristic.")

    return "\n".join(lines)


# ===========================================================================
# Batch scoring
# ===========================================================================

def score_many(applicants, scorecard, binning_spec, config=None,
               top_reasons=4):
    """Score a frame of applicants, returning one row each with reason codes.

    Reason codes are flattened into ``reason_1``..``reason_n`` columns so the
    output is a plain table a decisioning system or a reviewer can consume.
    """
    if len(applicants) == 0:
        raise ScorerError("No applicants to score.")

    rows = []
    for position in range(len(applicants)):
        single = applicants.iloc[[position]].copy()
        result = score_applicant(single, scorecard, binning_spec, config,
                                 top_reasons)

        row = {
            "score": result["score"],
            "probability_of_default": result["probability_of_default"],
        }
        for index in range(top_reasons):
            if index < len(result["reason_codes"]):
                code = result["reason_codes"][index]
                row[f"reason_{index + 1}"] = (
                    f"{code['characteristic']} ({code['points_lost']:.0f} pts)")
            else:
                row[f"reason_{index + 1}"] = ""
        rows.append(row)

    return pd.DataFrame(rows, index=applicants.index)


def score_applicant_legacy(applicant, scorecard=None):
    """Backwards-compatible name from the Phase 0 scaffold."""
    raise ScorerError(
        "Use score_applicant(applicant, scorecard, binning_spec, config)."
    )
