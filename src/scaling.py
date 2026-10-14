r"""
Stage 6 — Score scaling.  [Built in Phase 6, 14 Oct 2026]

Turns the logistic PD model into a **points table** — a list of plain statements
with a number beside each, which an applicant can be shown and a credit
committee can sign. The points table *is* the scorecard; everything before this
was the machinery that produced it.

The transform
-------------
.. math::
    \text{Score} = \text{Offset} + \text{Factor}\times\ln(\text{odds}_{\text{good:bad}})

with

.. math::
    \text{Factor} = \frac{\text{PDO}}{\ln 2},
    \qquad
    \text{Offset} = \text{BaseScore} - \text{Factor}\times\ln(\text{BaseOdds})

**PDO** is "points to double the odds". With `pdo = 20`, gaining 20 points means
the good:bad odds double — which is the one sentence that explains a scorecard to
a non-technical audience, and the reason this scaling is used at all rather than
just reporting the PD.

Why the score is linear in the points
-------------------------------------
Because the model is linear in the log-odds and the inputs are WOE, the whole
score decomposes into one additive contribution per characteristic:

.. math::
    \text{Points}_k = -\Big(\beta_k\,\text{WOE}_k + \frac{\beta_0}{n}\Big)
                      \times \text{Factor} + \frac{\text{Offset}}{n}

The intercept and offset are split evenly across the *n* model variables, which
is a presentational choice — it makes every characteristic's points comparable
and the total come out right. Nothing in the model depends on the split.

The sign chain, which must hold end to end
------------------------------------------
Phase 1 defined 1 = default. Phase 3 fixed WOE so positive = safer. Phase 4
therefore expects negative coefficients. Here, with :math:`\beta_k < 0`, a safer
bin (positive WOE) gives :math:`-\beta_k\,\text{WOE}_k > 0` — **more points**.

So **higher score = safer applicant**, which is the universal convention and the
opposite direction to the PD. If any link in that chain were flipped, the
scorecard would rank backwards while producing a perfectly plausible-looking
points table. `test_higher_score_means_lower_pd` is the end-to-end check.

The verification that makes this trustworthy
--------------------------------------------
A points table is only worth anything if it reproduces the model it came from.
`verify_against_model` scores every applicant both ways — through the points
table and through the logistic model directly — and reports the largest
disagreement. Before rounding it should be zero to floating-point precision; the
only difference afterwards is the rounding, and that is measured rather than
assumed.

See docs/methodology.md section 6.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


class ScalingError(Exception):
    """Raised when the points table cannot be built or fails verification."""


# ===========================================================================
# The scaling constants
# ===========================================================================

def scaling_constants(config=None):
    """Factor and offset from the configured PDO, base score and base odds.

    Returns
    -------
    dict
        ``factor``, ``offset``, and the three inputs, so a report can state the
        assumptions beside the numbers.
    """
    cfg = (config or {}).get("scaling", {})
    pdo = float(cfg.get("pdo", 20))
    base_score = float(cfg.get("base_score", 600))
    base_odds = float(cfg.get("base_odds", 50))

    if pdo <= 0:
        raise ScalingError("PDO must be positive.")
    if base_odds <= 0:
        raise ScalingError("Base odds must be positive.")

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    return {"pdo": pdo, "base_score": base_score, "base_odds": base_odds,
            "factor": float(factor), "offset": float(offset)}


def score_to_odds(score, constants):
    """Good:bad odds implied by a score. The inverse of the scaling."""
    return np.exp((np.asarray(score, dtype=float) - constants["offset"])
                  / constants["factor"])


def odds_to_score(odds, constants):
    """Score implied by good:bad odds."""
    return constants["offset"] + constants["factor"] * np.log(
        np.asarray(odds, dtype=float))


def score_to_pd(score, constants):
    """Probability of default implied by a score.

    Good:bad odds *o* means *o* goods for each bad, so
    ``PD = 1 / (1 + o)``.
    """
    odds = score_to_odds(score, constants)
    return 1.0 / (1.0 + odds)


def pd_to_score(probability_of_default, constants):
    """Score implied by a PD. Used to turn a cut-off PD into a cut-off score."""
    p = np.clip(np.asarray(probability_of_default, dtype=float), 1e-12, 1 - 1e-12)
    return odds_to_score((1 - p) / p, constants)


# ===========================================================================
# Building the table
# ===========================================================================

def build_points_table(pd_model, woe_spec, config=None, round_points=True):
    """Build the per-bin points table from the fitted model and the WOE tables.

    Parameters
    ----------
    pd_model : dict
        Output of `model.fit_pd_model`.
    woe_spec : dict
        Output of `woe.compute_woe_iv`, carrying the per-bin WOE values.
    config : dict | None
        Reads the ``scaling`` block.
    round_points : bool
        Round to whole points, as a real scorecard does. The rounding error is
        measured by `verify_against_model` either way.

    Returns
    -------
    dict
        The table (one row per feature-bin), the constants, and the score range.
    """
    constants = scaling_constants(config)
    coefficients = pd_model["coefficients"]["coefficient"]
    intercept = float(coefficients.get("const", pd_model.get("intercept", 0.0)))

    features = list(pd_model["features"])
    n = len(features)
    if n == 0:
        raise ScalingError("The model has no features, so there is no scorecard.")

    rows = []
    for feature_woe in features:
        # Model columns are named "<feature>_woe"; the WOE spec is keyed on the
        # bare feature name.
        feature = (feature_woe[:-4] if feature_woe.endswith("_woe")
                   else feature_woe)
        beta = float(coefficients[feature_woe])

        definition = woe_spec["features"].get(feature)
        if definition is None:
            raise ScalingError(
                f"No WOE table for '{feature}', so its points cannot be computed."
            )

        for bin_label, woe_value in definition["mapping"].items():
            raw_points = (-(beta * float(woe_value) + intercept / n)
                          * constants["factor"] + constants["offset"] / n)
            rows.append({
                "feature": feature,
                "bin": bin_label,
                "woe": float(woe_value),
                "coefficient": beta,
                "points_exact": float(raw_points),
                "points": int(round(raw_points)) if round_points
                          else float(raw_points),
            })

    table = pd.DataFrame(rows)

    # Within each characteristic, order bins by points so the table reads from
    # riskiest to safest — which is how a committee pack presents it.
    table = table.sort_values(["feature", "points_exact"]).reset_index(drop=True)

    # The attainable range: worst and best bin of every characteristic.
    worst = table.groupby("feature")["points"].min().sum()
    best = table.groupby("feature")["points"].max().sum()

    # Is the reference point even reachable?
    #
    # The subtlety worth understanding: raising `base_score` does NOT fix this.
    # The offset moves with it, so every point moves with it, so the attainable
    # range moves by exactly the same amount — the gap is invariant. The real
    # lever is `base_odds`.
    #
    # `base_odds = 50` asserts a reference applicant with a PD of 1/51 = 1.96%.
    # If the model never produces a PD that low, the scale is anchored at a
    # creditworthiness no applicant in this population has. The ranking is
    # unaffected (the scaling is linear) but every score reads oddly against the
    # stated reference, and a committee will ask about it on sight.
    base = constants["base_score"]
    base_attainable = bool(worst <= base <= best)

    # Express it the way it can be acted on: the PD range the model actually
    # spans, against the PD the reference odds imply.
    attainable_pd = {
        "best_applicant": float(1.0 / (1.0 + np.exp(
            (best - constants["offset"]) / constants["factor"]))),
        "worst_applicant": float(1.0 / (1.0 + np.exp(
            (worst - constants["offset"]) / constants["factor"]))),
        "base_odds_implies": float(1.0 / (1.0 + constants["base_odds"])),
    }

    return {
        "constants": constants,
        "features": features,
        "n_features": n,
        "intercept": intercept,
        "table": table,
        "score_range": {"minimum": int(worst), "maximum": int(best)},
        "base_score_attainable": base_attainable,
        "attainable_pd": attainable_pd,
        "rounded": bool(round_points),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ===========================================================================
# Scoring an applicant
# ===========================================================================

def score_applicants(binned, scorecard):
    """Score a frame of binned applicants by looking up and adding points.

    This is the whole point of the scorecard: scoring is a lookup and an
    addition. No model object, no library version, no floating-point pipeline —
    a clerk with the printed table and a calculator gets the same answer.

    Returns
    -------
    (pandas.Series, pandas.DataFrame)
        Total score per applicant, and the per-characteristic contributions —
        which are the raw material for the reason codes in Phase 7.
    """
    table = scorecard["table"]
    contributions = pd.DataFrame(index=binned.index)

    for feature_woe in scorecard["features"]:
        feature = (feature_woe[:-4] if feature_woe.endswith("_woe")
                   else feature_woe)
        column = f"{feature}_bin"
        if column not in binned.columns:
            raise ScalingError(f"Cannot score: '{column}' is missing.")

        lookup = (table[table["feature"] == feature]
                  .set_index("bin")["points"].to_dict())

        mapped = binned[column].astype(str).map(lookup)

        # A bin with no points is one the scorecard has never seen. It is given
        # the feature's own mean points rather than zero: zero is not neutral on
        # this scale (the offset is baked into every cell), so zero would be a
        # severe and arbitrary penalty.
        if mapped.isna().any():
            mapped = mapped.fillna(np.mean(list(lookup.values())))

        contributions[feature] = mapped.astype(float)

    return contributions.sum(axis=1), contributions


# ===========================================================================
# Verification — the step that makes the table trustworthy
# ===========================================================================

def rounding_tolerance(scorecard, points_per_feature=0.5, slack=0.01):
    """The largest disagreement rounding can legitimately cause.

    Each characteristic's points are rounded independently, so each contributes
    at most half a point of error, and the total score can be off by at most
    ``0.5 * n_features``.

    **This is why a fixed tolerance in points is the wrong design.** A
    two-characteristic card tolerates 1 point; an eleven-characteristic card
    needs 5.5. Setting one number for both either fails good tables or passes
    broken ones — the first real run tripped on exactly that, failing an
    eleven-feature card at 2.277 points against a flat 1.0 limit.
    """
    if not scorecard.get("rounded", True):
        return slack          # unrounded: only floating-point noise is allowed
    return points_per_feature * scorecard["n_features"] + slack


def verify_against_model(binned, woe_matrix, pd_model, scorecard,
                         tolerance=None, points_per_feature=0.5):
    """Check the points table reproduces the model it came from.

    Scores every applicant twice — once through the points table, once through
    the logistic model directly — converts both to a PD, and reports the largest
    disagreement.

    Unrounded, the two must agree to floating-point precision: the points table
    is an **algebraic rearrangement** of the model, not an approximation. Once
    points are rounded to whole numbers a small difference appears, bounded by
    half a point per characteristic.

    ``tolerance`` defaults to that bound (see `rounding_tolerance`) rather than a
    fixed number, because the admissible error grows with the number of
    characteristics.

    Raises
    ------
    ScalingError
        If the disagreement exceeds the tolerance, which means the table does
        not represent the model and must not be used.
    """
    from src import model as model_module

    constants = scorecard["constants"]
    if tolerance is None:
        tolerance = rounding_tolerance(scorecard, points_per_feature)

    card_score, _ = score_applicants(binned, scorecard)
    model_pd = model_module.predict_pd(pd_model, woe_matrix)

    # The model's own PD, expressed as a score on this scale.
    model_score = pd_to_score(model_pd, constants)

    difference = np.abs(card_score.to_numpy() - model_score)
    worst = float(np.max(difference))

    result = {
        "max_difference_points": worst,
        "mean_difference_points": float(np.mean(difference)),
        "tolerance_points": float(tolerance),
        "tolerance_basis": (f"{points_per_feature} points x "
                            f"{scorecard['n_features']} characteristics"
                            if scorecard.get("rounded", True)
                            else "unrounded: floating-point noise only"),
        "rounded": scorecard["rounded"],
        "n_checked": int(len(difference)),
        "passed": bool(worst <= tolerance),
    }

    if not result["passed"]:
        raise ScalingError(
            f"The points table disagrees with the model by up to {worst:.3f} "
            f"points, above the {tolerance:.3f} tolerance "
            f"({result['tolerance_basis']}). The table does not represent the "
            "model and must not be used. Rounding alone cannot cause this, so "
            "check the sign chain and the intercept split."
        )
    return result


def cutoff_table(scores, target, scorecard, n_bands=10):
    """Approval rate and bad rate at each candidate cut-off score.

    The table a credit committee actually decides on: *if we decline below 580,
    we approve 78% of applicants and the approved book has a 19% bad rate.*
    Every row is a policy option.
    """
    frame = pd.DataFrame({
        "score": np.asarray(scores, dtype=float),
        "bad": np.asarray(target, dtype=int),
    }).sort_values("score")

    candidates = np.quantile(frame["score"], np.linspace(0.05, 0.95, n_bands))

    rows = []
    for cutoff in candidates:
        approved = frame[frame["score"] >= cutoff]
        declined = frame[frame["score"] < cutoff]
        if approved.empty:
            continue
        rows.append({
            "cutoff_score": float(round(cutoff)),
            "approval_rate": float(len(approved) / len(frame)),
            "approved_bad_rate": float(approved["bad"].mean()),
            "declined_bad_rate": float(declined["bad"].mean())
                                  if len(declined) else np.nan,
            # Of all the bads in the population, how many does this cut-off
            # keep out? The other half of the trade-off.
            "bads_rejected": float(declined["bad"].sum() / max(frame["bad"].sum(), 1)),
            "goods_rejected": float((1 - declined["bad"]).sum()
                                     / max((1 - frame["bad"]).sum(), 1))
                               if len(declined) else 0.0,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Reporting and persistence
# ===========================================================================

def scorecard_report(scorecard, verification=None, cutoffs=None):
    """The points table, formatted the way a committee pack presents it."""
    constants = scorecard["constants"]
    lines = ["Scorecard — points table", "-" * 25]
    lines.append(f"  {'PDO (points to double the odds)':<34} "
                 f"{constants['pdo']:>8.0f}")
    lines.append(f"  {'base score':<34} {constants['base_score']:>8.0f}")
    lines.append(f"  {'at good:bad odds of':<34} {constants['base_odds']:>8.0f}:1")
    lines.append(f"  {'factor = PDO / ln 2':<34} {constants['factor']:>8.3f}")
    lines.append(f"  {'offset':<34} {constants['offset']:>8.3f}")
    lines.append(f"  {'characteristics':<34} {scorecard['n_features']:>8}")
    lines.append(f"  {'attainable score range':<34} "
                 f"{scorecard['score_range']['minimum']:>4} to "
                 f"{scorecard['score_range']['maximum']}")

    if not scorecard.get("base_score_attainable", True):
        pd_range = scorecard.get("attainable_pd", {})
        lines.append("")
        lines.append(f"  WARNING: the base score ({constants['base_score']:.0f}) "
                     "lies OUTSIDE the attainable range.")
        if pd_range:
            lines.append(f"    base_odds of {constants['base_odds']:.0f}:1 "
                         f"implies a reference PD of "
                         f"{pd_range['base_odds_implies']:.2%},")
            lines.append(f"    but this model spans "
                         f"{pd_range['best_applicant']:.2%} (best applicant) to "
                         f"{pd_range['worst_applicant']:.2%} (worst).")
        lines.append("    The ranking is unaffected — the scaling is linear — "
                     "but every score reads")
        lines.append("    oddly against the stated reference.")
        lines.append("    FIX: lower `base_odds` into the attainable PD range. "
                     "Raising base_score does")
        lines.append("    NOT help — the offset moves with it, so the whole "
                     "range shifts by the same")
        lines.append("    amount and the gap is unchanged.")

    lines.append("")
    lines.append("  Higher score = safer applicant. Each row's points are added.")
    lines.append("")

    table = scorecard["table"]
    for feature in table["feature"].unique():
        rows = table[table["feature"] == feature]
        lines.append(f"  {feature}")
        for _, row in rows.iterrows():
            label = str(row["bin"])[:46]
            lines.append(f"      {label:<48} {row['points']:>5}")
        lines.append("")

    if verification:
        lines.append("  Verification against the fitted model:")
        lines.append(f"    {'largest disagreement':<30} "
                     f"{verification['max_difference_points']:>8.4f} points")
        lines.append(f"    {'mean disagreement':<30} "
                     f"{verification['mean_difference_points']:>8.4f} points")
        lines.append(f"    {'applicants checked':<30} "
                     f"{verification['n_checked']:>8,}")
        lines.append(f"    {'rounded to whole points':<30} "
                     f"{str(verification['rounded']):>8}")
        lines.append(f"    {'tolerance':<30} "
                     f"{verification['tolerance_points']:>8.3f} points"
                     f"  ({verification['tolerance_basis']})")
        lines.append("    (unrounded this must be ~0: the table is an algebraic")
        lines.append("     rearrangement of the model, not an approximation)")

    if cutoffs is not None and not cutoffs.empty:
        lines.append("")
        lines.append("  Cut-off options (each row is a policy choice):")
        display = cutoffs[["cutoff_score", "approval_rate", "approved_bad_rate",
                           "bads_rejected", "goods_rejected"]]
        lines.append("    " + display.to_string(
            index=False, float_format=lambda v: f"{v:.3f}").replace("\n", "\n    "))

    return "\n".join(lines)


def save_scorecard(scorecard, config, filename=None):
    """Write the points table to data/processed/ as JSON.

    The frozen table plus the scaling constants are the deliverable: a scorecard
    has to be scoreable by hand, years later, with no code at all.
    """
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("scaling", {}).get(
        "scorecard_filename", "scorecard.json"))

    serialisable = {k: v for k, v in scorecard.items() if k != "table"}
    serialisable["table"] = scorecard["table"].to_dict(orient="records")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=_json_safe)
    return path


def load_scorecard(path):
    """Read a frozen points table back."""
    with open(path, "r", encoding="utf-8") as f:
        scorecard = json.load(f)
    scorecard["table"] = pd.DataFrame(scorecard["table"])
    return scorecard


def build_scorecard(model, config=None):
    """Backwards-compatible name from the Phase 0 scaffold."""
    raise ScalingError(
        "scaling.build_scorecard is superseded: use "
        "build_points_table(pd_model, woe_spec, config)."
    )


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)
