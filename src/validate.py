r"""
Stage 5 — Validation.  [Built in Phase 5, 13 Oct 2026]

**This is the first stage that touches the test set.** Everything before it —
binning, WOE, feature selection, the logistic fit — saw only the training half
of the frozen split. The numbers produced here are therefore the first honest
statement about how the model performs, and they get exactly one chance to be
honest: every time a modelling decision is changed in response to a test number,
the test set becomes a little more like training data.

The three questions, and why all three are needed
-------------------------------------------------
A credit committee asks three separate things, and a model can pass any one of
them while failing the others:

1. **Discrimination** — does the model separate good from bad? (KS, Gini/AUC)
2. **Calibration** — are the predicted probabilities right in level, not just in
   order? (reliability by band, Hosmer-Lemeshow style)
3. **Stability** — does the score distribution look the same out of sample?
   (PSI)

A model can rank perfectly and still be useless for pricing if every predicted
PD is double the truth. It can be beautifully calibrated on average and rank no
better than a coin. And it can do both on this sample and fall apart on next
quarter's applicants. Reporting only AUC — the most common shortcut — answers
one question out of three.

What is deliberately NOT done here
----------------------------------
No threshold is tuned, no feature is dropped, no hyperparameter is chosen on
these numbers. If the validation report is disappointing, the honest responses
are to revisit the training-set work or to report the result as it stands — not
to adjust until the test number improves. That is the difference between
validation and a second training run.

See docs/methodology.md section 5.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# PSI and WOE-style logs need a floor, or an empty band makes the result infinite.
PSI_FLOOR = 1e-6


class ValidationError(Exception):
    """Raised when the validation metrics cannot be computed."""


# ===========================================================================
# Discrimination
# ===========================================================================

def ks_statistic(scores, target, higher_is_riskier=True):
    """Kolmogorov-Smirnov: the widest gap between the good and bad distributions.

    .. math:: \\text{KS} = \\max_s \\big| F_{\\text{bad}}(s) - F_{\\text{good}}(s) \\big|

    Read it as: at the best possible cut-off, what share of bads could you reject
    while keeping what share of goods? A KS of 0.40 means there is a cut-off
    catching 40 percentage points more of the bads than of the goods.

    Industry bands for an application scorecard: below 0.20 is weak, 0.20-0.40 is
    usual, above 0.50 is unusually good — and above 0.70 is a reason to look for
    leakage rather than to celebrate.

    Returns
    -------
    dict
        The statistic, the score at which it occurs, and the two cumulative
        shares at that point — the cut-off a committee actually asks about.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(target, dtype=int)

    if len(np.unique(y)) < 2:
        raise ValidationError("Need both goods and bads to compute KS.")

    order = np.argsort(-s if higher_is_riskier else s)
    y_sorted = y[order]
    s_sorted = s[order]

    cumulative_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cumulative_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)

    gaps = np.abs(cumulative_bad - cumulative_good)
    at = int(np.argmax(gaps))

    return {
        "ks": float(gaps[at]),
        "at_score": float(s_sorted[at]),
        "bads_captured": float(cumulative_bad[at]),
        "goods_captured": float(cumulative_good[at]),
        "share_of_book": float((at + 1) / len(y)),
    }


def auc_and_gini(scores, target, higher_is_riskier=True):
    """AUC via the Mann-Whitney relationship, plus Gini = 2*AUC - 1.

    AUC is the probability that a randomly chosen bad scores riskier than a
    randomly chosen good. 0.5 is a coin flip.

    Computed from ranks rather than by integrating an ROC curve, because the
    rank identity is exact, handles ties correctly, and is three lines:

        AUC = (mean rank of bads - (n_bad + 1) / 2) / n_good
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(target, dtype=int)

    if len(np.unique(y)) < 2:
        raise ValidationError("Need both goods and bads to compute AUC.")

    risk = s if higher_is_riskier else -s

    # Average ranks so tied scores are handled correctly — ties are common once
    # a scorecard rounds to whole points, and ignoring them inflates AUC.
    order = np.argsort(risk)
    ranks = np.empty(len(risk), dtype=float)
    ranks[order] = np.arange(1, len(risk) + 1)

    unique, inverse, counts = np.unique(risk, return_inverse=True,
                                        return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            tied = inverse == index
            ranks[tied] = ranks[tied].mean()

    n_bad = int(y.sum())
    n_good = int(len(y) - n_bad)
    auc = (ranks[y == 1].mean() - (n_bad + 1) / 2) / n_good

    return {"auc": float(auc), "gini": float(2 * auc - 1),
            "n_good": n_good, "n_bad": n_bad}


def roc_curve_points(scores, target, higher_is_riskier=True, n_points=100):
    """Points for an ROC chart, thinned to something a plot can use."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(target, dtype=int)

    order = np.argsort(-s if higher_is_riskier else s)
    y_sorted = y[order]

    tpr = np.concatenate([[0.0], np.cumsum(y_sorted) / max(y_sorted.sum(), 1)])
    fpr = np.concatenate([[0.0],
                          np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)])

    step = max(1, len(tpr) // n_points)
    return pd.DataFrame({"fpr": fpr[::step], "tpr": tpr[::step]})


# ===========================================================================
# Calibration
# ===========================================================================

def calibration_by_band(predicted_pd, target, n_bands=10):
    """Predicted PD against realised default rate, by decile of predicted risk.

    This is the credit-risk version of a reliability diagram, and the basis of
    the Hosmer-Lemeshow test. A model that ranks well can still be wrong in
    level: if every band's predicted PD is double its realised rate, the ranking
    is fine and the pricing is not.

    Quantile bands (not equal-width) because the question is about the
    population: "of the riskiest tenth of applicants, how many actually
    defaulted?"
    """
    frame = pd.DataFrame({
        "predicted": np.asarray(predicted_pd, dtype=float),
        "bad": np.asarray(target, dtype=int),
    })

    frame["band"] = pd.qcut(frame["predicted"], q=n_bands, duplicates="drop",
                            labels=False)

    table = frame.groupby("band", observed=True).agg(
        accounts=("bad", "size"),
        bads=("bad", "sum"),
        predicted_pd=("predicted", "mean"),
    )
    table = table.sort_values("predicted_pd").reset_index(drop=True)
    table.insert(0, "band", range(1, len(table) + 1))

    table["realised_pd"] = table["bads"] / table["accounts"]
    table["gap"] = table["predicted_pd"] - table["realised_pd"]

    # Binomial standard error of the realised rate, so a gap can be judged
    # against the noise in its own band rather than a flat tolerance.
    table["std_error"] = np.sqrt(
        table["realised_pd"] * (1 - table["realised_pd"]) / table["accounts"])
    table["within_2se"] = table["gap"].abs() <= 2 * table["std_error"]

    return table


def hosmer_lemeshow(predicted_pd, target, n_bands=10):
    """The chi-square statistic behind the calibration table.

    .. math:: \\chi^2 = \\sum_b \\frac{(O_b - E_b)^2}{E_b (1 - E_b / n_b)}

    with observed and expected bad counts per band. Degrees of freedom are
    ``n_bands - 2``.

    A LARGE statistic (small p-value) means poor calibration — the opposite
    direction to most tests, and a routine source of misreading. Note also that
    the test is weak on small samples: with 300 accounts it will pass almost
    anything, so it is reported next to the band table rather than instead of it.
    """
    table = calibration_by_band(predicted_pd, target, n_bands)

    observed = table["bads"].to_numpy(dtype=float)
    expected = (table["predicted_pd"] * table["accounts"]).to_numpy(dtype=float)
    n = table["accounts"].to_numpy(dtype=float)

    denominator = expected * (1 - expected / n)
    usable = denominator > 0
    statistic = float(np.sum((observed[usable] - expected[usable]) ** 2
                             / denominator[usable]))

    degrees_of_freedom = max(int(usable.sum()) - 2, 1)
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(statistic, degrees_of_freedom))

    return {
        "chi_square": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        # Low p-value = poor calibration. Stated explicitly because the
        # direction catches people out.
        "calibration_rejected_at_5pct": bool(p_value < 0.05),
    }


# ===========================================================================
# Stability
# ===========================================================================

def population_stability_index(expected, actual, n_bands=10):
    """PSI between a development ("expected") and a validation ("actual") sample.

    .. math:: \\text{PSI} = \\sum_b (\\%A_b - \\%E_b)\\,\\ln(\\%A_b / \\%E_b)

    Bands are cut on the **expected** distribution and then applied to the
    actual one — that ordering is the whole point. Re-cutting bands on the
    actual sample would make any two distributions look identical, because each
    would have 10% of itself in each of its own deciles.

    Usual reading: below 0.10 stable, 0.10-0.25 a shift worth watching, above
    0.25 a major shift that normally triggers a redevelopment discussion.

    Returns
    -------
    (float, pandas.DataFrame)
        The index and the per-band contributions, so a single drifting band can
        be identified rather than just a total.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        raise ValidationError("PSI needs both samples to be non-empty.")

    # Cut on expected, apply to actual.
    _, edges = pd.qcut(expected, q=n_bands, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts = pd.cut(expected, bins=edges).value_counts().sort_index()
    actual_counts = pd.cut(actual, bins=edges).value_counts().sort_index()

    expected_share = np.maximum(expected_counts / len(expected), PSI_FLOOR)
    actual_share = np.maximum(actual_counts / len(actual), PSI_FLOOR)

    contribution = (actual_share - expected_share) * np.log(
        actual_share / expected_share)

    table = pd.DataFrame({
        "band": [str(i) for i in expected_counts.index],
        "expected_share": expected_share.to_numpy(),
        "actual_share": actual_share.to_numpy(),
        "contribution": contribution.to_numpy(),
    })
    return float(contribution.sum()), table


def psi_verdict(psi):
    """The conventional reading of a PSI value."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "shift — monitor"
    return "MAJOR SHIFT — consider redevelopment"


# ===========================================================================
# Rank ordering
# ===========================================================================

def rank_ordering(scores, target, n_bands=10, higher_is_riskier=True):
    """Bad rate across score bands. It must move in one direction.

    This is the check a credit committee does by eye before looking at any
    statistic, and the one that sinks a model fastest. A scorecard whose middle
    bands cross over cannot be used for cut-off policy: "band 6 is riskier than
    band 7" has no defensible business meaning, whatever the AUC says.
    """
    frame = pd.DataFrame({
        "score": np.asarray(scores, dtype=float),
        "bad": np.asarray(target, dtype=int),
    })

    frame["band"] = pd.qcut(frame["score"], q=n_bands, duplicates="drop",
                            labels=False)

    table = frame.groupby("band", observed=True).agg(
        accounts=("bad", "size"),
        bads=("bad", "sum"),
        mean_score=("score", "mean"),
    )
    table["bad_rate"] = table["bads"] / table["accounts"]

    # Always ordered lowest score to highest, which is how a score band table is
    # read. With a PD (higher = riskier) the bad rate should therefore RISE down
    # the table; with a points score (higher = safer) it should fall. Either is
    # monotonic — what matters is that it does not turn around in the middle.
    table = table.sort_values("mean_score").reset_index(drop=True)

    # Renumber 1..n by position, so "band 1" is always the lowest-score band
    # regardless of how qcut happened to label things. That is how a score band
    # table is read in a committee pack.
    table.insert(0, "band", range(1, len(table) + 1))

    rates = table["bad_rate"].to_numpy()
    rising = bool(np.all(np.diff(rates) >= 0))
    falling = bool(np.all(np.diff(rates) <= 0))
    monotonic = rising or falling

    # The overall direction, so a local reversal can be identified against it.
    overall = rates[-1] - rates[0]
    breaks = [int(i) for i in range(len(rates) - 1)
              if (rates[i + 1] - rates[i]) * overall < 0]

    return table, {
        "monotonic": monotonic,
        "direction": "rising" if rising else "falling" if falling else "mixed",
        "expected_direction": "rising" if higher_is_riskier else "falling",
        "reversals": breaks,
        "bands": int(len(table)),
    }


# ===========================================================================
# The whole report
# ===========================================================================

def evaluate(predicted_pd_train, target_train, predicted_pd_test, target_test,
             config=None):
    """Full validation: discrimination, calibration and stability.

    Train numbers are reported **only** as a comparison. A large train-to-test
    drop in Gini is the signature of overfitting, and that comparison is the
    reason to compute the train figures at all.

    Parameters
    ----------
    predicted_pd_train, predicted_pd_test : array-like
        Predicted probability of default.
    target_train, target_test : array-like
        1 = default.
    config : dict | None
        Reads ``validation.score_bands``, ``validation.psi_bands``,
        ``validation.min_gini``, ``validation.max_psi``.
    """
    config = config or {}
    cfg = config.get("validation", {})
    n_bands = int(cfg.get("score_bands", 10))
    psi_bands = int(cfg.get("psi_bands", 10))
    min_gini = float(cfg.get("min_gini", 0.30))
    max_psi = float(cfg.get("max_psi", 0.25))
    max_gini_drop = float(cfg.get("max_gini_drop", 0.15))

    train_pd = np.asarray(predicted_pd_train, dtype=float)
    test_pd = np.asarray(predicted_pd_test, dtype=float)
    y_train = np.asarray(target_train, dtype=int)
    y_test = np.asarray(target_test, dtype=int)

    if len(np.unique(y_test)) < 2:
        raise ValidationError(
            "The test set contains only one class, so nothing can be validated."
        )

    train_discrimination = auc_and_gini(train_pd, y_train)
    test_discrimination = auc_and_gini(test_pd, y_test)
    test_ks = ks_statistic(test_pd, y_test)

    calibration = calibration_by_band(test_pd, y_test, n_bands)
    hl = hosmer_lemeshow(test_pd, y_test, n_bands)

    psi, psi_table = population_stability_index(train_pd, test_pd, psi_bands)
    ordering_table, ordering = rank_ordering(test_pd, y_test, n_bands)

    gini_drop = train_discrimination["gini"] - test_discrimination["gini"]

    checks = {
        "gini_above_minimum": test_discrimination["gini"] >= min_gini,
        "psi_within_limit": psi <= max_psi,
        "rank_ordering_monotonic": ordering["monotonic"],
        "calibration_not_rejected": not hl["calibration_rejected_at_5pct"],
        "no_large_gini_drop": gini_drop <= max_gini_drop,
        "min_gini": min_gini,
        "max_psi": max_psi,
        "max_gini_drop": max_gini_drop,
    }
    checks["all_passed"] = all(
        v for k, v in checks.items()
        if isinstance(v, (bool, np.bool_)) and k != "all_passed")

    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_bad_rate": float(y_train.mean()),
        "test_bad_rate": float(y_test.mean()),
        "discrimination": {
            "train": train_discrimination,
            "test": test_discrimination,
            "gini_drop": float(gini_drop),
            "ks": test_ks,
        },
        "calibration": {"table": calibration, "hosmer_lemeshow": hl},
        "stability": {"psi": psi, "verdict": psi_verdict(psi), "table": psi_table},
        "rank_ordering": {"table": ordering_table, **ordering},
        "roc": roc_curve_points(test_pd, y_test),
        "checks": checks,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def validation_report(result):
    """The validation report — the Phase 5 deliverable."""
    discrimination = result["discrimination"]
    checks = result["checks"]
    ks = discrimination["ks"]

    lines = ["Validation (held-out test set)", "-" * 30]
    lines.append(f"  {'train / test accounts':<30} "
                 f"{result['n_train']:>6,} / {result['n_test']:,}")
    lines.append(f"  {'train / test bad rate':<30} "
                 f"{result['train_bad_rate']:>6.1%} / {result['test_bad_rate']:.1%}")

    lines.append("")
    lines.append("  Discrimination:")
    lines.append(f"    {'AUC  (test)':<26} {discrimination['test']['auc']:>8.4f}")
    lines.append(f"    {'Gini (test)':<26} {discrimination['test']['gini']:>8.4f}"
                 f"   (minimum {checks['min_gini']:.2f})")
    lines.append(f"    {'Gini (train)':<26} {discrimination['train']['gini']:>8.4f}")
    lines.append(f"    {'Gini drop train->test':<26} "
                 f"{discrimination['gini_drop']:>8.4f}"
                 f"   (limit {checks['max_gini_drop']:.2f})")
    lines.append(f"    {'KS   (test)':<26} {ks['ks']:>8.4f}")
    lines.append(f"      at that cut-off: rejects {ks['bads_captured']:.1%} of bads "
                 f"and {ks['goods_captured']:.1%} of goods "
                 f"({ks['share_of_book']:.1%} of the book)")

    lines.append("")
    lines.append("  Calibration by predicted-risk decile:")
    table = result["calibration"]["table"]
    display = table[["band", "accounts", "predicted_pd", "realised_pd", "gap",
                     "within_2se"]]
    lines.append("    " + display.to_string(
        index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n    "))
    hl = result["calibration"]["hosmer_lemeshow"]
    lines.append(f"    Hosmer-Lemeshow chi2 = {hl['chi_square']:.2f} "
                 f"(df {hl['degrees_of_freedom']}), p = {hl['p_value']:.4f}")
    lines.append("    (a LOW p-value means POOR calibration — the opposite of "
                 "most tests)")

    lines.append("")
    lines.append(f"  Stability: PSI = {result['stability']['psi']:.4f} "
                 f"— {result['stability']['verdict']}")

    lines.append("")
    ordering = result["rank_ordering"]
    lines.append(f"  Rank ordering across {ordering['bands']} bands: "
                 f"{'monotonic' if ordering['monotonic'] else 'NOT MONOTONIC'}")
    if ordering["reversals"]:
        lines.append(f"    reversals between bands: {ordering['reversals']}")
    lines.append("    " + ordering["table"][["band", "accounts", "mean_score",
                                             "bad_rate"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n    "))

    lines.append("")
    if checks["all_passed"]:
        lines.append("  All validation checks passed.")
    else:
        if not checks["gini_above_minimum"]:
            lines.append(f"  FAIL — test Gini below the {checks['min_gini']:.2f} "
                         "minimum. The model does not discriminate well enough.")
        if not checks["no_large_gini_drop"]:
            lines.append(f"  FAIL — Gini drops {discrimination['gini_drop']:.3f} "
                         "from train to test. That is the signature of "
                         "overfitting; revisit the TRAINING work, not this report.")
        if not checks["psi_within_limit"]:
            lines.append(f"  FAIL — PSI {result['stability']['psi']:.3f} above "
                         f"{checks['max_psi']:.2f}. The score distribution is "
                         "not stable out of sample.")
        if not checks["rank_ordering_monotonic"]:
            lines.append("  FAIL — bad rate is not monotonic across score bands, "
                         "so cut-off policy cannot be defended on it.")
        if not checks["calibration_not_rejected"]:
            lines.append("  WARNING — Hosmer-Lemeshow rejects calibration at 5%. "
                         "The ranking may still be sound; the LEVEL is off.")

    lines.append("")
    lines.append("  Reminder: nothing in this report may be used to choose a "
                 "feature, a threshold or a")
    lines.append("  hyperparameter. Doing so turns the test set into training "
                 "data and these numbers")
    lines.append("  stop being a forecast of anything.")
    return "\n".join(lines)


def save_report(result, config, filename=None):
    """Write the validation numbers to data/processed/ as JSON."""
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("validation", {}).get(
        "report_filename", "validation.json"))

    serialisable = {
        k: v for k, v in result.items()
        if k not in ("calibration", "stability", "rank_ordering", "roc")
    }
    serialisable["calibration"] = {
        "table": result["calibration"]["table"].to_dict(orient="records"),
        "hosmer_lemeshow": result["calibration"]["hosmer_lemeshow"],
    }
    serialisable["stability"] = {
        "psi": result["stability"]["psi"],
        "verdict": result["stability"]["verdict"],
        "table": result["stability"]["table"].to_dict(orient="records"),
    }
    serialisable["rank_ordering"] = {
        k: v for k, v in result["rank_ordering"].items() if k != "table"
    }
    serialisable["rank_ordering"]["table"] = (
        result["rank_ordering"]["table"].to_dict(orient="records"))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=_json_safe)
    return path


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)
