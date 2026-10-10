"""
Stage 3 — Weight of Evidence & Information Value.  [Built in Phase 3, 9 Oct 2026]

This stage turns bins into numbers the model can use, and decides which features
are worth using at all.

Weight of Evidence
------------------
For each bin *b* of a feature::

    WOE_b = ln( %good_b / %bad_b )

where ``%good_b`` is that bin's share of all good accounts and ``%bad_b`` its
share of all bads. Read it as: *how much more (or less) good-ish is this bin than
the book as a whole?*

**Sign convention, fixed here and never flipped again: positive WOE = safer.**
A bin holding a disproportionate share of goods scores positive. Because Phase 1
defined 1 = default, the logistic coefficients in Phase 4 are therefore expected
to come out **negative**. If they come out positive, something has been flipped
and the scorecard is ranking backwards — which is why Phase 4 asserts the sign
rather than hoping.

Information Value
-----------------
::

    IV = sum_b ( %good_b - %bad_b ) * WOE_b

One number per feature, summarising how much the bins separate good from bad.
The conventional bands (used in `config.yaml`):

======================  ===========================================
IV                      Interpretation
======================  ===========================================
< 0.02                  useless — drop
0.02 - 0.1              weak
0.1 - 0.3               medium
0.3 - 0.5               strong
> 0.5                   suspiciously strong — check for leakage
======================  ===========================================

That last band matters more than it looks. In a real bank build, a feature with
IV above 0.5 is usually not a brilliant predictor; it is usually a field that
encodes the outcome — a collections flag, a write-off date, a post-default
status. Investigating it is not pedantry, it is the difference between a model
that works and one that was reading the answer.

Why WOE rather than dummy variables
-----------------------------------
A WOE-encoded feature is one column, not one column per level, so the model stays
small and stable. It is monotone in risk by construction, so a coefficient has a
single interpretable meaning. And it maps straight onto a points table in
Phase 6, which is what makes the end product explainable to a customer.

**Fitted on train only**, like binning, for the same reason: WOE reads the target.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# A bin that appears at scoring time but never appeared in training gets this
# WOE. See `apply_woe` for the reasoning — it is a real decision, not a default.
NEUTRAL_WOE = 0.0

# Added to the good and bad counts of a bin that has none of one class, so the
# logarithm stays finite. Phase 2's minimum bin size makes this rare; when it
# does fire, the bin is flagged rather than quietly fixed.
ZERO_CELL_ADJUSTMENT = 0.5


class WOEError(Exception):
    """Raised when WOE cannot be computed from the input given."""


# ===========================================================================
# One feature at a time
# ===========================================================================

def woe_table(binned_values, target):
    """Build the WOE table for a single binned feature.

    Parameters
    ----------
    binned_values : array-like
        Bin labels, as produced by `binning.apply_bins`.
    target : array-like
        1 = default (bad), 0 = good.

    Returns
    -------
    pandas.DataFrame
        One row per bin: accounts, goods, bads, pct_good, pct_bad, woe, iv_part,
        bad_rate, and ``adjusted`` — True where the zero-cell correction fired.
    """
    frame = pd.DataFrame({
        "bin": pd.Series(binned_values).astype(str).values,
        "bad": pd.Series(target).astype(int).values,
    })

    if len(frame) == 0:
        raise WOEError("No rows to compute WOE from.")

    table = frame.groupby("bin", observed=True)["bad"].agg(
        accounts="size", bads="sum"
    )
    table["goods"] = table["accounts"] - table["bads"]

    # A bin with zero goods or zero bads makes the log infinite. The standard fix
    # is a small continuity correction on both counts; the alternative is to
    # merge the bin, which is Phase 2's job and may already have been tried.
    needs_adjustment = (table["goods"] == 0) | (table["bads"] == 0)
    goods = table["goods"].astype(float).copy()
    bads = table["bads"].astype(float).copy()
    goods[needs_adjustment] += ZERO_CELL_ADJUSTMENT
    bads[needs_adjustment] += ZERO_CELL_ADJUSTMENT

    table["pct_good"] = goods / goods.sum()
    table["pct_bad"] = bads / bads.sum()
    table["woe"] = np.log(table["pct_good"] / table["pct_bad"])
    table["iv_part"] = (table["pct_good"] - table["pct_bad"]) * table["woe"]
    table["bad_rate"] = table["bads"] / table["accounts"]
    table["share"] = table["accounts"] / table["accounts"].sum()
    table["adjusted"] = needs_adjustment

    return table


def information_value(table):
    """Sum the per-bin contributions into one number for the feature."""
    return float(table["iv_part"].sum())


def iv_band(iv, config=None):
    """Name the conventional band an IV falls in, using the configured cutoffs."""
    cfg = (config or {}).get("woe", {})
    iv_min = cfg.get("iv_min", 0.02)
    leakage = cfg.get("iv_leakage_flag", 0.50)

    if iv < iv_min:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < leakage:
        return "strong"
    return "SUSPICIOUS — check leakage"


# ===========================================================================
# The whole spec
# ===========================================================================

def compute_woe_iv(binned, target=None, config=None, bin_order=None):
    """Compute WOE tables, IV and the selection decision for every feature.

    Parameters
    ----------
    binned : pandas.DataFrame
        Output of `binning.apply_bins` on the TRAINING data — columns named
        ``<feature>_bin``.
    target : array-like
        1 = default. Required.
    config : dict | None
        Reads ``woe.iv_min`` and ``woe.iv_leakage_flag``.
    bin_order : dict | None
        Optional ``{feature: [bin labels in natural order]}`` from the binning
        spec, so the tables print in a readable order instead of alphabetically.

    Returns
    -------
    dict
        The WOE spec: metadata, plus per feature its table, IV, band, the
        bin → WOE mapping, and whether it was selected.
    """
    if target is None:
        raise WOEError(
            "WOE needs the target — it measures separation between good and bad, "
            "so it cannot be computed from the features alone."
        )

    cfg = (config or {}).get("woe", {})
    iv_min = float(cfg.get("iv_min", 0.02))
    leakage_flag = float(cfg.get("iv_leakage_flag", 0.50))

    y = pd.Series(target).astype(int)
    if y.nunique() < 2:
        raise WOEError("The target has only one class; WOE is undefined.")

    features = {}
    for column in binned.columns:
        feature = column[:-4] if column.endswith("_bin") else column
        table = woe_table(binned[column], y)

        if bin_order and feature in bin_order:
            order = [b for b in bin_order[feature] if b in table.index]
            extras = [b for b in table.index if b not in order]
            table = table.reindex(order + extras)

        iv = information_value(table)
        band = iv_band(iv, config)

        # Selection. Note what is NOT done here: a high-IV feature is flagged,
        # never dropped automatically. Deciding that a strong predictor is
        # leakage is a judgement about where the data came from, and a human
        # has to make it.
        selected = iv >= iv_min
        features[feature] = {
            "iv": iv,
            "band": band,
            "selected": bool(selected),
            "flagged_for_leakage": bool(iv >= leakage_flag),
            "bins": int(len(table)),
            "adjusted_bins": int(table["adjusted"].sum()),
            "mapping": {str(b): float(w) for b, w in table["woe"].items()},
            "table": table,
        }

    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "fitted_on_rows": int(len(binned)),
        "sign_convention": "WOE = ln(%good / %bad); positive = safer; "
                           "expect NEGATIVE logistic coefficients with 1 = default",
        "iv_min": iv_min,
        "iv_leakage_flag": leakage_flag,
        "features": features,
    }


def selected_features(spec):
    """The features that passed the IV floor, strongest first."""
    chosen = [(f, d["iv"]) for f, d in spec["features"].items() if d["selected"]]
    return [f for f, _ in sorted(chosen, key=lambda pair: pair[1], reverse=True)]


def apply_woe(binned, spec, features=None):
    """Replace bin labels with their WOE values — the model's actual input.

    A bin that was never seen during fitting (a new product code arriving in live
    data, or a rare level that only occurs in test) is given
    :data:`NEUTRAL_WOE` = 0.

    That choice is deliberate and worth being able to defend. WOE 0 means "this
    characteristic tells us nothing about this applicant", so they receive the
    population-average contribution for it. The alternatives are worse: assigning
    the worst bin's WOE punishes an applicant for a data gap that is the lender's
    fault, and assigning the best bin's WOE is exploitable by anyone who learns
    which values fall outside the training set. The honest cost of 0 is that a
    genuinely risky unseen level is scored as neutral — which is why the count of
    unseen bins is returned, and why it belongs on a monitoring report rather than
    being swallowed.

    Returns
    -------
    (pandas.DataFrame, dict)
        The WOE matrix (one numeric column per feature) and a dict of
        ``{feature: count}`` for bins that had no fitted WOE.
    """
    features = features if features is not None else selected_features(spec)
    out = pd.DataFrame(index=binned.index)
    unseen_counts = {}

    for feature in features:
        definition = spec["features"].get(feature)
        if definition is None:
            raise WOEError(f"No WOE fitted for feature '{feature}'.")

        column = f"{feature}_bin"
        if column not in binned.columns:
            raise WOEError(f"Binned column '{column}' is missing from the input.")

        mapped = binned[column].astype(str).map(definition["mapping"])
        unseen = int(mapped.isna().sum())
        if unseen:
            unseen_counts[feature] = unseen

        out[f"{feature}_woe"] = mapped.fillna(NEUTRAL_WOE).astype(float)

    return out, unseen_counts


# ===========================================================================
# Reporting
# ===========================================================================

def iv_ranking(spec):
    """The IV ranking table — a Phase 3 deliverable in its own right."""
    rows = [{
        "feature": feature,
        "iv": definition["iv"],
        "band": definition["band"],
        "bins": definition["bins"],
        "selected": definition["selected"],
        "leakage_flag": definition["flagged_for_leakage"],
    } for feature, definition in spec["features"].items()]

    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def woe_report(spec):
    """Readable summary: the ranking, what was dropped, and what was flagged."""
    ranking = iv_ranking(spec)
    lines = ["Information Value ranking", "-" * 25]
    lines.append(ranking.to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    dropped = ranking[~ranking["selected"]]["feature"].tolist()
    flagged = ranking[ranking["leakage_flag"]]["feature"].tolist()
    adjusted = [f for f, d in spec["features"].items() if d["adjusted_bins"]]

    lines.append("")
    lines.append(f"  selected: {int(ranking['selected'].sum())} of {len(ranking)} features")
    if dropped:
        lines.append(f"  dropped (IV < {spec['iv_min']}): {', '.join(dropped)}")
    if flagged:
        lines.append(f"  LEAKAGE CHECK NEEDED (IV >= {spec['iv_leakage_flag']}): "
                     f"{', '.join(flagged)}")
    if adjusted:
        lines.append(f"  zero-cell correction applied in: {', '.join(adjusted)}")
    lines.append(f"  sign convention: {spec['sign_convention']}")
    return "\n".join(lines)


# ===========================================================================
# Freezing
# ===========================================================================

def save_spec(spec, config, filename=None):
    """Write the WOE spec to data/processed/ as JSON and return the path.

    The per-bin tables are written too: a scorecard's WOE table is the artefact a
    model validator reads line by line, and it must survive outside this process.
    """
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("woe", {}).get(
        "spec_filename", "woe_spec.json"))

    serialisable = {k: v for k, v in spec.items() if k != "features"}
    serialisable["features"] = {
        feature: {
            **{k: v for k, v in definition.items() if k != "table"},
            "table": definition["table"].reset_index().to_dict(orient="records"),
        }
        for feature, definition in spec["features"].items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=_json_safe)
    return path


def load_spec(path):
    """Read a frozen WOE spec back, with its tables as DataFrames again."""
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    for definition in spec["features"].values():
        definition["table"] = pd.DataFrame(definition["table"]).set_index("bin")
    return spec


def _json_safe(value):
    """Make numpy and pandas types survive a JSON round trip."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)
