"""
Stage 1 — Data load & EDA.

Phase 0 (done): load the German Credit data and produce an EDA report — shape,
target balance, numeric summaries, categorical cardinalities, and null counts.

30 Sep 2026: added the REAL data path. The UCI file (``german.data``) is
space-delimited, has no header row, and codes every categorical value as A11,
A34, ... — unreadable as-is. The column order and the full code table live here
so decoding is one documented, testable step rather than guesswork spread
through the pipeline.

Phase 1 (next): define the good/bad target (raw 1 = good, 2 = bad -> 0/1) and
build a stratified train/test split. Look for the ``PHASE 1`` marker below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# The UCI file's 21 columns, IN ORDER. Names match config.yaml's schema block.
# Source: german.doc (the dataset's own documentation).
# ---------------------------------------------------------------------------
RAW_COLUMN_ORDER = [
    "checking_status",          # A11-A14   categorical
    "duration_months",          #           numeric (months)
    "credit_history",           # A30-A34   categorical
    "purpose",                  # A40-A410  categorical
    "credit_amount",            #           numeric (Deutsche Mark)
    "savings_status",           # A61-A65   categorical
    "employment_since",         # A71-A75   categorical
    "installment_rate",         #           numeric (% of disposable income)
    "personal_status_sex",      # A91-A95   categorical  <-- see the note below
    "other_debtors",            # A101-A103 categorical
    "residence_since",          #           numeric (years)
    "property",                 # A121-A124 categorical
    "age_years",                #           numeric (years)
    "other_installment_plans",  # A141-A143 categorical
    "housing",                  # A151-A153 categorical
    "existing_credits",         #           numeric (count at this bank)
    "job",                      # A171-A174 categorical
    "num_liable",               #           numeric (people liable for maintenance)
    "telephone",                # A191-A192 categorical
    "foreign_worker",           # A201-A202 categorical
    "target",                   #           1 = good, 2 = bad (recoded in Phase 1)
]

# ---------------------------------------------------------------------------
# The code table. Decoding at load time means every later stage — binning, WOE,
# the points table, the reason codes shown to an applicant — reads in plain
# English instead of "A34". That is not cosmetic: a scorecard has to be
# explainable to a customer and a regulator.
#
# NOTE on `personal_status_sex`: this column encodes SEX, a protected attribute
# under fair-lending rules. It is loaded so the data stays faithful to source,
# and the decision on whether to model with it is taken explicitly in Phase 3.
# ---------------------------------------------------------------------------
VALUE_LABELS = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assigned for 1+ year",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture / equipment",
        "A43": "radio / television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced / separated",
        "A92": "female: divorced / separated / married",
        "A93": "male: single",
        "A94": "male: married / widowed",
        "A95": "female: single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "registered in customer name",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}


def decode_values(df, labels=None):
    """Replace the A-codes with readable labels, failing loudly on anything new.

    An unrecognised code means the file is not what we think it is — a different
    dataset version, a shifted column, a corrupt download. Silently leaving it
    as "A99" would quietly create a junk bin in Phase 2, so we stop here instead.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw frame with columns named per :data:`RAW_COLUMN_ORDER`.
    labels : dict | None
        Code table; defaults to :data:`VALUE_LABELS`.

    Returns
    -------
    pandas.DataFrame
        A copy with categorical codes replaced by their labels.
    """
    labels = labels if labels is not None else VALUE_LABELS
    out = df.copy()

    for column, mapping in labels.items():
        if column not in out.columns:
            continue
        found = set(out[column].dropna().unique())
        unknown = sorted(found - set(mapping))
        if unknown:
            raise ValueError(
                f"Unknown codes in column '{column}': {unknown}. "
                f"Expected one of {sorted(mapping)}. "
                "Check the download is the UCI 'german.data' file."
            )
        out[column] = out[column].map(mapping)

    return out


def load_german_raw(path, config=None, decode=True):
    """Load the raw UCI ``german.data`` file into a readable DataFrame.

    The file is whitespace-delimited with no header, so column names come from
    :data:`RAW_COLUMN_ORDER` and categorical codes are decoded via
    :func:`decode_values`.

    The target is deliberately left in its raw form (1 = good, 2 = bad); the
    recode to 0/1 belongs to Phase 1, where the good/bad definition is stated.

    Returns
    -------
    pandas.DataFrame
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    # sep=r"\s+" handles one-or-more spaces; header=None because the file has no
    # header row at all — the very first line is already an applicant.
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=RAW_COLUMN_ORDER)

    # A shape check here is cheap insurance: if the file has a different number
    # of columns, everything downstream would be silently misaligned.
    if len(df.columns) != len(RAW_COLUMN_ORDER):
        raise ValueError(
            f"Expected {len(RAW_COLUMN_ORDER)} columns, found {len(df.columns)}."
        )

    if decode:
        df = decode_values(df)

    if config is not None:
        target = config.get("schema", {}).get("target_column")
        if target and target not in df.columns:
            raise ValueError(f"Target column '{target}' missing after load.")

    return df


def load_data(path, config=None):
    """Load the credit CSV into a DataFrame with a light target check.

    Parameters
    ----------
    path : str | Path
        Path to the CSV (readable-name German Credit format).
    config : dict | None
        Optional config. If given, the target column's presence is checked so a
        missing target fails here, loudly, rather than deep in the pipeline.

    Returns
    -------
    pandas.DataFrame
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if config is not None:
        target = config.get("schema", {}).get("target_column")
        if target and target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found. Columns: {list(df.columns)}"
            )

    # PHASE 1 goes here: recode target (2 -> 1 default, 1 -> 0 good) and build the
    # stratified train/test split, returning the pieces the model stage needs.
    return df


def build_eda_report(df, config=None):
    """Return a short, human-readable EDA summary of the credit data.

    Reports row/column counts, the target balance, numeric column ranges,
    categorical cardinalities, and per-column null counts. Written defensively so
    it works on the bundled sample before any Phase 1 cleaning.
    """
    lines = []
    lines.append(f"rows: {len(df):,}")
    lines.append(f"columns: {len(df.columns)}")

    target = config.get("schema", {}).get("target_column") if config else None
    if target and target in df.columns:
        counts = df[target].value_counts(dropna=False).sort_index()
        total = len(df)
        dist = ", ".join(f"{k}={v} ({v / total:.0%})" for k, v in counts.items())
        lines.append(f"target '{target}' distribution: {dist}")

    num_cols, cat_cols = [], []
    if config is not None:
        num_cols = config.get("schema", {}).get("numeric_columns", [])
        cat_cols = config.get("schema", {}).get("categorical_columns", [])
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    if num_cols:
        lines.append("numeric columns (min / mean / max):")
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            lines.append(f"  {c}: {s.min():.0f} / {s.mean():.1f} / {s.max():.0f}")

    if cat_cols:
        lines.append("categorical columns (distinct values):")
        for c in cat_cols:
            lines.append(f"  {c}: {df[c].nunique()}")

    null_counts = df.isna().sum()
    nulls = null_counts[null_counts > 0]
    lines.append(
        "nulls: none"
        if len(nulls) == 0
        else "nulls: " + ", ".join(f"{c}={n}" for c, n in nulls.items())
    )

    return "\n".join(lines)
