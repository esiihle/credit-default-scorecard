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

# Repo root = two levels up from this file (src/load.py -> src -> repo root).
# Used to resolve the relative paths in config.yaml the same way from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]

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


# ===========================================================================
# Phase 1 — target definition, validation, frozen train/test split
# ===========================================================================
#
# The order matters and is deliberate:
#
#   validate_raw -> clean -> define_target -> to_analysis_table -> split
#
# The target is defined BEFORE the split, and the split is frozen to disk with a
# manifest. Nothing after this point may re-split, re-sample, or peek at test.


class DataQualityError(Exception):
    """Raised when the input is too broken to model on.

    A distinct exception type so a caller can tell "this data is unusable" apart
    from an ordinary programming error.
    """


def validate_raw(df, config):
    """Structural checks before any cleaning. Raises on anything fatal.

    Returns
    -------
    list of str
        Non-fatal notes for the data-quality report.
    """
    notes = []
    schema = config.get("schema", {})
    target = schema.get("target_column", "target")

    if len(df) == 0:
        raise DataQualityError("The input has no rows at all.")

    expected = list(schema.get("numeric_columns", [])) + \
        list(schema.get("categorical_columns", [])) + [target]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataQualityError(
            f"Required columns are missing: {missing}. "
            f"Columns present: {sorted(df.columns)}"
        )

    raw_codes = set(df[target].dropna().unique())
    if not raw_codes <= {1, 2}:
        raise DataQualityError(
            f"Target column '{target}' holds unexpected codes: {sorted(raw_codes)}. "
            "Expected 1 (good) and 2 (bad)."
        )
    if len(raw_codes) < 2:
        raise DataQualityError(
            "The target has only one class — a scorecard cannot be built from it."
        )

    return notes


def clean(df, config):
    """Apply plausibility rules and count every row changed or removed.

    Nothing here is clever. It exists because a single 200-year-old applicant or
    a duplicated row will quietly distort a bin boundary in Phase 2, and that
    distortion is very hard to spot once it has reached a points table.
    """
    rules = config.get("validation", {})
    report = {"rows_in": len(df)}
    out = df.copy()

    # --- Numeric coercion -------------------------------------------------
    for col in config.get("schema", {}).get("numeric_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- Plausibility ranges ----------------------------------------------
    # Each limit lives in config.yaml, so a reviewer can argue with the number
    # without reading any code.
    checks = {
        "age_years": (rules.get("min_age", 18), rules.get("max_age", 100)),
        "duration_months": (rules.get("min_duration_months", 1),
                            rules.get("max_duration_months", 120)),
        "credit_amount": (rules.get("min_credit_amount", 1),
                          rules.get("max_credit_amount", 100000)),
    }

    implausible = pd.Series(False, index=out.index)
    for col, (low, high) in checks.items():
        if col not in out.columns:
            continue
        bad = out[col].isna() | (out[col] < low) | (out[col] > high)
        report[f"dropped_implausible_{col}"] = int(bad.sum())
        implausible = implausible | bad

    out = out[~implausible]

    # --- Duplicates -------------------------------------------------------
    # This dataset has no applicant ID, so a duplicate is an exact repeat of
    # every field. Two genuinely identical applications are possible but rare;
    # a repeated row is far more likely to be a loading error.
    if rules.get("drop_duplicate_rows", True):
        duplicated = out.duplicated(keep="first")
        report["dropped_duplicate_rows"] = int(duplicated.sum())
        out = out[~duplicated]
    else:
        report["dropped_duplicate_rows"] = 0

    report["rows_out"] = len(out)
    return out.reset_index(drop=True), report


def define_target(df, config, column="is_default"):
    """Create the modelled target: 1 = default (bad), 0 = good.

    The good/bad definition, stated plainly so it can be defended:

      * **Bad (1)** = the applicant is labelled 2 in the source, which the UCI
        documentation defines as a customer with *bad credit risk* — the loan did
        not perform.
      * **Good (0)** = labelled 1, a customer whose credit performed.

    Three things worth knowing about that definition, all recorded in the
    methodology rather than glossed over:

      1. The source gives no delinquency depth (30/60/90 days past due) and no
         performance window, so "bad" cannot be tightened to a Basel-style
         90-days-past-due rule. We inherit the publisher's definition.
      2. Every applicant in the file was **granted** credit, so this is an
         accepted-population model. Reject inference is out of scope.
      3. 1 is the event we model, so a higher predicted probability means a worse
         applicant. That direction has to stay consistent through WOE, the
         coefficients and the final points table.
    """
    schema = config.get("schema", {})
    target = schema.get("target_column", "target")
    bad_code = schema.get("bad_value_raw", 2)

    out = df.copy()
    out[column] = (out[target] == bad_code).astype(int)
    return out


def to_analysis_table(df, config, id_prefix="APP"):
    """Add a stable applicant ID and order the columns for analysis.

    German Credit has no identifier of its own, so one is built from row
    position: ``APP_0001`` and so on. That is only stable as long as the source
    file's row order is — which it is, because the file is a static published
    download and we never edit it. Noted here because an ID that silently
    changes meaning is worse than no ID at all.
    """
    schema = config.get("schema", {})
    out = df.copy()

    out.insert(0, "applicant_id",
               [f"{id_prefix}_{i + 1:04d}" for i in range(len(out))])

    ordered = (["applicant_id"]
               + list(schema.get("numeric_columns", []))
               + list(schema.get("categorical_columns", []))
               + [schema.get("target_column", "target"), "is_default"])
    ordered = [c for c in ordered if c in out.columns]
    return out[ordered]


def split_train_test(df, config, target_column="is_default"):
    """Split into train and test, stratified on the target, from the seed.

    Stratified because the bad rate is ~30%: an unstratified split can easily
    hand the test set a materially different bad rate, and every metric computed
    on it would then be measuring the split rather than the model.

    Returns
    -------
    (train, test) : tuple of DataFrame
    """
    from sklearn.model_selection import train_test_split

    split_cfg = config.get("split", {})
    seed = config.get("seed", 42)
    test_size = split_cfg.get("test_size", 0.30)
    stratify = df[target_column] if split_cfg.get("stratify", True) else None

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,      # the seed is what makes this reproducible
        stratify=stratify,
        shuffle=True,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_manifest(train, test, config, target_column="is_default"):
    """Describe the split precisely enough to prove a later run is identical.

    The fingerprint is a hash of the sorted applicant IDs in each half. If a
    future run produces the same two fingerprints, the split is byte-identical —
    which is how "reproducible from a seed" stops being a claim and becomes a
    check.
    """
    import hashlib
    from datetime import datetime, timezone

    def fingerprint(frame):
        ids = ",".join(sorted(frame["applicant_id"].astype(str)))
        return hashlib.sha256(ids.encode()).hexdigest()[:16]

    def describe(frame):
        return {
            "rows": int(len(frame)),
            "bad_rate": round(float(frame[target_column].mean()), 4),
            "bads": int(frame[target_column].sum()),
            "fingerprint": fingerprint(frame),
        }

    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config.get("seed", 42),
        "test_size": config.get("split", {}).get("test_size", 0.30),
        "stratified": bool(config.get("split", {}).get("stratify", True)),
        "out_of_time_slice": bool(config.get("split", {}).get("out_of_time", False)),
        "target_column": target_column,
        "target_definition": "1 = source code 2 (bad credit risk); 0 = source code 1 (good)",
        "train": describe(train),
        "test": describe(test),
    }


def save_splits(train, test, manifest, config):
    """Write train.csv, test.csv and the manifest. Returns the three paths."""
    import json

    split_cfg = config.get("split", {})
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / split_cfg.get("train_filename", "train.csv")
    test_path = out_dir / split_cfg.get("test_filename", "test.csv")
    manifest_path = out_dir / split_cfg.get("manifest_filename", "split_manifest.json")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return train_path, test_path, manifest_path


def quality_report(report, train=None, test=None, manifest=None):
    """Render the Phase 1 counts and the resulting split as readable text."""
    lines = ["Data-quality report", "-" * 19]
    labels = {
        "rows_in": "rows read",
        "dropped_implausible_age_years": "dropped: age outside plausible range",
        "dropped_implausible_duration_months": "dropped: loan term outside range",
        "dropped_implausible_credit_amount": "dropped: credit amount outside range",
        "dropped_duplicate_rows": "dropped: exact duplicate rows",
        "rows_out": "rows kept",
    }
    for key, label in labels.items():
        if key in report:
            lines.append(f"  {label:<40} {report[key]:>6,}")
    if report.get("rows_in"):
        lines.append(f"  {'share of rows kept':<40} "
                     f"{report['rows_out'] / report['rows_in']:>6.1%}")

    if train is not None and test is not None:
        lines += ["", "Frozen split", "-" * 12]
        lines.append(f"  {'train rows':<40} {len(train):>6,}")
        lines.append(f"  {'test rows':<40} {len(test):>6,}")
        if manifest:
            lines.append(f"  {'train bad rate':<40} "
                         f"{manifest['train']['bad_rate']:>6.1%}")
            lines.append(f"  {'test bad rate':<40} "
                         f"{manifest['test']['bad_rate']:>6.1%}")
            lines.append(f"  {'seed':<40} {manifest['seed']:>6}")
            lines.append(f"  train fingerprint: {manifest['train']['fingerprint']}")
            lines.append(f"  test  fingerprint: {manifest['test']['fingerprint']}")
    return "\n".join(lines)


def run_phase1(df_raw, config, save=True):
    """Run validation, target definition and the frozen split end to end.

    Returns
    -------
    (train, test, report, manifest)
    """
    rules = config.get("validation", {})

    notes = validate_raw(df_raw, config)
    cleaned, report = clean(df_raw, config)
    report["notes"] = notes

    # The guard rail: losing most of the file is a reason to stop, not to shrug.
    dropped_fraction = 1 - (report["rows_out"] / report["rows_in"])
    limit = rules.get("max_dropped_fraction", 0.05)
    if dropped_fraction > limit:
        raise DataQualityError(
            f"Dropped {dropped_fraction:.1%} of rows, above the {limit:.0%} limit. "
            "Inspect the raw file before continuing — do not relax this silently."
        )

    labelled = define_target(cleaned, config)
    table = to_analysis_table(labelled, config)
    train, test = split_train_test(table, config)
    manifest = split_manifest(train, test, config)

    if save:
        train_path, test_path, manifest_path = save_splits(train, test, manifest, config)
        report["saved_to"] = [str(train_path), str(test_path), str(manifest_path)]

    return train, test, report, manifest
