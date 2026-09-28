"""
Stage 1 — Data load & EDA.

Phase 0 (now): load the German Credit CSV and produce an EDA report — shape,
target balance, numeric summaries, categorical cardinalities, and null counts.

Phase 1 (next): define the good/bad target (raw 1 = good, 2 = bad -> 0/1) and
build a stratified train/test split. Look for the ``PHASE 1`` marker below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


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
