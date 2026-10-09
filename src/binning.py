"""
Stage 2 — Binning / coarse classing.  [Built in Phase 2, 5 Oct 2026]

What this stage does, and why a scorecard needs it at all:

A scorecard does not use raw values. It uses **bins** — "loan term 0-12 months",
"13-24 months", and so on — because bins give three things a credit committee
insists on:

1. **Explainability.** A points table reads as a handful of plain statements.
2. **Stability.** A bin holding 15% of the book moves slowly; a single raw value
   moves with every new applicant.
3. **Monotonicity.** "Longer term, more risk" is a claim a business can check
   against intuition. A bad rate that zig-zags across bins is almost always noise
   that has been fitted, and it is indefensible in a model review.

The method here is deliberately simple and auditable:

    fine-class (quantiles)  ->  merge until every constraint holds  ->  freeze

Merging is always done between **adjacent** bins, and always the pair that is
cheapest to merge (the smallest difference in bad rate), so the bins that survive
are the ones that genuinely separate risk.

Three constraints, all from config.yaml:
  * ``binning.min_bin_fraction`` — no bin smaller than this share of accounts
  * ``binning.max_bins``         — at most this many bins per feature
  * ``binning.enforce_monotonic`` — bad rate must move in one direction

**Fit on train only.** The returned spec is then applied unchanged to test, and
in Phase 7 to a single live applicant. That is the whole point of freezing it to
a file.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

MISSING_LABEL = "MISSING"      # value absent
UNSEEN_LABEL = "UNSEEN"        # a category never seen during fitting


class BinningError(Exception):
    """Raised when a feature cannot be binned under the configured constraints."""


# ===========================================================================
# Internal helpers — bins as a simple list of dicts, so every step is readable
# ===========================================================================

def _bin_stats(groups):
    """Attach count, bads and bad rate to each bin. Returns the same list."""
    for b in groups:
        b["count"] = int(b["count"])
        b["bads"] = int(b["bads"])
        b["bad_rate"] = (b["bads"] / b["count"]) if b["count"] else np.nan
    return groups


def _merge_pair(groups, i, numeric=True):
    """Merge bin i with bin i+1 and return the new, shorter list."""
    left, right = groups[i], groups[i + 1]
    merged = {
        "count": left["count"] + right["count"],
        "bads": left["bads"] + right["bads"],
    }
    if numeric:
        # Bins are adjacent on the number line, so the merged bin simply spans both.
        merged["left"] = left["left"]
        merged["right"] = right["right"]
    else:
        # Categorical bins are sets of levels; merging unions them.
        merged["levels"] = list(left["levels"]) + list(right["levels"])

    return _bin_stats(groups[:i] + [merged] + groups[i + 2:])


def _cheapest_pair(groups, candidates=None):
    """Index of the adjacent pair with the smallest bad-rate gap.

    Merging the closest pair first means the merges that happen are the ones that
    destroy the least information.
    """
    options = candidates if candidates is not None else range(len(groups) - 1)
    return min(options, key=lambda i: abs(groups[i]["bad_rate"] - groups[i + 1]["bad_rate"]))


def _monotonic_violations(groups):
    """Indices where the bad-rate trend reverses.

    Direction is taken from the overall trend (first bin vs last bin), then any
    adjacent pair moving the other way is a violation.
    """
    rates = [b["bad_rate"] for b in groups]
    if len(rates) < 3:
        return []

    rising = rates[-1] >= rates[0]
    violations = []
    for i in range(len(rates) - 1):
        step = rates[i + 1] - rates[i]
        if (rising and step < 0) or (not rising and step > 0):
            violations.append(i)
    return violations


def _apply_constraints(groups, total, min_fraction, max_bins, enforce_monotonic,
                       numeric=True):
    """Merge bins until every configured constraint is satisfied.

    Order matters: size first (a tiny bin's bad rate is noise, so it should not
    be allowed to drive the monotonicity decision), then count, then direction.
    """
    guard = 0
    while len(groups) > 1:
        guard += 1
        if guard > 200:                      # cannot happen; a safety net, not logic
            raise BinningError("Binning did not converge — check the input data.")

        # 1. Any bin too small?
        small = [i for i, b in enumerate(groups) if b["count"] / total < min_fraction]
        if small:
            i = small[0]
            # Merge with whichever neighbour it most resembles.
            if i == 0:
                groups = _merge_pair(groups, 0, numeric)
            elif i == len(groups) - 1:
                groups = _merge_pair(groups, i - 1, numeric)
            else:
                left_gap = abs(groups[i]["bad_rate"] - groups[i - 1]["bad_rate"])
                right_gap = abs(groups[i]["bad_rate"] - groups[i + 1]["bad_rate"])
                groups = _merge_pair(groups, i - 1 if left_gap <= right_gap else i, numeric)
            continue

        # 2. Too many bins?
        if len(groups) > max_bins:
            groups = _merge_pair(groups, _cheapest_pair(groups), numeric)
            continue

        # 3. Trend reversing?
        if enforce_monotonic:
            violations = _monotonic_violations(groups)
            if violations:
                groups = _merge_pair(groups, _cheapest_pair(groups, violations), numeric)
                continue

        break

    return groups


# ===========================================================================
# Numeric features
# ===========================================================================

def fit_numeric_bins(values, target, config=None, fine_bins=20):
    """Learn bin edges for one numeric feature.

    Steps: quantile fine-classing, then merging until the constraints hold.

    Returns
    -------
    dict
        ``{"type": "numeric", "edges": [...], "bins": [...]}``. Edges run from
        -inf to +inf so a value outside the training range still lands in a bin —
        a new applicant aged 19 must not crash the scorer.
    """
    cfg = (config or {}).get("binning", {})
    min_fraction = cfg.get("min_bin_fraction", 0.05)
    max_bins = cfg.get("max_bins", 5)
    enforce_monotonic = cfg.get("enforce_monotonic", True)

    x = pd.to_numeric(pd.Series(values), errors="coerce")
    y = pd.Series(target).astype(int)
    keep = x.notna()
    x, y = x[keep], y[keep]

    if len(x) == 0:
        raise BinningError("No non-missing values to bin.")

    # --- Fine classing: many small quantile bins to start from ---------------
    # duplicates="drop" handles features with few distinct values, where several
    # quantile edges land on the same number (e.g. installment_rate, 1-4).
    cut = pd.qcut(x, q=min(fine_bins, x.nunique()), duplicates="drop")
    frame = pd.DataFrame({"bin": cut, "y": y})
    stats = frame.groupby("bin", observed=True)["y"].agg(["size", "sum"])

    groups = _bin_stats([
        {"left": float(interval.left), "right": float(interval.right),
         "count": row["size"], "bads": row["sum"]}
        for interval, row in stats.iterrows()
    ])

    groups = _apply_constraints(groups, len(x), min_fraction, max_bins,
                                enforce_monotonic, numeric=True)

    # Open the outer edges so unseen extremes still bin.
    edges = [-np.inf] + [b["right"] for b in groups[:-1]] + [np.inf]
    return {"type": "numeric", "edges": edges, "bins": groups}


# ===========================================================================
# Categorical features
# ===========================================================================

def fit_categorical_bins(values, target, config=None):
    """Learn level groupings for one categorical feature.

    Levels are sorted by bad rate and then merged like a numeric feature, so each
    final group holds levels of similar risk and at least ``min_bin_fraction`` of
    the book. Sorting by risk is what makes grouping meaningful: putting "car
    (new)" with "car (used)" is only sensible if they default at similar rates.

    Monotonicity is NOT enforced here — a categorical has no natural order, so
    the groups are ordered by risk by construction.
    """
    cfg = (config or {}).get("binning", {})
    min_fraction = cfg.get("min_bin_fraction", 0.05)
    max_bins = cfg.get("max_bins", 5)

    x = pd.Series(values).astype(str)
    y = pd.Series(target).astype(int)

    stats = pd.DataFrame({"x": x, "y": y}).groupby("x", observed=True)["y"].agg(["size", "sum"])
    stats["bad_rate"] = stats["sum"] / stats["size"]
    stats = stats.sort_values("bad_rate")

    groups = _bin_stats([
        {"levels": [level], "count": row["size"], "bads": row["sum"]}
        for level, row in stats.iterrows()
    ])

    groups = _apply_constraints(groups, len(x), min_fraction, max_bins,
                               enforce_monotonic=False, numeric=False)

    mapping = {}
    for group in groups:
        label = " | ".join(sorted(group["levels"]))
        group["label"] = label
        for level in group["levels"]:
            mapping[level] = label

    return {"type": "categorical", "mapping": mapping, "bins": groups}


# ===========================================================================
# Fitting and applying a whole spec
# ===========================================================================

def fit_bins(train, config=None, target_column="is_default"):
    """Learn a binning spec for every configured feature, from TRAIN only.

    Parameters
    ----------
    train : pandas.DataFrame
        The training half of the frozen split. Never pass test here.
    config : dict
        Needs ``schema.numeric_columns``, ``schema.categorical_columns`` and the
        ``binning`` block.

    Returns
    -------
    dict
        The spec: metadata plus one entry per feature. Freeze it with
        :func:`save_spec` and apply it everywhere else.
    """
    config = config or {}
    schema = config.get("schema", {})

    if target_column not in train.columns:
        raise BinningError(
            f"Target column '{target_column}' not in the training frame. "
            "Binning must see the target — it is supervised, not just cutting."
        )

    y = train[target_column]
    features = {}

    for col in schema.get("numeric_columns", []):
        if col in train.columns:
            features[col] = fit_numeric_bins(train[col], y, config)

    for col in schema.get("categorical_columns", []):
        if col in train.columns:
            features[col] = fit_categorical_bins(train[col], y, config)

    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "fitted_on_rows": int(len(train)),
        "target_column": target_column,
        "settings": config.get("binning", {}),
        "features": features,
    }


def bin_label(edges):
    """Readable labels for numeric bins, e.g. '<= 12.0', '12.0 - 24.0', '> 24.0'."""
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isneginf(left):
            labels.append(f"<= {right:g}")
        elif np.isposinf(right):
            labels.append(f"> {left:g}")
        else:
            labels.append(f"{left:g} - {right:g}")
    return labels


def apply_bins(df, spec):
    """Apply a frozen spec to any frame — test, or one live applicant.

    Two deliberate behaviours, because both situations happen in production:

    * a missing value becomes the ``MISSING`` bin rather than an error;
    * a category never seen during fitting becomes ``UNSEEN``.

    Neither is silently dropped. Phase 3 assigns them a WOE explicitly, which is
    how a scorer stays alive when a new product code appears in live data.
    """
    out = pd.DataFrame(index=df.index)

    for feature, definition in spec["features"].items():
        if feature not in df.columns:
            continue

        if definition["type"] == "numeric":
            edges = [float(e) for e in definition["edges"]]
            labels = bin_label(edges)
            values = pd.to_numeric(df[feature], errors="coerce")
            binned = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
            binned = binned.astype(object).where(values.notna(), MISSING_LABEL)
        else:
            mapping = definition["mapping"]
            binned = df[feature].astype(str).map(mapping)
            binned = binned.where(binned.notna(), UNSEEN_LABEL)

        out[f"{feature}_bin"] = binned

    return out


def spec_bin_order(definition):
    """The natural order of a feature's bins.

    Numeric bins run low to high; categorical groups run safest to riskiest.
    Without this, a summary sorts labels alphabetically — "24 - 41" lands before
    "<= 24" — and a perfectly monotonic feature looks broken. Presentation order
    is not cosmetic when the whole point of the stage is a visible trend.
    """
    if definition["type"] == "numeric":
        return bin_label([float(e) for e in definition["edges"]])
    return [b["label"] for b in definition["bins"]]


def bin_summary(binned, target, spec=None, max_features=None):
    """Counts and bad rate per bin, for eyeballing before trusting anything.

    Pass the spec to get bins in their natural order rather than alphabetical.
    """
    y = pd.Series(target).astype(int)
    frames = []

    columns = list(binned.columns)
    if max_features:
        columns = columns[:max_features]

    for col in columns:
        feature = col.replace("_bin", "")
        table = pd.DataFrame({"bin": binned[col], "y": y.values})
        stats = table.groupby("bin", observed=True)["y"].agg(
            accounts="size", bads="sum"
        )

        if spec and feature in spec.get("features", {}):
            order = spec_bin_order(spec["features"][feature])
            # Keep MISSING / UNSEEN at the end if they appear.
            extras = [b for b in stats.index if b not in order]
            stats = stats.reindex([b for b in order if b in stats.index] + extras)

        stats["share"] = stats["accounts"] / len(table)
        stats["bad_rate"] = stats["bads"] / stats["accounts"]
        stats.insert(0, "feature", feature)
        frames.append(stats.reset_index())

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def check_monotonic(binned, target, spec):
    """Report which numeric features ended up monotonic. A fit-quality check.

    Returns a dict ``{feature: "rising" | "falling" | "not monotonic"}``.
    """
    y = pd.Series(target).astype(int).values
    result = {}

    for feature, definition in spec["features"].items():
        if definition["type"] != "numeric":
            continue
        col = f"{feature}_bin"
        if col not in binned.columns:
            continue

        labels = bin_label([float(e) for e in definition["edges"]])
        table = pd.DataFrame({"bin": binned[col], "y": y})
        rates = (table.groupby("bin", observed=True)["y"].mean()
                 .reindex(labels).dropna().values)

        if len(rates) < 2:
            result[feature] = "single bin"
        elif np.all(np.diff(rates) >= 0):
            result[feature] = "rising"
        elif np.all(np.diff(rates) <= 0):
            result[feature] = "falling"
        else:
            result[feature] = "not monotonic"

    return result


# ===========================================================================
# Freezing the spec
# ===========================================================================

def save_spec(spec, config, filename=None):
    """Write the spec to data/processed/ as JSON and return the path.

    JSON, and human-readable: the bin boundaries of a scorecard are exactly the
    thing a reviewer will want to read, argue with, and sign off.
    """
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("binning", {}).get(
        "spec_filename", "binning_spec.json"))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=_json_safe)
    return path


def load_spec(path):
    """Read a frozen spec back. Used by Phase 7's scorer and by the tests."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(value):
    """Make numpy types and infinities survive a JSON round trip."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
    return value
