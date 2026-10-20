r"""
Stage 8 — Gradient-boosted benchmark.  [Built in Phase 7, 20 Oct 2026]

One question: **what does the interpretability cost?**

The scorecard was not chosen because it is the most accurate model available. It
was chosen because a points table can be shown to an applicant, signed off by a
credit committee and audited by a regulator. That is a real benefit with a real
price, and the honest thing is to **measure the price** rather than assert it is
small.

So a gradient-boosted model is fitted on the same split, and the gap is reported.
If the GBM wins by a lot, that is a finding about the cost of the choice, not a
reason to abandon it — but a candidate who cannot state the number has not
really made the choice.

What makes this a fair comparison
---------------------------------
Three things, each easy to get wrong in a way that flatters one side:

1. **The same frozen split.** Both models train on `train` and are measured on
   `test`, the split frozen in Phase 1. A GBM scored on its own fresh split would
   be meaningless.
2. **The GBM gets the raw features, not the WOE.** Binning and WOE exist to make
   a *linear* model work; handing a tree model the pre-binned inputs would throw
   away its main advantage, which is finding non-linearity and interaction on its
   own. Giving it the raw data is what makes the comparison honest.
3. **The protected characteristic is excluded from both.** A GBM that uses sex to
   win would not be lawful either, and the comparison would be between a model
   that can be used and one that cannot.

Why scikit-learn rather than LightGBM or XGBoost
------------------------------------------------
`HistGradientBoostingClassifier` is a histogram-based gradient booster — the same
family and approach as LightGBM — and it ships with scikit-learn, which is
already a dependency. Adding a library to produce a benchmark number is a poor
trade when the benchmark is a comparison rather than a product. It handles
categoricals and missing values natively, which also keeps the preparation
honest: no imputation choices that could tilt the result.

See docs/methodology.md section 8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import validate


class BenchmarkError(Exception):
    """Raised when the benchmark cannot be fitted or compared."""


def prepare_features(frame, config=None, drop_protected=True):
    """Raw features for the tree model, with protected characteristics removed.

    Categoricals become pandas ``category`` dtype, which
    `HistGradientBoostingClassifier` handles natively — no one-hot expansion and
    no imputation decisions that could tilt the comparison.
    """
    config = config or {}
    schema = config.get("schema", {})
    numeric = [c for c in schema.get("numeric_columns", []) if c in frame.columns]
    categorical = [c for c in schema.get("categorical_columns", [])
                   if c in frame.columns]

    if drop_protected:
        protected = set(config.get("fair_lending", {}).get(
            "protected_characteristics", []))
        removed = [c for c in numeric + categorical if c in protected]
        numeric = [c for c in numeric if c not in protected]
        categorical = [c for c in categorical if c not in protected]
    else:
        removed = []

    if not numeric and not categorical:
        raise BenchmarkError("No usable features for the benchmark.")

    out = frame[numeric + categorical].copy()
    for column in categorical:
        out[column] = out[column].astype("category")
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return out, {
        "numeric": numeric,
        "categorical": categorical,
        "protected_removed": removed,
        # What the guard is watching, whether or not it had to act. A reader of
        # the report needs to know protected characteristics were considered,
        # not just that none were removed at this step.
        "protected_watched": sorted(config.get("fair_lending", {}).get(
            "protected_characteristics", [])),
    }


def fit_benchmark(train, test, config=None, target_column="is_default"):
    """Fit the gradient-boosted benchmark and score it on the frozen test half.

    Returns
    -------
    dict
        The fitted model, its predictions, and the feature sets used.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    config = config or {}
    cfg = config.get("benchmark", {})
    seed = int(config.get("seed", 42))

    if target_column not in train.columns:
        raise BenchmarkError(f"'{target_column}' is missing from the training data.")

    X_train, used = prepare_features(train, config)
    X_test, _ = prepare_features(test, config)
    y_train = train[target_column].astype(int)
    y_test = test[target_column].astype(int)

    # Deliberately modest settings. The point is a fair benchmark, not a tuned
    # competitor: a heavily tuned GBM on 700 rows would mostly be measuring how
    # much hyperparameter search the test set can absorb.
    estimator = HistGradientBoostingClassifier(
        max_iter=int(cfg.get("max_iter", 200)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 20)),
        l2_regularization=float(cfg.get("l2_regularization", 1.0)),
        early_stopping=bool(cfg.get("early_stopping", True)),
        validation_fraction=float(cfg.get("validation_fraction", 0.15)),
        categorical_features="from_dtype",
        random_state=seed,
    )
    estimator.fit(X_train, y_train)

    return {
        "model": estimator,
        "features": used,
        "train_pd": estimator.predict_proba(X_train)[:, 1],
        "test_pd": estimator.predict_proba(X_test)[:, 1],
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "settings": {
            "estimator": "HistGradientBoostingClassifier",
            "max_iter": estimator.max_iter,
            "learning_rate": estimator.learning_rate,
            "iterations_used": int(estimator.n_iter_),
        },
    }


def compare(scorecard_train_pd, scorecard_test_pd, benchmark, train_target,
            test_target, config=None):
    """Score both models on the same held-out half and report the gap.

    Returns
    -------
    dict
        A comparison table plus the headline gap and a verdict.
    """
    config = config or {}

    def metrics_for(train_pd, test_pd, label):
        result = validate.evaluate(train_pd, train_target, test_pd, test_target,
                                   config)
        discrimination = result["discrimination"]
        return {
            "model": label,
            "test_auc": discrimination["test"]["auc"],
            "test_gini": discrimination["test"]["gini"],
            "test_ks": discrimination["ks"]["ks"],
            "train_gini": discrimination["train"]["gini"],
            "gini_drop_train_to_test": discrimination["gini_drop"],
            "calibration_rejected": result["calibration"][
                "hosmer_lemeshow"]["calibration_rejected_at_5pct"],
            "psi": result["stability"]["psi"],
            "rank_monotonic": result["rank_ordering"]["monotonic"],
        }

    rows = [
        metrics_for(scorecard_train_pd, scorecard_test_pd, "Scorecard (logistic + WOE)"),
        metrics_for(benchmark["train_pd"], benchmark["test_pd"],
                    "Benchmark (gradient boosting)"),
    ]
    table = pd.DataFrame(rows)

    scorecard_gini = rows[0]["test_gini"]
    benchmark_gini = rows[1]["test_gini"]
    gap = benchmark_gini - scorecard_gini
    relative = gap / abs(scorecard_gini) if scorecard_gini else np.nan

    # The overfitting comparison matters as much as the raw gap. A GBM that wins
    # on test while dropping far more from train to test has bought its
    # advantage with variance, and on a book this size that advantage is the
    # first thing to disappear next quarter.
    overfit_gap = (rows[1]["gini_drop_train_to_test"]
                   - rows[0]["gini_drop_train_to_test"])

    return {
        "table": table,
        "scorecard_gini": scorecard_gini,
        "benchmark_gini": benchmark_gini,
        "gini_gap": float(gap),
        "relative_gap": float(relative),
        "overfit_gap": float(overfit_gap),
        "benchmark_wins": bool(gap > 0),
        "features_used": benchmark["features"],
        "settings": benchmark["settings"],
    }


def benchmark_report(comparison):
    """The accuracy-vs-interpretability table, with the trade-off spelled out."""
    table = comparison["table"]
    lines = ["Benchmark: scorecard vs gradient boosting", "-" * 41]

    display = table[["model", "test_gini", "test_auc", "test_ks",
                     "train_gini", "gini_drop_train_to_test",
                     "rank_monotonic"]]
    lines.append(display.to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))

    features = comparison["features_used"]
    lines.append("")
    lines.append(f"  benchmark: {comparison['settings']['estimator']}, "
                 f"{comparison['settings']['iterations_used']} iterations")
    lines.append(f"  features given to it: {len(features['numeric'])} numeric + "
                 f"{len(features['categorical'])} categorical, RAW (not binned)")
    if features["protected_removed"]:
        lines.append(f"  excluded from BOTH models here: "
                     f"{features['protected_removed']}")
    elif features.get("protected_watched"):
        # Nothing to remove at this stage because it was already excluded from
        # the schema upstream. Say so, otherwise a reader cannot tell whether
        # protected characteristics were considered at all.
        lines.append(f"  protected characteristics "
                     f"{features['protected_watched']}: already excluded from "
                     f"the schema, so neither model sees them")

    gap = comparison["gini_gap"]
    lines.append("")
    lines.append(f"  Gini gap (benchmark - scorecard): {gap:+.4f}"
                 f"   ({comparison['relative_gap']:+.1%} relative)")
    lines.append(f"  Overfitting gap (benchmark - scorecard, train->test drop): "
                 f"{comparison['overfit_gap']:+.4f}")

    lines.append("")
    lines.append("  What the scorecard buys for that gap:")
    lines.append("    - a points table an applicant can be shown")
    lines.append("    - reason codes read off the decision arithmetic itself, "
                 "not a surrogate")
    lines.append("    - monotonic risk by construction, so cut-off policy is "
                 "defensible")
    lines.append("    - scoreable by hand, years later, from a JSON file")

    lines.append("")
    if gap <= 0:
        lines.append("  The benchmark does NOT beat the scorecard on held-out "
                     "discrimination.")
        lines.append("  On a book this size that is a common and honest result: "
                     "there is not")
        lines.append("  enough data for a tree model's flexibility to pay for "
                     "itself. The")
        lines.append("  interpretability is free here.")
    elif gap < 0.05:
        lines.append(f"  The benchmark wins by {gap:.4f} Gini — a small margin.")
        lines.append("  That is a cheap price for everything in the list above, "
                     "and it is the")
        lines.append("  number to quote when asked why a logistic model was "
                     "chosen.")
    else:
        lines.append(f"  The benchmark wins by {gap:.4f} Gini — a MATERIAL "
                     "margin.")
        lines.append("  This is a real finding and should be reported, not "
                     "buried. The honest")
        lines.append("  position: the scorecard is still the right choice where "
                     "explainability is")
        lines.append("  a legal requirement, and the gap is what that "
                     "requirement costs.")

    if comparison["overfit_gap"] > 0.05:
        lines.append("")
        lines.append("  Note the overfitting gap: the benchmark drops further "
                     "from train to test")
        lines.append("  than the scorecard does, so part of its advantage is "
                     "variance rather than")
        lines.append("  signal — and variance is the first thing to disappear "
                     "next quarter.")

    return "\n".join(lines)


def run_benchmark(df, config=None):
    """Backwards-compatible name from the Phase 0 scaffold."""
    raise BenchmarkError(
        "Use fit_benchmark(train, test, config) then compare(...)."
    )
