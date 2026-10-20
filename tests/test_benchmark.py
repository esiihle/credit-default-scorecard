"""
Phase 7 tests — the gradient-boosted benchmark.

The tests here are about **fairness of the comparison**, not about the
benchmark's accuracy. Three things are easy to get wrong in a way that flatters
one side, and each has a test:

* the benchmark must use the same frozen split;
* it must get the RAW features, not the WOE (binning exists to make a linear
  model work; pre-binning a tree model throws away its advantage);
* the protected characteristic must be excluded from BOTH models.

Run with:  pytest tests/test_benchmark.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import benchmark, binning, model, woe


@pytest.fixture
def split():
    """A train/test split with real signal, plus a protected characteristic."""
    rng = np.random.default_rng(0)
    n = 1400

    duration = rng.integers(6, 60, n)
    checking = rng.choice(["none", "low", "medium", "high"], n,
                          p=[0.25, 0.3, 0.27, 0.18])
    sex = rng.choice(["male", "female"], n)

    checking_risk = pd.Series(checking).map(
        {"none": 0.10, "low": 0.50, "medium": 0.32, "high": 0.18})
    # The protected characteristic carries real signal, as it does in the
    # original data — which is exactly why excluding it has a measurable cost.
    sex_risk = pd.Series(sex).map({"male": 0.26, "female": 0.34})

    risk = np.clip(0.3 * checking_risk + 0.2 * sex_risk
                   + 0.006 * (duration - 30) + 0.12, 0.02, 0.95)
    frame = pd.DataFrame({
        "duration_months": duration,
        "checking_status": checking,
        "personal_status_sex": sex,
        "is_default": (rng.random(n) < risk).astype(int),
    })

    cfg = load_config()
    cfg["schema"] = dict(cfg["schema"],
                         numeric_columns=["duration_months"],
                         categorical_columns=["checking_status",
                                              "personal_status_sex"])
    cfg["fair_lending"] = {"protected_characteristics": ["personal_status_sex"],
                           "on_protected_reason_code": "refuse"}

    cut = int(n * 0.7)
    return {"config": cfg, "train": frame.iloc[:cut].reset_index(drop=True),
            "test": frame.iloc[cut:].reset_index(drop=True)}


def scorecard_predictions(split):
    """Fit the real scorecard chain, so the comparison is against the real thing."""
    cfg = dict(split["config"])
    # The lawful scorecard excludes the protected characteristic.
    cfg["schema"] = dict(cfg["schema"], categorical_columns=["checking_status"])

    bin_spec = binning.fit_bins(split["train"], cfg)
    train_binned = binning.apply_bins(split["train"], bin_spec)
    test_binned = binning.apply_bins(split["test"], bin_spec)

    woe_spec = woe.compute_woe_iv(train_binned, split["train"]["is_default"], cfg)
    chosen = woe.selected_features(woe_spec)
    train_woe, _ = woe.apply_woe(train_binned, woe_spec, chosen)
    test_woe, _ = woe.apply_woe(test_binned, woe_spec, chosen)

    pd_model = model.fit_pd_model(train_woe, split["train"]["is_default"], cfg)
    return (model.predict_pd(pd_model, train_woe),
            model.predict_pd(pd_model, test_woe))


# =========================================================================
# Fairness of the comparison
# =========================================================================

def test_protected_characteristic_is_excluded_from_the_benchmark(split):
    """A GBM that uses sex to win would not be lawful either, and the comparison
    would be between a model that can be used and one that cannot."""
    X, used = benchmark.prepare_features(split["train"], split["config"])

    assert "personal_status_sex" not in X.columns
    assert used["protected_removed"] == ["personal_status_sex"]


def test_benchmark_gets_raw_features_not_woe(split):
    """Binning and WOE exist to make a LINEAR model work. Handing a tree model
    pre-binned inputs throws away its main advantage."""
    X, used = benchmark.prepare_features(split["train"], split["config"])

    assert not any(column.endswith("_woe") for column in X.columns)
    assert not any(column.endswith("_bin") for column in X.columns)
    # Raw numeric values, not bin labels.
    assert X["duration_months"].nunique() > 10


def test_categoricals_are_passed_natively(split):
    """Category dtype, so the booster handles them without one-hot expansion or
    imputation choices that could tilt the result."""
    X, used = benchmark.prepare_features(split["train"], split["config"])

    for column in used["categorical"]:
        assert str(X[column].dtype) == "category"


def test_both_models_are_scored_on_the_same_frozen_split(split):
    """A GBM scored on its own fresh split would be meaningless."""
    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])

    assert len(fitted["test_pd"]) == len(test_pd) == len(split["test"])
    assert len(fitted["train_pd"]) == len(train_pd) == len(split["train"])
    np.testing.assert_array_equal(fitted["y_test"],
                                  split["test"]["is_default"].to_numpy())


def test_no_usable_features_is_refused():
    cfg = {"schema": {"numeric_columns": [], "categorical_columns": []}}
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.prepare_features(pd.DataFrame({"x": [1, 2]}), cfg)


def test_missing_target_is_refused(split):
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.fit_benchmark(split["train"].drop(columns=["is_default"]),
                                split["test"], split["config"])


# =========================================================================
# The benchmark itself
# =========================================================================

def test_benchmark_produces_valid_probabilities(split):
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])

    for predictions in (fitted["train_pd"], fitted["test_pd"]):
        assert ((predictions > 0) & (predictions < 1)).all()


def test_benchmark_is_reproducible(split):
    """Same seed, same model — the comparison has to be stable to be quotable."""
    first = benchmark.fit_benchmark(split["train"], split["test"],
                                    split["config"])
    second = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])

    np.testing.assert_allclose(first["test_pd"], second["test_pd"])


def test_benchmark_learns_something(split):
    """A benchmark that cannot beat a coin flip is not a benchmark."""
    from src import validate

    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])
    auc = validate.auc_and_gini(fitted["test_pd"],
                                split["test"]["is_default"])["auc"]
    assert auc > 0.55


# =========================================================================
# The comparison and its verdict
# =========================================================================

def test_comparison_reports_both_models_on_the_same_metrics(split):
    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])

    result = benchmark.compare(train_pd, test_pd, fitted,
                               split["train"]["is_default"],
                               split["test"]["is_default"], split["config"])

    assert len(result["table"]) == 2
    assert set(result["table"]["model"]) == {
        "Scorecard (logistic + WOE)", "Benchmark (gradient boosting)"}
    for column in ("test_gini", "test_auc", "test_ks",
                   "gini_drop_train_to_test"):
        assert column in result["table"].columns


def test_comparison_quantifies_the_gap_both_ways(split):
    """Absolute and relative, because a 0.02 gap means different things against
    a Gini of 0.20 and a Gini of 0.60."""
    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])

    result = benchmark.compare(train_pd, test_pd, fitted,
                               split["train"]["is_default"],
                               split["test"]["is_default"], split["config"])

    assert result["gini_gap"] == pytest.approx(
        result["benchmark_gini"] - result["scorecard_gini"])
    assert np.isfinite(result["relative_gap"])
    assert result["benchmark_wins"] == (result["gini_gap"] > 0)


def test_comparison_reports_the_overfitting_gap(split):
    """A GBM that wins on test while dropping further from train has bought its
    advantage with variance, and variance disappears next quarter."""
    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])

    result = benchmark.compare(train_pd, test_pd, fitted,
                               split["train"]["is_default"],
                               split["test"]["is_default"], split["config"])

    assert "overfit_gap" in result
    assert np.isfinite(result["overfit_gap"])


def test_report_states_what_the_scorecard_buys(split):
    """The report has to name the benefits, or the trade-off is only half shown."""
    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])
    result = benchmark.compare(train_pd, test_pd, fitted,
                               split["train"]["is_default"],
                               split["test"]["is_default"], split["config"])

    report = benchmark.benchmark_report(result)

    assert "points table an applicant can be shown" in report
    assert "reason codes" in report
    assert "excluded from BOTH models" in report
    assert "Gini gap" in report


def test_report_does_not_hide_a_material_loss(split):
    """If the benchmark wins by a lot, the report must say so plainly rather
    than reaching for the interpretability argument alone."""
    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"],
                                     split["config"])
    result = benchmark.compare(train_pd, test_pd, fitted,
                               split["train"]["is_default"],
                               split["test"]["is_default"], split["config"])

    # Force the material-loss branch and check the wording.
    result["gini_gap"] = 0.12
    report = benchmark.benchmark_report(result)

    assert "MATERIAL" in report
    assert "reported, not buried" in report


def test_old_entry_point_is_refused(split):
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.run_benchmark(split["train"], split["config"])


def test_report_says_protected_characteristics_were_considered(split):
    """Even when nothing needed removing at this step.

    If the characteristic was already dropped from the schema upstream,
    `protected_removed` is empty — and a reader of the report would have no way
    to tell whether protected characteristics were considered at all.
    """
    cfg = dict(split["config"])
    # The lawful setup: already excluded from the schema.
    cfg["schema"] = dict(cfg["schema"], categorical_columns=["checking_status"])

    train_pd, test_pd = scorecard_predictions(split)
    fitted = benchmark.fit_benchmark(split["train"], split["test"], cfg)
    result = benchmark.compare(train_pd, test_pd, fitted,
                               split["train"]["is_default"],
                               split["test"]["is_default"], cfg)

    assert fitted["features"]["protected_removed"] == []
    assert fitted["features"]["protected_watched"] == ["personal_status_sex"]

    report = benchmark.benchmark_report(result)
    assert "personal_status_sex" in report
    assert "neither model sees them" in report
