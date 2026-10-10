"""
Phase 4 tests — the logistic PD model.

The centrepiece is `test_all_coefficients_are_negative`. With WOE on the
positive-is-safer convention and 1 = default, every slope must be negative. A
flipped sign gives a scorecard that ranks applicants backwards and still prints
entirely plausible numbers, so it is asserted rather than eyeballed.

Run with:  pytest tests/test_model.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import binning, model, woe


@pytest.fixture
def config():
    return load_config()


def make_woe_training_set(n=1500, seed=0, n_noise=0):
    """A WOE matrix built the real way: through binning and WOE, not by hand.

    Two genuinely predictive features plus optional pure-noise columns, so the
    diagnostics have something true and something false to separate.
    """
    rng = np.random.default_rng(seed)

    duration = rng.integers(6, 60, n)
    checking = rng.choice(["none", "low", "medium", "high"], n,
                          p=[0.25, 0.3, 0.27, 0.18])
    checking_risk = pd.Series(checking).map(
        {"none": 0.12, "low": 0.48, "medium": 0.32, "high": 0.20})

    risk = np.clip(0.3 * checking_risk + 0.006 * (duration - 30) + 0.15, 0.02, 0.95)
    is_default = (rng.random(n) < risk).astype(int)

    frame = pd.DataFrame({
        "duration_months": duration,
        "checking_status": checking,
        "is_default": is_default,
    })
    for i in range(n_noise):
        frame[f"noise_{i}"] = rng.normal(0, 1, n)

    cfg = load_config()
    cfg["schema"] = dict(cfg["schema"],
                         numeric_columns=["duration_months"] +
                         [f"noise_{i}" for i in range(n_noise)],
                         categorical_columns=["checking_status"])

    bin_spec = binning.fit_bins(frame, cfg)
    binned = binning.apply_bins(frame, bin_spec)
    woe_spec = woe.compute_woe_iv(binned, frame["is_default"], cfg)

    # Take every feature, not just the IV-selected ones, so a test can look at
    # how the model treats a weak variable.
    all_features = list(woe_spec["features"].keys())
    matrix, _ = woe.apply_woe(binned, woe_spec, all_features)
    return matrix, frame["is_default"], woe_spec


# =========================================================================
# The sign check
# =========================================================================

def test_all_coefficients_are_negative(config):
    """WOE positive = safer, and 1 = default, so every slope must be negative.

    This is the single most important assertion in the project. A flipped sign
    produces a scorecard that ranks backwards and looks completely normal.
    """
    X, y, _ = make_woe_training_set()
    fitted = model.fit_pd_model(X[["duration_months_woe", "checking_status_woe"]],
                               y, config)

    slopes = fitted["coefficients"].drop(index="const")
    assert (slopes["coefficient"] < 0).all(), (
        f"positive coefficients found: "
        f"{slopes.index[slopes['coefficient'] > 0].tolist()}"
    )
    assert fitted["checks"]["wrong_sign"] == []


def test_flipped_woe_sign_is_caught(config):
    """Negate the WOE and the model must refuse, not quietly fit.

    This simulates the exact upstream mistake the check exists for.
    """
    X, y, _ = make_woe_training_set()
    flipped = -X[["duration_months_woe", "checking_status_woe"]]

    with pytest.raises(model.ModelError) as error:
        model.fit_pd_model(flipped, y, config, enforce_signs=True)

    assert "POSITIVE" in str(error.value)


def test_wrong_sign_is_reported_when_not_enforced(config):
    """With enforcement off the problem is reported rather than raised, so the
    diagnostics can be read before the feature set has settled."""
    X, y, _ = make_woe_training_set()
    flipped = -X[["duration_months_woe", "checking_status_woe"]]

    fitted = model.fit_pd_model(flipped, y, config, enforce_signs=False)

    assert len(fitted["checks"]["wrong_sign"]) == 2
    assert fitted["checks"]["all_passed"] is False
    assert "FAIL" in model.model_report(fitted)


# =========================================================================
# Diagnostics
# =========================================================================

def test_predictive_features_are_significant(config):
    X, y, _ = make_woe_training_set()
    fitted = model.fit_pd_model(X[["duration_months_woe", "checking_status_woe"]],
                               y, config)

    slopes = fitted["coefficients"].drop(index="const")
    assert (slopes["p_value"] < 0.05).all()
    assert fitted["checks"]["insignificant"] == []


def test_noise_feature_is_flagged_as_insignificant(config):
    X, y, _ = make_woe_training_set(n_noise=2)
    columns = [c for c in X.columns if "noise" in c] + ["duration_months_woe"]

    fitted = model.fit_pd_model(X[columns], y, config, enforce_signs=False)

    flagged = fitted["checks"]["insignificant"]
    assert any("noise" in name for name in flagged), (
        f"noise should be insignificant; flagged: {flagged}"
    )


def test_vif_is_one_for_independent_columns():
    """Uncorrelated predictors have no variance inflation, by definition."""
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame({
        "a": rng.normal(0, 1, 800),
        "b": rng.normal(0, 1, 800),
        "c": rng.normal(0, 1, 800),
    })
    vif = model.variance_inflation(matrix)

    assert (vif < 1.2).all(), f"independent columns should have VIF ~1: {vif.to_dict()}"


def test_vif_explodes_for_a_duplicated_column():
    """Two features saying the same thing is exactly what VIF is for."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 800)
    matrix = pd.DataFrame({
        "original": base,
        "near_copy": base + rng.normal(0, 0.01, 800),   # almost identical
        "unrelated": rng.normal(0, 1, 800),
    })
    vif = model.variance_inflation(matrix)

    assert vif["original"] > 10
    assert vif["near_copy"] > 10
    assert vif["unrelated"] < 1.2


def test_collinear_features_are_flagged(config):
    X, y, _ = make_woe_training_set()
    duplicated = X[["duration_months_woe", "checking_status_woe"]].copy()
    duplicated["duration_copy_woe"] = (
        duplicated["duration_months_woe"] + np.random.default_rng(2).normal(0, 0.001, len(X))
    )

    fitted = model.fit_pd_model(duplicated, y, config, enforce_signs=False)

    assert len(fitted["checks"]["collinear"]) >= 2
    assert "WARNING" in model.model_report(fitted)


def test_bootstrap_reports_sign_consistency(config):
    """A coefficient whose sign flips across resamples is fitted to the sample."""
    X, y, _ = make_woe_training_set()
    fitted = model.fit_pd_model(X[["duration_months_woe", "checking_status_woe"]],
                               y, config)

    assert fitted["checks"]["bootstrap_resamples_used"] > 0
    consistency = fitted["coefficients"]["sign_consistency"].drop(index="const")
    assert (consistency >= 0.95).all()


# =========================================================================
# Prediction
# =========================================================================

def test_predicted_pd_is_a_valid_probability(config):
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    fitted = model.fit_pd_model(X[columns], y, config)

    pd_hat = model.predict_pd(fitted, X[columns])

    assert ((pd_hat > 0) & (pd_hat < 1)).all()
    # The mean predicted PD must land close to the observed bad rate — a fitted
    # logistic regression is calibrated in-sample almost by construction, and if
    # it is not, something is wrong with the link or the intercept.
    assert pd_hat.mean() == pytest.approx(y.mean(), abs=0.02)


def test_safer_applicant_gets_a_lower_pd(config):
    """The direction that matters. If this reverses, the scorecard is backwards."""
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    fitted = model.fit_pd_model(X[columns], y, config)

    # Positive WOE = safer on both features.
    safe = pd.DataFrame([{c: 0.8 for c in columns}])
    risky = pd.DataFrame([{c: -0.8 for c in columns}])

    assert model.predict_pd(fitted, safe)[0] < model.predict_pd(fitted, risky)[0]


def test_scoring_without_the_needed_columns_fails_loudly(config):
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    fitted = model.fit_pd_model(X[columns], y, config)

    with pytest.raises(model.ModelError):
        model.predict_pd(fitted, X[["duration_months_woe"]])


# =========================================================================
# Guard rails and persistence
# =========================================================================

def test_fit_requires_the_target(config):
    X, _, _ = make_woe_training_set()
    with pytest.raises(model.ModelError):
        model.fit_pd_model(X, None, config)


def test_single_class_target_is_rejected(config):
    X, _, _ = make_woe_training_set()
    with pytest.raises(model.ModelError):
        model.fit_pd_model(X, pd.Series([0] * len(X)), config)


def test_no_features_is_rejected(config):
    _, y, _ = make_woe_training_set()
    with pytest.raises(model.ModelError):
        model.fit_pd_model(pd.DataFrame(index=range(len(y))), y, config)


def test_mismatched_lengths_are_rejected(config):
    X, y, _ = make_woe_training_set()
    with pytest.raises(model.ModelError):
        model.fit_pd_model(X.head(100), y, config)


def test_model_survives_a_json_round_trip(config, tmp_path):
    """A scorecard must be scoreable years later, so no pickles."""
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    fitted = model.fit_pd_model(X[columns], y, config)

    cfg = dict(config)
    cfg["data"] = dict(config["data"], processed_dir=str(tmp_path))
    reloaded = model.load_model(model.save_model(fitted, cfg))

    np.testing.assert_allclose(
        model.predict_pd(fitted, X[columns]),
        model.predict_pd(reloaded, X[columns]),
        rtol=1e-12,
    )


def test_drop_feature_returns_the_remaining_names(config):
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    fitted = model.fit_pd_model(X[columns], y, config)

    remaining = model.drop_feature(fitted, "duration_months_woe")
    assert remaining == ["checking_status_woe"]


def test_report_renders(config):
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    report = model.model_report(model.fit_pd_model(X[columns], y, config))

    assert "Logistic PD model" in report
    assert "must be NEGATIVE" in report
    assert "pseudo R2" in report


# =========================================================================
# Regression tests for the 10 Oct 2026 sample-data failure
# =========================================================================

def test_too_few_events_per_variable_is_refused(config):
    """The exact situation that crashed a real run: 14 features, 5 events.

    On one library version statsmodels raised `LinAlgError: Singular matrix`; on
    another it quietly returned coefficients of -21.9 with NaN standard errors
    and VIF of 299. Silent nonsense is the more dangerous of the two, so the
    model now refuses before fitting.
    """
    X, y, _ = make_woe_training_set(n=1500)

    # 14 feature columns, but only 16 rows and about 5 events.
    wide = X.copy()
    while wide.shape[1] < 14:
        wide[f"filler_{wide.shape[1]}_woe"] = np.random.default_rng(0).normal(
            0, 1, len(wide))
    tiny_X = wide.head(16)
    tiny_y = pd.Series([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])

    with pytest.raises(model.ModelError) as error:
        model.fit_pd_model(tiny_X, tiny_y, config)

    message = str(error.value)
    assert "events per variable" in message
    assert "bundled sample" in message       # points at the actual cause
    assert "do not lower this threshold" in message


def test_events_per_variable_is_computed_from_the_minority_class(config):
    """EPV counts events, not rows. A 10,000-row book with 8 bads is still
    unfittable, and the row count would hide that completely."""
    X, y, _ = make_woe_training_set(n=1500)
    columns = ["duration_months_woe", "checking_status_woe"]

    # Plenty of rows, almost no events.
    y_rare = pd.Series([0] * (len(y) - 8) + [1] * 8)

    with pytest.raises(model.ModelError) as error:
        model.fit_pd_model(X[columns], y_rare, config)
    assert "events per variable" in str(error.value)


def test_epv_is_reported_on_a_successful_fit(config):
    """So a thin-but-passable fit is visible rather than taken on trust."""
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe", "checking_status_woe"]
    fitted = model.fit_pd_model(X[columns], y, config)

    stats = fitted["fit_statistics"]
    assert stats["events"] == int(y.sum())
    assert stats["events_per_variable"] == pytest.approx(y.sum() / 2)
    assert "events per variable" in model.model_report(fitted)


def test_a_single_event_is_refused(config):
    X, y, _ = make_woe_training_set()
    columns = ["duration_months_woe"]
    y_one = pd.Series([0] * (len(y) - 1) + [1])

    with pytest.raises(model.ModelError) as error:
        model.fit_pd_model(X[columns], y_one, config)
    assert "event" in str(error.value).lower()


def test_perfect_separation_is_refused(config):
    """Perfect separation does not raise and does not produce NaNs.

    A feature that splits the classes exactly has no finite best coefficient, so
    the optimiser walks off toward infinity: statsmodels returned a coefficient
    of -13.4 with a standard error of 16,342 here. Nothing errored, nothing was
    NaN, and the coefficient was meaningless. The giveaway is the standard
    error — on the WOE scale a real one is of order 0.5 to 2.
    """
    rng = np.random.default_rng(3)
    n = 400
    # A feature that separates the classes perfectly: no finite coefficient fits.
    y_sep = pd.Series([0] * (n // 2) + [1] * (n // 2))
    X_sep = pd.DataFrame({
        "separator_woe": np.where(y_sep == 1, -2.0, 2.0) + rng.normal(0, 1e-6, n),
    })

    with pytest.raises(model.ModelError) as error:
        model.fit_pd_model(X_sep, y_sep, config, enforce_signs=False)

    message = str(error.value).lower()
    assert "not identified" in message
    assert "separation" in message
