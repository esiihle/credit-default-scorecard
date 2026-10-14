"""
Phase 5 tests — validation metrics.

The metrics here are what a credit committee reads, so the tests check them
against cases where the right answer is known independently: a perfect model, a
coin flip, a hand-computed KS, and PSI between two samples drawn from the same
distribution (which must be ~0) versus a deliberately shifted one.

Run with:  pytest tests/test_validate.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import validate


@pytest.fixture
def config():
    return load_config()


def make_scored_population(n=2000, seed=0, signal=1.6, bad_rate=0.3):
    """Predicted PDs with a known amount of real signal in them.

    ``signal`` controls how well the score separates: 0 is a coin flip, higher
    gives a stronger model. Returns (predicted_pd, target).
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)

    # Probability of default rises with the latent risk factor.
    log_odds = np.log(bad_rate / (1 - bad_rate)) + signal * latent
    true_pd = 1 / (1 + np.exp(-log_odds))
    target = (rng.random(n) < true_pd).astype(int)

    return true_pd, target


# =========================================================================
# Discrimination
# =========================================================================

def test_perfect_model_has_auc_one():
    """Bads score strictly riskier than every good."""
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    target = np.array([1, 1, 1, 0, 0, 0])

    result = validate.auc_and_gini(scores, target)
    assert result["auc"] == pytest.approx(1.0)
    assert result["gini"] == pytest.approx(1.0)


def test_reversed_model_has_auc_zero():
    """A perfectly backwards model — the symptom of a flipped sign upstream."""
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    target = np.array([1, 1, 1, 0, 0, 0])

    result = validate.auc_and_gini(scores, target)
    assert result["auc"] == pytest.approx(0.0)
    assert result["gini"] == pytest.approx(-1.0)


def test_random_model_has_auc_about_half():
    rng = np.random.default_rng(1)
    scores = rng.random(5000)
    target = rng.integers(0, 2, 5000)

    result = validate.auc_and_gini(scores, target)
    assert result["auc"] == pytest.approx(0.5, abs=0.03)
    assert abs(result["gini"]) < 0.06


def test_auc_handles_ties_correctly():
    """Once a scorecard rounds to whole points, ties are everywhere. Ignoring
    them inflates AUC, so average ranks are used."""
    # Every score identical: the model has no information at all.
    scores = np.ones(100)
    target = np.array([1] * 30 + [0] * 70)

    result = validate.auc_and_gini(scores, target)
    assert result["auc"] == pytest.approx(0.5, abs=1e-9)


def test_gini_is_twice_auc_minus_one():
    """The identity, on real-ish data."""
    predicted, target = make_scored_population(1500, seed=2)
    result = validate.auc_and_gini(predicted, target)
    assert result["gini"] == pytest.approx(2 * result["auc"] - 1)


def test_ks_by_hand():
    """Four accounts, worked out on paper.

    Sorted riskiest first: bad, bad, good, good.
      after 1: F_bad = 0.5, F_good = 0.0  -> gap 0.5
      after 2: F_bad = 1.0, F_good = 0.0  -> gap 1.0   <- maximum
      after 3: F_bad = 1.0, F_good = 0.5  -> gap 0.5
      after 4: F_bad = 1.0, F_good = 1.0  -> gap 0.0
    """
    result = validate.ks_statistic([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])

    assert result["ks"] == pytest.approx(1.0)
    assert result["bads_captured"] == pytest.approx(1.0)
    assert result["goods_captured"] == pytest.approx(0.0)


def test_ks_is_zero_for_a_useless_model():
    rng = np.random.default_rng(3)
    scores = rng.random(4000)
    target = rng.integers(0, 2, 4000)

    assert validate.ks_statistic(scores, target)["ks"] < 0.06


def test_ks_reports_the_cut_off_a_committee_asks_about():
    predicted, target = make_scored_population(2000, seed=4)
    result = validate.ks_statistic(predicted, target)

    assert 0 < result["ks"] < 1
    assert result["bads_captured"] > result["goods_captured"]
    assert 0 < result["share_of_book"] <= 1


def test_single_class_is_refused():
    with pytest.raises(validate.ValidationError):
        validate.auc_and_gini([0.3, 0.5, 0.7], [0, 0, 0])
    with pytest.raises(validate.ValidationError):
        validate.ks_statistic([0.3, 0.5, 0.7], [1, 1, 1])


# =========================================================================
# Calibration
# =========================================================================

def test_calibration_table_tracks_reality_for_an_honest_model():
    """The PDs here ARE the true probabilities, so predicted and realised must
    agree within sampling noise in every band."""
    predicted, target = make_scored_population(6000, seed=5)
    table = validate.calibration_by_band(predicted, target, n_bands=10)

    assert table["accounts"].sum() == 6000
    # Judged against each band's own binomial standard error, not a flat number.
    assert table["within_2se"].mean() >= 0.8, table


def test_auc_is_blind_to_a_level_error_but_calibration_is_not():
    """Why all three question areas are needed, as a single demonstration.

    AUC depends only on the ORDER of the scores, so it is invariant to any
    strictly monotonic transform. Shifting every prediction in log-odds space
    leaves the ranking identical to the last decimal and makes every predicted PD
    badly too high. A model judged on AUC alone would look untouched.

    Note the transform has to be strictly monotonic for that claim to hold. An
    earlier version of this test used `clip(pd * 2, 0, 0.999)`, which ties every
    high-risk account at the ceiling, destroys real ranking information, and
    moved AUC from 0.820 to 0.804. Clipping is not a monotonic transform at the
    boundary — a small thing, and the reason this test now uses a logit shift.
    """
    predicted, target = make_scored_population(4000, seed=6)

    # A pure level shift in log-odds: strictly increasing, stays inside (0, 1).
    log_odds = np.log(predicted / (1 - predicted))
    inflated = 1 / (1 + np.exp(-(log_odds + 1.2)))

    honest_auc = validate.auc_and_gini(predicted, target)["auc"]
    inflated_auc = validate.auc_and_gini(inflated, target)["auc"]
    assert inflated_auc == pytest.approx(honest_auc, abs=1e-9), (
        "a strictly monotonic transform must leave AUC exactly unchanged"
    )

    # Calibration, in contrast, sees it immediately.
    table = validate.calibration_by_band(inflated, target, n_bands=10)
    assert (table["gap"] > 0).mean() > 0.8, "should be over-predicting everywhere"
    assert table["within_2se"].mean() < 0.5, "and calibration should detect it"


def test_hosmer_lemeshow_passes_a_calibrated_model():
    predicted, target = make_scored_population(4000, seed=7)
    result = validate.hosmer_lemeshow(predicted, target, n_bands=10)

    assert result["p_value"] > 0.05
    assert result["calibration_rejected_at_5pct"] is False


def test_hosmer_lemeshow_rejects_a_miscalibrated_model():
    """And the direction is the one that catches people out: LOW p = BAD."""
    predicted, target = make_scored_population(4000, seed=8)
    doubled = np.clip(predicted * 2, 0.001, 0.999)

    result = validate.hosmer_lemeshow(doubled, target, n_bands=10)
    assert result["p_value"] < 0.05
    assert result["calibration_rejected_at_5pct"] is True


# =========================================================================
# Stability
# =========================================================================

def test_psi_is_near_zero_for_two_samples_from_one_distribution():
    rng = np.random.default_rng(9)
    expected = rng.normal(0, 1, 3000)
    actual = rng.normal(0, 1, 3000)

    psi, table = validate.population_stability_index(expected, actual)

    assert psi < 0.10
    assert validate.psi_verdict(psi) == "stable"
    assert len(table) > 0


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(10)
    expected = rng.normal(0, 1, 3000)
    actual = rng.normal(1.0, 1, 3000)         # a whole standard deviation

    psi, _ = validate.population_stability_index(expected, actual)

    assert psi > 0.25
    assert "MAJOR SHIFT" in validate.psi_verdict(psi)


def test_psi_bands_are_cut_on_the_expected_sample():
    """The ordering that makes PSI work at all.

    Re-cutting bands on the actual sample would make any two distributions look
    identical, because each would hold 10% of itself in each of its own deciles.
    A shifted sample must therefore produce UNEVEN actual shares.
    """
    rng = np.random.default_rng(11)
    expected = rng.normal(0, 1, 3000)
    actual = rng.normal(1.2, 1, 3000)

    _, table = validate.population_stability_index(expected, actual, n_bands=10)

    # Expected shares are ~0.1 by construction; actual ones must not be.
    assert table["expected_share"].std() < 0.01
    assert table["actual_share"].std() > 0.03


def test_psi_survives_an_empty_band():
    """A band with no accounts would make the log infinite without a floor."""
    expected = np.concatenate([np.zeros(500), np.ones(500)])
    actual = np.zeros(500)                     # nothing in the upper bands

    psi, table = validate.population_stability_index(expected, actual, n_bands=5)
    assert np.isfinite(psi)
    assert np.isfinite(table["contribution"]).all()


def test_psi_verdict_bands():
    assert validate.psi_verdict(0.05) == "stable"
    assert "monitor" in validate.psi_verdict(0.15)
    assert "MAJOR" in validate.psi_verdict(0.40)


# =========================================================================
# Rank ordering
# =========================================================================

def test_rank_ordering_is_monotonic_for_a_good_model():
    predicted, target = make_scored_population(4000, seed=12)
    table, summary = validate.rank_ordering(predicted, target, n_bands=10)

    assert summary["monotonic"] is True
    assert summary["reversals"] == []
    # Bands run lowest predicted PD to highest, so the bad rate must rise.
    assert summary["direction"] == "rising"
    assert summary["direction"] == summary["expected_direction"]
    assert table["bad_rate"].is_monotonic_increasing


def test_rank_ordering_catches_a_reversal():
    """A model whose middle bands cross over cannot support cut-off policy,
    whatever its AUC says."""
    rng = np.random.default_rng(13)
    n = 3000
    score = rng.uniform(0, 1, n)
    # Deliberately non-monotonic: the middle of the range is made safest.
    risk = np.where((score > 0.4) & (score < 0.6), 0.05, 0.1 + 0.5 * score)
    target = (rng.random(n) < risk).astype(int)

    _, summary = validate.rank_ordering(score, target, n_bands=10)
    assert summary["monotonic"] is False
    assert len(summary["reversals"]) > 0


# =========================================================================
# The whole report
# =========================================================================

def test_evaluate_reports_all_three_question_areas(config):
    predicted_train, target_train = make_scored_population(1400, seed=14)
    predicted_test, target_test = make_scored_population(600, seed=15)

    result = validate.evaluate(predicted_train, target_train,
                               predicted_test, target_test, config)

    assert "discrimination" in result
    assert "calibration" in result
    assert "stability" in result
    assert "rank_ordering" in result
    assert result["n_test"] == 600

    report = validate.validation_report(result)
    assert "Discrimination" in report
    assert "Calibration" in report
    assert "Stability" in report
    assert "may be used to choose a feature" in report    # the discipline note


def test_evaluate_flags_a_large_train_to_test_gini_drop(config):
    """The signature of overfitting, and the reason train numbers are computed."""
    # Strong signal in train, almost none in test.
    predicted_train, target_train = make_scored_population(1400, seed=16, signal=2.5)
    predicted_test, target_test = make_scored_population(600, seed=17, signal=0.05)

    result = validate.evaluate(predicted_train, target_train,
                               predicted_test, target_test, config)

    assert result["discrimination"]["gini_drop"] > config["validation"]["max_gini_drop"]
    assert result["checks"]["no_large_gini_drop"] is False
    assert "overfitting" in validate.validation_report(result)


def test_evaluate_fails_a_model_that_does_not_discriminate(config):
    rng = np.random.default_rng(18)
    predicted_train = rng.uniform(0.2, 0.4, 1400)
    target_train = rng.integers(0, 2, 1400)
    predicted_test = rng.uniform(0.2, 0.4, 600)
    target_test = rng.integers(0, 2, 600)

    result = validate.evaluate(predicted_train, target_train,
                               predicted_test, target_test, config)

    assert result["checks"]["gini_above_minimum"] is False
    assert result["checks"]["all_passed"] is False
    assert "FAIL" in validate.validation_report(result)


def test_evaluate_refuses_a_single_class_test_set(config):
    predicted_train, target_train = make_scored_population(1000, seed=19)
    with pytest.raises(validate.ValidationError):
        validate.evaluate(predicted_train, target_train,
                          np.full(200, 0.3), np.zeros(200, dtype=int), config)


def test_report_survives_a_json_round_trip(config, tmp_path):
    predicted_train, target_train = make_scored_population(1400, seed=20)
    predicted_test, target_test = make_scored_population(600, seed=21)
    result = validate.evaluate(predicted_train, target_train,
                               predicted_test, target_test, config)

    cfg = dict(config)
    cfg["data"] = dict(config["data"], processed_dir=str(tmp_path))
    path = validate.save_report(result, cfg)

    import json
    with open(path) as f:
        reloaded = json.load(f)

    assert reloaded["discrimination"]["test"]["gini"] == pytest.approx(
        result["discrimination"]["test"]["gini"])
    assert reloaded["stability"]["psi"] == pytest.approx(result["stability"]["psi"])
