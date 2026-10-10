"""
Phase 3 tests — Weight of Evidence and Information Value.

The most valuable tests here are the two hand-computed ones: a WOE and an IV
worked out on paper, asserted against the code. Everything else in a scorecard
is built on these two numbers, so they are the place to be certain rather than
confident.

Run with:  pytest tests/test_woe.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import binning, woe


@pytest.fixture
def config():
    return load_config()


def make_binned_frame(n=1000, seed=0):
    """Binned training data where one feature predicts and one is pure noise."""
    rng = np.random.default_rng(seed)

    # A predictive feature: risk falls as the level improves.
    levels = rng.choice(["poor", "fair", "good", "excellent"], n,
                        p=[0.3, 0.3, 0.25, 0.15])
    risk = pd.Series(levels).map(
        {"poor": 0.55, "fair": 0.35, "good": 0.18, "excellent": 0.08}
    )
    target = (rng.random(n) < risk).astype(int)

    # A useless feature: assigned at random, unrelated to the target.
    noise = rng.choice(["a", "b", "c"], n)

    return pd.DataFrame({
        "credit_grade_bin": levels,
        "coin_flip_bin": noise,
    }), pd.Series(target)


# --- The hand-computed anchors ---------------------------------------------

def test_woe_matches_a_hand_computation():
    """A tiny book, worked out on paper.

        bin A: 60 good,  20 bad
        bin B: 40 good,  80 bad
        totals: 100 good, 100 bad

        WOE_A = ln( (60/100) / (20/100) ) = ln 3  =  1.0986
        WOE_B = ln( (40/100) / (80/100) ) = ln 0.5 = -0.6931
    """
    bins = ["A"] * 80 + ["B"] * 120
    target = [0] * 60 + [1] * 20 + [0] * 40 + [1] * 80

    table = woe.woe_table(bins, target)

    assert table.loc["A", "woe"] == pytest.approx(np.log(3), abs=1e-6)
    assert table.loc["B", "woe"] == pytest.approx(np.log(0.5), abs=1e-6)


def test_information_value_matches_a_hand_computation():
    """Same book.

        IV = (0.60 - 0.20) * ln 3  +  (0.40 - 0.80) * ln 0.5
           = 0.40 * 1.0986  +  (-0.40) * (-0.6931)
           = 0.43944 + 0.27726 = 0.7167
    """
    bins = ["A"] * 80 + ["B"] * 120
    target = [0] * 60 + [1] * 20 + [0] * 40 + [1] * 80

    iv = woe.information_value(woe.woe_table(bins, target))
    expected = 0.40 * np.log(3) + (-0.40) * np.log(0.5)

    assert iv == pytest.approx(expected, abs=1e-6)
    assert iv == pytest.approx(0.7167, abs=1e-3)


# --- Sign convention -------------------------------------------------------

def test_positive_woe_means_safer():
    """The convention the whole scorecard depends on. If this flips, the model
    ranks backwards and still looks plausible."""
    binned, target = make_binned_frame()
    table = woe.woe_table(binned["credit_grade_bin"], target)

    assert table.loc["excellent", "woe"] > 0        # mostly goods
    assert table.loc["poor", "woe"] < 0             # mostly bads
    assert table.loc["excellent", "woe"] > table.loc["good", "woe"]
    assert table.loc["good", "woe"] > table.loc["fair", "woe"]


def test_woe_ordering_mirrors_bad_rate_ordering():
    """Higher bad rate must mean lower WOE, bin for bin."""
    binned, target = make_binned_frame()
    table = woe.woe_table(binned["credit_grade_bin"], target).sort_values("bad_rate")

    assert table["woe"].is_monotonic_decreasing


# --- Information Value and selection ---------------------------------------

def test_predictive_feature_beats_noise(config):
    binned, target = make_binned_frame(n=2000)
    spec = woe.compute_woe_iv(binned, target, config)

    assert spec["features"]["credit_grade"]["iv"] > spec["features"]["coin_flip"]["iv"]
    assert spec["features"]["credit_grade"]["selected"] is True


def test_useless_feature_is_dropped(config):
    """A coin flip must fall below the IV floor and be deselected."""
    binned, target = make_binned_frame(n=2000)
    spec = woe.compute_woe_iv(binned, target, config)

    assert spec["features"]["coin_flip"]["iv"] < config["woe"]["iv_min"]
    assert spec["features"]["coin_flip"]["selected"] is False
    assert "coin_flip" not in woe.selected_features(spec)


def test_suspiciously_strong_feature_is_flagged_not_dropped(config):
    """A leaking feature must be surfaced for a human, never auto-removed."""
    n = 1000
    rng = np.random.default_rng(1)
    target = pd.Series(rng.integers(0, 2, n))
    # A "feature" that almost is the target — i.e. obvious leakage.
    leak = np.where(target == 1, "will_default", "will_pay")
    binned = pd.DataFrame({"collections_flag_bin": leak})

    spec = woe.compute_woe_iv(binned, target, config)
    definition = spec["features"]["collections_flag"]

    assert definition["iv"] >= config["woe"]["iv_leakage_flag"]
    assert definition["flagged_for_leakage"] is True
    assert definition["selected"] is True          # flagged, still selected
    assert "SUSPICIOUS" in definition["band"]


def test_iv_bands_use_the_configured_cutoffs(config):
    assert woe.iv_band(0.005, config) == "useless"
    assert woe.iv_band(0.05, config) == "weak"
    assert woe.iv_band(0.2, config) == "medium"
    assert woe.iv_band(0.4, config) == "strong"
    assert "SUSPICIOUS" in woe.iv_band(0.9, config)


# --- Edge cases ------------------------------------------------------------

def test_bin_with_no_bads_does_not_produce_infinity():
    """Zero in a cell makes the log blow up. It must be corrected and flagged."""
    bins = ["clean"] * 50 + ["mixed"] * 100
    target = [0] * 50 + [0] * 60 + [1] * 40      # 'clean' has zero bads

    table = woe.woe_table(bins, target)

    assert np.isfinite(table["woe"]).all()
    assert bool(table.loc["clean", "adjusted"]) is True
    assert bool(table.loc["mixed", "adjusted"]) is False


def test_single_bin_feature_has_zero_iv(config):
    """A feature that collapsed to one bin in Phase 2 carries no information."""
    binned = pd.DataFrame({"constant_bin": ["only_value"] * 500})
    target = pd.Series(np.random.default_rng(0).integers(0, 2, 500))

    spec = woe.compute_woe_iv(binned, target, config)

    assert spec["features"]["constant"]["iv"] == pytest.approx(0.0, abs=1e-9)
    assert spec["features"]["constant"]["selected"] is False


def test_woe_requires_the_target(config):
    binned, _ = make_binned_frame()
    with pytest.raises(woe.WOEError):
        woe.compute_woe_iv(binned, None, config)


def test_single_class_target_is_rejected(config):
    binned, _ = make_binned_frame()
    with pytest.raises(woe.WOEError):
        woe.compute_woe_iv(binned, pd.Series([0] * len(binned)), config)


# --- Applying the spec -----------------------------------------------------

def test_apply_woe_produces_one_numeric_column_per_feature(config):
    binned, target = make_binned_frame()
    spec = woe.compute_woe_iv(binned, target, config)

    matrix, unseen = woe.apply_woe(binned, spec)

    assert list(matrix.columns) == ["credit_grade_woe"]      # noise was dropped
    assert matrix["credit_grade_woe"].notna().all()
    assert unseen == {}


def test_unseen_bin_scores_neutral_and_is_counted(config):
    """A level that never appeared in training gets WOE 0 — and is reported."""
    binned, target = make_binned_frame()
    spec = woe.compute_woe_iv(binned, target, config)

    live = pd.DataFrame({"credit_grade_bin": ["brand_new_grade"],
                         "coin_flip_bin": ["a"]})
    matrix, unseen = woe.apply_woe(live, spec)

    assert matrix.loc[0, "credit_grade_woe"] == woe.NEUTRAL_WOE
    assert unseen["credit_grade"] == 1


def test_applying_to_test_data_does_not_change_the_spec(config):
    """No leakage through the WOE tables, same as for the bins."""
    binned, target = make_binned_frame(n=800, seed=2)
    other, _ = make_binned_frame(n=400, seed=99)

    spec = woe.compute_woe_iv(binned, target, config)
    before = {f: dict(d["mapping"]) for f, d in spec["features"].items()}

    woe.apply_woe(other, spec)

    after = {f: dict(d["mapping"]) for f, d in spec["features"].items()}
    assert before == after


# --- Freezing --------------------------------------------------------------

def test_spec_survives_a_json_round_trip(config, tmp_path):
    binned, target = make_binned_frame()
    spec = woe.compute_woe_iv(binned, target, config)

    cfg = dict(config)
    cfg["data"] = dict(config["data"], processed_dir=str(tmp_path))
    path = woe.save_spec(spec, cfg)
    reloaded = woe.load_spec(path)

    before, _ = woe.apply_woe(binned, spec)
    after, _ = woe.apply_woe(binned, reloaded)
    pd.testing.assert_frame_equal(before, after)


# --- End to end with the real binning stage --------------------------------

def test_works_on_output_from_the_binning_stage(config):
    """Phases 2 and 3 must actually fit together, not just pass in isolation."""
    rng = np.random.default_rng(4)
    n = 1200
    duration = rng.integers(6, 60, n)
    risk = np.clip(0.05 + 0.008 * (duration - 6), 0.02, 0.95)
    train = pd.DataFrame({
        "duration_months": duration,
        "purpose": rng.choice(["car", "furniture", "education"], n),
        "is_default": (rng.random(n) < risk).astype(int),
    })

    bin_spec = binning.fit_bins(train, config)
    binned = binning.apply_bins(train, bin_spec)

    order = {f: binning.spec_bin_order(d) for f, d in bin_spec["features"].items()}
    spec = woe.compute_woe_iv(binned, train["is_default"], config, bin_order=order)

    assert spec["features"]["duration_months"]["iv"] > 0.1
    # Bins came out in natural order, so WOE should fall as the term lengthens.
    table = spec["features"]["duration_months"]["table"]
    assert table["woe"].is_monotonic_decreasing

    matrix, unseen = woe.apply_woe(binned, spec)
    assert len(matrix) == len(train)
    assert unseen == {}
