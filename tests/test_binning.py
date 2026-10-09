"""
Phase 2 tests — binning and coarse classing.

Each test builds data with a known shape and asserts the binner does the right
thing with it. If one fails, the name says which guarantee broke.

Run with:  pytest tests/test_binning.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import binning


@pytest.fixture
def config():
    return load_config()


def make_training_frame(n=1000, seed=0):
    """A frame where risk genuinely rises with duration and falls with age."""
    rng = np.random.default_rng(seed)
    duration = rng.integers(6, 60, n)
    age = rng.integers(20, 70, n)
    purpose = rng.choice(["car", "furniture", "education", "business", "rare"],
                         n, p=[0.35, 0.3, 0.2, 0.13, 0.02])

    risk = 0.05 + 0.008 * (duration - 6) - 0.004 * (age - 20)
    risk = np.clip(risk, 0.02, 0.95)

    return pd.DataFrame({
        "duration_months": duration,
        "age_years": age,
        "purpose": purpose,
        "is_default": (rng.random(n) < risk).astype(int),
    })


# --- Numeric binning -------------------------------------------------------

def test_numeric_bins_respect_max_bins(config):
    df = make_training_frame()
    spec = binning.fit_numeric_bins(df["duration_months"], df["is_default"], config)

    assert len(spec["bins"]) <= config["binning"]["max_bins"]


def test_every_bin_holds_at_least_the_minimum_share(config):
    df = make_training_frame()
    spec = binning.fit_numeric_bins(df["duration_months"], df["is_default"], config)

    total = len(df)
    min_fraction = config["binning"]["min_bin_fraction"]
    for b in spec["bins"]:
        assert b["count"] / total >= min_fraction, f"bin too small: {b}"


def test_monotonic_trend_is_enforced(config):
    """A deliberate mid-range blip must be merged away, not preserved."""
    rng = np.random.default_rng(1)
    n = 1200
    duration = rng.integers(6, 60, n)
    risk = 0.05 + 0.008 * (duration - 6)
    # A bump: one slice of the range made artificially safe, against the trend.
    risk = np.where((duration > 28) & (duration < 34), 0.02, risk)
    df = pd.DataFrame({
        "duration_months": duration,
        "is_default": (rng.random(n) < np.clip(risk, 0.01, 0.95)).astype(int),
    })

    spec = binning.fit_numeric_bins(df["duration_months"], df["is_default"], config)
    rates = [b["bad_rate"] for b in spec["bins"]]

    assert np.all(np.diff(rates) >= 0) or np.all(np.diff(rates) <= 0), (
        f"bad rate is not monotonic across bins: {rates}"
    )


def test_edges_are_open_at_both_ends(config):
    """A value outside the training range must still bin, not crash the scorer."""
    df = make_training_frame()
    spec = binning.fit_numeric_bins(df["duration_months"], df["is_default"], config)

    assert np.isneginf(spec["edges"][0])
    assert np.isposinf(spec["edges"][-1])


def test_few_distinct_values_do_not_break_binning(config):
    """installment_rate only takes 1-4; quantile edges collide. Must still work."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "installment_rate": rng.integers(1, 5, 800),
        "is_default": rng.integers(0, 2, 800),
    })
    spec = binning.fit_numeric_bins(df["installment_rate"], df["is_default"], config)

    assert len(spec["bins"]) >= 1


# --- Categorical binning ---------------------------------------------------

def test_thin_levels_are_merged_away(config):
    """'rare' holds ~2% of the book and must not survive as its own group."""
    df = make_training_frame(n=2000)
    spec = binning.fit_categorical_bins(df["purpose"], df["is_default"], config)

    total = len(df)
    for b in spec["bins"]:
        assert b["count"] / total >= config["binning"]["min_bin_fraction"]
    # Every original level is still mapped somewhere — nothing silently vanishes.
    assert set(spec["mapping"]) == set(df["purpose"].unique())


# --- Applying the spec -----------------------------------------------------

def test_apply_puts_unseen_category_in_its_own_bin(config):
    """A product code that didn't exist at fit time must not crash scoring."""
    train = make_training_frame()
    spec = binning.fit_bins(train, config)

    live = pd.DataFrame({"purpose": ["a_brand_new_purpose"],
                         "duration_months": [24], "age_years": [40]})
    binned = binning.apply_bins(live, spec)

    assert binned.loc[0, "purpose_bin"] == binning.UNSEEN_LABEL


def test_apply_puts_missing_value_in_its_own_bin(config):
    train = make_training_frame()
    spec = binning.fit_bins(train, config)

    live = pd.DataFrame({"purpose": ["car"], "duration_months": [np.nan],
                         "age_years": [40]})
    binned = binning.apply_bins(live, spec)

    assert binned.loc[0, "duration_months_bin"] == binning.MISSING_LABEL


def test_apply_is_deterministic(config):
    train = make_training_frame()
    spec = binning.fit_bins(train, config)

    first = binning.apply_bins(train, spec)
    second = binning.apply_bins(train, spec)

    pd.testing.assert_frame_equal(first, second)


# --- The frozen spec -------------------------------------------------------

def test_spec_survives_a_json_round_trip(config, tmp_path):
    """Infinite edges and numpy types must come back unchanged, or test scoring
    would differ from training — the classic silent scorecard bug."""
    train = make_training_frame()
    spec = binning.fit_bins(train, config)

    path = tmp_path / "spec.json"
    import json
    with open(path, "w") as f:
        json.dump(spec, f, default=binning._json_safe)
    reloaded = binning.load_spec(path)

    before = binning.apply_bins(train, spec)
    after = binning.apply_bins(train, reloaded)
    pd.testing.assert_frame_equal(before, after)


def test_fit_requires_the_target(config):
    """Binning is supervised. Without the target it is just cutting."""
    train = make_training_frame().drop(columns=["is_default"])
    with pytest.raises(binning.BinningError):
        binning.fit_bins(train, config)


def test_spec_fitted_on_train_is_unchanged_by_test_data(config):
    """Applying to test must never alter the spec — no leakage through binning."""
    train = make_training_frame(n=800, seed=3)
    test = make_training_frame(n=400, seed=99)

    spec = binning.fit_bins(train, config)
    edges_before = [list(f.get("edges", [])) for f in spec["features"].values()]

    binning.apply_bins(test, spec)

    edges_after = [list(f.get("edges", [])) for f in spec["features"].values()]
    assert edges_before == edges_after


def test_summary_lists_bins_in_natural_order(config):
    """Alphabetical order would print '24 - 41' before '<= 24' and hide the trend."""
    train = make_training_frame()
    spec = binning.fit_bins(train, config)
    binned = binning.apply_bins(train, spec)

    summary = binning.bin_summary(binned, train["is_default"], spec)
    duration = summary[summary["feature"] == "duration_months"]

    expected = [b for b in binning.spec_bin_order(spec["features"]["duration_months"])
                if b in set(duration["bin"])]
    assert list(duration["bin"]) == expected
    # And in that order the bad rate should move one way, since risk rises with term.
    rates = duration["bad_rate"].values
    assert np.all(np.diff(rates) >= 0) or np.all(np.diff(rates) <= 0)
