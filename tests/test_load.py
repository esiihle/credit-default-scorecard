"""
Phase 1 tests — validation, target definition and the frozen split.

Behaviour tests, not smoke tests: each one creates a specific situation and
asserts what the code does about it. If one fails, its name tells you what broke.

Run with:  pytest tests/test_load.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import load


# --------------------------------------------------------------------------
# A clean synthetic book. Each test corrupts a copy, so what is being tested is
# always the one change that test makes.
# --------------------------------------------------------------------------
def make_applicants(n=200, bad_rate=0.3, seed=0):
    """Build a frame with the same columns as the decoded German Credit data."""
    rng = np.random.default_rng(seed)
    config = load_config()
    schema = config["schema"]

    data = {
        "duration_months": rng.integers(6, 48, n),
        "credit_amount": rng.integers(500, 15000, n),
        "installment_rate": rng.integers(1, 5, n),
        "residence_since": rng.integers(1, 5, n),
        "age_years": rng.integers(20, 70, n),
        "existing_credits": rng.integers(1, 4, n),
        "num_liable": rng.integers(1, 3, n),
    }
    for col in schema["categorical_columns"]:
        data[col] = rng.choice([f"{col}_a", f"{col}_b", f"{col}_c"], n)

    # Raw coding: 1 = good, 2 = bad.
    n_bad = int(round(n * bad_rate))
    target = np.array([2] * n_bad + [1] * (n - n_bad))
    rng.shuffle(target)
    data[schema["target_column"]] = target

    return pd.DataFrame(data)


@pytest.fixture
def config():
    return load_config()


# --- Structural validation -------------------------------------------------

def test_missing_column_stops_the_run(config):
    df = make_applicants().drop(columns=["age_years"])
    with pytest.raises(load.DataQualityError) as err:
        load.validate_raw(df, config)
    assert "age_years" in str(err.value)


def test_unexpected_target_code_stops_the_run(config):
    """A 3 in the target means the file isn't what we think it is."""
    df = make_applicants()
    df.loc[0, "target"] = 3
    with pytest.raises(load.DataQualityError) as err:
        load.validate_raw(df, config)
    assert "3" in str(err.value)


def test_single_class_target_stops_the_run(config):
    df = make_applicants()
    df["target"] = 1                      # every applicant is good
    with pytest.raises(load.DataQualityError):
        load.validate_raw(df, config)


# --- Cleaning --------------------------------------------------------------

def test_implausible_age_is_dropped(config):
    df = make_applicants()
    df.loc[0, "age_years"] = 5            # below validation.min_age
    df.loc[1, "age_years"] = 150          # above validation.max_age

    cleaned, report = load.clean(df, config)

    assert report["dropped_implausible_age_years"] == 2
    assert len(cleaned) == len(df) - 2


def test_exact_duplicate_rows_are_dropped(config):
    df = make_applicants()
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)

    cleaned, report = load.clean(df, config)

    assert report["dropped_duplicate_rows"] == 1
    assert len(cleaned) == len(df) - 1


# --- Target definition -----------------------------------------------------

def test_target_recode_direction(config):
    """1 = default. Getting this backwards inverts the entire scorecard."""
    df = make_applicants()
    labelled = load.define_target(df, config)

    # Raw 2 (bad) must become 1; raw 1 (good) must become 0.
    assert set(labelled.loc[labelled["target"] == 2, "is_default"]) == {1}
    assert set(labelled.loc[labelled["target"] == 1, "is_default"]) == {0}
    # And the overall bad rate is unchanged by the recode.
    assert labelled["is_default"].mean() == pytest.approx((df["target"] == 2).mean())


# --- The split -------------------------------------------------------------

def test_split_sizes_match_config(config):
    df = load.to_analysis_table(load.define_target(make_applicants(400), config), config)
    train, test = load.split_train_test(df, config)

    expected_test = round(len(df) * config["split"]["test_size"])
    assert len(test) == pytest.approx(expected_test, abs=1)
    assert len(train) + len(test) == len(df)


def test_split_is_stratified(config):
    """Both halves must carry the same bad rate, or metrics measure the split."""
    df = load.to_analysis_table(load.define_target(make_applicants(1000), config), config)
    train, test = load.split_train_test(df, config)

    assert train["is_default"].mean() == pytest.approx(test["is_default"].mean(), abs=0.01)


def test_no_applicant_appears_in_both_halves(config):
    """The most important property of a split, and the easiest to break."""
    df = load.to_analysis_table(load.define_target(make_applicants(500), config), config)
    train, test = load.split_train_test(df, config)

    overlap = set(train["applicant_id"]) & set(test["applicant_id"])
    assert overlap == set(), f"{len(overlap)} applicants leaked across the split"


def test_split_is_reproducible_from_the_seed(config):
    """Same seed, same split — proven by fingerprint, not by eye."""
    df = load.to_analysis_table(load.define_target(make_applicants(500), config), config)

    first = load.split_manifest(*load.split_train_test(df, config), config)
    second = load.split_manifest(*load.split_train_test(df, config), config)

    assert first["train"]["fingerprint"] == second["train"]["fingerprint"]
    assert first["test"]["fingerprint"] == second["test"]["fingerprint"]


def test_different_seed_gives_a_different_split(config):
    """The seed must actually be doing something."""
    df = load.to_analysis_table(load.define_target(make_applicants(500), config), config)

    other = dict(config)
    other["seed"] = config.get("seed", 42) + 1

    base = load.split_manifest(*load.split_train_test(df, config), config)
    changed = load.split_manifest(*load.split_train_test(df, other), other)

    assert base["test"]["fingerprint"] != changed["test"]["fingerprint"]


# --- End to end ------------------------------------------------------------

def test_run_phase1_end_to_end(config):
    df = make_applicants(600)
    train, test, report, manifest = load.run_phase1(df, config, save=False)

    assert len(train) + len(test) == report["rows_out"]
    assert manifest["target_definition"].startswith("1 =")
    assert "Data-quality report" in load.quality_report(report, train, test, manifest)


def test_run_phase1_refuses_to_silently_discard_most_of_the_file(config):
    df = make_applicants(100)
    df.loc[0:59, "age_years"] = 200       # 60% of the book made implausible

    with pytest.raises(load.DataQualityError) as err:
        load.run_phase1(df, config, save=False)
    assert "%" in str(err.value)
