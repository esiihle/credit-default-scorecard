"""
Phase 6 tests — score scaling and the points table.

Two tests carry the weight:

* `test_pdo_doubles_the_odds` — the defining property of the scaling, and the one
  sentence that explains a scorecard to a non-technical audience.
* `test_points_table_reproduces_the_model_exactly` — the points table is an
  algebraic rearrangement of the logistic model, not an approximation. Unrounded
  it must agree to floating-point precision.

Run with:  pytest tests/test_scaling.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import binning, model, scaling, woe


@pytest.fixture
def config():
    return load_config()


def build_full_pipeline(n=2000, seed=0):
    """Run binning -> WOE -> PD model, the way the real pipeline does.

    Returns everything the scaling stage needs, so these tests exercise the real
    chain rather than a hand-made stand-in.
    """
    rng = np.random.default_rng(seed)

    duration = rng.integers(6, 60, n)
    checking = rng.choice(["none", "low", "medium", "high"], n,
                          p=[0.25, 0.3, 0.27, 0.18])
    checking_risk = pd.Series(checking).map(
        {"none": 0.10, "low": 0.50, "medium": 0.32, "high": 0.18})

    risk = np.clip(0.35 * checking_risk + 0.006 * (duration - 30) + 0.15,
                   0.02, 0.95)
    is_default = (rng.random(n) < risk).astype(int)

    frame = pd.DataFrame({
        "duration_months": duration,
        "checking_status": checking,
        "is_default": is_default,
    })

    cfg = load_config()
    cfg["schema"] = dict(cfg["schema"],
                         numeric_columns=["duration_months"],
                         categorical_columns=["checking_status"])

    bin_spec = binning.fit_bins(frame, cfg)
    binned = binning.apply_bins(frame, bin_spec)
    woe_spec = woe.compute_woe_iv(binned, frame["is_default"], cfg)
    chosen = woe.selected_features(woe_spec)
    woe_matrix, _ = woe.apply_woe(binned, woe_spec, chosen)
    pd_model = model.fit_pd_model(woe_matrix, frame["is_default"], cfg)

    return {"frame": frame, "binned": binned, "woe_spec": woe_spec,
            "woe_matrix": woe_matrix, "pd_model": pd_model, "config": cfg}


# =========================================================================
# The scaling constants
# =========================================================================

def test_factor_is_pdo_over_ln_two(config):
    constants = scaling.scaling_constants(config)
    expected = config["scaling"]["pdo"] / np.log(2)
    assert constants["factor"] == pytest.approx(expected)


def test_base_score_maps_to_base_odds(config):
    """The anchor point of the whole scale, by construction."""
    constants = scaling.scaling_constants(config)

    odds_at_base = scaling.score_to_odds(constants["base_score"], constants)
    assert odds_at_base == pytest.approx(constants["base_odds"])

    score_at_base_odds = scaling.odds_to_score(constants["base_odds"], constants)
    assert score_at_base_odds == pytest.approx(constants["base_score"])


def test_pdo_doubles_the_odds(config):
    """The defining property. With pdo = 20, +20 points doubles the good:bad odds.

    This is the sentence that explains a scorecard to someone who will never see
    a logistic regression, and it is the entire reason for this scaling rather
    than just reporting the PD.
    """
    constants = scaling.scaling_constants(config)
    pdo = constants["pdo"]

    for score in (500, 560, 600, 680):
        odds_here = scaling.score_to_odds(score, constants)
        odds_up = scaling.score_to_odds(score + pdo, constants)
        odds_down = scaling.score_to_odds(score - pdo, constants)

        assert odds_up == pytest.approx(2 * odds_here)
        assert odds_down == pytest.approx(odds_here / 2)


def test_score_and_pd_round_trip(config):
    constants = scaling.scaling_constants(config)

    for probability in (0.01, 0.05, 0.2, 0.5, 0.8):
        score = scaling.pd_to_score(probability, constants)
        assert scaling.score_to_pd(score, constants) == pytest.approx(probability)


def test_higher_score_means_lower_pd(config):
    """The sign chain, end to end: 1 = default, WOE positive = safer, negative
    coefficients, more points for safer bins, higher score = safer applicant."""
    constants = scaling.scaling_constants(config)
    scores = np.array([400, 500, 600, 700, 800])
    probabilities = scaling.score_to_pd(scores, constants)

    assert np.all(np.diff(probabilities) < 0), probabilities


def test_invalid_scaling_settings_are_refused():
    with pytest.raises(scaling.ScalingError):
        scaling.scaling_constants({"scaling": {"pdo": 0}})
    with pytest.raises(scaling.ScalingError):
        scaling.scaling_constants({"scaling": {"base_odds": -1}})


# =========================================================================
# The points table
# =========================================================================

def test_points_table_covers_every_bin_of_every_feature():
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    table = scorecard["table"]
    for feature_woe in parts["pd_model"]["features"]:
        feature = feature_woe.replace("_woe", "")
        expected_bins = set(parts["woe_spec"]["features"][feature]["mapping"])
        actual_bins = set(table[table["feature"] == feature]["bin"])
        assert actual_bins == expected_bins, feature


def test_safer_bins_earn_more_points():
    """Within a characteristic, the bin with the higher WOE must score higher.
    If this reverses, the scorecard ranks backwards."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    table = scorecard["table"]
    for feature in table["feature"].unique():
        rows = table[table["feature"] == feature].sort_values("woe")
        # Sorted by WOE ascending (riskiest first), points must rise.
        assert rows["points_exact"].is_monotonic_increasing, (
            f"{feature}: points do not rise with WOE\n{rows}"
        )


def test_points_table_reproduces_the_model_exactly():
    """Unrounded, the table is an algebraic rearrangement of the model.

    If this disagrees by anything beyond floating-point noise, the intercept
    split or the sign chain is wrong — and the table would be a plausible-looking
    object that does not represent the model.
    """
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"],
        round_points=False)

    verification = scaling.verify_against_model(
        parts["binned"], parts["woe_matrix"], parts["pd_model"], scorecard,
        tolerance=1e-6)

    assert verification["max_difference_points"] < 1e-6, verification
    assert verification["passed"] is True


def test_rounding_error_is_small_and_measured():
    """Real scorecards use whole points. The cost of that must be known, not
    assumed — with a handful of characteristics it is well under a point."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"],
        round_points=True)

    verification = scaling.verify_against_model(
        parts["binned"], parts["woe_matrix"], parts["pd_model"], scorecard)

    assert verification["passed"] is True
    # Rounding to integers cannot cost more than half a point per characteristic,
    # which is exactly the default tolerance.
    bound = 0.5 * scorecard["n_features"]
    assert verification["max_difference_points"] <= bound + 1e-9
    assert verification["tolerance_points"] == pytest.approx(bound + 0.01)


def test_tolerance_scales_with_the_number_of_characteristics():
    """Why a fixed tolerance is the wrong design.

    The first real run failed an eleven-characteristic card at 2.277 points
    against a flat 1.0 limit. Rounding error is bounded by half a point per
    characteristic, so the admissible total grows with the card — one number for
    all cards either fails good tables or passes broken ones.
    """
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    small = dict(scorecard, n_features=2)
    large = dict(scorecard, n_features=11)

    assert scaling.rounding_tolerance(large) > scaling.rounding_tolerance(small)
    assert scaling.rounding_tolerance(small) == pytest.approx(1.01)
    assert scaling.rounding_tolerance(large) == pytest.approx(5.51)

    # Unrounded, only floating-point noise is admissible.
    exact = dict(scorecard, rounded=False, n_features=11)
    assert scaling.rounding_tolerance(exact) < 0.1


def test_verification_fails_loudly_on_a_tampered_table():
    """A table that does not match its model must be refused, not used."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    # Break one characteristic's points.
    scorecard["table"].loc[0, "points"] = int(scorecard["table"].loc[0, "points"]) + 60

    with pytest.raises(scaling.ScalingError) as error:
        scaling.verify_against_model(parts["binned"], parts["woe_matrix"],
                                     parts["pd_model"], scorecard, tolerance=1.0)
    assert "must not be used" in str(error.value)


def test_score_range_is_sensible():
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    low = scorecard["score_range"]["minimum"]
    high = scorecard["score_range"]["maximum"]

    assert low < high
    # Every applicant must land inside the attainable range, by construction.
    scores, _ = scaling.score_applicants(parts["binned"], scorecard)
    assert scores.min() >= low - 1 and scores.max() <= high + 1


def test_unreachable_base_score_is_flagged_not_ignored():
    """A real finding from this fixture, now a check.

    With only two characteristics the attainable range came out 460 to 552,
    while the configured base score is 600 — a score no applicant can reach.
    The ranking is unaffected because the scaling is linear, but every score
    reads oddly against the stated reference odds and a committee will ask about
    it on sight. So it is flagged rather than silently accepted.
    """
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    low = scorecard["score_range"]["minimum"]
    high = scorecard["score_range"]["maximum"]
    base = scorecard["constants"]["base_score"]
    inside = low <= base <= high

    assert scorecard["base_score_attainable"] == inside

    if not inside:
        assert "WARNING" in scaling.scorecard_report(scorecard)


def test_raising_the_base_score_does_not_fix_an_unreachable_reference():
    """The trap, pinned.

    The obvious fix for "base score outside the attainable range" is to move the
    base score. It does nothing: the offset moves with it, so every point moves
    with it, so the range shifts by exactly the same amount. The gap is
    invariant.
    """
    parts = build_full_pipeline()
    cfg = parts["config"]

    gaps = []
    for base_score in (600, 500, 520):
        scorecard = scaling.build_points_table(
            parts["pd_model"], parts["woe_spec"],
            dict(cfg, scaling=dict(cfg["scaling"], base_score=base_score)))
        gaps.append(base_score - scorecard["score_range"]["maximum"])

    assert len(set(gaps)) == 1, (
        f"the gap should be invariant to base_score, got {gaps}"
    )


def test_lowering_base_odds_is_what_fixes_it():
    """`base_odds` is the real lever.

    base_odds = 50 asserts a reference applicant with a PD of 1/51 = 1.96%. If
    the model never produces a PD that low, the scale is anchored at a
    creditworthiness nobody in the population has. Lowering base_odds into the
    attainable PD range is the fix.
    """
    parts = build_full_pipeline()
    cfg = parts["config"]

    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], cfg)

    if scorecard["base_score_attainable"]:
        pytest.skip("this fixture's reference is already attainable")

    # Pick base_odds from the middle of the PD range the model actually spans.
    pd_range = scorecard["attainable_pd"]
    target_pd = (pd_range["best_applicant"] + pd_range["worst_applicant"]) / 2
    sensible_odds = (1 - target_pd) / target_pd

    rebased = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"],
        dict(cfg, scaling=dict(cfg["scaling"], base_odds=sensible_odds)))

    assert rebased["base_score_attainable"] is True
    # And the warning is gone from the report.
    assert "WARNING" not in scaling.scorecard_report(rebased)


def test_attainable_pd_range_is_reported():
    """So the warning can be acted on rather than just noted."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    pd_range = scorecard["attainable_pd"]
    assert 0 < pd_range["best_applicant"] < pd_range["worst_applicant"] < 1
    assert 0 < pd_range["base_odds_implies"] < 1


def test_no_features_is_refused(config):
    with pytest.raises(scaling.ScalingError):
        scaling.build_points_table(
            {"features": [], "coefficients": pd.DataFrame(
                {"coefficient": []}), "intercept": 0.0},
            {"features": {}}, config)


# =========================================================================
# Scoring applicants
# =========================================================================

def test_scoring_is_a_lookup_and_an_addition():
    """The property that makes a scorecard explainable: the total is the sum of
    the per-characteristic points, and nothing else."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    total, contributions = scaling.score_applicants(parts["binned"], scorecard)

    np.testing.assert_allclose(total.to_numpy(),
                               contributions.sum(axis=1).to_numpy())
    assert list(contributions.columns) == [
        f.replace("_woe", "") for f in scorecard["features"]]


def test_contributions_are_the_raw_material_for_reason_codes():
    """Phase 7 needs to say WHY an applicant scored low. The lowest-scoring
    characteristic is that answer, and it has to be available per applicant."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    _, contributions = scaling.score_applicants(parts["binned"], scorecard)

    worst_characteristic = contributions.idxmin(axis=1)
    assert len(worst_characteristic) == len(parts["binned"])
    assert worst_characteristic.isin(contributions.columns).all()


def test_an_unseen_bin_gets_the_feature_mean_not_zero():
    """Zero is not neutral on this scale — the offset is baked into every cell,
    so zero points would be a severe and arbitrary penalty."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    live = parts["binned"].head(1).copy()
    feature = scorecard["features"][0].replace("_woe", "")
    live[f"{feature}_bin"] = "a_bin_that_never_existed"

    total, contributions = scaling.score_applicants(live, scorecard)

    feature_points = scorecard["table"][
        scorecard["table"]["feature"] == feature]["points"]
    assert contributions.iloc[0][feature] == pytest.approx(
        feature_points.mean(), abs=1e-9)
    assert total.iloc[0] > 0


def test_scoring_without_a_needed_column_fails_loudly():
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    feature = scorecard["features"][0].replace("_woe", "")
    with pytest.raises(scaling.ScalingError):
        scaling.score_applicants(
            parts["binned"].drop(columns=[f"{feature}_bin"]), scorecard)


# =========================================================================
# Cut-off policy
# =========================================================================

def test_cutoff_table_trades_approval_rate_against_bad_rate():
    """Each row is a policy option. A higher cut-off must approve fewer and
    cleaner applicants — if not, the score is not ranking."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])
    scores, _ = scaling.score_applicants(parts["binned"], scorecard)

    table = scaling.cutoff_table(scores, parts["frame"]["is_default"], scorecard)

    assert table["approval_rate"].is_monotonic_decreasing
    # And the approved book gets cleaner as the bar rises.
    assert table["approved_bad_rate"].iloc[-1] < table["approved_bad_rate"].iloc[0]
    assert (table["bads_rejected"] >= 0).all()


# =========================================================================
# Persistence
# =========================================================================

def test_scorecard_survives_a_json_round_trip(tmp_path):
    """A scorecard has to be scoreable years later with no code at all, so the
    frozen artefact is plain JSON."""
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])

    cfg = dict(parts["config"])
    cfg["data"] = dict(parts["config"]["data"], processed_dir=str(tmp_path))
    reloaded = scaling.load_scorecard(scaling.save_scorecard(scorecard, cfg))

    before, _ = scaling.score_applicants(parts["binned"], scorecard)
    after, _ = scaling.score_applicants(parts["binned"], reloaded)

    pd.testing.assert_series_equal(before, after)


def test_report_renders():
    parts = build_full_pipeline()
    scorecard = scaling.build_points_table(
        parts["pd_model"], parts["woe_spec"], parts["config"])
    verification = scaling.verify_against_model(
        parts["binned"], parts["woe_matrix"], parts["pd_model"], scorecard)

    report = scaling.scorecard_report(scorecard, verification)
    assert "points table" in report
    assert "Higher score = safer" in report
    assert "PDO" in report
