"""
Phase 7 tests — the applicant scorer and reason codes.

Reason codes are a legal artefact, so the tests are about correctness and
fairness rather than performance: the right characteristic named, the right
reference point, nothing listed that did not cost points, and an incomplete
application still scoreable.

Run with:  pytest tests/test_scorecard.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import binning, model, scaling, scorecard as scorer, woe


@pytest.fixture
def pipeline():
    """Run the real chain: binning -> WOE -> PD model -> points table."""
    rng = np.random.default_rng(0)
    n = 2000

    duration = rng.integers(6, 60, n)
    checking = rng.choice(["none", "low", "medium", "high"], n,
                          p=[0.25, 0.3, 0.27, 0.18])
    checking_risk = pd.Series(checking).map(
        {"none": 0.10, "low": 0.50, "medium": 0.32, "high": 0.18})

    risk = np.clip(0.35 * checking_risk + 0.006 * (duration - 30) + 0.15,
                   0.02, 0.95)
    frame = pd.DataFrame({
        "duration_months": duration,
        "checking_status": checking,
        "is_default": (rng.random(n) < risk).astype(int),
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
    points = scaling.build_points_table(pd_model, woe_spec, cfg)

    return {"config": cfg, "binning_spec": bin_spec, "woe_spec": woe_spec,
            "points": points, "frame": frame, "binned": binned}


def best_and_worst_bins(points_table, characteristic):
    """The highest- and lowest-scoring bin labels for one characteristic."""
    rows = points_table["table"]
    rows = rows[rows["feature"] == characteristic]
    return (str(rows.loc[rows["points"].idxmax(), "bin"]),
            str(rows.loc[rows["points"].idxmin(), "bin"]))


# =========================================================================
# Scoring
# =========================================================================

def test_scores_one_applicant_with_pd_and_breakdown(pipeline):
    result = scorer.score_applicant(
        {"duration_months": 12, "checking_status": "high"},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    assert pipeline["points"]["score_range"]["minimum"] - 1 <= result["score"]
    assert result["score"] <= pipeline["points"]["score_range"]["maximum"] + 1
    assert 0 < result["probability_of_default"] < 1
    assert len(result["breakdown"]) == pipeline["points"]["n_features"]


def test_the_score_is_the_sum_of_the_breakdown(pipeline):
    """The property that makes the explanation honest: the arithmetic shown to
    the applicant is the arithmetic that produced the decision."""
    result = scorer.score_applicant(
        {"duration_months": 30, "checking_status": "low"},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    assert result["score"] == pytest.approx(result["breakdown"]["points"].sum())


def test_a_safer_applicant_scores_higher_and_has_a_lower_pd(pipeline):
    best_checking, worst_checking = best_and_worst_bins(
        pipeline["points"], "checking_status")

    safe = scorer.score_applicant(
        {"duration_months": 6, "checking_status": best_checking.split(" | ")[0]},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])
    risky = scorer.score_applicant(
        {"duration_months": 58, "checking_status": worst_checking.split(" | ")[0]},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    assert safe["score"] > risky["score"]
    assert safe["probability_of_default"] < risky["probability_of_default"]


def test_an_incomplete_application_is_still_scoreable(pipeline):
    """Incomplete applications are the normal case, not an error. Phase 2's
    MISSING bin and Phase 3's WOE for it are what make this work."""
    result = scorer.score_applicant(
        {"checking_status": "low"},                 # no duration at all
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    assert np.isfinite(result["score"])
    assert 0 < result["probability_of_default"] < 1


def test_a_completely_blank_application_is_still_scoreable(pipeline):
    result = scorer.score_applicant(
        {}, pipeline["points"], pipeline["binning_spec"], pipeline["config"])
    assert np.isfinite(result["score"])


def test_an_unknown_attribute_is_refused(pipeline):
    """A typo in an attribute name must not be silently ignored — it would
    score the applicant as though that information were missing."""
    with pytest.raises(scorer.ScorerError) as error:
        scorer.score_applicant(
            {"duration_month": 12},                 # missing the 's'
            pipeline["points"], pipeline["binning_spec"], pipeline["config"])
    assert "duration_month" in str(error.value)


def test_scoring_is_deterministic(pipeline):
    applicant = {"duration_months": 24, "checking_status": "medium"}
    first = scorer.score_applicant(applicant, pipeline["points"],
                                   pipeline["binning_spec"], pipeline["config"])
    second = scorer.score_applicant(applicant, pipeline["points"],
                                    pipeline["binning_spec"], pipeline["config"])
    assert first["score"] == second["score"]
    assert first["reason_codes"] == second["reason_codes"]


# =========================================================================
# Reason codes — the legal artefact
# =========================================================================

def test_reason_codes_name_the_characteristic_that_cost_the_most(pipeline):
    """Ranked by points lost against the best available bin, not by low points."""
    best_checking, worst_checking = best_and_worst_bins(
        pipeline["points"], "checking_status")

    result = scorer.score_applicant(
        # Best possible duration, worst possible checking status.
        {"duration_months": 6, "checking_status": worst_checking.split(" | ")[0]},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    codes = result["reason_codes"]
    assert len(codes) >= 1
    assert codes[0]["characteristic"] == "checking_status", (
        f"the worst characteristic should lead the reasons; got {codes}"
    )
    # Ranked by points lost, descending.
    lost = [code["points_lost"] for code in codes]
    assert lost == sorted(lost, reverse=True)


def test_best_possible_applicant_has_no_reason_codes(pipeline):
    """An applicant holding the best bin everywhere has lost nothing.

    This falls out of the arithmetic rather than needing a special case, which
    is exactly why it is worth asserting: a reason code saying "your perfect
    credit history counted against you" would be worse than no notice at all.
    """
    table = pipeline["points"]["table"]
    applicant = {}
    for characteristic in table["feature"].unique():
        rows = table[table["feature"] == characteristic]
        best_bin = str(rows.loc[rows["points"].idxmax(), "bin"])

        if characteristic in pipeline["config"]["schema"]["categorical_columns"]:
            applicant[characteristic] = best_bin.split(" | ")[0]
        else:
            # Pick a value inside the best numeric band.
            definition = pipeline["binning_spec"]["features"][characteristic]
            labels = binning.bin_label([float(e) for e in definition["edges"]])
            index = labels.index(best_bin)
            edges = [float(e) for e in definition["edges"]]
            low, high = edges[index], edges[index + 1]
            if np.isneginf(low):
                applicant[characteristic] = high - 1
            elif np.isposinf(high):
                applicant[characteristic] = low + 1
            else:
                applicant[characteristic] = (low + high) / 2

    result = scorer.score_applicant(applicant, pipeline["points"],
                                    pipeline["binning_spec"],
                                    pipeline["config"])

    assert result["reason_codes"] == [], (
        f"best-everywhere applicant should have no reasons:\n"
        f"{result['breakdown']}"
    )
    assert "No reason codes" in scorer.explain(result)


def test_reason_codes_are_capped_at_the_requested_number(pipeline):
    _, worst_checking = best_and_worst_bins(pipeline["points"],
                                            "checking_status")
    result = scorer.score_applicant(
        {"duration_months": 58, "checking_status": worst_checking.split(" | ")[0]},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"],
        top_reasons=1)

    assert len(result["reason_codes"]) <= 1


def test_trivial_losses_are_not_reported_as_reasons(pipeline):
    """Being one point short of the best bin is not why anyone was declined,
    and listing it would bury the real cause."""
    breakdown = pd.DataFrame([
        {"characteristic": "a", "applicant_bin": "x", "points": 50,
         "best_available": 50.4, "worst_available": 20, "points_lost": 0.4,
         "best_bin": "y"},
        {"characteristic": "b", "applicant_bin": "p", "points": 20,
         "best_available": 60, "worst_available": 20, "points_lost": 40,
         "best_bin": "q"},
    ])

    codes = scorer.reason_codes(breakdown, top_n=4, minimum_points_lost=1.0)
    assert [c["characteristic"] for c in codes] == ["b"]


def test_every_reason_code_carries_what_the_applicant_needs(pipeline):
    """A notice has to say what their value was and what would score better."""
    _, worst_checking = best_and_worst_bins(pipeline["points"],
                                            "checking_status")
    result = scorer.score_applicant(
        {"duration_months": 58, "checking_status": worst_checking.split(" | ")[0]},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    for code in result["reason_codes"]:
        assert set(code) == {"rank", "characteristic", "your_value",
                             "points_lost", "best_value"}
        assert code["your_value"] != ""
        assert code["best_value"] != ""
        assert code["points_lost"] > 0


# =========================================================================
# Decisions
# =========================================================================

def test_decision_approves_above_the_cutoff_and_declines_below(pipeline):
    result = scorer.score_applicant(
        {"duration_months": 24, "checking_status": "medium"},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    lenient = scorer.decision(result, result["score"] - 20)
    strict = scorer.decision(result, result["score"] + 20)

    assert lenient["approved"] is True
    assert strict["approved"] is False
    assert lenient["margin"] == pytest.approx(20)
    assert strict["margin"] == pytest.approx(-20)


def test_reasons_are_attached_only_on_a_decline(pipeline):
    """Reasons are required when an application is refused. Attaching them to an
    approval would be noise at best and confusing at worst."""
    _, worst_checking = best_and_worst_bins(pipeline["points"],
                                            "checking_status")
    result = scorer.score_applicant(
        {"duration_months": 58, "checking_status": worst_checking.split(" | ")[0]},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    assert scorer.decision(result, result["score"] + 50)["reason_codes"] != []
    assert scorer.decision(result, result["score"] - 50)["reason_codes"] == []


# =========================================================================
# Explanation and batch
# =========================================================================

def test_explanation_is_readable_and_complete(pipeline):
    result = scorer.score_applicant(
        {"duration_months": 48, "checking_status": "low"},
        pipeline["points"], pipeline["binning_spec"], pipeline["config"])

    text = scorer.explain(result, cutoff_score=500)

    assert "Score:" in text
    assert "Probability of default:" in text
    assert "How the score was built:" in text
    assert ("APPROVE" in text) or ("DECLINE" in text)
    # Every characteristic appears in the breakdown.
    for characteristic in result["breakdown"]["characteristic"]:
        assert characteristic in text


def test_batch_scoring_matches_single_scoring(pipeline):
    """The batch path must not be a separate implementation that can drift."""
    applicants = pipeline["frame"].head(25).drop(columns=["is_default"])
    batch = scorer.score_many(applicants, pipeline["points"],
                              pipeline["binning_spec"], pipeline["config"])

    for position in range(5):
        single = scorer.score_applicant(
            applicants.iloc[[position]], pipeline["points"],
            pipeline["binning_spec"], pipeline["config"])
        assert batch.iloc[position]["score"] == pytest.approx(single["score"])


def test_batch_scoring_fills_every_reason_column(pipeline):
    applicants = pipeline["frame"].head(10).drop(columns=["is_default"])
    batch = scorer.score_many(applicants, pipeline["points"],
                              pipeline["binning_spec"], pipeline["config"],
                              top_reasons=3)

    for index in (1, 2, 3):
        assert f"reason_{index}" in batch.columns
    assert len(batch) == 10


def test_empty_batch_is_refused(pipeline):
    with pytest.raises(scorer.ScorerError):
        scorer.score_many(pd.DataFrame(), pipeline["points"],
                          pipeline["binning_spec"], pipeline["config"])


def test_scorer_works_from_frozen_artefacts_alone(pipeline, tmp_path):
    """The point of the whole design: scoring must not need the training data,
    the model object, or the pipeline — only the frozen JSON."""
    cfg = dict(pipeline["config"])
    cfg["data"] = dict(pipeline["config"]["data"], processed_dir=str(tmp_path))

    bin_path = binning.save_spec(pipeline["binning_spec"], cfg)
    card_path = scaling.save_scorecard(pipeline["points"], cfg)

    reloaded_bins = binning.load_spec(bin_path)
    reloaded_card = scaling.load_scorecard(card_path)

    applicant = {"duration_months": 36, "checking_status": "low"}
    original = scorer.score_applicant(applicant, pipeline["points"],
                                      pipeline["binning_spec"], cfg)
    from_disk = scorer.score_applicant(applicant, reloaded_card,
                                       reloaded_bins, cfg)

    assert original["score"] == pytest.approx(from_disk["score"])
    assert original["reason_codes"] == from_disk["reason_codes"]


# =========================================================================
# Fair lending — found on a live explanation, 18 Oct 2026
# =========================================================================

def test_a_protected_characteristic_in_a_reason_code_is_refused(pipeline):
    """The finding that stopped Phase 7.

    Running the scorer on real German Credit produced this as reason code 4:

        personal_status_sex: you are in 'female: divorced / separated /
        married', which costs 18 points against the best available
        ('male: divorced / separated')

    That is an adverse-action notice telling an applicant their SEX counted
    against them — unlawful under ECOA / Regulation B in the US, and under the
    National Credit Act and Equality Act in South Africa.

    The code was working perfectly. The problem is that the MODEL contains a
    protected characteristic, which was flagged as an open decision in Phase 3
    and never resolved. The scorer now refuses to emit the notice.
    """
    cfg = dict(pipeline["config"])
    cfg["fair_lending"] = {
        "protected_characteristics": ["checking_status"],   # stand-in for the test
        "on_protected_reason_code": "refuse",
    }

    with pytest.raises(scorer.FairLendingError) as error:
        scorer.score_applicant(
            {"duration_months": 6, "checking_status": "low"},
            pipeline["points"], pipeline["binning_spec"], cfg)

    message = str(error.value)
    assert "checking_status" in message
    # The error must say that suppressing the notice is not the fix.
    assert "would NOT make the model lawful" in message
    assert "Fix it upstream" in message


def test_warn_policy_suppresses_the_code_but_not_the_problem(pipeline):
    """Suppressing the notice does not make the model lawful, and the
    explanation has to keep saying so."""
    cfg = dict(pipeline["config"])
    cfg["fair_lending"] = {
        "protected_characteristics": ["checking_status"],
        "on_protected_reason_code": "warn",
    }

    result = scorer.score_applicant(
        {"duration_months": 6, "checking_status": "low"},
        pipeline["points"], pipeline["binning_spec"], cfg)

    # The protected characteristic is gone from the reasons...
    assert all(code["characteristic"] != "checking_status"
               for code in result["reason_codes"])
    # ...but it is still named as present in the model.
    assert "checking_status" in result["protected_in_model"]

    text = scorer.explain(result)
    assert "WARNING" in text
    assert "not fit for lending use" in text


def test_no_protected_characteristics_means_no_warning(pipeline):
    """A clean model must not carry a spurious fair-lending warning."""
    cfg = dict(pipeline["config"])
    cfg["fair_lending"] = {"protected_characteristics": [],
                           "on_protected_reason_code": "refuse"}

    result = scorer.score_applicant(
        {"duration_months": 24, "checking_status": "medium"},
        pipeline["points"], pipeline["binning_spec"], cfg)

    assert result["protected_in_model"] == []
    assert "WARNING" not in scorer.explain(result)


def test_a_protected_characteristic_below_the_threshold_does_not_trigger(pipeline):
    """The guard fires on what would actually be CITED, not on mere presence in
    the model — that is what `protected_in_model` reports separately."""
    breakdown = pd.DataFrame([
        {"characteristic": "personal_status_sex", "applicant_bin": "x",
         "points": 50, "best_available": 50.2, "worst_available": 20,
         "points_lost": 0.2, "best_bin": "y"},
        {"characteristic": "duration_months", "applicant_bin": "p",
         "points": 20, "best_available": 60, "worst_available": 20,
         "points_lost": 40, "best_bin": "q"},
    ])
    cfg = {"fair_lending": {"protected_characteristics": ["personal_status_sex"],
                            "on_protected_reason_code": "refuse"}}

    codes = scorer.reason_codes(breakdown, 4, minimum_points_lost=1.0,
                                config=cfg)
    assert [c["characteristic"] for c in codes] == ["duration_months"]
