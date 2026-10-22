"""
Phase 8 tests — the README must stay true.

A README is the one file nobody re-reads after the code changes, which is how a
portfolio project ends up claiming numbers its own pipeline no longer produces.
These tests make that failure loud.

Two kinds of test here:

* Always run — the README references only charts the chart script writes,
  carries no leftover placeholders, names the real repository, and still states
  the fair-lending finding and the failing validation gates.
* Run when ``docs/charts/numbers.json`` is present — every headline number in
  the README matches the run that produced the charts. Without the real
  german.data there is no numbers.json, so those skip with instructions rather
  than fail on a fresh clone.

Run with:  pytest tests/test_docs.py -v
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

README = REPO_ROOT / "README.md"
CHART_DIR = REPO_ROOT / "docs" / "charts"
NUMBERS = CHART_DIR / "numbers.json"

# The README quotes discrimination metrics to four decimal places, so anything
# inside half of the last place is the same number.
TOLERANCE = 0.00005


@pytest.fixture(scope="module")
def readme():
    return README.read_text(encoding="utf-8")


def collapse(text):
    """Whitespace-normalised text, for matching a phrase that may be wrapped.

    A phrase check against the raw file fails the moment the sentence wraps
    across a line — which says nothing about the README and everything about
    the test. Collapsing runs of whitespace to one space tests the prose.
    """
    return " ".join(text.split()).lower()


@pytest.fixture(scope="module")
def chart_script():
    """Import scripts/make_charts.py for its CHARTS manifest.

    Imported by path rather than duplicated here, so the manifest has one
    definition and this test cannot drift from it.
    """
    spec = importlib.util.spec_from_file_location(
        "make_charts", REPO_ROOT / "scripts" / "make_charts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def numbers():
    if not NUMBERS.exists():
        pytest.skip(
            "docs/charts/numbers.json is absent — run "
            "`python scripts/make_charts.py` on real data first. On a clone "
            "without data/raw/german.data this is expected."
        )
    recorded = json.loads(NUMBERS.read_text())

    # A numbers.json built from the bundled sample is itself the bug, not a
    # reason to skip: it means sample output reached the published folder. Fail
    # here with the cause, rather than letting every downstream test die on a
    # KeyError for a figure the sample run never produced.
    if recorded.get("real_data") is not True:
        pytest.fail(
            "docs/charts/numbers.json came from the bundled sample "
            f"({recorded.get('matches', recorded.get('n_train', '?'))} rows). "
            "Delete docs/charts/, put the real data in data/raw/, and run "
            "`python scripts/make_charts.py` again."
        )
    return recorded


# --- Always run ------------------------------------------------------------

def test_readme_only_links_charts_the_script_writes(readme, chart_script):
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))
    produced = set(chart_script.CHARTS.values())

    unknown = linked - produced
    assert not unknown, (
        f"README links charts that make_charts.py does not write: {sorted(unknown)}"
    )
    assert linked, "the README should show the charts"


def test_every_chart_the_script_writes_is_shown(chart_script, readme):
    """A chart nobody looks at is a chart nobody maintains."""
    produced = set(chart_script.CHARTS.values())
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))

    unused = produced - linked
    assert not unused, (
        f"make_charts.py writes charts the README never shows: {sorted(unused)}"
    )


def test_no_placeholders_survive(readme):
    forbidden = ["🚧", "TODO", "TBD", "Planned headline outputs",
                 "in active development", "<your-username>", "coming soon"]
    found = [marker for marker in forbidden if marker.lower() in readme.lower()]
    assert not found, f"placeholder text left in the README: {found}"


def test_readme_names_the_real_repository(readme):
    assert "github.com/esiihle/credit-default-scorecard" in readme


def test_readme_states_every_gate_and_its_verdict(readme):
    """The README must report each validation gate against its threshold.

    NOT a hardcoded number. The first version of this test asserted the literal
    string "0.2751" — the train-to-test Gini drop — and "fail". Both came from a
    run that had never happened on real data, so the test spent a day ENFORCING
    the wrong figure rather than catching it: the README said the model failed
    two gates, the real run passed all four, and the suite was green on the lie.

    A test may check that a claim is PRESENT and that it AGREES with the
    recorded run. The moment it hardcodes the value, it stops being a check and
    becomes a second copy of the thing it was meant to verify.
    """
    lowered = readme.lower()
    assert "threshold" in lowered
    for gate in ("gini", "psi", "drop"):
        assert gate in lowered, f"the README should report the {gate} gate"


def test_readme_keeps_the_fair_lending_finding(readme):
    """The protected-characteristic finding is the project's best material.

    It is also the part a nervous author would soften. The removal, the guard
    and the lesson must all stay. The measured COST is checked separately,
    against the file that records it, never against a literal here.
    """
    lowered = readme.lower()
    assert "personal_status_sex" in lowered
    assert "fair lending" in lowered
    assert ("suppressing a disclosure is not the same as removing a variable"
            in collapse(readme))


def test_readme_explains_the_benchmark_trade_off(readme):
    """A benchmark reported without its overfitting gap is half the story."""
    lowered = collapse(readme)
    assert "reason code" in lowered
    assert "gradient boosting" in lowered
    assert "overfit" in lowered or "train \u2192 test" in lowered or "drop" in lowered


def test_chart_manifest_has_a_purpose_for_every_chart(chart_script):
    assert set(chart_script.CHARTS) == set(chart_script.CHART_PURPOSE)


# --- Run when a real chart run is committed --------------------------------

def assert_quoted(readme, value, label, places=4):
    # Values are compared at the precision the README quotes them, never by
    # hardcoding the digits here — see the note in
    # test_readme_states_every_gate_and_its_verdict.
    """The README must quote `value` somewhere, to its own precision."""
    text = f"{value:.{places}f}"
    assert text in readme, (
        f"the README does not quote {label} = {text} from numbers.json. "
        "Re-run scripts/make_charts.py and update the tables."
    )


def test_readme_quotes_the_test_gini(readme, numbers):
    assert_quoted(readme, numbers["gini"], "test Gini")


def test_readme_quotes_the_train_gini_and_drop(readme, numbers):
    assert_quoted(readme, numbers["train_gini"], "train Gini")
    assert_quoted(readme, numbers["gini_drop"], "the train-to-test Gini drop")


def test_readme_quotes_ks_and_auc(readme, numbers):
    assert_quoted(readme, numbers["ks"], "test KS")
    assert_quoted(readme, numbers["auc"], "test AUC")


def test_readme_quotes_the_benchmark_comparison(readme, numbers):
    if "benchmark_test_gini" not in numbers:
        pytest.skip("the benchmark did not run in that chart run")
    assert_quoted(readme, numbers["benchmark_test_gini"], "the benchmark's test Gini")
    assert_quoted(readme, numbers["benchmark_gini_drop"],
                  "the benchmark's train-to-test drop")


def test_readme_quotes_psi_and_calibration(readme, numbers):
    """Stability and calibration are gates too, and were never quoted before."""
    assert_quoted(readme, numbers["psi"], "PSI")
    if numbers.get("hosmer_lemeshow_p") is not None:
        assert_quoted(readme, numbers["hosmer_lemeshow_p"],
                      "the Hosmer-Lemeshow p-value", places=3)


def test_readme_gate_verdicts_match_the_config_thresholds(readme, numbers):
    """The README's pass/fail marks must agree with config.yaml and the run.

    This is the test that would have caught yesterday's error on its own: the
    README claimed two failures, and the recorded run cleared every threshold.
    """
    from src.config import load_config
    thresholds = load_config().get("validation", {})

    gates = {
        "Gini >= minimum": numbers["gini"] >= thresholds.get("min_gini", 0.30),
        "PSI <= maximum": numbers["psi"] <= thresholds.get("max_psi", 0.25),
        "Gini drop <= maximum":
            numbers["gini_drop"] <= thresholds.get("max_gini_drop", 0.15),
    }
    passed = sum(gates.values())

    # The README marks each gate with a tick or a cross. Count them and check
    # the totals agree with what the recorded run actually did.
    ticks, crosses = readme.count("\u2705"), readme.count("\u274c")
    assert ticks + crosses >= len(gates), (
        "the README should mark each validation gate with \u2705 or \u274c; "
        f"found {ticks} ticks and {crosses} crosses for {len(gates)} gates"
    )
    assert ticks >= passed, (
        f"the recorded run passes {passed} of {len(gates)} gates "
        f"({gates}) but the README shows only {ticks} ticks"
    )
    assert crosses <= len(gates) - passed, (
        f"the README shows {crosses} failed gates; the recorded run fails "
        f"{len(gates) - passed}"
    )


def test_readme_quotes_the_scaling_constants(readme, numbers):
    """PDO, base score and base odds define every number on the points table."""
    for key, label in (("pdo", "PDO"), ("base_score", "the base score"),
                       ("base_odds", "the base odds")):
        assert f"{numbers[key]:.0f}" in readme, f"the README should state {label}"


def test_committed_charts_came_from_real_data(numbers):
    """Charts built on the 24-row sample must never be committed as results.

    make_charts.py stamps them ILLUSTRATIVE, but a stamped PNG is still a PNG.
    This is the test that stops one reaching the front page.
    """
    assert numbers.get("real_data") is True, (
        "docs/charts/ was generated from the bundled sample. Download "
        "german.data and run `python scripts/make_charts.py` again."
    )


def test_the_protected_characteristic_is_not_in_the_model(numbers):
    """The 20 Oct removal, asserted against the run that drew the charts."""
    assert numbers.get("protected_excluded"), (
        "config.fair_lending.protected_characteristics should name the "
        "characteristics excluded from the model"
    )


def test_every_chart_the_readme_shows_exists_on_disk(readme):
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))
    if not any((CHART_DIR / name).exists() for name in linked):
        pytest.skip("no charts committed yet — run scripts/make_charts.py")

    missing = [name for name in linked if not (CHART_DIR / name).exists()]
    assert not missing, f"README shows charts that are not in the repo: {missing}"


# --- Added 22 Oct 2026 -----------------------------------------------------

FAIR_LENDING = CHART_DIR / "fair_lending.json"


def test_the_ci_badge_points_at_a_workflow_that_exists(readme):
    """A badge is a claim about the project; the workflow is what makes it true.

    The README carried a CI badge for a day while .github/workflows/tests.yml
    did not exist, so the badge rendered broken on the repository's front page.
    """
    if "actions/workflows/" not in readme:
        pytest.skip("no CI badge in the README")

    referenced = set(re.findall(r"actions/workflows/([\w.-]+)/badge\.svg", readme))
    assert referenced, "could not parse the workflow name out of the badge URL"
    for name in referenced:
        path = REPO_ROOT / ".github" / "workflows" / name
        assert path.exists(), (
            f"the README's CI badge points at .github/workflows/{name}, "
            "which does not exist"
        )


def test_ci_runs_the_suite_on_more_than_one_python_version():
    workflow = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    if not workflow.exists():
        pytest.skip("no CI workflow")

    text = workflow.read_text()
    assert "pytest" in text, "CI must run the test suite"
    versions = re.findall(r'"3\.\d+"', text)
    assert len(set(versions)) >= 2, (
        f"CI should test at least two Python versions, found {sorted(set(versions))}"
    )


def test_readme_matches_the_measured_fair_lending_cost(readme):
    """When the cost has been measured, the README must quote THAT measurement.

    Skips until `scripts/measure_fair_lending_cost.py` has been run on real
    data. It never checks a literal written here — that mistake is what let the
    old figures survive for a day with a green suite.
    """
    if not FAIR_LENDING.exists():
        pytest.skip(
            "docs/charts/fair_lending.json is absent — run "
            "`python scripts/measure_fair_lending_cost.py` on real data first."
        )

    measured = json.loads(FAIR_LENDING.read_text())
    with_it = measured["with_protected"]["test_gini"]
    without = measured["without_protected"]["test_gini"]

    # If the characteristic was never selected the two are the same model, and
    # there is no before/after pair to quote.
    if abs(with_it - without) < 1e-12:
        assert "not selected" in collapse(readme) or "iv" in collapse(readme), (
            "when removing the characteristic changes nothing, the README "
            "should say why rather than showing two identical figures"
        )
        return

    for value, label in ((with_it, "test Gini with the characteristic"),
                         (without, "test Gini without it")):
        assert f"{value:.4f}" in readme, (
            f"the README does not quote {label} = {value:.4f} from "
            "fair_lending.json"
        )


# --- Added 22 Oct 2026: the guard that was missing -------------------------

def test_sample_charts_cannot_be_written_into_the_published_folder(chart_script):
    """docs/charts is what the README publishes, so sample data may not reach it.

    On 21 Oct 2026 the football project committed three ILLUSTRATIVE-stamped
    charts built from a 12-match sample, and shipped a README referencing two
    more that did not exist. The stamp had worked exactly as designed — and a
    stamp is a warning, which is something a person can walk past.

    `refuse_illustrative_output` turns the warning into a refusal: it returns
    True (refuse) for the published folder and False (allow) for anywhere else,
    so previewing a chart's layout on sample data stays easy and publishing one
    stops being possible by accident.
    """
    published = REPO_ROOT / "docs" / "charts"

    # The published folder, without the override: refused.
    assert chart_script.refuse_illustrative_output(published, False) is True

    # The same folder with the override asked for by name: allowed.
    assert chart_script.refuse_illustrative_output(published, True) is False

    # Anywhere else: always allowed, override or not.
    elsewhere = REPO_ROOT / "build" / "preview"
    assert chart_script.refuse_illustrative_output(elsewhere, False) is False


def test_the_refusal_is_decided_before_any_work_is_done(chart_script):
    """`will_use_sample` must answer from the filesystem alone.

    The first version of this guard ran after `gather()`, which meant a run
    doomed to be refused still paid for a full pipeline — on the football
    project, an entire walk-forward backtest — before saying so.
    """
    from src.config import load_config
    config = load_config()

    # Forced sample mode is knowable without touching the pipeline.
    assert chart_script.will_use_sample(config, forced=True) is True
    # And the unforced answer is a bool, derived from whether raw data exists.
    assert isinstance(chart_script.will_use_sample(config, forced=False), bool)
