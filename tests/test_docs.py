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
    return json.loads(NUMBERS.read_text())


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


def test_readme_reports_the_failing_gates(readme):
    """Two validation gates fail. The README may not quietly stop saying so.

    This test exists because the temptation is real: the easiest way to make a
    portfolio project look stronger is to delete the sentence admitting it did
    not clear a threshold. Here that deletes a passing test with it.
    """
    lowered = readme.lower()
    assert "fail" in lowered
    assert "0.2751" in readme, "the train-to-test Gini drop must be stated"
    assert "threshold" in lowered


def test_readme_keeps_the_fair_lending_finding(readme):
    """The protected-characteristic finding is the project's best material.

    It is also the part a nervous author would soften. Both the removal and its
    measured cost have to stay in the README.
    """
    lowered = readme.lower()
    assert "personal_status_sex" in lowered
    assert "fair lending" in lowered
    assert "0.1997" in readme and "0.2100" in readme, (
        "the README must state the Gini before and after removing the "
        "protected characteristic"
    )
    assert ("suppressing a disclosure is not the same as removing a variable"
            in collapse(readme))


def test_readme_explains_the_benchmark_trade_off(readme):
    """A benchmark reported without its overfitting gap is half the story."""
    assert "0.2137" in readme, "the benchmark's test Gini"
    assert "0.6487" in readme, "the benchmark's train-to-test drop"
    assert "reason code" in readme.lower()


def test_chart_manifest_has_a_purpose_for_every_chart(chart_script):
    assert set(chart_script.CHARTS) == set(chart_script.CHART_PURPOSE)


# --- Run when a real chart run is committed --------------------------------

def assert_quoted(readme, value, label, places=4):
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
