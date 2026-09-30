"""
Phase 0 smoke tests.

These don't test model correctness (those tests arrive with each phase). They
check the scaffolding holds together: config loads, every module imports, the
sample data loads, and the EDA report renders. A green run here means a fresh
clone is wired correctly.

Run with:  pytest
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import load, binning, woe, model, validate, scaling, scorecard, benchmark


def test_config_loads():
    """The config file parses and has the sections we rely on."""
    config = load_config()
    assert "data" in config
    assert "schema" in config


def test_sample_data_loads_and_reports():
    """The bundled sample loads and the EDA report renders."""
    config = load_config()
    sample = REPO_ROOT / config["data"]["sample_path"]
    df = load.load_data(sample, config)
    assert len(df) > 0
    report = load.build_eda_report(df, config)
    assert "rows:" in report
    assert "target" in report  # target distribution line is present


def test_missing_target_fails_loudly():
    """Data without the configured target column raises a clear error."""
    import pandas as pd

    bad = pd.DataFrame({"age_years": [30, 40]})  # no 'target' column
    bad_path = REPO_ROOT / "data" / "sample" / "_tmp_bad.csv"
    bad.to_csv(bad_path, index=False)
    try:
        config = load_config()
        with pytest.raises(ValueError):
            load.load_data(bad_path, config)
    finally:
        bad_path.unlink(missing_ok=True)


def test_downstream_stages_declared_but_pending():
    """Every downstream stage exists and cleanly signals 'not built yet'."""
    for call in (
        lambda: binning.fit_bins(None),
        lambda: woe.compute_woe_iv(None),
        lambda: model.fit_pd_model(None),
        lambda: validate.evaluate(None),
        lambda: scaling.build_scorecard(None),
        lambda: scorecard.score_applicant(None),
        lambda: benchmark.run_benchmark(None),
    ):
        with pytest.raises(NotImplementedError):
            call()


# --- Added 30 Sep 2026: real-data loading and the code table ---------------


def test_raw_column_order_matches_the_config_schema():
    """Every column the config expects is produced by the raw loader."""
    config = load_config()
    schema = config["schema"]
    expected = set(schema["numeric_columns"]) | set(schema["categorical_columns"])
    expected.add(schema["target_column"])

    assert expected == set(load.RAW_COLUMN_ORDER), (
        "config.yaml and RAW_COLUMN_ORDER have drifted apart: "
        f"{expected.symmetric_difference(set(load.RAW_COLUMN_ORDER))}"
    )


def test_decode_values_translates_codes_to_labels():
    """A-codes become readable labels."""
    import pandas as pd

    df = pd.DataFrame({"checking_status": ["A11", "A14"], "housing": ["A152", "A151"]})
    out = load.decode_values(df)

    assert out["checking_status"].tolist() == ["< 0 DM", "no checking account"]
    assert out["housing"].tolist() == ["own", "rent"]


def test_unknown_code_fails_loudly():
    """An unrecognised code stops the run instead of creating a junk bin."""
    import pandas as pd

    df = pd.DataFrame({"housing": ["A151", "A999"]})
    with pytest.raises(ValueError) as err:
        load.decode_values(df)
    assert "A999" in str(err.value)
