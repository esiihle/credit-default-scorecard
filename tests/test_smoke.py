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


def test_every_pipeline_stage_is_now_built():
    """Phase 0 kept a list of stages that still raised NotImplementedError.

    As of Phase 7 (20 Oct 2026) that list is empty: load, binning, woe, model,
    validate, scaling, scorecard and benchmark are all implemented and each has
    its own test module. This replaces the old "declared but pending" check so
    the suite records that the scaffold is fully filled in.
    """
    import inspect

    stages = {
        "load": load.run_phase1,
        "binning": binning.fit_bins,
        "woe": woe.compute_woe_iv,
        "model": model.fit_pd_model,
        "validate": validate.evaluate,
        "scaling": scaling.build_points_table,
        "scorecard": scorecard.score_applicant,
        "benchmark": benchmark.fit_benchmark,
    }

    for name, function in stages.items():
        source = inspect.getsource(function)
        assert "NotImplementedError" not in source, (
            f"{name} still raises NotImplementedError"
        )

def test_config_schema_is_a_subset_of_what_the_raw_file_contains():
    """The schema says what we MODEL; RAW_COLUMN_ORDER says what the file HAS.

    These were identical until 20 Oct 2026, when `personal_status_sex` was
    removed from the schema as a protected characteristic. The file still
    contains it — the source data is not edited — so the right relationship is
    now subset, not equality.

    Both directions still matter:
      * every column the schema names must exist in the file, or loading breaks;
      * a column in the file but not the schema is a deliberate exclusion.
    """
    config = load_config()
    schema = config["schema"]

    modelled = (set(schema["numeric_columns"])
                | set(schema["categorical_columns"])
                | {schema["target_column"]})
    in_file = set(load.RAW_COLUMN_ORDER)

    unknown = modelled - in_file
    assert not unknown, (
        f"the schema names columns the raw file does not contain: {sorted(unknown)}"
    )

    # Anything excluded must be excluded on purpose, which for now means it is a
    # declared protected characteristic.
    excluded = in_file - modelled
    protected = set(config.get("fair_lending", {}).get(
        "protected_characteristics", []))
    unexplained = excluded - protected
    assert not unexplained, (
        f"columns are in the file but not modelled, with no reason recorded: "
        f"{sorted(unexplained)}. Either model them or add them to "
        f"fair_lending.protected_characteristics."
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
