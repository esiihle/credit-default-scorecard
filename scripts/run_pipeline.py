"""
Credit Default Scorecard — end-to-end pipeline runner (config-driven).

Phase 0: this runs the implemented stages (load + EDA report) fully, then walks
the remaining stages and reports each as [done] or [pending]. As each phase is
built, its stage lights up here automatically — so this runner doubles as the
project's live progress indicator.

30 Sep 2026: the runner now prefers the REAL German Credit file in data/raw/.
If it isn't there it falls back to the committed sample and prints the download
link, so a fresh clone always runs.

Usage
-----
    python scripts/run_pipeline.py                      # real data if present, else sample
    python scripts/run_pipeline.py --config config.yaml # explicit config
    python scripts/run_pipeline.py --data path/to/other.csv  # one specific file
    python scripts/run_pipeline.py --sample             # force the bundled sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable whether this is run from the root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed
from src import load, binning, woe, model, validate, scaling, benchmark


def _run_stage(name, fn, *args, **kwargs):
    """Run one pipeline stage, reporting done/pending without crashing the run.

    A stage that isn't built yet raises NotImplementedError; we catch it and
    mark the stage pending, so the whole pipeline always completes cleanly.
    """
    try:
        result = fn(*args, **kwargs)
        print(f"  [done]    {name}")
        return result
    except NotImplementedError as exc:
        print(f"  [pending] {name}  ->  {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run the scorecard pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--data", default=None, help="Override the input CSV path")
    parser.add_argument("--sample", action="store_true",
                        help="Force the bundled sample instead of data/raw")
    parser.add_argument("--no-save", action="store_true",
                        help="Run the checks but don't write train/test to disk")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    print(f"\nScorecard pipeline — config loaded, seed={config.get('seed', 42)}")

    # --- Stage 1: load + EDA ---
    # Three ways in, in priority order:
    #   1. --data <file>  : one explicit CSV
    #   2. data/raw/german.data : the real UCI file (space-delimited, coded)
    #   3. the committed sample : so a fresh clone always runs
    print("Stage 1 — Load & EDA")
    data_cfg = config["data"]
    raw_path = REPO_ROOT / data_cfg["raw_dir"] / data_cfg.get("raw_file", "german.data")

    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        print(f"  source: explicit file — {data_path}")
        df = load.load_data(data_path, config)

    elif raw_path.exists() and not args.sample:
        print(f"  source: real data — {raw_path.name} ({data_cfg.get('source', 'UCI')})")
        df = load.load_german_raw(raw_path, config)

    else:
        sample_path = REPO_ROOT / data_cfg["sample_path"]
        if not args.sample:
            print(f"  MISSING: {raw_path.name}  ->  download {data_cfg['download_url']}")
            print(f"           and save it as {raw_path}")
        reason = "forced by --sample" if args.sample else "real file not downloaded yet"
        print(f"  source: bundled sample ({reason}) — {sample_path.name}")
        df = load.load_data(sample_path, config)

    print()
    print(load.build_eda_report(df, config))

    # --- Phase 1: validate, define the target, freeze the split --------------
    # Everything downstream trains on `train` and never looks at `test`.
    print()
    train, test, report, manifest = load.run_phase1(df, config, save=not args.no_save)
    print(load.quality_report(report, train, test, manifest))
    if "saved_to" in report:
        print("\n  written:")
        for path in report["saved_to"]:
            print(f"    {path}")
    print()

    # --- Downstream stages (progressively implemented) ---
    print("Downstream stages:")
    binned = _run_stage("Stage 2 — Binning", binning.fit_bins, df, config)
    woe_feats = _run_stage("Stage 3 — WOE / IV", woe.compute_woe_iv, df, config)
    pd_model = _run_stage("Stage 4 — PD model", model.fit_pd_model, woe_feats, config)
    _run_stage("Stage 5 — Validation", validate.evaluate, pd_model, config)
    _run_stage("Stage 6 — Score scaling", scaling.build_scorecard, pd_model, config)
    _run_stage("Stage 8 — Benchmark", benchmark.run_benchmark, df, config)

    print("\nPhase 0 complete: pipeline runs end-to-end and reports stage status.")


if __name__ == "__main__":
    main()
