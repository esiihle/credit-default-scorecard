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
    on_sample = False          # set in the fallback branch below
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
        on_sample = True
        print()
        print("  NOTE: the sample is 24 rows. The data stages below are real and")
        print("  worth reading, but the MODEL stages cannot run on it — there are")
        print("  not enough events to fit a logistic regression. Download the real")
        print("  file to exercise the whole pipeline.")

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

    # --- Phase 2: binning, fitted on TRAIN ONLY ------------------------------
    print("Stage 2 — Binning (coarse classing)")
    spec = binning.fit_bins(train, config)

    train_binned = binning.apply_bins(train, spec)
    test_binned = binning.apply_bins(test, spec)      # applied, never re-fitted

    trend = binning.check_monotonic(train_binned, train["is_default"], spec)
    rising = [f for f, d in trend.items() if d in ("rising", "falling")]
    print(f"  features binned: {len(spec['features'])}"
          f"  ({len(rising)} of {len(trend)} numeric features monotonic)")
    for feature, direction in trend.items():
        if direction not in ("rising", "falling"):
            print(f"  WARNING: {feature} is {direction} after binning")

    summary = binning.bin_summary(train_binned, train["is_default"], spec)
    print()
    print(summary.head(12).to_string(index=False,
                                     float_format=lambda v: f"{v:.3f}"))
    print(f"  ... {len(summary)} bins across all features")

    if not args.no_save:
        spec_path = binning.save_spec(spec, config)
        print(f"\n  binning spec written to: {spec_path}")

    # --- Phase 3: WOE / IV, fitted on TRAIN ONLY -----------------------------
    print()
    print("Stage 3 — WOE / Information Value")
    bin_order = {f: binning.spec_bin_order(d) for f, d in spec["features"].items()}
    woe_spec = woe.compute_woe_iv(train_binned, train["is_default"], config,
                                  bin_order=bin_order)
    print(woe.woe_report(woe_spec))

    chosen = woe.selected_features(woe_spec)
    train_woe, unseen_train = woe.apply_woe(train_binned, woe_spec, chosen)
    test_woe, unseen_test = woe.apply_woe(test_binned, woe_spec, chosen)

    if unseen_test:
        print(f"\n  bins in test with no fitted WOE (scored neutral): {unseen_test}")

    if not args.no_save:
        woe_path = woe.save_spec(woe_spec, config)
        print(f"\n  WOE spec written to: {woe_path}")

    # --- Phase 4: logistic PD model, trained on TRAIN ONLY -------------------
    print()
    print("Stage 4 — Logistic PD model")
    pd_model = None
    try:
        pd_model = model.fit_pd_model(train_woe, train["is_default"], config)
    except model.ModelError as error:
        # Not an unexpected failure: on the bundled sample there are far too few
        # events to identify the parameters, and refusing is the correct answer.
        # The run continues so the rest of the pipeline still reports its status.
        print("  CANNOT FIT — and that is the right outcome here:")
        for line in str(error).splitlines():
            print(f"    {line}")

    if pd_model is not None:
        print(model.model_report(pd_model))

        # In-sample PD, as a sanity check only. The real numbers come from
        # Phase 5 on the held-out half; test_woe exists here but is deliberately
        # not scored.
        train_pd = model.predict_pd(pd_model, train_woe)
        print()
        print(f"  mean predicted PD on train   {train_pd.mean():.1%}")
        print(f"  observed bad rate on train   {train['is_default'].mean():.1%}")
        print("  (a fitted logistic is calibrated in-sample by construction — this")
        print("   says nothing about performance. Phase 5 uses the held-out half.)")

        if not args.no_save:
            pd_path = model.save_model(pd_model, config)
            print(f"\n  PD model written to: {pd_path}")

    # --- Phase 5: validation on the HELD-OUT test set ------------------------
    # The first and only point at which test data is used. Nothing here may
    # feed back into a modelling decision.
    print()
    print("Stage 5 — Validation (held-out test set)")
    if pd_model is None:
        print("  SKIPPED — there is no fitted PD model to validate.")
    else:
        train_pd = model.predict_pd(pd_model, train_woe)
        test_pd = model.predict_pd(pd_model, test_woe)

        try:
            validation = validate.evaluate(
                train_pd, train["is_default"], test_pd, test["is_default"], config)
            print(validate.validation_report(validation))

            if not args.no_save:
                report_path = validate.save_report(validation, config)
                print(f"\n  validation report written to: {report_path}")
        except validate.ValidationError as error:
            print(f"  CANNOT VALIDATE: {error}")

    # --- Phase 6: score scaling into the points table ------------------------
    print()
    print("Stage 6 — Score scaling (points table)")
    if pd_model is None:
        print("  SKIPPED — there is no fitted PD model to scale.")
    else:
        scale_cfg = config.get("scaling", {})
        scorecard = scaling.build_points_table(
            pd_model, woe_spec, config,
            round_points=bool(scale_cfg.get("round_points", True)))

        # The table must reproduce the model it came from, or it is a
        # plausible-looking object that represents nothing.
        verification = scaling.verify_against_model(
            train_binned, train_woe, pd_model, scorecard,
            points_per_feature=float(scale_cfg.get(
                "verification_tolerance_per_feature", 0.5)))

        # Cut-off options are computed on the HELD-OUT half: a policy choice has
        # to be made on data the model has not seen.
        test_scores, _ = scaling.score_applicants(test_binned, scorecard)
        cutoffs = scaling.cutoff_table(test_scores, test["is_default"], scorecard)

        print(scaling.scorecard_report(scorecard, verification, cutoffs))

        if not args.no_save:
            card_path = scaling.save_scorecard(scorecard, config)
            print(f"\n  points table written to: {card_path}")

    # --- Downstream stages (progressively implemented) ---
    print()
    print("Downstream stages:")
    _run_stage("Stage 8 — Benchmark", benchmark.run_benchmark, df, config)

    print("\nPhase 0 complete: pipeline runs end-to-end and reports stage status.")


if __name__ == "__main__":
    main()
