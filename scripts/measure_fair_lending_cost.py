#!/usr/bin/env python3
"""
Measure what removing the protected characteristic costs.

`personal_status_sex` encodes SEX and was removed from the model on
20 October 2026. The README and the methodology both state a cost for that
removal. This script is what makes that a MEASUREMENT rather than a claim: it
fits the whole pipeline twice — once with the characteristic, once without,
everything else held fixed — and writes the comparison to
`docs/charts/fair_lending.json`.

Why a script and not a note in a file
-------------------------------------
A number typed into a comment is a number nobody can check. On 22 October 2026
the figures recorded in this project's config, README and methodology were found
to be wrong by a factor of three: they had come from a run that was never made
on real data, and nothing in the repository could contradict them. Every figure
the documentation quotes now has a command that regenerates it.

The characteristic is put back ONLY inside this script's own copy of the config,
and only for the comparison run. `config.yaml` is never modified, so there is no
way to leave the model in the wrong state by forgetting to undo something.

Usage
-----
    python scripts/measure_fair_lending_cost.py
    python scripts/measure_fair_lending_cost.py --json-only   # no markdown block
"""
import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed
from src import load, binning, woe, model, validate


def fit_and_validate(frame, config):
    """Run load → bin → WOE → PD → validate and return the test metrics.

    The whole chain is rerun for each variant rather than reusing bins or WOE
    tables. Adding a characteristic changes which features are selected and can
    change the bins of the others, so sharing any stage between the two runs
    would understate the difference.
    """
    train, test, _, _ = load.run_phase1(frame, config, save=False)

    bin_spec = binning.fit_bins(train, config)
    binned_train = binning.apply_bins(train, bin_spec)
    binned_test = binning.apply_bins(test, bin_spec)

    order = {f: binning.spec_bin_order(d) for f, d in bin_spec["features"].items()}
    woe_spec = woe.compute_woe_iv(binned_train, train["is_default"], config,
                                  bin_order=order)
    selected = woe.selected_features(woe_spec)

    woe_train, _ = woe.apply_woe(binned_train, woe_spec)
    woe_test, _ = woe.apply_woe(binned_test, woe_spec)

    pd_model = model.fit_pd_model(woe_train, train["is_default"], config)
    result = validate.evaluate(
        model.predict_pd(pd_model, woe_train), train["is_default"],
        model.predict_pd(pd_model, woe_test), test["is_default"], config)

    # The characteristic's own Information Value, where it is present — the
    # standalone measure of how much it carried before anything else saw it.
    protected_iv = {}
    for name, definition in woe_spec["features"].items():
        protected_iv[name] = float(definition["iv"])

    return {
        "test_gini": float(result["discrimination"]["test"]["gini"]),
        "test_auc": float(result["discrimination"]["test"]["auc"]),
        "test_ks": float(result["discrimination"]["ks"]["ks"]),
        "train_gini": float(result["discrimination"]["train"]["gini"]),
        "gini_drop": float(result["discrimination"]["gini_drop"]),
        "psi": float(result["stability"]["psi"]),
        "features_selected": len(selected),
        "selected": sorted(selected),
        "information_values": protected_iv,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--json-only", action="store_true",
                        help="Write the JSON without printing the markdown block")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    protected = list((config.get("fair_lending", {}) or {})
                     .get("protected_characteristics", []))
    if not protected:
        print("config.fair_lending.protected_characteristics is empty — "
              "nothing to measure.")
        return 1

    raw_path = REPO_ROOT / config["data"]["raw_dir"] / config["data"]["raw_file"]
    if not raw_path.exists():
        print(f"No raw data at {raw_path}")
        print(f"  download {config['data']['download_url']}")
        print(f"       to {raw_path}")
        print("\nThis measurement is meaningless on the bundled 24-row sample,")
        print("so it is not attempted. Nothing was written.")
        return 1

    frame = load.load_german_raw(raw_path, config)

    print("\nMeasuring the cost of removing the protected characteristic")
    print(f"  characteristic(s): {', '.join(protected)}")
    print(f"  data: {raw_path.name}, {len(frame):,} rows\n")

    # --- Variant A: the model as it ships, without the characteristic ------
    print("  fitting WITHOUT the protected characteristic (the shipped model)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        without = fit_and_validate(frame, config)

    # --- Variant B: the same pipeline with it added back -------------------
    # A deep copy, so config.yaml on disk and the loaded config used above are
    # both untouched. The model can never be left in this state by accident.
    with_config = copy.deepcopy(config)
    categorical = list(with_config["schema"].get("categorical_columns", []))
    added = [c for c in protected if c not in categorical]
    with_config["schema"]["categorical_columns"] = categorical + added
    # The guard would refuse to produce a reason code citing it; this run never
    # produces reason codes, only metrics, so the guard is not in play.

    print(f"  fitting WITH {', '.join(added)} added back (comparison only)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_protected = fit_and_validate(frame, with_config)

    cost = {
        "protected_characteristics": protected,
        "with_protected": with_protected,
        "without_protected": without,
        "cost": {
            "test_gini": without["test_gini"] - with_protected["test_gini"],
            "test_auc": without["test_auc"] - with_protected["test_auc"],
            "test_ks": without["test_ks"] - with_protected["test_ks"],
        },
        "standalone_iv": {
            name: with_protected["information_values"].get(name)
            for name in protected
        },
        "rows": int(len(frame)),
        "note": ("Negative cost means the model WITHOUT the characteristic "
                 "scored lower. A difference smaller than the standard error "
                 "of the metric is not evidence of a difference."),
    }

    out_dir = Path(args.out) if args.out else REPO_ROOT / "docs" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fair_lending.json"
    path.write_text(json.dumps(cost, indent=2) + "\n")

    print(f"\n  wrote {path.relative_to(REPO_ROOT)}\n")

    if args.json_only:
        return 0

    # --- The markdown block, ready to paste into the README ----------------
    name = ", ".join(protected)
    iv = cost["standalone_iv"].get(protected[0])
    print("  " + "-" * 68)
    print("  Paste this into the README's fair-lending section:")
    print("  " + "-" * 68)
    print()
    print(f"| | With `{name}` | Without |")
    print("|---|---|---|")
    for label, key in (("Test Gini", "test_gini"), ("Test AUC", "test_auc"),
                       ("Test KS", "test_ks")):
        a = with_protected[key]
        b = without[key]
        print(f"| {label} | {a:.4f} | {b:.4f} |")
    print()
    delta = cost["cost"]["test_gini"]

    # If the characteristic never made it into the model, the two runs are the
    # same model and the identical figures are a result, not a bug. Say so —
    # otherwise an unexplained row of identical numbers reads as broken code.
    not_selected = [n for n in protected
                    if n not in with_protected["selected"]]
    if not_selected and abs(delta) < 1e-12:
        floor = config.get("woe", {}).get("iv_min")
        print(f"The two runs are identical because `{name}` was not selected: "
              f"its Information Value of "
              + (f"{iv:.4f} " if iv is not None else "")
              + (f"fell below the iv_min floor of {floor}. " if floor else ". ")
              + "Removing a characteristic the model had already discarded "
                "costs exactly nothing, which is worth stating rather than "
                "leaving as an unexplained row of equal numbers.")
    else:
        direction = "costs" if delta < 0 else "gains"
        print(f"Removing it {direction} {abs(delta):.4f} Gini on the held-out set"
              + (f", and its standalone IV was {iv:.4f}." if iv is not None else "."))
        print()
        print("A difference smaller than the metric's standard error is not "
              "evidence of a difference — quote it with that caveat.")
    print()
    print("  " + "-" * 68)
    print("  Then run: pytest tests/test_docs.py")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
