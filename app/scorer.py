"""
Interactive applicant scorer (CLI).

Scores an applicant against the frozen scorecard and explains the result in
plain language, including reason codes.

It reads only the FROZEN artefacts written by the pipeline — the binning spec,
the WOE spec and the points table — and never refits anything. That is the
point: scoring an applicant must not depend on having the training data, the
library versions, or the pipeline available. A scorecard that can only be used
by the person who built it is not a scorecard.

Four ways to run it, because different situations need different ones
--------------------------------------------------------------------
    python app/scorer.py                      # interactive prompts
    python app/scorer.py --examples           # three contrasting applicants
    python app/scorer.py --batch file.csv     # score a whole file
    python app/scorer.py --table              # print the points table

Add --cutoff 520 to any of them for an approve/decline decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import binning, scaling, scorecard as scorer


def explain_or_report_refusal(applicant, points_table, binning_spec, config,
                              reasons, cutoff):
    """Score and explain, turning a fair-lending refusal into a clear message.

    A `FairLendingError` is not a crash — it is the scorer declining to produce
    an explanation that could not lawfully be given. It deserves a readable
    message and a non-zero exit code, not a traceback.
    """
    try:
        result = scorer.score_applicant(applicant, points_table, binning_spec,
                                        config, reasons)
    except scorer.FairLendingError as error:
        print("  CANNOT EXPLAIN THIS DECISION — refused on fair-lending grounds:")
        for line in str(error).splitlines():
            print(f"    {line}")
        return False

    print(scorer.explain(result, cutoff))
    return True


# ---------------------------------------------------------------------------
# Loading the frozen artefacts
# ---------------------------------------------------------------------------

def load_artefacts(config):
    """Load the frozen binning spec and points table, or explain what's missing."""
    processed = REPO_ROOT / config["data"]["processed_dir"]

    needed = {
        "binning spec": processed / config.get("binning", {}).get(
            "spec_filename", "binning_spec.json"),
        "points table": processed / config.get("scaling", {}).get(
            "scorecard_filename", "scorecard.json"),
    }

    missing = [name for name, path in needed.items() if not path.exists()]
    if missing:
        sys.exit(
            f"Cannot score: the {' and '.join(missing)} have not been built yet.\n"
            f"Run the pipeline first:\n\n    python scripts/run_pipeline.py\n"
        )

    return (binning.load_spec(needed["binning spec"]),
            scaling.load_scorecard(needed["points table"]))


# ---------------------------------------------------------------------------
# The three ways of supplying an applicant
# ---------------------------------------------------------------------------

def prompt_for_applicant(config, binning_spec):
    """Ask for each attribute, showing the valid options as we go.

    Pressing Enter leaves an attribute blank, which is deliberate: a real
    application is often incomplete, and the scorecard handles that through the
    MISSING bin rather than refusing to score.
    """
    schema = config["schema"]
    applicant = {}

    print("\nEnter the applicant's details. Press Enter to leave any blank.\n")

    for column in schema.get("numeric_columns", []):
        definition = binning_spec["features"].get(column)
        hint = ""
        if definition and definition["type"] == "numeric":
            edges = [float(e) for e in definition["edges"]]
            finite = [e for e in edges if abs(e) != float("inf")]
            if finite:
                hint = f"  (bands at {', '.join(f'{e:g}' for e in finite)})"
        raw = input(f"  {column}{hint}: ").strip()
        if raw:
            try:
                applicant[column] = float(raw)
            except ValueError:
                print(f"    '{raw}' is not a number — leaving blank.")

    for column in schema.get("categorical_columns", []):
        definition = binning_spec["features"].get(column)
        options = sorted(definition["mapping"]) if definition else []
        if options:
            print(f"\n  {column} — options:")
            for index, option in enumerate(options, start=1):
                print(f"      {index:>2}. {option}")
            raw = input(f"  choose 1-{len(options)} (or type the value): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                applicant[column] = options[int(raw) - 1]
            elif raw:
                applicant[column] = raw
        else:
            raw = input(f"  {column}: ").strip()
            if raw:
                applicant[column] = raw

    return applicant


def example_applicants(binning_spec, config):
    """Three contrasting applicants built from the scorecard's own bins.

    Constructed rather than hard-coded, so they stay valid when the model is
    refitted and the bins move. A demo applicant with stale bin labels would
    score through the MISSING path and quietly show nothing useful.
    """
    schema = config["schema"]
    examples = {}

    for label, pick in (("Strongest profile", "best"),
                        ("Weakest profile", "worst"),
                        ("Partially completed application", "sparse")):
        applicant = {}

        for column in schema.get("numeric_columns", []):
            definition = binning_spec["features"].get(column)
            if not definition or definition["type"] != "numeric":
                continue
            edges = [float(e) for e in definition["edges"]]
            finite = [e for e in edges if abs(e) != float("inf")]
            if not finite:
                continue
            if pick == "best":
                applicant[column] = finite[0] * 0.5           # below the lowest band
            elif pick == "worst":
                applicant[column] = finite[-1] * 1.5          # above the highest
            # "sparse" leaves every numeric blank.

        for column in schema.get("categorical_columns", []):
            definition = binning_spec["features"].get(column)
            if not definition:
                continue
            options = sorted(definition["mapping"])
            if not options:
                continue
            if pick == "best":
                applicant[column] = options[0]
            elif pick == "worst":
                applicant[column] = options[-1]
            elif column in list(schema.get("categorical_columns", []))[:2]:
                applicant[column] = options[0]                # only the first two

        examples[label] = applicant

    return examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score an applicant against the frozen scorecard.")
    parser.add_argument("--examples", action="store_true",
                        help="Score three contrasting example applicants")
    parser.add_argument("--batch", metavar="CSV",
                        help="Score every row of a CSV of raw applicant data")
    parser.add_argument("--table", action="store_true",
                        help="Print the points table and exit")
    parser.add_argument("--cutoff", type=float,
                        help="Approve/decline against this cut-off score")
    parser.add_argument("--reasons", type=int, default=4,
                        help="How many reason codes to show (default 4)")
    args = parser.parse_args()

    config = load_config()
    binning_spec, points_table = load_artefacts(config)

    # --- Just show the table ------------------------------------------
    if args.table:
        print(scaling.scorecard_report(points_table))
        return

    print("=" * 66)
    print(" Credit Default Scorecard — applicant scorer")
    print(f" points table built {points_table.get('created_utc', 'unknown')}")
    print(f" score range {points_table['score_range']['minimum']}"
          f"-{points_table['score_range']['maximum']}")
    print("=" * 66)

    # --- Batch --------------------------------------------------------
    if args.batch:
        path = Path(args.batch)
        if not path.exists():
            sys.exit(f"File not found: {path}")

        applicants = pd.read_csv(path)
        try:
            scored = scorer.score_many(applicants, points_table, binning_spec,
                                       config, args.reasons)
        except scorer.FairLendingError as error:
            print("CANNOT SCORE THIS FILE — refused on fair-lending grounds:")
            for line in str(error).splitlines():
                print(f"  {line}")
            sys.exit(2)

        if args.cutoff is not None:
            scored["decision"] = [
                "APPROVE" if s >= args.cutoff else "DECLINE"
                for s in scored["score"]
            ]

        print(f"\nScored {len(scored)} applicants from {path.name}:\n")
        print(scored.to_string(float_format=lambda v: f"{v:.3f}"))

        out_path = path.with_name(path.stem + "_scored.csv")
        scored.to_csv(out_path, index=True)
        print(f"\nWritten to: {out_path}")
        return

    # --- Examples -----------------------------------------------------
    if args.examples:
        refused = False
        for label, applicant in example_applicants(binning_spec, config).items():
            print(f"\n{'-' * 66}")
            print(f" {label}")
            print(f"{'-' * 66}")
            if applicant:
                shown = {k: v for k, v in list(applicant.items())[:4]}
                print(f"  attributes supplied: {len(applicant)} "
                      f"(e.g. {shown})")
            else:
                print("  attributes supplied: none — everything blank")
            print()
            ok = explain_or_report_refusal(applicant, points_table,
                                           binning_spec, config,
                                           args.reasons, args.cutoff)
            if not ok:
                refused = True
        if refused:
            sys.exit(2)
        return

    # --- Interactive --------------------------------------------------
    applicant = prompt_for_applicant(config, binning_spec)
    print(f"\n{'-' * 66}")
    if not explain_or_report_refusal(applicant, points_table, binning_spec,
                                     config, args.reasons, args.cutoff):
        sys.exit(2)


if __name__ == "__main__":
    main()
