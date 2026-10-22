#!/usr/bin/env python3
"""
Phase 8 — generate every chart the README reports, from one real pipeline run.

Running

    python scripts/make_charts.py

rebuilds every PNG in docs/charts/ *and* writes docs/charts/numbers.json, which
records the headline validation figures. The README quotes numbers from that
file and tests/test_docs.py checks the two agree. If a rerun moves a number, the
README is wrong and the test says so.

THE SAFETY RULE IN THIS FILE
----------------------------
Charts built from the 24-row bundled sample are meaningless, and a meaningless
chart that looks real is worse than no chart. When the real german.data is
absent, every chart is stamped ILLUSTRATIVE across its face and numbers.json
records `"real_data": false`. A test refuses to let those be committed.

Usage
-----
    python scripts/make_charts.py                     # real data from data/raw
    python scripts/make_charts.py --sample            # force the bundled sample
    python scripts/make_charts.py --only separation   # rebuild one chart
    python scripts/make_charts.py --list              # what it writes, and why
"""
import argparse
import json
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed
from src import (load, binning, woe, model, validate, scaling, benchmark)


# --- The chart manifest ----------------------------------------------------
# One source of truth. The README links these filenames and tests/test_docs.py
# reads this dict, so a chart cannot be renamed in one place and left stale in
# the other.
CHARTS = {
    "separation": "score-separation.png",
    "discrimination": "discrimination-roc-ks.png",
    "points": "points-table.png",
    "calibration": "calibration.png",
    "benchmark": "scorecard-vs-benchmark.png",
}

CHART_PURPOSE = {
    "separation": "score distributions of repaid vs defaulted accounts, with "
                  "the cutoff — does the card separate?",
    "discrimination": "ROC curve and KS plot, the two standard discrimination "
                      "views side by side",
    "points": "the scorecard itself — points awarded by characteristic and bin",
    "calibration": "predicted vs realised default rate by score band",
    "benchmark": "scorecard vs gradient boosting: the accuracy bought and the "
                 "overfitting paid for it",
}


# --- Visual grammar --------------------------------------------------------
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"        # the scorecard / repaid
ORANGE = "#eb6834"      # the benchmark / defaulted
BAD = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 9,
    "font.size": 10, "figure.dpi": 150,
})


def new_figure(figsize, ncols=1):
    """Every figure is created here, with constrained layout.

    Constrained layout keeps title, subtitle, legend and tick labels from
    colliding at any figure size. Hand-placed text in figure coordinates works
    at one size and silently breaks at the next.

    Do not call fig.autofmt_xdate() or fig.subplots_adjust() on these figures —
    both switch constrained layout off, and the first symptom is a subtitle
    printed through a title.
    """
    return plt.subplots(1, ncols, figsize=figsize, layout="constrained")


def show(path):
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def finish(fig, ax_or_axes, path, illustrative, title=None, subtitle=None):
    """Title, stamp and save. Every chart goes through here, so the stamp
    cannot be forgotten on one."""
    axes = np.ravel(ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray))
                    else [ax_or_axes])
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        axes[0].set_title("\n".join(textwrap.wrap(subtitle, 76)), fontsize=9,
                          color=INK_SOFT, fontweight="normal", pad=8)
    if illustrative:
        for ax in axes:
            ax.text(0.5, 0.5, "ILLUSTRATIVE\nbundled sample — not results",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=BAD, alpha=0.30,
                    rotation=18, zorder=50)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"    wrote {show(path)}")


# --- Chart 1: score separation ---------------------------------------------

def chart_separation(data, illustrative, out_dir):
    """The chart a credit manager actually looks at.

    Two score distributions. If the card works they are shifted apart, and the
    overlap is the business problem: every applicant in it is one the card
    cannot confidently call either way.
    """
    scores = data["test_scores"]
    target = data["test_target"]

    fig, ax = new_figure((7.2, 4.4))
    if scores is None or len(scores) == 0:
        ax.text(0.5, 0.5, "no scored accounts", transform=ax.transAxes,
                ha="center", color=INK_SOFT)
        finish(fig, ax, out_dir / CHARTS["separation"], illustrative,
               title="Score separation")
        return

    good = np.asarray(scores)[np.asarray(target) == 0]
    bad = np.asarray(scores)[np.asarray(target) == 1]
    edges = np.histogram_bin_edges(scores, bins=22)

    # Density, not counts: there are far more goods than bads, and raw counts
    # would hide the bad distribution entirely.
    #
    # Fill at low alpha PLUS a solid outline. Two translucent fills alone turn
    # the overlap into a third muddy colour that belongs to neither group; the
    # outline keeps each distribution's shape readable through the other.
    for values, colour, label in (
            (good, BLUE, f"repaid  (n={len(good)})"),
            (bad, ORANGE, f"defaulted  (n={len(bad)})")):
        ax.hist(values, bins=edges, density=True, color=colour, alpha=0.38,
                zorder=3)
        ax.hist(values, bins=edges, density=True, histtype="step",
                color=colour, linewidth=2.0, label=label, zorder=4)

    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)      # headroom for the cutoff label

    for values, colour in ((good, BLUE), (bad, ORANGE)):
        if len(values):
            ax.axvline(np.mean(values), color=colour, linewidth=1.5,
                       linestyle=":", zorder=5)

    cutoff = data["numbers"].get("illustrative_cutoff")
    if cutoff is not None:
        ax.axvline(cutoff, color=INK, linewidth=1.8, zorder=6)
        # Only the marker label sits in the plot; the policy numbers go in the
        # subtitle, where they cannot collide with the legend or run off the
        # left edge when the cutoff lands low in the range.
        ax.annotate(f"cutoff {cutoff:.0f}", xy=(cutoff, top * 1.10),
                    xytext=(7, 0), textcoords="offset points", fontsize=9,
                    color=INK, fontweight="bold", va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                              edgecolor="none"))

    ax.set_xlabel("score  (higher = safer)")
    ax.set_ylabel("density")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    finish(fig, ax, out_dir / CHARTS["separation"], illustrative,
           title="Does the scorecard separate good from bad?",
           subtitle=f"held-out test set · KS {data['numbers']['ks']:.4f} · "
                    f"at this illustrative cutoff "
                    f"{data['numbers']['approval_rate']:.0%} are approved and "
                    f"{data['numbers']['approved_bad_rate']:.1%} of those default")


# --- Chart 2: ROC and KS ---------------------------------------------------

def chart_discrimination(data, illustrative, out_dir):
    """ROC and KS, the two standard views, side by side.

    Two panels rather than one chart with two y-axes. They answer related
    questions on different scales and a shared axis would misrepresent both.
    """
    roc = data["roc"]
    numbers = data["numbers"]

    fig, axes = new_figure((9.0, 4.4), ncols=2)
    left, right = axes

    # --- ROC ---
    left.plot([0, 1], [0, 1], "--", color=INK_SOFT, linewidth=1.3,
              label="random", zorder=2)
    if roc is not None and len(roc):
        x = roc["fpr"].to_numpy()
        y = roc["tpr"].to_numpy()
        left.plot(x, y, color=BLUE, linewidth=2.0, label="scorecard", zorder=4)
        left.fill_between(x, x, y, color=BLUE, alpha=0.12, zorder=3)
    left.set_xlim(0, 1); left.set_ylim(0, 1)
    left.set_xlabel("false positive rate")
    left.set_ylabel("true positive rate")
    left.set_title(f"ROC — AUC {numbers['auc']:.4f} · Gini {numbers['gini']:.4f}",
                   fontsize=10, color=INK_SOFT, pad=8)
    left.grid(alpha=0.6); left.set_axisbelow(True)
    left.legend(loc="lower right")

    # --- KS ---
    # Drawn on the PREDICTED PD, not the rounded points, so the gap on the
    # chart is exactly the KS the validation report quotes. Rounding to whole
    # points creates ties, and ties move KS slightly: numbers.json records both
    # figures and §9 of the methodology reports the difference as the measured
    # cost of rounding.
    scores = -np.asarray(data["test_pd"], dtype=float)   # higher PD = worse
    target = np.asarray(data["test_target"], dtype=int)
    if len(scores):
        order = np.argsort(scores)
        sorted_target = target[order]
        cumulative_bad = np.cumsum(sorted_target) / max(sorted_target.sum(), 1)
        cumulative_good = (np.cumsum(1 - sorted_target)
                           / max((1 - sorted_target).sum(), 1))
        axis = np.arange(1, len(scores) + 1) / len(scores)

        right.plot(axis, cumulative_bad, color=ORANGE, linewidth=2.0,
                   label="cumulative defaulted", zorder=4)
        right.plot(axis, cumulative_good, color=BLUE, linewidth=2.0,
                   label="cumulative repaid", zorder=4)

        gap = np.abs(cumulative_bad - cumulative_good)
        at = int(np.argmax(gap))
        right.vlines(axis[at], cumulative_good[at], cumulative_bad[at],
                     color=INK, linewidth=2.0, zorder=5)
        right.annotate(f"KS {numbers['ks']:.4f}",
                       (axis[at], (cumulative_good[at] + cumulative_bad[at]) / 2),
                       xytext=(10, 0), textcoords="offset points",
                       fontsize=10, fontweight="bold", color=INK, va="center")

    right.set_xlim(0, 1); right.set_ylim(0, 1)
    right.set_xlabel("share of accounts, worst score first")
    right.set_ylabel("cumulative share of group")
    right.set_title("KS — the widest gap between the two curves",
                    fontsize=10, color=INK_SOFT, pad=8)
    right.grid(alpha=0.6); right.set_axisbelow(True)
    right.legend(loc="lower right")

    finish(fig, axes, out_dir / CHARTS["discrimination"], illustrative,
           title="Discrimination on the held-out test set")


# --- Chart 3: the points table ---------------------------------------------

def chart_points(data, illustrative, out_dir):
    """The scorecard itself.

    This is the artefact that makes the whole approach defensible: an applicant
    can be shown exactly which attribute cost them how many points, and the
    arithmetic is addition.
    """
    table = data["points_table"]

    if table is None or len(table) == 0:
        fig, ax = new_figure((7.2, 4.0))
        ax.text(0.5, 0.5, "no points table", transform=ax.transAxes,
                ha="center", color=INK_SOFT)
        finish(fig, ax, out_dir / CHARTS["points"], illustrative,
               title="The scorecard")
        return

    # One row per bin, grouped by characteristic, with the characteristic name
    # as its own bold header row. The name used to sit in the left margin at a
    # fixed offset, which collided with the longest bin labels — and the longest
    # labels are exactly the ones a reader most needs to read.
    def shorten(label, limit=52):
        label = str(label)
        return label if len(label) <= limit else label[:limit - 1].rstrip() + "…"

    ordered = table.sort_values(["feature", "points"])
    labels, values, headers = [], [], []
    for feature, group in ordered.groupby("feature", sort=True):
        headers.append(len(labels))
        labels.append(feature.replace("_", " ").upper())
        values.append(np.nan)                       # header row: no bar
        for _, row in group.iterrows():
            labels.append("   " + shorten(row["bin"]))
            values.append(row["points"])
        labels.append("")                           # blank row between groups
        values.append(np.nan)

    values = np.array(values, dtype=float)
    y = np.arange(len(values))
    height = max(5.0, 0.23 * len(values))
    fig, ax = plt.subplots(figsize=(9.6, height), layout="constrained")

    finite = ~np.isnan(values)
    ax.barh(y[finite], values[finite], height=0.66, color=BLUE, alpha=0.85,
            zorder=3)
    for index in y[finite]:
        ax.annotate(f"{values[index]:.0f}", (values[index], index),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()

    # Bold the header rows, so the grouping reads without a margin column.
    for index, tick in enumerate(ax.get_yticklabels()):
        if index in headers:
            tick.set_fontweight("bold")
            tick.set_fontsize(9)
            tick.set_color(INK)

    ax.set_xlabel("points awarded")
    ax.set_xlim(0, float(np.nanmax(values)) * 1.10)
    ax.grid(axis="x", alpha=0.6); ax.set_axisbelow(True)

    constants = data["numbers"]
    finish(fig, ax, out_dir / CHARTS["points"], illustrative,
           title="The scorecard: points by characteristic and bin",
           subtitle=f"PDO {constants['pdo']:.0f} · base score "
                    f"{constants['base_score']:.0f} at {constants['base_odds']:.0f}:1 "
                    f"· attainable range {constants['score_min']:.0f}"
                    f"–{constants['score_max']:.0f}")


# --- Chart 4: calibration --------------------------------------------------

def chart_calibration(data, illustrative, out_dir):
    """Predicted vs realised default rate. Ranking is not enough.

    A card can rank perfectly and still price wrongly. Calibration is what makes
    the predicted PD usable for provisioning and pricing rather than only for
    ordering applicants.
    """
    table = data["calibration_table"]

    fig, ax = new_figure((6.0, 5.4))
    ax.plot([0, 1], [0, 1], "--", color=INK_SOFT, linewidth=1.3,
            label="perfect calibration", zorder=2)

    if table is not None and len(table):
        predicted = table["predicted_pd"].to_numpy(dtype=float)
        realised = table["realised_pd"].to_numpy(dtype=float)
        counts = table["accounts"].to_numpy(dtype=float)
        sizes = 30 + 220 * (counts / counts.max())

        ax.scatter(predicted, realised, s=sizes, color=BLUE, alpha=0.85,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4,
                   label="score band (area = accounts)")
        ax.plot(predicted, realised, color=BLUE, linewidth=1.5, alpha=0.5,
                zorder=3)

        top = max(predicted.max(), realised.max()) * 1.15
        ax.set_xlim(0, top); ax.set_ylim(0, top)

    ax.set_xlabel("predicted default rate")
    ax.set_ylabel("realised default rate")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    hl = data["numbers"].get("hosmer_lemeshow_p")
    subtitle = (f"Hosmer-Lemeshow p = {hl:.3f} — "
                + ("calibration not rejected" if hl is not None and hl > 0.05
                   else "calibration rejected at 5%")) if hl is not None else None

    finish(fig, ax, out_dir / CHARTS["calibration"], illustrative,
           title="Calibration: are the predicted PDs the right size?",
           subtitle=subtitle)


# --- Chart 5: the benchmark ------------------------------------------------

def chart_benchmark(data, illustrative, out_dir):
    """The trade-off, in the only two numbers that matter for it.

    TWO PANELS, NOT TWO SERIES ON ONE AXIS. Test Gini and the train-to-test drop
    are different quantities answering different questions; sharing an axis
    invites the eye to compare heights that are not comparable.
    """
    numbers = data["numbers"]
    panels = [
        ("Test Gini", "discrimination on unseen accounts — higher is better",
         numbers.get("gini"), numbers.get("benchmark_test_gini")),
        ("Train → test Gini drop", "how much was memorised — LOWER is better",
         numbers.get("gini_drop"), numbers.get("benchmark_gini_drop")),
    ]

    fig, axes = new_figure((8.6, 4.6), ncols=2)
    for ax, (name, explanation, scorecard_value, benchmark_value) in zip(axes, panels):
        values = [scorecard_value if scorecard_value is not None else np.nan,
                  benchmark_value if benchmark_value is not None else np.nan]
        ax.bar([0, 1], values, width=0.6, color=[BLUE, ORANGE], zorder=3)

        for position, value in zip([0, 1], values):
            if np.isfinite(value):
                ax.annotate(f"{value:.4f}", (position, value), ha="center",
                            va="bottom", xytext=(0, 5),
                            textcoords="offset points", fontsize=11,
                            fontweight="bold", color=INK)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["scorecard", "benchmark"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{name}\n{explanation}", fontsize=9.5, color=INK_SOFT,
                     pad=8)
        ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.22)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles,
               ["scorecard — logistic on WOE, with a points table",
                "benchmark — gradient boosting, no reason codes"],
               loc="outside lower center", ncol=1, frameon=False, fontsize=9)

    finish(fig, axes, out_dir / CHARTS["benchmark"], illustrative,
           title="What interpretability costs")


BUILDERS = {
    "separation": chart_separation,
    "discrimination": chart_discrimination,
    "points": chart_points,
    "calibration": chart_calibration,
    "benchmark": chart_benchmark,
}


# --- Running the pipeline to get the numbers -------------------------------

def will_use_sample(config, forced):
    """Will this run fall back to the bundled sample? Answerable before any work."""
    if forced:
        return True
    raw = REPO_ROOT / config["data"]["raw_dir"] / config["data"]["raw_file"]
    return not raw.exists()


def refuse_illustrative_output(out_dir, allow_sample):
    """Stop sample-built charts from ever landing in the published folder.

    The ILLUSTRATIVE stamp was a warning, and a warning is something a person
    can commit anyway — on 21 Oct 2026 exactly that happened on the companion
    football project: stamped charts from a 12-match sample were committed and
    shown on the repository's front page, with two charts missing entirely.

    A warning that can be walked past is not a control. Writing into the
    published charts folder from sample data is now refused outright. The
    override exists (--allow-sample) because looking at a chart's LAYOUT on
    sample data is a legitimate thing to want; it has to be asked for by name.
    """
    published = (REPO_ROOT / "docs" / "charts").resolve()
    try:
        targets_published = out_dir.resolve() == published
    except OSError:
        targets_published = False

    if targets_published and not allow_sample:
        print("\n  REFUSED — these charts would be built from the bundled sample,")
        print("  and docs/charts/ is what the README publishes.")
        print()
        print("  Either:")
        print("    * put the real data in place and run this again, or")
        print("    * send the sample charts somewhere else, e.g.")
        print("        python scripts/make_charts.py --sample --out /tmp/preview")
        print("    * or, to overrule this deliberately:")
        print("        python scripts/make_charts.py --sample --allow-sample")
        print()
        return True
    return False


def gather(config, use_sample, verbose=False):
    """Run the stages the charts need and collect everything in one dict."""
    numbers = {}

    raw_path = (REPO_ROOT / config["data"]["raw_dir"] / config["data"]["raw_file"])
    if use_sample or not raw_path.exists():
        print("  loading the bundled sample")
        if not use_sample:
            print(f"    NO RAW DATA at {raw_path}")
            print(f"      download {config['data']['download_url']}")
            print(f"           to {raw_path}")
        frame = load.load_data(REPO_ROOT / config["data"]["sample_path"], config)
        illustrative = True
    else:
        print("  loading german.data")
        frame = load.load_german_raw(raw_path, config)
        illustrative = False

    train, test, _, _ = load.run_phase1(frame, config, save=False)
    numbers["n_train"], numbers["n_test"] = int(len(train)), int(len(test))

    print("  binning and WOE")
    bin_spec = binning.fit_bins(train, config)
    binned_train = binning.apply_bins(train, bin_spec)
    binned_test = binning.apply_bins(test, bin_spec)

    order = {f: binning.spec_bin_order(d) for f, d in bin_spec["features"].items()}
    woe_spec = woe.compute_woe_iv(binned_train, train["is_default"], config,
                                  bin_order=order)
    numbers["features_selected"] = len(woe.selected_features(woe_spec))

    woe_train, _ = woe.apply_woe(binned_train, woe_spec)
    woe_test, _ = woe.apply_woe(binned_test, woe_spec)

    print("  PD model")
    pd_model = model.fit_pd_model(woe_train, train["is_default"], config)
    train_pd = model.predict_pd(pd_model, woe_train)
    test_pd = model.predict_pd(pd_model, woe_test)

    print("  validation")
    result = validate.evaluate(train_pd, train["is_default"],
                               test_pd, test["is_default"], config)
    numbers["auc"] = result["discrimination"]["test"]["auc"]
    numbers["gini"] = result["discrimination"]["test"]["gini"]
    numbers["train_gini"] = result["discrimination"]["train"]["gini"]
    numbers["gini_drop"] = result["discrimination"]["gini_drop"]
    numbers["ks"] = result["discrimination"]["ks"]["ks"]
    numbers["psi"] = result["stability"]["psi"]
    numbers["psi_verdict"] = result["stability"]["verdict"]
    numbers["hosmer_lemeshow_p"] = result["calibration"]["hosmer_lemeshow"].get("p_value")
    numbers["test_bad_rate"] = result["test_bad_rate"]

    print("  score scaling")
    scorecard = scaling.build_points_table(pd_model, woe_spec, config)
    points_table = pd.DataFrame(scorecard["table"])
    numbers["pdo"] = scorecard["constants"]["pdo"]
    numbers["base_score"] = scorecard["constants"]["base_score"]
    numbers["base_odds"] = scorecard["constants"]["base_odds"]
    numbers["score_min"] = float(scorecard["score_range"]["minimum"])
    numbers["score_max"] = float(scorecard["score_range"]["maximum"])
    numbers["base_odds_implies_pd"] = float(
        scorecard["attainable_pd"]["base_odds_implies"])
    numbers["n_characteristics"] = int(scorecard["n_features"])

    # score_applicants returns (total, per-characteristic contributions).
    totals, _ = scaling.score_applicants(binned_test, scorecard)
    test_scores = np.asarray(totals, dtype=float)

    # KS on the ROUNDED points, beside the official KS on the continuous PD.
    # Rounding a scorecard to whole points is a real operational choice and it
    # has a real, small cost in discrimination. Measuring it beats assuming it.
    numbers["ks_on_rounded_score"] = float(validate.ks_statistic(
        test_scores, test["is_default"].to_numpy(), higher_is_riskier=False)["ks"])
    # Signed: rounded minus continuous. A positive value does NOT mean rounding
    # improved the model — coarsening a score creates ties, and KS on a coarser
    # scale can land either side of the continuous figure by chance. It is
    # reported so the size of the wobble is visible rather than assumed away.
    numbers["ks_shift_from_rounding"] = float(
        numbers["ks_on_rounded_score"] - numbers["ks"])

    # A cutoff to draw on the separation chart: the band nearest 70% approval,
    # chosen for illustration only — the real cutoff is a business decision.
    cutoffs = scaling.cutoff_table(test_scores, test["is_default"].to_numpy(),
                                   scorecard)
    numbers["illustrative_cutoff"] = None
    if len(cutoffs):
        pick = (cutoffs.assign(distance=(cutoffs["approval_rate"] - 0.70).abs())
                .nsmallest(1, "distance").iloc[0])
        numbers["illustrative_cutoff"] = float(pick["cutoff_score"])
        numbers["approval_rate"] = float(pick["approval_rate"])
        numbers["approved_bad_rate"] = float(pick["approved_bad_rate"])

    print("  benchmark")
    try:
        # fit_benchmark does its own feature preparation, including dropping
        # the protected characteristic. Passing the raw frames keeps that guard
        # in one place rather than duplicating it here.
        fitted = benchmark.fit_benchmark(train, test, config)
        comparison = benchmark.compare(train_pd, test_pd, fitted,
                                       train["is_default"], test["is_default"],
                                       config)
        numbers["benchmark_test_gini"] = comparison["benchmark_gini"]
        numbers["gini_gap"] = comparison["gini_gap"]
        numbers["relative_gap"] = comparison["relative_gap"]
        numbers["overfit_gap"] = comparison["overfit_gap"]
        row = comparison["table"]
        bench_row = row[row["model"].str.contains("Benchmark")].iloc[0]
        numbers["benchmark_gini_drop"] = float(bench_row["gini_drop_train_to_test"])
    except Exception as error:
        print(f"    benchmark unavailable: {error}")

    numbers["real_data"] = not illustrative
    numbers["protected_excluded"] = list(
        (config.get("fair_lending", {}) or {}).get("protected_characteristics", []))

    return {
        "numbers": numbers,
        "roc": result["roc"],
        "calibration_table": result["calibration"]["table"],
        "points_table": points_table,
        "test_scores": test_scores,
        "test_pd": np.asarray(test_pd, dtype=float),
        "test_target": test["is_default"].to_numpy(),
        "illustrative": illustrative,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--allow-sample", action="store_true",
                        help="Permit writing sample-built charts into "
                             "docs/charts (refused by default)")
    parser.add_argument("--sample", action="store_true",
                        help="Force the bundled sample (charts get stamped)")
    parser.add_argument("--only", choices=sorted(CHARTS), action="append",
                        help="Build only these charts (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="Show what would be written, and why, then exit")
    parser.add_argument("--out", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nCharts this script writes into docs/charts/:\n")
        for key, filename in CHARTS.items():
            print(f"  {filename:<30} {CHART_PURPOSE[key]}")
        print("\nPlus numbers.json — the headline validation figures, which"
              "\ntests/test_docs.py checks the README against.\n")
        return 0

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    out_dir = Path(args.out) if args.out else REPO_ROOT / "docs" / "charts"

    print("\nCredit scorecard — building the README charts\n")

    # Checked BEFORE any work: a run that cannot produce publishable charts
    # should say so in a second, not after fitting every stage.
    if will_use_sample(config, args.sample) and refuse_illustrative_output(
            out_dir, args.allow_sample):
        return 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = gather(config, use_sample=args.sample, verbose=args.verbose)

    print("\n  drawing")
    for key in (args.only or list(CHARTS)):
        BUILDERS[key](data, data["illustrative"], out_dir)

    numbers_path = out_dir / "numbers.json"
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers_path.write_text(json.dumps(data["numbers"], indent=2, default=str) + "\n")
    print(f"    wrote {show(numbers_path)}")

    print()
    if data["illustrative"]:
        print("  THESE CHARTS ARE STAMPED ILLUSTRATIVE. Do not commit them as")
        print("  results — download german.data and run again.")
    else:
        print("  Real data. Commit docs/charts/ and run pytest tests/test_docs.py")
        print("  to check the README's numbers against numbers.json.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
