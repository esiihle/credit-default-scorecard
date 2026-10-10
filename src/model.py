"""
Stage 4 — PD model.  [Built in Phase 4, 10 Oct 2026]

Logistic regression on the WOE-transformed features:

.. math::
    \\ln\\!\\left(\\frac{p}{1-p}\\right) = \\beta_0 + \\sum_k \\beta_k\\,\\text{WOE}_k

where *p* is the probability of default.

Why this model, when a gradient-boosted tree would score better
---------------------------------------------------------------
Because the inputs are WOE, the model is **linear in the log-odds of default**.
That single property is what makes Phase 6 possible: a linear score can be
rescaled into a points table, and a points table is something a customer can be
shown, a credit committee can sign off, and a regulator can audit. Phase 8
benchmarks a gradient-boosted model precisely to measure what that
interpretability costs — and reporting that number honestly is the point, not
hiding it.

The sign check is the headline diagnostic
-----------------------------------------
Phase 3 fixed the convention: WOE = ln(%good / %bad), so **positive WOE = safer**.
Phase 1 defined **1 = default**. Therefore every coefficient must come out
**negative**: more good-ish characteristic, lower log-odds of default.

A positive coefficient is not a curiosity. It means either the feature's
relationship reverses once other variables are in the model (usually
multicollinearity), or a sign has been flipped somewhere upstream. A scorecard
with a flipped sign ranks applicants backwards and still produces entirely
plausible output — which is why this is asserted, not eyeballed.

What else is checked, and why a credit committee asks for each
-------------------------------------------------------------
* **Significance (p-values).** A variable that could plausibly have zero effect
  does not belong in a model that will be used for years.
* **VIF (multicollinearity).** Two features saying the same thing make
  coefficients unstable: they swap magnitudes, and sometimes signs, on a
  slightly different sample. WOE reduces this but does not remove it.
* **Stability.** Coefficients refitted on bootstrap resamples should not move
  much. One that does is fitted to this particular sample.

**Trained on train only.** The test half is untouched until Phase 5.

See docs/methodology.md section 4.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


# On the WOE scale a believable logistic coefficient is of order 0.5 to 2, so a
# standard error above this is a symptom, not a number. See the check in
# `fit_pd_model` for why it matters.
MAX_CREDIBLE_STD_ERROR = 1000.0


class ModelError(Exception):
    """Raised when the PD model cannot be fitted or fails a hard check."""


# ===========================================================================
# Diagnostics
# ===========================================================================

def variance_inflation(matrix):
    """VIF for every column: how much a coefficient's variance is inflated by
    the other predictors.

    VIF_k = 1 / (1 - R²_k), where R²_k is from regressing column k on all the
    others. Rules of thumb: under 5 is comfortable, 5-10 is worth a look, above
    10 means two features are telling the same story and the model cannot decide
    how to split the credit between them.

    Computed by hand rather than imported, because it is three lines and the
    formula is worth being able to write on a whiteboard.
    """
    columns = list(matrix.columns)
    results = {}

    for column in columns:
        others = [c for c in columns if c != column]
        if not others:
            results[column] = 1.0
            continue

        y = matrix[column].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(matrix)),
                             matrix[others].to_numpy(dtype=float)])

        # Least squares, then R² from the residual and total sums of squares.
        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ coefficients
        ss_residual = float(np.sum(residual ** 2))
        ss_total = float(np.sum((y - y.mean()) ** 2))

        r_squared = 0.0 if ss_total == 0 else 1.0 - ss_residual / ss_total
        results[column] = np.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)

    return pd.Series(results, name="vif")


def bootstrap_stability(X, y, n_resamples=50, seed=42):
    """Refit on bootstrap resamples and report how much each coefficient moves.

    A coefficient whose sign flips across resamples is not a finding about
    credit risk; it is a finding about this particular sample.

    Returns a frame with the mean, standard deviation, and the share of
    resamples in which each coefficient kept its sign.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    design = sm.add_constant(X, has_constant="add")
    draws = []

    for _ in range(n_resamples):
        rows = rng.integers(0, len(X), len(X))
        try:
            fit = sm.Logit(y.iloc[rows].to_numpy(),
                           design.iloc[rows].to_numpy()).fit(disp=0)
            draws.append(fit.params)
        except Exception:
            # A resample can be separable or singular. Skipping it is honest;
            # the count of usable resamples is reported alongside.
            continue

    if not draws:
        return pd.DataFrame(), 0

    frame = pd.DataFrame(draws, columns=design.columns)
    summary = pd.DataFrame({
        "mean": frame.mean(),
        "std": frame.std(),
        "sign_consistency": (np.sign(frame) == np.sign(frame.mean())).mean(),
    })
    return summary, len(draws)


# ===========================================================================
# Fitting
# ===========================================================================

def fit_pd_model(woe_features, target=None, config=None, enforce_signs=None):
    """Fit the logistic PD model on WOE features.

    Parameters
    ----------
    woe_features : pandas.DataFrame
        WOE-encoded TRAINING features — one numeric column per feature, as
        produced by `woe.apply_woe`.
    target : array-like
        1 = default. Required.
    config : dict | None
        Reads ``model.p_value_max``, ``model.vif_max``,
        ``model.bootstrap_resamples``, ``model.enforce_negative_signs``.
    enforce_signs : bool | None
        Overrides the config. When True, a positive coefficient raises instead of
        being reported — appropriate once the feature set has settled.

    Returns
    -------
    dict
        Coefficients with standard errors and p-values, VIF, stability, fit
        statistics, and the list of checks that failed.
    """
    import statsmodels.api as sm

    if target is None:
        raise ModelError("fit_pd_model needs the target (1 = default).")

    config = config or {}
    model_cfg = config.get("model", {})
    p_max = float(model_cfg.get("p_value_max", 0.05))
    vif_max = float(model_cfg.get("vif_max", 5.0))
    n_resamples = int(model_cfg.get("bootstrap_resamples", 50))
    min_epv = float(model_cfg.get("min_events_per_variable", 10.0))
    if enforce_signs is None:
        enforce_signs = bool(model_cfg.get("enforce_negative_signs", False))

    X = woe_features.copy()
    y = pd.Series(target).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if len(X) != len(y):
        raise ModelError(f"Features have {len(X)} rows but the target has {len(y)}.")
    if y.nunique() < 2:
        raise ModelError("The target has only one class; a PD model is undefined.")
    if X.shape[1] == 0:
        raise ModelError("No features to fit on — Phase 3 selected nothing.")

    # --- Is there enough data to fit this many parameters at all? --------
    # The binding constraint on a logistic regression is not the row count, it
    # is the number of EVENTS — here, bads. The long-standing rule of thumb in
    # credit risk and biostatistics is at least 10 events per variable (EPV);
    # below that, standard errors are unreliable and the fit may not be
    # identified at all.
    #
    # This guard exists because of a real failure on 10 Oct 2026: a run against
    # the bundled 24-row sample reached this point with 14 features and 5 bads
    # (EPV 0.36, 15 parameters from 16 rows). On one library version it raised
    # `LinAlgError: Singular matrix`; on another it quietly returned
    # coefficients of -21.9 with NaN standard errors and VIF of 299. Silent
    # nonsense is the worse of the two, so the model now refuses up front.
    n_events = int(min(y.sum(), len(y) - y.sum()))
    epv = n_events / X.shape[1] if X.shape[1] else 0.0

    if n_events < 2:
        raise ModelError(
            f"Only {n_events} event(s) in the minority class — a PD model "
            "cannot be fitted."
        )

    if epv < min_epv:
        raise ModelError(
            f"Not enough data for {X.shape[1]} features: {n_events} events "
            f"gives {epv:.2f} events per variable, below the required "
            f"{min_epv:g} (model.min_events_per_variable).\n"
            f"  {len(y)} training rows, {X.shape[1] + 1} parameters to estimate.\n"
            "  Most likely cause: the pipeline is running on the bundled sample "
            "rather than the real dataset.\n"
            "  Fix: download the real data to data/raw/, or reduce the feature "
            "count — do not lower this threshold to make the error go away."
        )

    design = sm.add_constant(X, has_constant="add")

    # Perfect separation and singular Hessians are the standard failure modes
    # here. Both are symptoms of too many parameters for the data, so they are
    # translated into an explanation rather than re-raised as linear algebra.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            fit = sm.Logit(y.to_numpy(), design.to_numpy()).fit(disp=0)
    except np.linalg.LinAlgError as error:
        raise ModelError(
            f"The logistic fit failed ({error}). The Hessian is singular, which "
            "means the parameters are not identified — usually too many "
            f"features ({X.shape[1]}) for the data ({len(y)} rows, "
            f"{n_events} events), or two features that are near-duplicates. "
            "Check the VIF table and reduce the feature set."
        ) from error
    except Exception as error:
        raise ModelError(
            f"The logistic fit failed: {error}. With {X.shape[1]} features and "
            f"{n_events} events this is most likely perfect separation — some "
            "combination of features predicts the outcome exactly, which a "
            "logistic regression cannot fit with finite coefficients."
        ) from error

    # Even when the optimiser returns, the fit can be degenerate. Two symptoms,
    # and it is worth knowing both because they look nothing alike:
    #
    #   * NaN standard errors — the Hessian could not be inverted at all.
    #   * ENORMOUS standard errors — the usual signature of perfect separation.
    #     A feature that splits the classes exactly has no finite best
    #     coefficient, so the optimiser walks off toward infinity and statsmodels
    #     reports, say, a coefficient of -13.4 with a standard error of 16,342.
    #     Nothing is NaN, nothing raises, and the coefficient is meaningless.
    #
    # On the WOE scale a real coefficient is of order 0.5 to 2, so a standard
    # error in the thousands is never a result.
    unidentified = (not np.all(np.isfinite(fit.bse))
                    or float(np.max(np.abs(fit.bse))) > MAX_CREDIBLE_STD_ERROR)
    if unidentified:
        raise ModelError(
            "The fit returned standard errors that are not credible "
            f"(max {float(np.nanmax(np.abs(fit.bse))):.3g}), so the parameters "
            "are not identified. This is the signature of perfect separation or "
            "a singular Hessian: some feature combination predicts the outcome "
            f"exactly. {len(y)} rows, {n_events} events, {X.shape[1]} features. "
            "Reduce the feature set, merge the offending bins, or use more data."
        )

    names = list(design.columns)
    coefficients = pd.DataFrame({
        "coefficient": fit.params,
        "std_error": fit.bse,
        "z": fit.tvalues,
        "p_value": fit.pvalues,
    }, index=names)

    # --- The sign check --------------------------------------------------
    # The intercept is exempt: it carries the base log-odds of the book, which
    # is legitimately positive or negative depending on the bad rate.
    slopes = coefficients.drop(index="const", errors="ignore")
    wrong_sign = slopes.index[slopes["coefficient"] > 0].tolist()

    if wrong_sign and enforce_signs:
        raise ModelError(
            f"Coefficients came out POSITIVE for {wrong_sign}. With WOE "
            "(positive = safer) and 1 = default, every slope must be negative. "
            "Either the relationship reverses once other variables are in the "
            "model (check VIF), or a sign has been flipped upstream."
        )

    # --- Significance and multicollinearity ------------------------------
    insignificant = slopes.index[slopes["p_value"] > p_max].tolist()
    vif = variance_inflation(X)
    collinear = vif.index[vif > vif_max].tolist()

    stability, usable_resamples = bootstrap_stability(X, y, n_resamples)
    unstable = []
    if len(stability):
        unstable = stability.drop(index="const", errors="ignore")
        unstable = unstable.index[unstable["sign_consistency"] < 0.95].tolist()

    coefficients["vif"] = vif.reindex(coefficients.index)
    if len(stability):
        coefficients["bootstrap_std"] = stability["std"].reindex(coefficients.index)
        coefficients["sign_consistency"] = (
            stability["sign_consistency"].reindex(coefficients.index))

    return {
        "features": list(X.columns),
        "coefficients": coefficients,
        "intercept": float(fit.params[0]),
        "fit_statistics": {
            "n_observations": int(len(y)),
            "n_features": int(X.shape[1]),
            "bad_rate": float(y.mean()),
            "events": n_events,
            "events_per_variable": float(epv),
            "log_likelihood": float(fit.llf),
            "null_log_likelihood": float(fit.llnull),
            # McFadden pseudo-R². Not comparable with an OLS R²; useful only
            # against other models on the same data.
            "pseudo_r2": float(fit.prsquared),
            "aic": float(fit.aic),
            "converged": bool(fit.mle_retvals.get("converged", True)),
        },
        "checks": {
            "wrong_sign": wrong_sign,
            "insignificant": insignificant,
            "collinear": collinear,
            "unstable_sign": unstable,
            "p_value_max": p_max,
            "vif_max": vif_max,
            "bootstrap_resamples_used": usable_resamples,
            "all_passed": not (wrong_sign or insignificant or collinear or unstable),
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def predict_pd(model, woe_features):
    """Probability of default for each row.

    Applies the fitted coefficients directly rather than keeping a statsmodels
    object alive, so a saved JSON model is enough to score an applicant. Phase 7
    needs exactly this: no pickles, no version pinning, no surprises.
    """
    coefficients = model["coefficients"]["coefficient"]
    intercept = float(coefficients.get("const", model.get("intercept", 0.0)))

    missing = [f for f in model["features"] if f not in woe_features.columns]
    if missing:
        raise ModelError(f"Cannot score: missing WOE columns {missing}.")

    log_odds = np.full(len(woe_features), intercept, dtype=float)
    for feature in model["features"]:
        log_odds += woe_features[feature].to_numpy(dtype=float) * float(
            coefficients[feature])

    return 1.0 / (1.0 + np.exp(-log_odds))


def drop_feature(model_spec, feature):
    """The names to refit with after removing one feature.

    Feature selection in a scorecard is iterative: fit, read the diagnostics,
    drop the worst offender, refit. This helper keeps that loop explicit rather
    than burying it in an automatic stepwise routine — which would hide exactly
    the judgement a credit committee wants to see.
    """
    return [f for f in model_spec["features"] if f != feature]


# ===========================================================================
# Reporting
# ===========================================================================

def model_report(model):
    """The coefficient table with diagnostics — the Phase 4 deliverable."""
    stats = model["fit_statistics"]
    checks = model["checks"]

    lines = ["Logistic PD model", "-" * 17]
    lines.append(f"  {'observations':<28} {stats['n_observations']:>9,}")
    lines.append(f"  {'features':<28} {stats['n_features']:>9}")
    lines.append(f"  {'bad rate':<28} {stats['bad_rate']:>9.1%}")
    if "events_per_variable" in stats:
        lines.append(f"  {'events (bads)':<28} {stats['events']:>9,}")
        lines.append(f"  {'events per variable':<28} "
                     f"{stats['events_per_variable']:>9.1f}"
                     f"   (10+ wanted)")
    lines.append(f"  {'log-likelihood':<28} {stats['log_likelihood']:>9.2f}")
    lines.append(f"  {'pseudo R2 (McFadden)':<28} {stats['pseudo_r2']:>9.4f}")
    lines.append(f"  {'AIC':<28} {stats['aic']:>9.1f}")
    lines.append(f"  {'converged':<28} {str(stats['converged']):>9}")

    lines.append("")
    lines.append("  Coefficients (all slopes must be NEGATIVE):")
    table = model["coefficients"].copy()
    display = [c for c in ["coefficient", "std_error", "p_value", "vif",
                           "sign_consistency"] if c in table.columns]
    lines.append("    " + table[display].to_string(
        float_format=lambda v: f"{v:>9.4f}").replace("\n", "\n    "))

    lines.append("")
    if checks["all_passed"]:
        lines.append("  All diagnostics passed.")
    else:
        if checks["wrong_sign"]:
            lines.append(f"  FAIL — positive coefficient (ranks backwards): "
                         f"{', '.join(checks['wrong_sign'])}")
        if checks["insignificant"]:
            lines.append(f"  WARNING — not significant at p < {checks['p_value_max']}: "
                         f"{', '.join(checks['insignificant'])}")
        if checks["collinear"]:
            lines.append(f"  WARNING — VIF above {checks['vif_max']} "
                         f"(features overlap): {', '.join(checks['collinear'])}")
        if checks["unstable_sign"]:
            lines.append(f"  WARNING — sign not stable across bootstrap resamples: "
                         f"{', '.join(checks['unstable_sign'])}")
        lines.append("  Next step: drop the worst offender, refit, re-read. "
                     "Do not automate this — the judgement is the deliverable.")

    return "\n".join(lines)


def save_model(model, config, filename=None):
    """Write the fitted model to data/processed/ as JSON.

    JSON, not a pickle: a scorecard has to be scoreable years later, and a
    pickle ties that to one library version. Coefficients plus the WOE tables
    are the whole model.
    """
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("model", {}).get(
        "model_filename", "pd_model.json"))

    serialisable = {k: v for k, v in model.items() if k != "coefficients"}
    serialisable["coefficients"] = (
        model["coefficients"].reset_index()
        .rename(columns={"index": "term"}).to_dict(orient="records"))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=_json_safe)
    return path


def load_model(path):
    """Read a fitted model back, with its coefficient table as a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    model["coefficients"] = (
        pd.DataFrame(model["coefficients"]).set_index("term"))
    return model


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)
