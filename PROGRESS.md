# Progress Log — Credit Default Scorecard

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus
> `docs/ROADMAP.md`) and you know exactly where the project stands. Newest entry
> at the top.

**Current phase:** Phase 6 — Score scaling & points table (Nov 9 – Nov 15), started early
**Overall status:** 🟢 Pipeline runs end-to-end on the **real** 1,000-row UCI file;
tests green (7); exploration notebook written and run.
**Next action:** Run the exploration notebook and fill in its findings table
(carried over from 1 Oct), then review the Phase 1 data-quality report and the
split manifest.
**Schedule:** baselined from 28 Sep 2026; target presentation **27 Nov 2026**
(Python), 11 Dec 2026 with the optional SAS build.
**Cadence:** one commit per working day.

---

## Open decisions
- [ ] **SAS build (Phase 9):** in or out? Depends on SAS OnDemand access.
- [x] **Real data:** done 30 Sep 2026 — UCI `german.data` (1,000 rows) loads via
      `src/load.py`, with the column order and A-code table applied at load time.
- [x] **Good/bad definition:** done 2 Oct 2026 — 1 = default (source code 2),
      0 = good. Documented in methodology §1.4 with its three limitations
      (no delinquency depth, accepted-applicants only, direction consistency).
- [ ] **`personal_status_sex`:** encodes sex, a protected attribute. Decide in
      Phase 3 whether it enters the model, and record the reasoning — a panel
      will ask.

## Phase checklist
- [x] Documentation (OVERVIEW, README, ROADMAP, methodology, LICENSE, .gitignore)
- [x] Phase 0 — Foundations & EDA
- [ ] Phase 1 — Target definition & split
- [ ] Phase 2 — Binning / monotonic classing
- [ ] Phase 3 — WOE / IV & feature selection
- [ ] Phase 4 — Logistic PD model
- [ ] Phase 5 — Validation (KS/Gini/AUC/PSI)
- [ ] Phase 6 — Score scaling & points table
- [ ] Phase 7 — Interactive scorer + benchmark
- [ ] Phase 8 — Docs & presentation
- [ ] Phase 9 — (optional) SAS parallel build

## How to run (current state)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py     # real data from data/raw if present, else the sample
python scripts/run_pipeline.py --sample   # force the committed sample
python scripts/make_sample.py      # rebuild the committed sample from the real file
python app/scorer.py               # placeholder until Phase 7
pytest                             # 124 tests (smoke + Phases 1-6), all green
python scripts/run_pipeline.py --no-save  # run the checks without writing splits
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

### 2026-10-14 — Phase 6: score scaling and the points table
- `src/scaling.py` built: `scaling_constants`, `score_to_odds` / `odds_to_score`,
  `score_to_pd` / `pd_to_score`, `build_points_table`, `score_applicants`,
  `verify_against_model`, `rounding_tolerance`, `cutoff_table`,
  `scorecard_report`, `save_scorecard` / `load_scorecard`.
- **The points table is the scorecard.** Scoring is now a lookup and an
  addition — a clerk with the printed table and a calculator gets the same
  answer as the pipeline.
- Verified against the model: unrounded the table must agree to floating-point
  precision, because it is an algebraic rearrangement, not an approximation.
  First real run: largest disagreement 2.277 points, mean 0.792, across 700
  applicants and 11 characteristics.

#### A design error found by the first real run
The verification tolerance was a **fixed** 1.0 points, and it failed an
eleven-characteristic card at 2.277 — a card that was entirely correct. Rounding
error is bounded by half a point PER CHARACTERISTIC, so the admissible total
grows with the card: one number either fails good tables or passes broken ones.
The setting is now `verification_tolerance_per_feature: 0.5` and
`rounding_tolerance()` turns it into a total.
`test_tolerance_scales_with_the_number_of_characteristics` pins it.

#### A second finding: the reference point has to be reachable
With a two-characteristic test fixture the attainable range came out 460-552
while `base_score` was 600 — a score no applicant could reach. The subtlety:
**raising base_score does not fix it.** The offset moves with it, so the whole
range shifts by the same amount and the gap is invariant (measured at 600, 500
and 520 — identical gap). `base_odds` is the lever: 50:1 asserts a reference PD
of 1.96%, and if the model never gets that low the anchor is outside the
population. The report now states both PD ranges and names the fix.
- Unseen bins score the characteristic's MEAN points, not zero: zero is not
  neutral on this scale because the offset is baked into every cell.
- `cutoff_table` computed on the HELD-OUT half, reporting approval rate,
  approved bad rate, and BOTH sides of the trade-off (bads rejected and goods
  rejected) — a cut-off that rejects most bads by rejecting most of everyone is
  not a policy.
- `tests/test_scaling.py` (new): 22 tests — factor identity, base anchor,
  **PDO doubling the odds at four score levels**, score/PD round trip, higher
  score = lower PD, every bin covered, safer bins earning more points,
  **the table reproducing the model exactly when unrounded**, rounding error
  bounded, tampered table refused, tolerance scaling, base-score trap,
  base_odds as the real lever, lookup-and-addition, reason-code material,
  unseen bin handling, cut-off trade-off, JSON round trip. Suite now 124 green.
- **Next:** Phase 7 — the interactive applicant scorer with reason codes, built
  on the per-characteristic contributions this phase produces.

### 2026-10-13 — Phase 5: validation on the held-out set
- `src/validate.py` built: `ks_statistic`, `auc_and_gini`, `roc_curve_points`,
  `calibration_by_band`, `hosmer_lemeshow`, `population_stability_index`,
  `psi_verdict`, `rank_ordering`, `evaluate`, `validation_report`, `save_report`.
- **First stage to touch the test set.** The report ends with a printed reminder
  that nothing in it may be used to choose a feature, threshold or
  hyperparameter — doing so turns the test set into training data.
- **The first run immediately caught overfitting:** train Gini 0.51, test Gini
  0.19, a drop of 0.32 against a 0.15 limit. Rank ordering was not monotonic
  (reversals at bands 2, 6, 7) and Hosmer-Lemeshow rejected calibration. Exactly
  what this phase is for, on the first attempt.
- AUC computed from average ranks via Mann-Whitney, so **ties are handled** —
  they are everywhere once a scorecard rounds to points, and ignoring them
  inflates AUC.
- **A test-design bug worth recording.** To show that AUC is blind to a level
  error I first used `clip(pd * 2, 0, 0.999)` — but clipping ties every
  high-risk account at the ceiling, destroys real ranking information and moved
  AUC from 0.820 to 0.804. AUC is invariant to *strictly* monotonic transforms
  only; the test now uses a log-odds shift and AUC is unchanged to 1e-9.
- Calibration gaps judged against each band's own binomial standard error, not a
  flat tolerance. Hosmer-Lemeshow's direction (low p = poor calibration) is
  stated in the output because it is routinely misread.
- PSI bands cut on the DEVELOPMENT sample and applied to validation — re-cutting
  on validation would make any two distributions look identical.
  `test_psi_bands_are_cut_on_the_expected_sample` pins it.
- `tests/test_validate.py` (new): 25 tests — perfect/reversed/random AUC, ties,
  the Gini identity, KS by hand, KS on a useless model, calibration tracking
  reality, **AUC blind to level error while calibration is not**, Hosmer-Lemeshow
  both directions, PSI ~0 on same-distribution and >0.25 on shifted, PSI band
  ordering, empty-band survival, rank ordering monotonic and reversal detection,
  the full report, the Gini-drop flag, a non-discriminating model failing,
  single-class refusal, JSON round trip. Suite now 99 green.
- `config.yaml`: `validation` block — `score_bands`, `psi_bands`, `min_gini`,
  `max_psi`, `max_gini_drop`, `report_filename`. Noted in the file as acceptance
  thresholds, not tuning knobs.
- **Next:** Phase 6 — score scaling (PDO / factor / offset) into a points table.

### 2026-10-10 (later) — Phase 4 hardening: events-per-variable guard
- A real run against the bundled 24-row sample crashed Stage 4 with
  `LinAlgError: Singular matrix` — 14 features, 5 bads, 15 parameters from 16
  rows. Reproduced here and found to be **worse than a crash on other library
  versions**: it returned coefficients of -21.9 with NaN standard errors and VIF
  of 299, with no error at all. Silent nonsense beats a traceback for danger.
- `src/model.py`: added an **events-per-variable guard**
  (`model.min_events_per_variable`, default 10). EPV counts the minority class,
  not rows — a 10,000-row book with 8 bads is still unfittable, and a row count
  hides that completely. The error names the likely cause (running on the
  sample), the fix, and says explicitly not to lower the threshold.
- Also refused now: standard errors in the thousands. Perfect separation does
  not raise and does not produce NaNs — the optimiser walks toward infinity and
  statsmodels reports a coefficient of -13.4 with a standard error of 16,342.
  On the WOE scale a believable coefficient is of order 0.5 to 2, so the
  standard error is the tell. `LinAlgError` is now translated into an
  explanation rather than re-raised as linear algebra.
- EPV is printed on every successful fit, so a thin-but-passable model is
  visible rather than taken on trust.
- `scripts/run_pipeline.py`: Stage 1 now says plainly when it is on the sample
  and that model stages cannot run on it; Stage 4 catches `ModelError` and
  continues, so a fresh clone still completes the whole run.
- `tests/test_model.py`: 5 regression tests — EPV refused, EPV computed from the
  minority class, EPV reported on success, single event refused, perfect
  separation refused. Suite now 74 green.

### 2026-10-10 — Phase 4: logistic PD model with diagnostics
- `src/model.py` built: `fit_pd_model`, `predict_pd`, `variance_inflation`,
  `bootstrap_stability`, `drop_feature`, `model_report`, `save_model` /
  `load_model`. statsmodels added to requirements (sklearn fits the model but
  gives no coefficient p-values, which a credit committee asks for first).
- **The sign check is the headline.** WOE positive = safer and 1 = default, so
  every slope must be negative. A flipped sign ranks applicants backwards while
  producing entirely plausible PDs, a plausible score distribution and a
  plausible points table — there is no symptom. Asserted, not eyeballed, and
  `test_flipped_woe_sign_is_caught` feeds the model negated WOE and requires it
  to refuse.
- Diagnostics: p-values, VIF (computed by hand — it is three lines and worth
  knowing), and bootstrap sign consistency across 50 resamples.
- **Selection is iterative and deliberately not automated.** The report names
  every failing check and says to drop the worst offender and refit. A stepwise
  routine would hide the reasoning, which is the part a committee wants.
- `predict_pd` applies saved coefficients directly — no pickles, so the JSON
  model plus the WOE tables are the whole model and remain scoreable years later.
  Phase 7's applicant scorer depends on this.
- First fit on realistic data: all 11 slopes negative, VIF ~1.0 throughout,
  pseudo R2 0.153, and four features flagged insignificant with two showing
  unstable signs — a real selection situation to work through.
- `tests/test_model.py` (new): 19 tests — all-coefficients-negative, flipped
  sign caught, wrong sign reported when enforcement is off, predictive features
  significant, noise flagged, VIF = 1 for independent columns, VIF explodes for
  a near-duplicate, collinear features flagged, bootstrap consistency, PD is a
  valid probability and matches the bad rate in-sample, safer applicant scores
  lower, missing columns rejected, target/class/feature/length guards, JSON
  round trip, drop_feature, report renders. Suite now 69 green.
- `config.yaml`: `model` block — `p_value_max`, `vif_max`,
  `bootstrap_resamples`, `enforce_negative_signs`, `model_filename`.
- `scripts/run_pipeline.py`: Stage 4 fits, prints the coefficient table with
  diagnostics, and writes `data/processed/pd_model.json`.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run.
- **Next:** Phase 5 — validation on the held-out half (KS, Gini/AUC, PSI). This
  is the first time test data is touched.

### 2026-10-09 — Phase 3: WOE, Information Value and feature selection
- `src/woe.py` built: `woe_table`, `information_value`, `iv_band`,
  `compute_woe_iv`, `selected_features`, `apply_woe`, `iv_ranking`,
  `woe_report`, `save_spec` / `load_spec`.
- **Sign convention fixed and documented:** WOE = ln(%good / %bad), positive =
  safer. With 1 = default, Phase 4's logistic coefficients must therefore come
  out negative — Phase 4 will assert that rather than assume it.
- **Leakage policy:** IV >= 0.5 is flagged for a human, never auto-dropped.
  Deciding a strong predictor is leakage is a judgement about where the field
  came from, not something to automate.
- **Unseen bins score WOE 0** (population-average contribution), with the count
  reported — the answer to the open question from Phase 2. Worst-bin WOE punishes
  an applicant for the lender's data gap; best-bin WOE is exploitable.
- Zero-cell bins get a 0.5 continuity correction and are flagged as `adjusted`.
- Fitted on train only; spec frozen to `data/processed/woe_spec.json` with the
  full per-bin tables, because that table is what a model validator reads.
- `tests/test_woe.py` (new): 17 tests, including **two hand-computed anchors** —
  a WOE worked out on paper (ln 3 and ln 0.5) and an IV (0.7167) — plus sign
  convention, WOE ordering mirrors bad-rate ordering, noise dropped, leakage
  flagged not dropped, IV bands, zero-cell finiteness, single-bin feature,
  target required, unseen bin neutral and counted, no spec change from test data,
  JSON round trip, and an end-to-end check against the Phase 2 binner.
- Phase 0 smoke test updated; suite now 50 green.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run. Its
  hand-computed IV for `checking_status` is the number this stage should
  reproduce — worth doing before Phase 4.
- **Next:** Phase 4 — logistic PD model on the WOE matrix.

### 2026-10-05 — Phase 2: binning and monotonic coarse classing
- `src/binning.py` built: quantile fine-classing, then adjacent-pair merging
  until the config constraints hold — minimum bin size, then maximum bin count,
  then monotonic trend, in that order and for that reason.
  - `fit_numeric_bins`, `fit_categorical_bins`, `fit_bins` (whole spec),
    `apply_bins`, `bin_summary`, `check_monotonic`, `save_spec` / `load_spec`.
  - `MISSING` and `UNSEEN` bins always exist; numeric edges run -inf to +inf, so
    an unseen category or an out-of-range value never crashes the scorer.
  - Spec frozen to `data/processed/binning_spec.json` — readable JSON, because
    bin boundaries are what a reviewer signs off.
- **Fitted on train only**, applied to test. Binning reads the target, so fitting
  it on all the data leaks test information through the bin boundaries — the
  subtlest leak in scorecard work.
- `scripts/run_pipeline.py`: Stage 2 runs, prints bins in natural order with
  share and bad rate, and warns about any numeric feature that did not come out
  monotonic.
- `config.yaml`: `binning.fine_bins` and `binning.spec_filename` added.
- `tests/test_binning.py` (new): 13 tests — max bins, min share, monotonicity
  enforced against a deliberate blip, open outer edges, few-distinct-values
  features, thin categorical levels merged, UNSEEN, MISSING, determinism, JSON
  round trip, target required, spec unchanged by test data, natural bin order.
- Phase 0 smoke test updated: binning no longer belongs in the "not built yet"
  list. Suite now 33 green.
- **Next:** Phase 3 — WOE / IV and feature selection. The hand-computed IV from
  the exploration notebook is the fixture it must reproduce.

### 2026-10-02 — Phase 1: target definition, validation, frozen split
- `src/load.py`: Phase 1 flow — `validate_raw` → `clean` → `define_target` →
  `to_analysis_table` → `split_train_test` → `save_splits`, orchestrated by
  `run_phase1`.
  - Structural faults raise `DataQualityError`: missing column, a target code
    that isn't 1 or 2, or a target with only one class.
  - Cleaning: numeric coercion, plausibility ranges for age / term / amount
    (all in `config.yaml`), exact duplicate rows dropped, every count reported.
  - `define_target`: **1 = default**, with the definition and its three
    limitations written into the docstring and methodology §1.4.
  - `applicant_id` added (APP_0001 …), since the source has no identifier.
  - 70/30 stratified split from the seed, written to `data/processed/` with
    `split_manifest.json` — seed, counts, bad rate per half, and a **fingerprint**
    hash of each half's IDs, so reproducibility is checkable, not asserted.
  - Guard rail: dropping more than `validation.max_dropped_fraction` (5%) of rows
    stops the run.
- `config.yaml`: new `validation` block; `split` block extended with
  `out_of_time: false` and the reason (no date column in this dataset).
- `scripts/run_pipeline.py`: runs Phase 1 and prints the quality report plus the
  frozen-split summary (`--no-save` to skip writing).
- `tests/test_load.py` (new): 13 behaviour tests — missing column, bad target
  code, single-class target, implausible age, duplicate rows, **recode
  direction**, split sizes, stratification, no train/test overlap, same-seed
  reproducibility, different-seed difference, end to end, and the drop-fraction
  guard. Suite now 20 green.
- **Not yet done:** the exploration notebook from 1 Oct still needs running and
  its findings table filling in.
- **Next:** Phase 2 — binning and monotonic coarse classing.

### 2026-10-01 — Exploration notebook
- Added `notebooks/01_exploration.ipynb`, importing from `src/` rather than
  duplicating loader logic; runs on the real file when present, the sample
  otherwise.
- Sections, each tied to the phase it justifies: target balance and good:bad
  odds (metric choice, Phase 5; reference odds, Phase 6); bad rate by level for
  every categorical (the raw material of WOE, Phase 3); **WOE and IV worked by
  hand for `checking_status`** as a fixture for Phase 3's code to reproduce; bad
  rate by quantile bin for every numeric, flagging non-monotonic features
  (Phase 2); data quality — duplicates, nulls, levels under 5% of book
  (`binning.min_bin_fraction`); bad rate by `personal_status_sex` with the
  fair-lending questions to answer in Phase 3.
- Added `ipykernel` to requirements.txt.

#### Findings from the real data — FILL IN after running the notebook
| Measurement | Value | Note |
|---|---|---|
| Overall bad rate | | expect ~30% on German Credit |
| Good:bad odds | | compare with `scaling.base_odds` (50) in config |
| Top 3 categorical features by bad-rate spread | | |
| Hand-computed IV, `checking_status` | | Phase 3 must reproduce this exactly |
| Numeric features NOT monotonic | | these are the Phase 2 to-do list |
| Thin levels (<5% of book) | | each needs merging in Phase 2 |

- **Next:** Phase 1 (opens 5 Oct) — target recode + stratified split.

### 2026-09-30 — Real data loading + fresh-clone fix + schedule re-baseline
- **Bug found and fixed:** `.gitignore` had `data/*`, which silently excluded the
  bundled sample from the repo. A fresh clone crashed on `run_pipeline.py` and
  2 of 4 tests failed. Now only `data/raw/` and `data/processed/` are ignored;
  `data/sample/` is committed on purpose.
- Removed the leftover dotless `gitignore` file (it was never doing anything).
- Settled the data source: UCI Statlog German Credit (id 144), `german.data`.
- `src/load.py`: added `RAW_COLUMN_ORDER` (the file has no header),
  `VALUE_LABELS` (the full A-code table), `decode_values()` — which raises on an
  unknown code instead of letting it become a junk bin — and
  `load_german_raw()`. The target is left raw (1/2); recoding belongs to Phase 1.
- `scripts/make_sample.py`: rebuilds the committed 24-row sample from the real
  file, stratified on the target with a fixed seed, so the sample is real data
  and anyone can regenerate it.
- `scripts/run_pipeline.py`: uses the real file when present, else the sample,
  printing the download link (`--sample` forces the sample).
- `docs/ROADMAP.md` re-baselined from 28 Sep; Python presentation now **27 Nov
  2026**. `docs/methodology.md` §1 filled in: source, the disputed code table
  (Grömping 2019), selection bias, vintage, and the protected-attribute flag.
- Tests: 7 green (added column-order/config drift, code decoding, unknown-code).
- **Next:** Phase 1 — target recode + stratified split.

### 2026-09-28 — Phase 0 scaffolding
- Built the runnable skeleton: `config.py` (+ `config.yaml`), implemented
  `load.py` (loader + EDA report), and stubs for every downstream stage.
- `scripts/run_pipeline.py` runs end-to-end and reports each stage as
  done/pending — it doubles as a live progress indicator.
- Added `app/scorer.py` (friendly placeholder), `tests/test_smoke.py`
  (4 tests, green), and a bundled 24-row German Credit sample.
- Verified: pipeline runs clean on a fresh checkout; tests pass.
- **Next:** Phase 1 — target recode + stratified split.

### 2026-09-01 — Project inception
- Created repo and committed all documentation (overview, README, roadmap,
  methodology skeleton, LICENSE, .gitignore, PROGRESS).

<!--
Template for each new entry (copy this, newest at the top):

### YYYY-MM-DD — <short title>
- What I did this session:
- Decisions made:
- Anything broken / to revisit:
- Next action:
-->
