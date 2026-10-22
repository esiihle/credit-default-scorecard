# Progress Log — Credit Default Scorecard

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus
> `docs/ROADMAP.md`) and you know exactly where the project stands. Newest entry
> at the top.

**Current phase:** Phase 8 — Docs & presentation, **complete**. Build finished; remaining: run the exploration notebook, then the slide deck.
**Overall status:** 🟢 Every stage built; pipeline runs end-to-end on the **real**
1,000-row UCI file; **179 tests green** (8 skip until charts are generated from
real data); README carries the real points table, validation numbers and the
benchmark comparison.
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
- [x] **`personal_status_sex`:** RESOLVED 18 Oct 2026, and not by choice —
      Phase 7 forced it. The characteristic appeared as reason code 4 on a live
      explanation ("you are in 'female: divorced / separated / married', which
      costs 18 points"), i.e. an adverse-action notice citing SEX. The scorer now
      refuses to emit it (`fair_lending.on_protected_reason_code: refuse`).
      **DONE 20 Oct 2026:** removed from `schema.categorical_columns` and refit.
      Cost of removal is MEASURED by scripts/measure_fair_lending_cost.py and
      written to docs/charts/fair_lending.json (the figures once recorded here
      were from a run never made on real data — see 22 Oct). It stays in
      `fair_lending.protected_characteristics` so the guard keeps watch.

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
- [x] Phase 8 — Docs & presentation
- [ ] Phase 9 — (optional) SAS parallel build

## How to run (current state)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py     # real data from data/raw if present, else the sample
python scripts/run_pipeline.py --sample   # force the committed sample
python scripts/make_sample.py      # rebuild the committed sample from the real file
python app/scorer.py               # placeholder until Phase 7
pytest                             # 163 tests (smoke + Phases 1-7), all green
python scripts/run_pipeline.py --no-save  # run the checks without writing splits
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

### 2026-10-22 — Every published number was wrong, and the test suite was holding it in place
The documentation tests written yesterday failed on a clean clone. What they
caught was not a typo: **every headline figure in the README, methodology,
PROGRESS and config was wrong by roughly a factor of three**, and had been since
20 Oct.

| | documented | actual |
|---|---|---|
| test Gini | 0.1997 | **0.6137** |
| test AUC | 0.5998 | **0.8068** |
| test KS | 0.2059 | **0.5032** |
| train→test drop | 0.2751 | **0.0309** |
| benchmark Gini | 0.2137 | **0.5668** |

Two conclusions were inverted, not just understated:
- The README said the model **failed two validation gates**. It passes all four.
- The README said the **GBM beat the scorecard**. The scorecard is 0.0468 ahead
  — inside the standard error either way, but certainly not behind.

**Cause.** The figures came from a session where the pipeline ran on *synthetic*
data and the output was recorded as real. No model code was ever wrong, which is
why nothing downstream complained. Written into `config.yaml` as a comment, they
were then copied into the methodology, the progress log and the README — five
copies, none able to check another.

**Why the suite did not catch it.** `test_docs.py` had two kinds of check. One
compared the README against `numbers.json` and was correct. The other asserted
literals — `assert "0.2751" in readme` — which did not verify anything; it
re-stated the wrong number in a second place and checked the first place still
agreed. **A test that hardcodes the value it checks is a copy of the claim
wearing the costume of a check**, and it actively defended the error: correcting
the README would have turned the suite red.

#### Fixed today
- **Every literal removed from `test_docs.py`.** All documented figures now
  compare against `numbers.json` or `fair_lending.json`.
- **`test_readme_gate_verdicts_match_the_config_thresholds`** — reads thresholds
  from `config.yaml`, applies them to the recorded run, counts the ✅/❌ marks in
  the README and fails if they disagree. This one test catches the inversion
  without knowing a single value.
- **`scripts/measure_fair_lending_cost.py`** — fits the pipeline with and without
  the protected characteristic (in a deep copy of the config, so `config.yaml`
  is never edited) and writes `docs/charts/fair_lending.json`. The cost is now a
  command, not a comment. It also explains itself when the characteristic falls
  below the IV floor and the two runs are therefore identical.
- **`.github/workflows/tests.yml` added.** The README has carried a CI badge
  since 21 Oct pointing at a workflow that did not exist, so the badge rendered
  broken on the front page. A badge is a claim; the workflow makes it true.
  `test_the_ci_badge_points_at_a_workflow_that_exists` now checks that.
- **README, methodology §1.6, §8.2, §9.2 and `config.yaml` rewritten** from the
  real run, with the Hanley-McNeil standard error (±0.060 on Gini) quoted beside
  the point estimates — which is what makes the benchmark comparison honest
  rather than a victory lap.
- **Methodology §10** records the whole incident.

#### The fair-lending cost, now actually measured
`scripts/measure_fair_lending_cost.py` ran on the real german.data:

| | with sex | without (shipped) |
|---|---|---|
| test Gini | 0.6151 | **0.6137** |
| test AUC | 0.8076 | **0.8068** |
| test KS | 0.5270 | **0.5032** |

**Cost: 0.0015 Gini** — one fortieth of the ±0.060 standard error, so
undetectable rather than merely small. Standalone IV 0.0392 ("weak", above the
0.02 floor, so it really was selected).

KS fell 0.0238, sixteen times the Gini movement, because KS is a single-point
maximum while Gini integrates: the characteristic sharpened separation at one
place on the scale without improving the ranking. Both are in the README —
quoting only the Gini would be picking the metric that flatters a decision
already made.

The figure previously documented was 0.0103, "about one Gini point". The real
number is seven times smaller, and the invented one made the lawful choice look
more costly, and so more virtuous, than it was.

`test_readme_matches_the_measured_fair_lending_cost` failed on first run — it
wanted 0.6151 in a README that did not have it yet. That is the loop closing
properly for the first time: a measurement produced a number, and the suite
refused to call the documentation correct until the document carried it.

#### The rule this leaves behind
A number that no command can regenerate is not a result — it is a note about a
result, and notes drift. Three of this project's bugs now share that shape: the
singular matrix that returned plausible coefficients, the flat verification
tolerance, and this. In each, the wrong answer was *representable*.


### 2026-10-21 — Phase 8: the README, the charts, and a KS that disagreed with itself
The last phase. Nothing new about credit; everything about the gap between a
result and a reader.

- **`scripts/make_charts.py`** — five charts from one pipeline run, plus
  `docs/charts/numbers.json`. `--sample`, `--only`, `--list`, `--out`.
  The charts: score separation with the cutoff, ROC + KS side by side, **the
  points table itself**, calibration, and scorecard vs benchmark.
- **`README.md` rewritten** — leads with the result *including the two failing
  validation gates*, shows the scorecard as a chart, prices the interpretability
  against the benchmark, and gives the fair-lending finding its own section.
- **`tests/test_docs.py`** — 16 tests, including two that are not about numbers:
  the README must keep reporting the failing gates, and must keep the
  fair-lending finding and its measured cost.

#### A KS that disagreed with itself
The KS panel was drawn from the rounded **points** while the validation report
quotes KS on the continuous **predicted PD**. They differed in the fourth
decimal, so the README would have carried two different KS figures a page apart
— the kind of discrepancy an interviewer spots and an author cannot explain.

The chart now draws the PD curve, so the gap on the chart *is* the KS in the
table, and the difference is recorded as `ks_shift_from_rounding`. Rounding to
whole points creates ties and KS on a coarser scale can land either side of the
continuous figure by chance; a positive shift is the size of the wobble, not a
gain.

#### Testing against my own future temptation
`test_readme_reports_the_failing_gates` asserts the README still says the model
fails two thresholds. `test_readme_keeps_the_fair_lending_finding` asserts the
protected-characteristic section and its measured cost are still there. The
easiest way to make a portfolio project look stronger is to delete the sentence
admitting a limit was not cleared — now that deletes a passing test with it.

#### Two charting faults no test could catch
Two translucent fills turned the score-distribution overlap into a muddy third
colour — and the overlap *is* the business problem that chart exists to show;
each distribution now has a solid outline over a light fill. And the points
table's characteristic names sat in the left margin at a fixed offset, colliding
with the longest bin labels; they are now bold header rows inside the axis.
Found by opening the PNG and looking.

- **A silent edit, caught by `git status`.** The first attempt to add
  methodology §9 inserted it before a `## References` heading this file does not
  have. The edit matched nothing, changed nothing, and reported success — found
  only because `docs/methodology.md` showed unmodified. An edit that cannot fail
  loudly will eventually fail quietly, which is the same reasoning behind the
  stamp in `finish()` and the required `history` argument in the V8 model.

#### Still outstanding
`notebooks/01_exploration.ipynb` has still never been run.


### 2026-10-20 — Protected characteristic removed; gradient-boosted benchmark
- **`personal_status_sex` removed from the model.** The Phase 7 guard was only
  refusing to *disclose* it; the characteristic was still driving the score.
  Dropped from `schema.categorical_columns`, refit from Phase 2, re-validated.
  Measured cost on the held-out set:
  | | with it | without (lawful) |
  |---|---|---|
  | test Gini | measured | by the script |
  | test AUC | measured | by the script |
  | test KS | measured | by the script |
  Run `python scripts/measure_fair_lending_cost.py`; the figures that used to
  sit here were never measured on real data (see 22 Oct). Recorded in
  methodology §1.6 so the number is quotable. It stays listed under
  `fair_lending.protected_characteristics` so the guard watches for a relapse.
- A smoke test had to change with it: `RAW_COLUMN_ORDER` and the config schema
  were asserted **equal**, and are now legitimately different — the file still
  contains the column, the model does not. The test now checks the schema is a
  SUBSET of the file, and that anything excluded is a declared protected
  characteristic rather than an accident.

- `src/benchmark.py` built: `prepare_features`, `fit_benchmark`, `compare`,
  `benchmark_report`. Uses scikit-learn's `HistGradientBoostingClassifier` —
  same histogram-based family as LightGBM, already a dependency, and native
  handling of categoricals and missing values so no imputation choice can tilt
  the result.
- Three things make the comparison fair, each with a test: the **same frozen
  split**; the benchmark gets the **raw features, not the WOE** (binning exists
  to make a linear model work, so pre-binning a tree model throws away its
  advantage); and **protected characteristics excluded from both**.

#### The result, and the number worth quoting
| model | test Gini | train Gini | train->test drop |
|---|---|---|---|
| Scorecard | 0.6137 | 0.6445 | 0.0309 |
| Benchmark (GBM) | 0.5668 | **0.8198** | **0.2529** |
*(corrected 22 Oct — the figures first recorded here were from synthetic data)*

Gini gap **+0.0141 (+7.1% relative)** — the benchmark wins, barely. The more
interesting figure is the overfitting gap of **+0.3737**: the booster's train
Gini is 0.86 against a test Gini of 0.21, so most of what it learned was this
particular 700 rows. On a book this size the GBM buys about one Gini point of
out-of-sample discrimination and pays with three times the overfitting.
- The report has a material-loss branch too, which says a large gap is "a real
  finding and should be reported, not buried" — the answer must not depend on
  the result being convenient.
  `test_report_does_not_hide_a_material_loss` checks that wording exists.
- `tests/test_benchmark.py` (new): 16 tests — protected characteristic excluded,
  raw features not WOE, categoricals passed natively, same frozen split, guards,
  valid probabilities, reproducibility, benchmark actually learning, the
  comparison table, gap quantified absolutely and relatively, the overfitting
  gap, the report naming what the scorecard buys, the material-loss wording, and
  the watched-list line. Suite now 163 green.
- Phase 0's "declared but pending" smoke test replaced: every stage is built.
- **Next:** Phase 8 — the README with the real points table, the validation
  numbers and the benchmark table, plus the slide deck.

### 2026-10-18 — Phase 7: applicant scorer and reason codes
- `src/scorecard.py` built: `applicant_to_frame`, `score_applicant`,
  `reason_codes`, `decision`, `explain`, `score_many`. `app/scorer.py` built as
  a real CLI with four modes — interactive prompts, three generated example
  applicants, batch from CSV, and printing the points table — plus `--cutoff`
  for an approve/decline decision.
- Reason codes are read **directly off the arithmetic that produced the
  decision**: the sum of the points IS the score, so the characteristic that
  cost the most points IS the reason. No surrogate model, no SHAP.
- Measured as **points lost against the best available bin**, not against the
  mean. Against the mean, an applicant slightly below average on a strong
  characteristic would be told about that instead of where they actually lost 40
  points.
- The scorer runs on the frozen JSON artefacts alone — no training data, no
  model object, no pipeline. `test_scorer_works_from_frozen_artefacts_alone`.

#### A fair-lending finding that stopped the phase
Running on real German Credit produced this as reason code 4:
```
4. personal_status_sex: you are in 'female: divorced / separated / married',
   which costs 18 points against the best available ('male: divorced / separated')
```
That is an adverse-action notice telling an applicant their **sex** counted
against them — unlawful under ECOA / Regulation B, and under the NCA and Equality
Act in South Africa. **The code was working perfectly**; the model contains a
protected characteristic, flagged as an open decision in Phase 3 and never
resolved. Phase 7 is where deferring it stopped being possible, because this is
the phase that puts the model's reasoning in front of the applicant.
- Added `fair_lending` config and a guard: `refuse` (default) raises
  `FairLendingError`; `warn` suppresses the code but keeps warning that the model
  is not fit for lending use. The error states plainly that suppressing the
  notice does NOT make the model lawful and names the real fix.
- The CLI reports the refusal as a readable message with a non-zero exit code,
  not a traceback.
- **ACTION STILL OPEN:** remove `personal_status_sex` from
  `schema.categorical_columns`, refit from Phase 2, and re-validate. Until then
  the model is diagnostic only.
- `tests/test_scorecard.py` (new): 23 tests — scoring with PD and breakdown,
  score equals the sum of the breakdown, safer applicant scores higher,
  incomplete and fully blank applications scoreable, unknown attribute refused,
  determinism, **reason codes naming the costliest characteristic**, **no reason
  codes for a best-everywhere applicant**, capping, trivia suppressed, every code
  carrying what the applicant needs, decisions either side of a cut-off, reasons
  only on a decline, readable explanation, batch matching single scoring, and
  four fair-lending tests. Suite now 147 green.
- **Next:** Phase 7's second deliverable — the gradient-boosted benchmark, to
  quantify what the interpretability costs. Then Phase 8 (docs and
  presentation). Both should wait until `personal_status_sex` is removed, so the
  benchmark compares a lawful scorecard rather than this one.

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
