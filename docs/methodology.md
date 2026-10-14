# Methodology — Credit Default Scorecard

> **Living document.** This is the mathematical spine of the scorecard. It is
> scaffolded now and filled in as each phase lands (see `ROADMAP.md`). Each
> section carries a **Status** tag so the reader always knows what is settled and
> what is pending. Metrics are reported only from held-out validation.

---

## 1. Data & target definition
**Status: 🟡 skeleton — finalised in Phase 1.**

### 1.1 Source (settled 30 Sep 2026)

**UCI Statlog (German Credit Data)**, dataset id 144 — 1,000 applicants, 20
attributes, binary good/bad outcome. Downloaded as `german.data` from
`https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`.

The file is **space-delimited, has no header row**, and codes every categorical
value as `A11`, `A34`, `A143` and so on. The column order and the full code
table therefore live in `src/load.py` and are applied at load time, so every
later stage — binning, WOE, the points table, the reason codes an applicant
would see — reads in plain English. An unrecognised code raises rather than
passing through, because a stray code would silently become a junk bin in
Phase 2.

Lending Club was ruled out earlier for target leakage. Home Credit remains an
optional follow-up once this build is complete.

### 1.2 Known caveats of this dataset

- **The published code table is disputed.** Grömping (2019) showed that the
  widely used UCI code table mislabels several attribute levels, and released a
  corrected version ("South German Credit"). We use the standard UCI coding so
  the work is comparable with the published literature, and flag it here. The
  issue affects the *interpretation* of certain levels, not the mechanics of the
  scorecard.
- **Selection bias by construction.** Applicants in this data were all granted
  credit, so it describes accepted applicants only. Reject inference is the
  standard bank remedy and is out of scope here — but the limitation is stated
  rather than glossed over.
- **Vintage.** The loans date from the 1970s and are denominated in Deutsche
  Mark. The methodology transfers; the coefficients do not.
- **`personal_status_sex` encodes sex**, a protected attribute under fair-lending
  rules. It is loaded so the data stays faithful to source; whether it enters the
  model is decided explicitly in Phase 3 and recorded there. This is the kind of
  decision a credit-risk panel will ask about.

### 1.3 Target definition

- The **good/bad target** and its rationale are stated here once fixed in Phase 1.
  Raw coding is 1 = good, 2 = bad; the recode to 0/1 (1 = default = the modelled
  event) happens in Phase 1, not at load, so the raw file and the modelling
  decision stay separable.
- Records are split into train / test, stratified on the target.

### 1.4 Target definition (Phase 1, 2 Oct 2026)

**Bad (1) = default = the modelled event.** Source code 2, which the UCI
documentation defines as a customer with bad credit risk. **Good (0)** = source
code 1. The recode happens in `src/load.py::define_target`, after loading and
before the split, and it is the only place the direction is set.

Three limitations are inherited with that definition, stated rather than hidden:

1. **No delinquency depth or performance window.** The source gives no
   days-past-due and no observation period, so "bad" cannot be tightened to a
   Basel-style 90-days-past-due rule. We inherit the publisher's label.
2. **Accepted applicants only.** Every record was granted credit, so this is an
   accept-population model. Reject inference is the standard bank remedy and is
   out of scope; the consequence is that the model describes risk among people a
   lender already said yes to.
3. **Direction must stay consistent.** 1 is the event, so a higher predicted
   probability means a worse applicant. If that direction flips anywhere between
   here and the points table, the scorecard silently ranks backwards — which is
   why the recode is tested (`test_target_recode_direction`).

### 1.5 The frozen split (Phase 1)

- **70 / 30 train / test, stratified on the target, seeded** from
  `config.yaml` (`seed: 42`). Stratified because the bad rate is ~30%: an
  unstratified split can hand the test set a materially different bad rate, and
  every metric computed on it would then be measuring the split rather than the
  model.
- Written to `data/processed/train.csv` and `test.csv`, with
  `split_manifest.json` recording the seed, row counts, bad rate per half, and a
  **fingerprint** — a hash of the sorted applicant IDs in each half. A later run
  producing the same fingerprints proves the split is identical, which turns
  "reproducible from a seed" from a claim into a check.
- **No out-of-time slice is possible.** German Credit carries no application or
  performance date, so there is nothing to split time on. Stability is instead
  tested in Phase 5 with PSI across the random split. This is a real limitation
  of the dataset, not of the method: on a bank's own book, an out-of-time slice
  would be the first thing a credit committee asked for.
- `applicant_id` (APP_0001 …) is built from row position, because the source has
  no identifier. That is stable only because the file is a static published
  download that we never edit.

**Nothing downstream may re-split, re-sample, or read the test set** until Phase
5 validation. The split is frozen here on purpose.
- Notation: for a bin $b$ of a feature, let $g_b, n_b$ be the counts of goods and
  bads; $G, N$ the totals. Distributions $\%good_b = g_b / G$,
  $\%bad_b = n_b / N$.

## 2. Binning / coarse classing  *(built 5 Oct 2026)*

A scorecard never uses raw values. Each feature is cut into a few bins, and the
model learns one number per bin. That buys three things a credit committee cares
about more than a decimal of AUC:

1. **Explainability** — the points table reads as a short list of plain statements.
2. **Stability** — a bin holding 15% of the book moves slowly; a raw value moves
   with every applicant.
3. **Monotonicity** — "longer term, more risk" can be checked against business
   intuition. A bad rate that zig-zags is usually fitted noise and is very hard
   to defend in a model review.

### 2.1 Method

    fine-class (quantiles)  ->  merge until constraints hold  ->  freeze to JSON

Merging always joins **adjacent** bins, and always the pair with the smallest
bad-rate gap, so the splits that survive are the ones that genuinely separate
risk. Constraints are applied in a fixed order, from `config.yaml`:

| Order | Constraint | Setting | Why first/last |
|---|---|---|---|
| 1 | Minimum bin size | `binning.min_bin_fraction` (5%) | a tiny bin's bad rate is noise, so it must not be allowed to drive the monotonicity decision |
| 2 | Maximum bin count | `binning.max_bins` (5) | more bins means a longer, less stable points table |
| 3 | Monotonic trend | `binning.enforce_monotonic` | applied last, on bins already big enough to trust |

Categorical features are sorted by bad rate and merged the same way, so each
group holds levels of similar risk. Monotonicity is not enforced for them: a
category has no natural order, and ordering by risk is the ordering.

### 2.2 Fitted on train only

Bins are learned from the training half of the frozen split and then **applied**
to test, never re-fitted. Binning is supervised — it reads the target — so
fitting it on all the data would leak test information into the model through
the bin boundaries themselves. This is the subtlest leak in scorecard work,
because nothing downstream looks wrong.

The spec is frozen to `data/processed/binning_spec.json`: readable JSON, because
bin boundaries are exactly what a reviewer wants to read and sign off.

### 2.3 Two bins that always exist

- `MISSING` — a value that is absent. Not an error, not an imputation: a bin,
  which Phase 3 gives its own WOE.
- `UNSEEN` — a category that did not exist at fit time. A new product code in
  live data must not crash the scorer.

Numeric bin edges run from -inf to +inf for the same reason: an applicant
younger or older than anyone in training still scores.

### 2.4 Features that cannot be binned

If a feature collapses to a single bin under these constraints (too few distinct
values, or no separation), it carries no information. It is left in place and
dropped in Phase 3 by the IV threshold, rather than deleted here — one place
decides what enters the model.

## 3. WOE / IV & feature selection  *(built 9 Oct 2026)*

### 3.1 Weight of Evidence

For each bin *b* of a feature:

$$\text{WOE}_b = \ln\!\left(\frac{\%\text{good}_b}{\%\text{bad}_b}\right)$$

where $\%\text{good}_b$ is that bin's share of all good accounts and
$\%\text{bad}_b$ its share of all bads. In words: how much more, or less,
good-ish is this bin than the book as a whole.

**Sign convention, fixed once and never flipped: positive WOE = safer.** Because
Phase 1 defined 1 = default, the logistic coefficients in Phase 4 are therefore
expected to be **negative**. Phase 4 asserts that rather than assuming it — a
scorecard that ranks backwards still produces entirely plausible-looking output,
and this is the cheapest place to catch it.

Why WOE rather than one dummy column per level:

- one column per feature, so the model stays small and stable;
- monotone in risk by construction, so each coefficient has a single meaning;
- it maps straight onto the points table in Phase 6, which is what makes the
  final product explainable to a customer and a regulator.

### 3.2 Information Value

$$\text{IV} = \sum_b \left(\%\text{good}_b - \%\text{bad}_b\right)\,\text{WOE}_b$$

One number per feature. Conventional bands, with the cutoffs in `config.yaml`:

| IV | Reading | Action |
|---|---|---|
| < 0.02 | useless | drop (`woe.iv_min`) |
| 0.02 – 0.1 | weak | keep, expect little |
| 0.1 – 0.3 | medium | keep |
| 0.3 – 0.5 | strong | keep |
| > 0.5 | suspiciously strong | **investigate for leakage** (`woe.iv_leakage_flag`) |

### 3.3 The leakage check

A feature with IV above 0.5 is usually not a brilliant predictor. In a real bank
build it is usually a field that encodes the outcome — a collections flag, a
write-off date, a post-default status — and it will evaporate the moment the
model meets an applicant whose outcome is not yet known.

Such a feature is **flagged, never dropped automatically**. Deciding whether a
strong predictor is leakage is a judgement about where the field came from and
when it is populated, and a person has to make it. Automating that decision would
either silently discard a genuinely good feature or silently keep a fatal one.

### 3.4 Bins with no bads (or no goods)

The logarithm is undefined when a bin holds none of one class. A continuity
correction of 0.5 is added to both counts of that bin, and the bin is flagged as
`adjusted` in the table and in the run report. Phase 2's minimum bin size makes
this rare; when it does fire it is a signal that the bin should probably have
been merged, not a result to accept quietly.

### 3.5 Bins never seen at fit time

A bin present at scoring time but absent from training — a new product code in
live data — is given **WOE = 0**, and the count is reported.

The reasoning, which is worth being able to defend: WOE 0 means "this
characteristic tells us nothing about this applicant", so they receive the
population-average contribution for it. The alternatives are worse. Assigning the
worst bin's WOE punishes an applicant for a data gap that is the lender's fault.
Assigning the best bin's WOE is exploitable by anyone who discovers which values
fall outside the training set. The honest cost of 0 is that a genuinely risky
unseen level is scored as neutral, which is precisely why the count is surfaced
and belongs on a monitoring report.

### 3.6 Fitted on train only

WOE reads the target, so like binning it is fitted on the training half and
applied unchanged to test. The spec is frozen to `data/processed/woe_spec.json`,
including the full per-bin tables — a scorecard's WOE table is the artefact a
model validator reads line by line, and it has to survive outside the process
that produced it.

## 4. PD model  *(built 10 Oct 2026)*

Logistic regression on the WOE-transformed features:

$$\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{k} \beta_k \,\text{WOE}_k$$

where $p$ is the probability of default.

### 4.1 Why this model and not a better-scoring one

Because the inputs are WOE, the model is **linear in the log-odds of default**.
That one property is what makes §6 possible: a linear score rescales into a
points table, and a points table is something a customer can be shown, a credit
committee can sign off and a regulator can audit.

A gradient-boosted model would almost certainly separate better. Phase 8
benchmarks one for exactly that reason — to **measure** what the
interpretability costs rather than assert that it is worth it. Reporting that
number honestly is the deliverable; hiding it would defeat the point.

### 4.2 The sign check is the headline diagnostic

§3 fixed the convention — WOE = ln(%good/%bad), so positive WOE is safer — and
§1.4 defined 1 = default. Therefore **every slope must be negative**: a more
good-ish characteristic must lower the log-odds of default.

A positive coefficient means one of two things, and both matter:

1. the relationship reverses once other variables are in the model, which is
   almost always multicollinearity; or
2. a sign has been flipped somewhere upstream.

A scorecard with a flipped sign **ranks applicants backwards and still produces
entirely plausible output** — plausible PDs, a plausible score distribution, a
plausible points table. There is no symptom. That is why
`test_all_coefficients_are_negative` asserts it, and why
`test_flipped_woe_sign_is_caught` feeds the model deliberately negated WOE and
requires it to refuse.

The intercept is exempt: it carries the base log-odds of the book, which is
legitimately signed either way depending on the bad rate.

`model.enforce_negative_signs` is left **off** during feature selection so the
whole diagnostic picture can be read at once, and turned on once the feature set
has settled, at which point a positive slope stops the run.

### 4.3 Diagnostics, and what each answers

| Check | Setting | The question it answers |
|---|---|---|
| Coefficient sign | — | Does the model rank the right way round? |
| p-value | `model.p_value_max` (0.05) | Could this variable's true effect be zero? |
| VIF | `model.vif_max` (5.0) | Are two features telling the same story? |
| Bootstrap sign consistency | `model.bootstrap_resamples` (50) | Is this coefficient a fact about credit risk, or about this sample? |

**VIF** is $1/(1-R^2_k)$, where $R^2_k$ comes from regressing feature $k$ on all
the others. Under 5 is comfortable; above 10 means the model cannot decide how to
split the credit between two overlapping variables, so their coefficients swap
magnitudes — and sometimes signs — on a slightly different sample. WOE reduces
this but does not remove it. It is computed here by hand rather than imported,
because it is three lines and worth being able to write on a whiteboard.

**Bootstrap stability** refits on resamples and reports the share of fits in
which each coefficient keeps its sign. Below 95% is a warning: that variable is
being fitted to this particular 700 rows.

### 4.4 Selection is iterative, and deliberately not automated

The run reports every failing check and then says: drop the worst offender,
refit, re-read. `model.drop_feature` supports that loop explicitly.

A stepwise routine would do it faster and hide the thing a credit committee most
wants to see — *why* each variable is in the final model. On the first fit of the
real data, four features came back insignificant and two had unstable signs.
Working through that by hand, and recording the reasoning, is the Phase 4
deliverable; the coefficient table is just the artefact.

### 4.5 Scoring, and no pickles

`predict_pd` applies the saved coefficients directly rather than keeping a
statsmodels object alive, so the frozen JSON model plus the WOE tables are the
*whole* model. A scorecard has to be scoreable years later, and a pickle ties
that to one library version. Phase 7's applicant scorer depends on this.

### 4.5a Enough data to fit at all: events per variable

The binding constraint on a logistic regression is not the row count, it is the
number of **events** — here, bads. The long-standing rule of thumb in credit risk
and biostatistics is at least **10 events per variable (EPV)**; below that,
standard errors are unreliable and the parameters may not be identified at all.
`model.min_events_per_variable` enforces it and the figure is printed on every
fit.

This guard exists because of a real failure. A run against the bundled 24-row
sample reached Stage 4 with 14 features and 5 bads — EPV 0.36, fifteen
parameters from sixteen rows. On one library version it raised
`LinAlgError: Singular matrix`; on another it returned, without complaint,
coefficients of -21.9 with VIF of 299. **The silent version is the dangerous
one**, because a coefficient table that merely looks strange still looks like a
result.

Two degenerate outcomes are now refused explicitly:

| Symptom | What it means |
|---|---|
| EPV below the threshold | not enough events to identify the parameters; stop before fitting |
| Standard errors in the thousands | perfect separation — some feature combination predicts the outcome exactly, so no finite coefficient is best |

Perfect separation deserves its own note, because it does not raise and does not
produce NaNs: the optimiser simply walks toward infinity, and statsmodels reports
a coefficient of, say, -13.4 with a standard error of 16,342. On the WOE scale a
believable coefficient is of order 0.5 to 2, so the standard error is the tell.

Lowering the threshold to make the error go away is the wrong fix, and the error
message says so.

### 4.6 Trained on train only

The test half is untouched until §5. In-sample PD is reported in the run only as
a sanity check — a fitted logistic regression is calibrated in-sample almost by
construction, so a mean predicted PD equal to the observed bad rate says the link
and intercept are right and says **nothing whatever** about performance.

## 5. Validation  *(built 13 Oct 2026)*

**This is the first stage that touches the test set.** Everything before it —
binning, WOE, selection, the logistic fit — saw only the training half of the
frozen split. These numbers are therefore the first honest statement about the
model, and they get exactly one chance to be honest: every time a modelling
decision is changed in response to a test number, the test set becomes a little
more like training data.

### 5.1 Three questions, not one

A committee asks three separate things, and a model can pass any one while
failing the others:

| Question | Metrics | What failing it means |
|---|---|---|
| **Discrimination** — does it separate good from bad? | KS, Gini / AUC | the model has no signal |
| **Calibration** — are the PDs right in *level*? | reliability by decile, Hosmer-Lemeshow | the ranking is fine, the pricing is wrong |
| **Stability** — does it hold up out of sample? | PSI | it works on this sample and not the next |

Reporting AUC alone — the common shortcut — answers one question in three.

### 5.2 KS

$$\text{KS} = \max_s \big| F_{\text{bad}}(s) - F_{\text{good}}(s) \big|$$

At the best cut-off, how many more percentage points of bads can you reject than
of goods. The report prints that cut-off in business terms — *rejects X% of bads
and Y% of goods, which is Z% of the book* — because that is the sentence a credit
committee actually discusses.

Bands for an application scorecard: below 0.20 weak, 0.20–0.40 usual, above 0.50
unusually good, above 0.70 a reason to look for leakage rather than celebrate.

### 5.3 Gini and AUC

AUC is the probability a randomly chosen bad scores riskier than a randomly
chosen good; $\text{Gini} = 2\,\text{AUC} - 1$. Computed from average ranks via
the Mann-Whitney identity, which is exact and handles **ties** correctly. Ties
matter: once a scorecard rounds to whole points they are everywhere, and
ignoring them inflates AUC.

**The key limitation, worth stating in an interview:** AUC depends only on the
*order* of the scores, so it is invariant to any strictly monotonic transform.
Shift every prediction in log-odds space and AUC does not move by a single
decimal while every PD becomes badly too high.
`test_auc_is_blind_to_a_level_error_but_calibration_is_not` demonstrates exactly
that. It is the reason calibration is a separate section rather than a footnote.

### 5.4 Calibration

Predicted PD against realised default rate by decile of predicted risk — the
credit version of a reliability diagram. Each band's gap is judged against its
own **binomial standard error**, $\sqrt{p(1-p)/n}$, rather than a flat
tolerance, because a 30-account band is noisy and a 3,000-account band is not.

The **Hosmer-Lemeshow** chi-square summarises it:

$$\chi^2 = \sum_b \frac{(O_b - E_b)^2}{E_b\,(1 - E_b/n_b)}$$

Two things about it are routinely misread, so the report says both out loud:

* a **low** p-value means **poor** calibration — the opposite direction to most
  tests;
* the test is weak on small samples, so it is printed *beside* the band table,
  never instead of it.

### 5.5 PSI

$$\text{PSI} = \sum_b (\%A_b - \%E_b)\,\ln(\%A_b / \%E_b)$$

Bands are cut on the **development** sample and then applied to the validation
one. That ordering is the entire mechanism: re-cutting bands on the validation
sample would make any two distributions look identical, because each would hold
10% of itself in each of its own deciles.
`test_psi_bands_are_cut_on_the_expected_sample` pins it.

Reading: below 0.10 stable, 0.10–0.25 a shift worth monitoring, above 0.25 a
major shift that normally triggers a redevelopment discussion. Per-band
contributions are reported too, so a single drifting characteristic can be
identified rather than just a total.

### 5.6 Rank ordering

Bad rate across score bands, ordered lowest score to highest. It must move in one
direction. This is the check a committee does by eye before looking at any
statistic, and the one that sinks a model fastest: *"band 6 is riskier than band
7"* has no defensible business meaning, whatever the AUC says, so cut-off policy
cannot be built on it. The report names the bands where it reverses rather than
only pronouncing a verdict.

### 5.7 Train numbers are reported only as a comparison

A large **train-to-test Gini drop** is the signature of overfitting, and that
comparison is the only reason the training figures are computed here. The first
real run made the point immediately:

```
Gini (train)              0.5100
Gini (test)               0.1886
Gini drop train->test     0.3214   (limit 0.15)
```

The model learned the training sample, not the problem. The honest response is to
go back to the **training-set** work — fewer features, heavier binning, a
stricter IV floor — and *not* to adjust anything in response to the test number.

### 5.8 What is deliberately not done

No threshold is tuned, no feature dropped, no hyperparameter chosen on these
numbers. If the report is disappointing, the two honest responses are to revisit
the training work or to report the result as it stands. That distinction is the
difference between validation and a second training run, and the report ends with
that reminder printed, every time.

## 6. Score scaling
**Status: 🟡 skeleton — built in Phase 6.**

Model log-odds are mapped to a human-readable score with the standard
**PDO / factor / offset** transform:

$$\text{Score} = \text{Offset} + \text{Factor}\times \ln(\text{odds}),
\qquad \text{Factor} = \frac{\text{PDO}}{\ln 2}$$

where **PDO** is the points needed to double the odds. Distributing across the $n$
model variables gives the **points per bin**:

$$\text{Points}_k = -\left(\beta_k \,\text{WOE}_k + \frac{\beta_0}{n}\right)\times \text{Factor} + \frac{\text{Offset}}{n}$$

The full set of per-bin points **is the scorecard**.

## 7. Interactive scorer & reason codes
**Status: 🟡 skeleton — built in Phase 7.**

Given an applicant, the scorer returns their PD, their scaled score, and **reason
codes** — the bins contributing the largest negative points relative to a
reference (the adverse-action / explainability output a bank must provide). Built
to cover many applicant scenarios interactively, not a single demo row.

## 8. Benchmark & the trade-off
**Status: 🟡 skeleton — built in Phase 8.**

A gradient-boosted model (e.g. LightGBM) is trained on the same data and compared
to the scorecard on KS / Gini / AUC. The expected pattern — the GBM edges ahead on
raw discrimination while losing on interpretability, stability and auditability —
is quantified and discussed. This section is the core interview talking point:
*why a bank knowingly accepts a small accuracy cost for an explainable model.*

## References
- Siddiqi, N. (2017). *Intelligent Credit Scoring.* Wiley.
- Thomas, L., Crook, J. & Edelman, D. (2017). *Credit Scoring and Its
  Applications.* SIAM.
- Additional references added as the methodology is written up.
