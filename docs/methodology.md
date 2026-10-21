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
  rules. It is still *loaded*, so the data stays faithful to source, but it was
  **removed from the model on 20 Oct 2026** — see §7.3 for how Phase 7 forced the
  decision, and §1.6 for the measured cost.

### 1.6 Cost of removing the protected characteristic  *(20 Oct 2026)*

`personal_status_sex` was dropped from `schema.categorical_columns`. Measured on
the held-out set, with everything else unchanged:

| | with it | without it (lawful) |
|---|---|---|
| test Gini | 0.2100 | **0.1997** |
| test AUC | 0.6050 | 0.5998 |
| test KS | 0.2255 | 0.2059 |
| features selected | 9 | 8 |

Its standalone IV was 0.0414 — "weak" on the conventional scale. **The cost is
about one Gini point.** It remains listed under
`fair_lending.protected_characteristics`, so the Phase 7 guard keeps watch if
anyone ever puts it back.

The number matters more than its size. A model that cannot lawfully be used has
no Gini worth quoting, so this is not a trade-off between accuracy and compliance
— it is the price of having a usable model at all.

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

## 6. Score scaling  *(built 14 Oct 2026)*

The points table **is** the scorecard. Everything before this was the machinery
that produced it.

### 6.1 The transform

$$\text{Score} = \text{Offset} + \text{Factor}\times\ln(\text{odds}_{\text{good:bad}})$$

$$\text{Factor} = \frac{\text{PDO}}{\ln 2}, \qquad
\text{Offset} = \text{BaseScore} - \text{Factor}\times\ln(\text{BaseOdds})$$

**PDO** is "points to double the odds". With `pdo = 20`, gaining 20 points
doubles the good:bad odds — the one sentence that explains a scorecard to someone
who will never see a logistic regression, and the entire reason for this scaling
rather than reporting the PD directly. `test_pdo_doubles_the_odds` checks it at
four different score levels.

### 6.2 Points per bin

Because the model is linear in the log-odds and the inputs are WOE, the score
decomposes into one additive contribution per characteristic:

$$\text{Points}_k = -\Big(\beta_k\,\text{WOE}_k + \frac{\beta_0}{n}\Big)
                      \times \text{Factor} + \frac{\text{Offset}}{n}$$

The intercept and offset are split evenly across the $n$ variables. That is a
*presentational* choice — it makes every characteristic's points comparable and
the total come out right — and nothing in the model depends on it.

Scoring an applicant is then a **lookup and an addition**. A clerk with the
printed table and a calculator gets the same answer as the pipeline, which is
what "explainable" actually means in practice.

### 6.3 The sign chain, end to end

| Phase | Convention |
|---|---|
| 1 | 1 = default |
| 3 | WOE positive = safer |
| 4 | therefore coefficients negative |
| 6 | $-\beta_k\text{WOE}_k > 0$ for a safer bin → **more points** |

So **higher score = safer applicant**, the universal convention and the opposite
direction to the PD. If any link were flipped the scorecard would rank backwards
while producing a perfectly plausible points table.
`test_higher_score_means_lower_pd` and `test_safer_bins_earn_more_points` are the
end-to-end checks.

### 6.4 Verification: the table must reproduce the model

A points table is worth nothing unless it reproduces the model it came from.
`verify_against_model` scores every applicant both ways — through the table and
through the logistic model — and reports the largest disagreement.

**Unrounded it must be zero to floating-point precision**, because the table is
an algebraic rearrangement of the model, not an approximation. Rounding to whole
points, as a real scorecard does, introduces a bounded error: each
characteristic rounds independently, so each contributes at most half a point.

**The tolerance therefore scales with the number of characteristics**, and a
fixed one is a design error. The first real run failed an eleven-characteristic
card at 2.277 points against a flat 1.0 limit — a table that was entirely
correct. `verification_tolerance_per_feature` (0.5) is the right form of the
setting; `rounding_tolerance` turns it into a total.

`test_verification_fails_loudly_on_a_tampered_table` adds 60 points to one cell
and requires the check to refuse the table.

### 6.5 The reference point has to be reachable

`base_odds = 50` asserts a reference applicant with a PD of $1/51 = 1.96\%$. If
the model never produces a PD that low, the scale is anchored at a
creditworthiness nobody in the population has. The ranking is unaffected — the
scaling is linear — but every score reads oddly against the stated reference, and
a committee will ask on sight.

**The subtlety: raising `base_score` does not fix it.** The offset moves with it,
so every point moves with it, so the attainable range shifts by exactly the same
amount and the gap is invariant.
`test_raising_the_base_score_does_not_fix_an_unreachable_reference` measures that
the gap is identical at base scores of 600, 500 and 520.

**`base_odds` is the lever.** Lowering it into the PD range the model actually
spans is the fix, and the report states both ranges so the warning can be acted
on rather than merely noted.

### 6.6 Unseen bins score the characteristic's mean, not zero

Zero is not neutral on this scale — the offset is baked into every cell — so zero
points would be a severe and arbitrary penalty for a data gap that is the
lender's fault. An unseen bin receives the characteristic's mean points.

### 6.7 Cut-off options

The table a credit committee actually decides on, computed on the **held-out**
half because a policy choice must be made on data the model has not seen. Each
row is a policy option:

| cut-off | approval rate | approved bad rate | bads rejected | goods rejected |
|---|---|---|---|---|
| 472 | 95.0% | 29.1% | 5.7% | 4.7% |
| 514 | 56.0% | 23.2% | 55.7% | 39.2% |
| 552 | 15.3% | 15.2% | 92.0% | 81.6% |

Read as: *declining below 514 approves 56% of applicants, the approved book runs
a 23.2% bad rate, and we keep out 55.7% of the bads at the cost of turning away
39.2% of the goods.* Both sides of that trade-off are reported, because a
cut-off that rejects most of the bads by rejecting most of everyone is not a
policy.

### 6.8 The frozen artefact

`data/processed/scorecard.json` — the points table plus the scaling constants, as
plain JSON. A scorecard has to be scoreable by hand, years later, with no code at
all. That is why nothing in this pipeline is pickled.

## 7. Interactive scorer & reason codes  *(built 18 Oct 2026)*

Given one applicant, return their score, their PD, and **reason codes** — the
characteristics where they lost the most points.

### 7.1 Reason codes are a legal artefact

A declined applicant is entitled to be told why. In the US that is the
adverse-action notice required by ECOA / Regulation B; in South Africa the
National Credit Act obliges a lender to give reasons on request. *"The model said
no"* is not an answer in either jurisdiction.

This is the clearest practical reason the project uses a points table rather than
a gradient-boosted model. Reason codes here are not a surrogate model, a LIME
explanation or a SHAP value — they are **read directly off the arithmetic that
produced the decision**. The sum of the points *is* the score, so the
characteristic that cost the most points *is* the reason.

### 7.2 Lost points, not low points

Each characteristic is measured against the **best bin available in that
characteristic**:

$$\text{points lost}_k = \max_b(\text{points}_{kb}) - \text{points awarded}_k$$

Measuring against the mean would be wrong in a specific and unfair way: an
applicant slightly below average on a strongly predictive characteristic would be
told about *that*, instead of the characteristic where they actually lost 40
points. The regulatory question is where the points went, and the best attainable
bin is the only reference that answers it.

A characteristic where the applicant already holds the best bin has lost nothing
and can never be cited. That falls out of the arithmetic rather than needing a
special case, which is exactly why
`test_best_possible_applicant_has_no_reason_codes` asserts it — a notice saying
*"your perfect credit history counted against you"* would be worse than none.

Losses below `minimum_points_lost` (1 point) are suppressed: being one point
short of the best bin is not why anyone was declined, and listing it would bury
the real cause.

### 7.3 Fair lending: a finding that stopped this phase

Running the scorer on real German Credit produced this as the **fourth reason
code** on a live explanation:

```
4. personal_status_sex: you are in 'female: divorced / separated / married',
   which costs 18 points against the best available ('male: divorced / separated')
```

That is an adverse-action notice telling an applicant their **sex** counted
against them. It is unlawful under ECOA / Regulation B, and under the National
Credit Act and the Equality Act in South Africa.

The code was working perfectly. The problem is that **the model contains a
protected characteristic** — flagged as an open decision in Phase 3 (§1.2,
§3) and never resolved. Phase 7 is where deferring it stopped being possible,
because this is the phase that puts the model's reasoning in front of the
applicant.

The scorer now refuses. `fair_lending.protected_characteristics` names the
characteristics that may not be cited, and
`fair_lending.on_protected_reason_code` sets the behaviour:

* **`refuse`** (default) — raise `FairLendingError`. An explanation that cannot
  lawfully be given should not be produced at all.
* **`warn`** — suppress the code, and keep saying in the explanation that the
  model is not fit for lending use.

**Suppressing the notice does not make the model lawful.** The characteristic is
still driving the score; only the disclosure has been hidden, which is arguably
worse. The error message says so and names the real fix: drop it from the schema,
refit from Phase 2, and re-validate. The `warn` path exists for diagnostic work
only, and `test_warn_policy_suppresses_the_code_but_not_the_problem` checks the
warning survives.

The guard fires on what would actually be **cited**, not on mere presence in the
model; presence is reported separately as `protected_in_model`.

### 7.4 The scorer runs on frozen artefacts alone

`app/scorer.py` reads only `binning_spec.json` and `scorecard.json`. It never
refits anything and does not need the training data, the model object, or the
pipeline. A scorecard that can only be used by the person who built it is not a
scorecard — `test_scorer_works_from_frozen_artefacts_alone` is the check.

Four ways to run it, because different situations need different ones:

```bash
python app/scorer.py                      # interactive prompts
python app/scorer.py --examples           # three contrasting applicants
python app/scorer.py --batch file.csv     # score a whole file
python app/scorer.py --table              # print the points table
```

`--cutoff 520` adds an approve/decline decision to any of them. Reasons are
attached **only on a decline**, which is when they are required.

An incomplete application is scoreable, because incomplete applications are the
normal case: Phase 2's `MISSING` bin and Phase 3's WOE for it carry it through.
An *unknown* attribute name, by contrast, is refused — a typo must not silently
score the applicant as though that information were missing.

## 8. Benchmark & the trade-off  *(built 20 Oct 2026)*

One question: **what does the interpretability cost?**

The scorecard was not chosen because it is the most accurate model available. It
was chosen because a points table can be shown to an applicant, signed off by a
credit committee and audited by a regulator. That is a real benefit with a real
price, and the honest thing is to **measure the price**.

### 8.1 What makes the comparison fair

Three things, each easy to get wrong in a way that flatters one side:

1. **The same frozen split.** Both models train on `train` and are measured on
   `test`, the split frozen in §1.5. A benchmark scored on its own fresh split
   would be meaningless.
2. **The benchmark gets the RAW features, not the WOE.** Binning and WOE exist
   to make a *linear* model work; handing a tree model pre-binned inputs throws
   away its main advantage, which is finding non-linearity and interaction by
   itself.
3. **Protected characteristics are excluded from both.** A benchmark that uses
   sex to win would not be lawful either, and the comparison would be between a
   model that can be used and one that cannot. The report names the watched list
   whether or not it had to act, so a reader can tell the question was asked.

**scikit-learn's `HistGradientBoostingClassifier`**, not LightGBM — the same
histogram-based family, already a dependency, and it handles categoricals and
missing values natively, so no imputation choice can tilt the result. Settings
are deliberately modest: a heavily tuned booster on 700 rows would mostly be
measuring how much hyperparameter search the test set can absorb.

### 8.2 The result

| model | test Gini | test AUC | test KS | train Gini | train→test drop |
|---|---|---|---|---|---|
| Scorecard (logistic + WOE) | 0.1997 | 0.5998 | 0.2059 | 0.4747 | 0.2751 |
| Benchmark (gradient boosting) | **0.2137** | 0.6069 | 0.1949 | **0.8625** | **0.6487** |

**Gini gap: +0.0141 (+7.1% relative).** The benchmark wins, and barely.

The second number is the more interesting one. The benchmark's **train Gini is
0.8625 against a test Gini of 0.2137** — it drops 0.65 from train to test, where
the scorecard drops 0.28. The overfitting gap is **+0.3737**. Most of what the
booster learned was this particular 700 rows, and variance is the first thing to
disappear next quarter.

So the honest summary is: *on a book this size, the gradient-boosted model buys
about one Gini point of out-of-sample discrimination and pays for it with three
times the overfitting.* The interpretability is close to free here, and the
number to quote is +0.0141.

### 8.3 What the scorecard buys for that gap

* a points table an applicant can be shown;
* reason codes read off the decision arithmetic itself, not a surrogate model;
* monotonic risk by construction, so cut-off policy is defensible;
* scoreable by hand, years later, from a JSON file.

### 8.4 If the gap had been large

It is worth saying what the position would be, because the answer must not
depend on the result being convenient. The report has three branches, and the
material-loss one says plainly that a large gap is *a real finding and should be
reported, not buried* — the scorecard would still be the right choice where
explainability is a legal requirement, and the gap is what that requirement
costs. `test_report_does_not_hide_a_material_loss` checks that wording exists.

## 9. Reporting  *(built 21 Oct 2026)*

The last phase produces nothing new about credit. It is about the gap between a
result and a reader, which is where quantitative work is usually lost.

### 9.1 Charts as a build artefact

`scripts/make_charts.py` regenerates every chart in the README from one pipeline
run and writes `docs/charts/numbers.json` alongside them. The README quotes
numbers from that file and `tests/test_docs.py` checks the two agree, to the
precision the README quotes.

This closes a loop usually left open. A README's numbers drift out of date
silently, and the drift is only ever found by a reader — in this case, an
interviewer. Here a stale number fails the test suite.

Two of those tests are not about numbers at all:

* `test_readme_reports_the_failing_gates` — the model fails two validation
  thresholds, and the README must keep saying so. The easiest way to make a
  portfolio project look stronger is to delete the sentence admitting it did not
  clear a limit; here that deletes a passing test with it.
* `test_readme_keeps_the_fair_lending_finding` — the protected-characteristic
  work, its measured cost, and the sentence stating the lesson must all remain.

Writing a test against your own future temptation is an odd thing to do, and it
is the right thing to do.

### 9.2 Why the failing gates stay failing

Test Gini 0.1997 against a 0.30 minimum, and a train-to-test Gini drop of 0.2751
against a 0.15 limit. Both fail.

German Credit is 1,000 rows with a 30% bad rate: 300 events across 11
characteristics. The drop is the signature of a model fitted on too little data,
not of a modelling error — and the honest response is to report it against
thresholds set *before* the model was fitted. Relaxing a limit until a model
passes is how a validation process becomes theatre, and `config.yaml` is where
that would have happened, so `config.yaml` is where the limits stay.

### 9.3 A number that had to be made exact

The KS panel of the discrimination chart was first drawn from the rounded
**points**, while the validation report quotes KS on the continuous **predicted
PD**. The two differed in the fourth decimal place, so the README would have
carried two different KS figures a page apart — exactly the kind of discrepancy
an interviewer notices and an author cannot explain on the spot.

The chart now draws the PD curve, so the gap on the chart *is* the KS in the
table. The difference itself is measured and recorded as
`ks_shift_from_rounding`: rounding a scorecard to whole points creates ties, and
KS on a coarser scale can land either side of the continuous figure by chance. A
positive shift is not an improvement from rounding; it is the size of the wobble,
reported rather than assumed away.

### 9.4 The stamp

Charts built on the 24-row bundled sample are meaningless, and a meaningless
chart that looks real is worse than none. When `german.data` is absent,
`make_charts.py` falls back to the sample, stamps **ILLUSTRATIVE** across every
chart, and records `"real_data": false`.
`test_committed_charts_came_from_real_data` refuses the commit.

The stamp is applied in `finish()`, which every chart is saved through, so it
cannot be forgotten on one — the same structural argument as the
per-characteristic rounding tolerance in §6: make the correct behaviour the
default rather than something to remember.

### 9.5 Two charting mistakes worth naming

* **Two translucent fills on the score-separation chart** turned the overlap
  into a third muddy colour belonging to neither group — and the overlap is the
  business problem the chart exists to show. Each distribution now carries a
  solid outline over a light fill, so its shape reads through the other.
* **The characteristic names on the points table** were placed in the left
  margin at a fixed offset, which collided with the longest bin labels — and the
  longest labels are the ones a reader most needs. They are now bold header rows
  inside the axis.

Neither fault raises, logs, or fails a unit test. Both were found by opening the
PNG and looking at it, which is the only check that catches them.

### 9.6 A silent edit, caught by the file itself

Worth recording because it is the same class of error as everything else in this
document. The first attempt to add this section inserted it before a
`## References` heading — which this file does not have. The edit matched
nothing, changed nothing, and reported success. It was caught only because
`git status` showed `docs/methodology.md` unmodified.

An edit that cannot fail loudly will eventually fail quietly. The same reasoning
produced the `history` argument in the football model's feature stage and the
stamp in `finish()`: arrange the work so the wrong outcome is impossible or
noisy, never merely unlikely.
