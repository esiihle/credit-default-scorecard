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

## 2. Binning / classing
**Status: 🟡 skeleton — built in Phase 2.**

Each feature is **fine-classed** (many bins) then **coarse-classed** into a small
number of bins chosen so the bad rate is **monotonic** across the ordered bins.
Continuous and categorical features are handled, with explicit special bins for
missing / sentinel values. Monotonicity is what makes the final points table
stable and defensible to a regulator.

## 3. Weight of Evidence & Information Value
**Status: 🟡 skeleton — built in Phase 3.**

### 3.1 Weight of Evidence
Each bin is recoded to its log-odds of being good vs. bad:

$$\text{WOE}_b = \ln\!\left(\frac{\%good_b}{\%bad_b}\right)$$

WOE puts every feature on one monotonic, model-friendly scale and is what lets a
logistic model translate cleanly into a points table.

### 3.2 Information Value
Feature-level predictive strength:

$$\text{IV} = \sum_b \left(\%good_b - \%bad_b\right)\,\text{WOE}_b$$

Selection guide: $<0.02$ drop; $0.02\text{–}0.1$ weak; $0.1\text{–}0.3$ medium;
$0.3\text{–}0.5$ strong; $>0.5$ **suspiciously strong — check for leakage** before
trusting it.

## 4. PD model
**Status: 🟡 skeleton — built in Phase 4.**

Logistic regression on the WOE-transformed features:

$$\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{k} \beta_k \,\text{WOE}_k$$

where $p$ is the probability of default. Because inputs are WOE, the model is
linear in the log-odds — the property that makes §6 possible. Diagnostics:
coefficient **signs** must be intuitive, **VIF** controls multicollinearity, and
each variable must be significant and stable.

## 5. Validation
**Status: 🟡 skeleton — built in Phase 5.**

- **KS statistic** — maximum gap between the cumulative good and bad score
  distributions: $\text{KS} = \max_s \left| F_{good}(s) - F_{bad}(s) \right|$.
- **Gini / AUC** — rank-ordering power, with $\text{Gini} = 2\,\text{AUC} - 1$.
- **Calibration** — predicted PD vs. realised default rate (reliability curve).
- **PSI (Population Stability Index)** — drift between development and validation:
  $\text{PSI} = \sum_b (\%A_b - \%E_b)\,\ln(\%A_b / \%E_b)$, read with the usual
  $<0.1$ stable / $0.1\text{–}0.25$ shift / $>0.25$ major-shift bands.
- **Rank-ordering** — monotonic bad rate across score bands.

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
