# Project Overview: Credit Default Scorecard

> Read this first. It orients anyone picking up the project, whether that is a
> collaborator, a recruiter, or me a year from now. It explains what the thing
> is, why the industry builds it this way, and how the pieces fit together. The
> maths is in `methodology.md`, the build history is in `ROADMAP.md` and
> `PROGRESS.md`, and the headline results are in the README.

---

## 1. What this is

A **probability of default scorecard**: the standard, regulator friendly credit
risk model that retail banks use to decide who gets a loan. Given an
applicant's attributes it produces a probability of default and a scaled score,
built from a transparent points table where every attribute contributes a known
number of points.

It is built the way a bank builds one: weight of evidence feature engineering,
then logistic regression, then validation against thresholds fixed in advance,
then score scaling, then an interactive scorer with reason codes. Finally it is
benchmarked against a gradient booster, so the cost of staying interpretable is
a number rather than an assertion.

## 2. Why it exists

In regulated retail credit, interpretability, stability and auditability matter
as much as raw accuracy. A bank has to explain to a regulator, and to the
customer, why an application was declined. A black box that is slightly more
accurate but cannot produce reason codes is often simply not usable. That is
why the WOE and logistic scorecard, decades old, is still the workhorse.

This project builds that workhorse properly, validates it to the standard a
model validation team would expect, and then measures what interpretability
actually costs.

**On this data it cost nothing measurable.** The gradient booster came in 0.0468
Gini *lower* than the scorecard, and it overfitted eightfold worse. The
difference is inside the standard error either way, so the honest claim is that
there is no evidence the benchmark is better here. That result is the opposite
of what the textbook answer predicts, and being able to say why is more useful
than having guessed right.

## 3. Why a recruiter should care

This project is a South African bank credit risk role in miniature. The
vocabulary and the deliverables map one to one onto retail credit risk work:

| This project | The real deliverable |
|---|---|
| Good and bad definition | Target definition and performance window |
| Monotonic binning | Coarse classing for stable, explainable risk trends |
| WOE and Information Value | Feature transformation and selection |
| Logistic PD model | The application or behavioural scorecard |
| KS, Gini, AUC | Discrimination testing |
| PSI | Population stability and drift monitoring |
| PDO, factor, offset scaling | Turning odds into a score a customer understands |
| Reason codes | Adverse action and explainability requirements |
| Removing a protected characteristic, and pricing it | Fair lending review |

It also feeds my [credit risk ECL engine](https://github.com/esiihle/credit-risk-ecl-engine),
which takes a probability of default and turns it into an IFRS 9 balance sheet
provision. This project produces the PD. That one consumes it.

## 4. The pipeline

Eight stages, each independently testable.

### Stage 1: data and target (`load.py`)
Load the data, decode the A-codes to plain English so every later stage reads in
English, define the good and bad target, and split train and test behind a
fingerprinted manifest. An unrecognised code raises rather than passing through,
because a stray code would silently become a junk bin in the next stage.

**German Credit**, 1,000 rows, is the only dataset used. It is small, clean and
fully documented, which is what makes it possible to get the methodology exactly
right. Its size is also the binding constraint on every result, which is why the
standard errors are quoted rather than hidden.

**Lending Club is deliberately avoided.** Its public versions carry well known
target leakage, and recognising that is a better move than scoring on it.

### Stage 2: binning and coarse classing (`binning.py`)
Fine classing first, then coarse classing into a small number of bins with a
monotonic risk trend, so the bad rate moves consistently in one direction.
Continuous and categorical features are both handled, with explicit special and
missing bins. Monotonicity is what makes the final scorecard stable and
defensible to someone who has to explain it.

### Stage 3: WOE and Information Value (`woe.py`)
Each bin becomes its weight of evidence, `ln(%good / %bad)`, which recodes every
feature onto one log odds scale. Information Value then selects features:

- below 0.02 is unpredictive and dropped
- 0.02 to 0.1 weak, 0.1 to 0.3 medium, 0.3 to 0.5 strong
- above 0.5 is suspiciously strong and gets **flagged for a human rather than
  auto-dropped**, because that is the signature of leakage

### Stage 4: PD model (`model.py`)
Logistic regression on the WOE features, which is linear in the log odds of
default and is exactly what makes the points table possible later.

The headline diagnostic is the **sign check**. Because positive WOE means safer
and 1 means default, every slope must be negative. A scorecard with a flipped
sign ranks applicants backwards and still produces entirely plausible output, so
there is no symptom to notice. Alongside it: an events per variable guard, VIF
for multicollinearity, and a perfect separation check.

### Stage 5: validation (`validate.py`)
The credibility layer, every threshold set in `config.yaml` before the model was
fitted:

- **KS**, the widest separation between the good and bad score distributions
- **Gini and AUC**, rank ordering power, where `Gini = 2·AUC - 1`
- **Calibration**, whether predicted PDs match realised default rates, which
  starts to matter the moment the PD is used for pricing rather than only for
  accept and decline
- **PSI**, whether the score distribution has drifted, which is the early
  warning for a decaying model
- **Rank ordering**, a monotonic bad rate across score bands

### Stage 6: score scaling (`scaling.py`)
Model log odds become a readable score through the standard transform,
`Score = Offset + Factor · ln(odds)` with `Factor = PDO / ln 2`. The output is
the points table, and that table *is* the scorecard.

Verification is against the model's own PD, within a tolerance of half a point
**per characteristic** rather than a flat one, because rounding error scales
with the number of characteristics.

### Stage 7: interactive scorer (`scorecard.py`, `app/scorer.py`)
Feed in an applicant and get back a PD, a score, a decision and reason codes,
which are the bins that cost them the most points. Change an input and watch the
score and the explanation move.

Reason codes are a legal artefact rather than a convenience feature, which is
how the fair lending finding surfaced: the scorer produced a decline notice
citing sex. The guard that stops it is still there, and the characteristic is
also out of the model, because suppressing a disclosure is not the same as
removing a variable.

### Stage 8: benchmark (`benchmark.py`)
A `HistGradientBoostingClassifier` on the same split, compared on the same
metrics. scikit-learn rather than LightGBM or XGBoost, so the repository has no
extra heavyweight dependency for one comparison.

## 5. Design principles

- **Interpretable by construction.** Every point on the card traces back to a
  WOE bin and a coefficient.
- **Interactive.** The scorer handles many applicant scenarios, not one demo
  row.
- **Simple and commented**, so it can be walked through line by line.
- **Thresholds before results.** Every gate was written into `config.yaml`
  before the model was fitted, because a limit that moves to accommodate a
  result was never a limit.
- **No number in a document that a command cannot regenerate.** The charts write
  `numbers.json` and the test suite checks the README against it. This is not a
  principle I started with; it is one I adopted after publishing figures that
  were wrong by a factor of three and watching a green test suite hold them in
  place for two days.
- **Reproducible.** Fixed seeds, pinned dependencies, config driven.

## 6. Repository structure

```
credit-default-scorecard/
├── README.md              the results, the charts, and the bugs worth reading about
├── config.yaml            every threshold, set before the model was fitted
├── src/
│   ├── load.py            data load, good and bad target, frozen split
│   ├── binning.py         fine and coarse classing, monotonic bins
│   ├── woe.py             WOE transform, IV, feature selection
│   ├── model.py           logistic PD model and diagnostics
│   ├── validate.py        KS, Gini, AUC, PSI, calibration
│   ├── scaling.py         PDO, factor and offset into a points table
│   ├── scorecard.py       applicant scorer and reason codes
│   └── benchmark.py       gradient-boosted comparison
├── app/scorer.py          the interactive scorer, CLI
├── scripts/               run_pipeline · make_charts · measure_fair_lending_cost
├── tests/                 ten modules; every bug above is pinned by one
├── notebooks/             01_exploration.ipynb
└── docs/                  OVERVIEW · ROADMAP · methodology · charts
```

## 7. What was left out, and why

- **No out-of-time validation slice.** German Credit carries no usable
  origination date, so train and test are a random split. Out-of-time testing is
  the better stability check and it is not possible on this data.
- **No reject inference.** Everyone in this dataset was granted credit, so it
  describes accepted applicants only. Reject inference is the standard bank
  remedy and is out of scope here.
- **No SAS implementation.** It was scoped as a stretch phase here, and I built
  it in the ECL engine instead, where `PROC SQL` has more to say than
  `PROC LOGISTIC` would have here.
- **No Home Credit follow-up.** It was scoped as an optional second dataset.
  Finishing the methodology properly on one dataset was worth more than running
  a half-finished one on two.

## 8. What done looked like

All of these hold:

- one command runs load, bins, WOE, model, validation and the scaled scorecard
- the scorer returns PD, score, decision and reason codes for any applicant
- validation is reported against thresholds fixed in advance, with the
  uncertainty on each figure stated
- the benchmark is run and the trade-off is priced rather than asserted
- every figure in the README is checked against `numbers.json` by the test suite
