# Roadmap: Credit Default Scorecard

> The build plan, and what actually happened against it. I wrote the schedule
> before starting and I am leaving it here unedited, because a plan that is
> quietly rewritten to match the outcome is not a plan.

## Planning assumptions, as written on 28 September 2026

- **Inception:** 28 September 2026. Documentation went up on 1 September; the
  build itself started on the 28th and the schedule was baselined from then.
- **Cadence:** one commit per working day, with the commit history as evidence.
- **Effort:** about 1 hour a day, 6 days a week, so roughly 6 focused hours a
  week.
- **Risk:** this is a well trodden method on a small dataset, so the risk was
  methodology precision rather than runtime.
- Dates were targets, not promises. The commit history is the real record.

## Plan against actual

| Phase | Focus | Target window | Landed | |
|---|---|---|---|---|
| **0** | Foundations and EDA | Sep 28 to Oct 4 | **30 Sep** | on time |
| **1** | Target definition and split | Oct 5 to 11 | **3 Oct** | 8 days early |
| **2** | Binning and monotonic classing | Oct 12 to 18 | **9 Oct** | 9 days early |
| **3** | WOE, IV and feature selection | Oct 19 to 25 | **10 Oct** | 15 days early |
| **4** | Logistic PD model | Oct 26 to Nov 1 | **10 Oct** | 22 days early |
| **5** | Validation | Nov 2 to 8 | **14 Oct** | 25 days early |
| **6** | Score scaling and points table | Nov 9 to 15 | **14 Oct** | 32 days early |
| **7** | Interactive scorer and benchmark | Nov 16 to 22 | **18 Oct** | 35 days early |
| **8** | Docs and presentation | Nov 23 to 29 | **21 Oct** | 39 days early |

The presentation milestone was targeted at 27 November and reached on 21
October, about five weeks ahead. Two reasons, and only one of them is flattering.

The honest one: a well specified method on a 1,000 row dataset runs fast, and
most phases were an evening rather than a week once the scaffolding and the
config discipline were in place.

The less flattering one: the original plan allocated a week per phase because I
did not know how long each would take. The estimate was not conservative, it was
uninformed. The useful figure out of this is that one focused hour with a
written definition of done moved the project further than I had assumed a week
would.

## What landed after the plan ended

Two things, both triggered by findings rather than scheduled:

- **20 to 22 October: the protected characteristic.** The applicant scorer
  produced a decline notice citing sex. That forced a guard, then removal of the
  characteristic from the model, then a script to measure what the removal cost.
  None of it was on the plan and it is the part of the project I would lead with.
- **22 October: every published number was wrong.** The README reported a model
  that failed two gates it had actually passed, and the test suite was holding
  the error in place because it asserted the literal figures rather than
  checking them against the run. See `PROGRESS.md` for the full story.

Neither was foreseeable from a roadmap, which is the point of keeping a progress
log alongside one.

## Phase detail, as originally written

### Phase 0: foundations and EDA
Scaffold the repo, environment and config, load German Credit, first EDA.
**Done when** the dataset loads reproducibly and its shape and target balance
are documented.

### Phase 1: target definition and split
Define good and bad, build train and test splits.
**Done when** the rule is explicit and defensible and the splits are
reproducible from a seed.

### Phase 2: binning and monotonic classing
Fine then coarse classing, monotonic risk trends, special and missing bins.
**Done when** each selected feature bins into a small, monotonic, interpretable
set of buckets.

### Phase 3: WOE, IV and feature selection
WOE transform, Information Value, drop the weak and investigate the
suspiciously strong.
**Done when** every model input is WOE encoded and justified by IV, with
leakage checks on high IV variables.

### Phase 4: logistic PD model
Fit on WOE, check signs, VIF and significance.
**Done when** all signs are intuitive, multicollinearity is controlled and the
fit is stable.

### Phase 5: validation
KS, Gini and AUC, ROC, calibration, PSI, rank ordering.
**Done when** discrimination, calibration and stability are all reported on
held-out data.

### Phase 6: score scaling and points table
The PDO, factor and offset transform into a points table.
**Done when** any log odds maps to a score and the points table is complete and
readable.

### Phase 7: interactive scorer and benchmark
Applicant scorer with reason codes, then the gradient-boosted comparison.
**Done when** the scorer returns PD, score and reason codes for any applicant
and the accuracy against interpretability trade-off is quantified.

### Phase 8: docs and presentation
README with real charts, complete methodology.
**Done when** the project can be presented as fluently as it can be run.

## The phase that did not happen here

Phase 9 was scoped as an optional SAS parallel build, `PROC LOGISTIC` with WOE
and IV in data steps, reconciled against the Python output.

I built it in my [credit risk ECL engine](https://github.com/esiihle/credit-risk-ecl-engine)
instead. The reasoning: `PROC SQL` has considerably more to say about what SAS
can and cannot do than `PROC LOGISTIC` would have had here. That project's port
had to rebuild `LEAD`, `LAG`, a bounded window frame and a recursive CTE,
because PROC SQL has none of them, and the reconciliation is against a full ECL
chain rather than one regression.

A SAS logistic scorecard remains a reasonable extension. It is just not the one
that teaches the most.

## What was deliberately left out

- **Out-of-time validation.** German Credit carries no usable origination date,
  so train and test are a random split. Out-of-time is the better stability
  check and it is not possible on this data.
- **Reject inference.** Everyone in the dataset was granted credit, so it
  describes accepted applicants only.
- **A second dataset.** Home Credit was scoped as an optional follow-up.
  Finishing the methodology properly on one dataset was worth more than running
  a half-finished one on two.
