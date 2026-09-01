# Project Overview — Credit Default Scorecard

> **Read this first.** This document orients anyone picking up the project — a
> collaborator, a recruiter, or myself later. It explains *what* we are building,
> *why* the industry builds it this way, and *how* the pieces fit together. The
> maths lives in `methodology.md`; the schedule lives in `ROADMAP.md`.

---

## 1. What this project is

This is a **probability-of-default (PD) scorecard** — the standard, regulator-
friendly credit-risk model that retail banks use to decide who gets a loan.
Given an applicant's attributes, it produces:

- a **probability of default**, and
- a **scaled credit score** (e.g. "score of 680"), built from a transparent
  **points table** where every attribute contributes a known, explainable number
  of points.

It is built the way a real bank builds one: **weight-of-evidence (WOE) feature
engineering → logistic regression → rigorous validation → score scaling →
interactive scorer**. Then, as a mature final step, it is **benchmarked against a
gradient-boosted model** to quantify the accuracy-versus-interpretability
trade-off that sits at the heart of credit-risk modelling.

## 2. Why it exists (the thesis)

In regulated retail credit, **interpretability, stability and auditability matter
as much as raw accuracy.** A bank must be able to explain to a regulator, and to
the customer, *why* an application was declined. A black-box model that is 2%
more accurate but cannot give reason codes is often not usable. That is why the
WOE-logistic scorecard — decades old — remains the workhorse.

This project builds that workhorse correctly, validates it to the standard a
model-validation team would expect, and then honestly measures what accuracy is
left on the table by choosing interpretability. **Understanding *why* the bank
chooses the interpretable model is the point** — and it is exactly the
conversation a credit-risk interview turns on.

## 3. Project Overview

This project *is* the South African bank job in miniature. Its vocabulary and
deliverables map one-to-one onto retail credit-risk work:

| This project | The real deliverable |
|---|---|
| Good/bad definition | Target definition & performance window |
| Monotonic binning | Coarse classing for stable, explainable risk trends |
| WOE / Information Value | Feature transformation & selection |
| Logistic PD model | The application/behavioural scorecard |
| KS / Gini / AUC | Discrimination testing |
| PSI | Population stability / drift monitoring |
| PDO / factor / offset scaling | Turning odds into a score customers understand |
| Reason codes | Adverse-action / explainability requirements |

## 4. How it works — the pipeline

A linear pipeline of well-separated, independently testable stages.

### Stage 1 — Data & target (`load.py`)
Load the dataset, define the **good/bad target** (who counts as a "default"),
and split into train/test — with an out-of-time slice where the data allows, to
test stability, not just fit.
- **Primary dataset: German Credit** — small, clean, fully documented. It lets us
  nail the *methodology* end to end in a system that runs in seconds.
- **Optional follow-up: Home Credit** — larger and messier, to show the same
  pipeline survives realistic data.
- **Deliberately avoided: Lending Club** — its public versions are riddled with
  target leakage; using it well means spotting that, not scoring on it.

### Stage 2 — Binning / classing (`binning.py`)
Split each feature into bins. **Fine classing** first (many bins), then **coarse
classing** into a small number of bins with a **monotonic** risk trend — bad rate
moving consistently in one direction across the bins. Continuous and categorical
features are handled, with explicit special/missing bins. Monotonicity is what
makes the final scorecard stable and defensible.

### Stage 3 — WOE & Information Value (`woe.py`)
Transform each bin to its **weight of evidence** — `ln(%good / %bad)` — which
recodes every feature onto a single, model-friendly, log-odds scale. Compute
**Information Value (IV)** per feature for selection:

- IV < 0.02 → unpredictive, drop.
- 0.02–0.1 → weak. 0.1–0.3 → medium. 0.3–0.5 → strong.
- IV > 0.5 → *suspiciously* strong — investigate for leakage before trusting it.

### Stage 4 — PD model (`model.py`)
Fit **logistic regression** on the WOE-transformed features. Because the inputs
are WOE, the model is linear in the log-odds of default, which is precisely what
makes the later points table possible. Diagnostics: coefficient **signs** must
match intuition, **multicollinearity** (VIF) is checked, and each variable must
be significant and stable.

### Stage 5 — Validation (`validate.py`)
The credibility layer:
- **KS statistic** — maximum separation between good and bad score distributions.
- **Gini / AUC / ROC** — rank-ordering power (`Gini = 2·AUC − 1`).
- **Calibration** — do predicted PDs match realised default rates?
- **PSI (Population Stability Index)** — does the score distribution drift between
  development and validation samples? The early-warning metric for a decaying
  model.
- **Rank-ordering** — monotonic bad rate across score bands.

### Stage 6 — Score scaling (`scaling.py`)
Convert model log-odds into a human-readable score with the standard
**PDO / factor / offset** transform:
`Score = Offset + Factor · ln(odds)`, where `Factor = PDO / ln(2)` and PDO is the
"points to double the odds." The output is a **points table**: for every
attribute bin, the exact number of points it adds. *This table is the scorecard.*

### Stage 7 — Interactive scorer (`scorecard.py`, `app/scorer.py`)
Feed in an applicant, get back their PD, their score, and **reason codes** — the
bins that most reduced their score, i.e. *why* they scored as they did. This is
the interactive, scenario-covering piece: change an input, watch the score and
the explanation move.

### Stage 8 — Benchmark (`benchmark.py`)
Train a **gradient-boosted model** (e.g. LightGBM) on the same data and compare
KS/Gini/AUC against the scorecard. Typically the GBM wins on raw discrimination
and loses on interpretability, stability and auditability. The write-up makes the
trade-off explicit — the single strongest talking point for a bank interview.

## 5. Design principles

- **Interpretable by construction** — every point on the scorecard is traceable
  to a WOE bin and a coefficient.
- **Interactive** — the scorer covers many applicant scenarios, not one demo row.
- **Simple and well-commented** — readable enough to walk a recruiter through
  line by line.
- **Validated to standard** — the metrics a real model-validation team asks for,
  reported honestly, including where the scorecard underperforms the benchmark.
- **Reproducible** — fixed seeds, pinned deps, config-driven.

## 6. Planned repository structure

```
credit-default-scorecard/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
├── src/
│   ├── load.py        # data load + good/bad target + split
│   ├── binning.py     # fine/coarse classing + monotonic bins
│   ├── woe.py         # WOE transform + IV + feature selection
│   ├── model.py       # logistic PD model + diagnostics
│   ├── validate.py    # KS, Gini, AUC, PSI, calibration
│   ├── scaling.py     # PDO/factor/offset -> points table
│   ├── scorecard.py   # applicant scorer + reason codes
│   └── benchmark.py   # gradient-boosted comparison
├── app/
│   └── scorer.py      # interactive applicant scorer (CLI / Streamlit)
├── tests/
│   └── test_woe.py
├── notebooks/
│   └── 01_eda.ipynb
├── sas/               # optional parallel SAS implementation (PROC LOGISTIC)
│   └── scorecard.sas
└── docs/
    ├── OVERVIEW.md    # this file
    ├── ROADMAP.md     # dated build plan
    └── methodology.md # the maths, in detail
```

## 7. On SAS (optional differentiator)

Many SA banks still run credit models in SAS. An optional parallel implementation
(`PROC LOGISTIC`, WOE/IV in data steps) is scoped as a stretch phase in the
roadmap: it doesn't change the methodology, but demonstrating it in *both*
languages is a genuine differentiator for those roles. Python is the primary
build; SAS is added if access (e.g. SAS OnDemand for Academics) is available.

## 8. What "done" looks like

- One command runs load → bins → WOE → model → validation → scaled scorecard.
- The interactive scorer returns PD, score and reason codes for any applicant.
- Validation (KS/Gini/AUC/PSI/calibration) is reported honestly, alongside the
  GBM benchmark and a clear discussion of the trade-off.
- README, methodology write-up and a slide deck exist.

See `ROADMAP.md` for the week-by-week plan.
