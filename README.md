# 💳 Credit Default Scorecard

**An interpretable probability-of-default scorecard built the way a bank builds one: WOE/IV feature engineering, logistic PD model, full validation (KS/Gini/AUC/PSI), score scaling, and an interactive applicant scorer — then benchmarked against a gradient-boosted model.**

![status](https://img.shields.io/badge/status-in%20active%20development-orange)
![python](https://img.shields.io/badge/python-3.11+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## The problem

Retail banks must decide who receives credit, and must be able to **explain every
decision** to a regulator and to the customer. Raw predictive power isn't enough:
the model has to be interpretable, stable, and auditable. This project builds the
industry-standard answer — a **WOE-logistic scorecard** — end to end, validates
it to a model-validation standard, and honestly measures the accuracy it trades
away for that interpretability.

## The method

A clean, one-directional pipeline:

1. **Data & target** — define good/bad default, split train/test (+ out-of-time).
2. **Binning** — fine then coarse classing into **monotonic** risk bins.
3. **WOE / IV** — weight-of-evidence transform; Information Value for selection.
4. **PD model** — logistic regression on WOE features, with sign/VIF diagnostics.
5. **Validation** — KS, Gini/AUC, calibration, and **PSI** for stability.
6. **Score scaling** — PDO/factor/offset → a transparent **points table**.
7. **Interactive scorer** — PD, score, and **reason codes** for any applicant.
8. **Benchmark** — a gradient-boosted model, to quantify the
   accuracy-vs-interpretability trade-off.

Full detail in [`docs/OVERVIEW.md`](docs/OVERVIEW.md) and
[`docs/methodology.md`](docs/methodology.md).

## Dataset

**German Credit** (primary) — clean, documented, runs in seconds; ideal for
nailing the methodology. **Home Credit** is an optional follow-up for realistic,
messy data. **Lending Club is deliberately avoided** — its public versions carry
well-known target leakage; the mature move is to recognise that rather than score
on it.

## Results

> 🚧 **In active development.** Populated as each stage lands. Metrics come only
> from held-out validation — no placeholder numbers.

Planned headline outputs:

- **KS / Gini / AUC** table for the scorecard.
- **Reliability (calibration) curve** — predicted vs. realised default rate.
- **PSI report** — development vs. validation stability.
- **The scorecard itself** — the points table.
- **Scorecard vs. gradient-boosted benchmark** — the trade-off, quantified.

## Why this project

It is the SA bank credit-risk role in miniature. Every step maps to a real
deliverable:

| This repo | The real deliverable |
|---|---|
| WOE / IV | Feature transformation & selection |
| Logistic PD model | The application scorecard |
| KS / Gini / AUC | Discrimination testing |
| PSI | Stability / drift monitoring |
| PDO scaling | Customer-facing score |
| Reason codes | Adverse-action explainability |

## Repository structure

```
src/        pipeline (load, binning, woe, model, validate, scaling, scorecard, benchmark)
app/        interactive applicant scorer
tests/      unit tests
notebooks/  EDA
sas/        optional parallel SAS implementation
docs/       OVERVIEW, ROADMAP, methodology
```

## Quickstart

```bash
git clone https://github.com/<your-username>/credit-default-scorecard.git
cd credit-default-scorecard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --config config.yaml   # runs the pipeline; each stage lights up as it's built
python app/scorer.py                                   # interactive applicant scorer (from Phase 7)
```

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the dated, week-by-week build plan
from inception to presentation.

## Author

**Sihle** — BSc Computer Science (Wits), data analyst. Portfolio project built in the open.

## License

MIT — see [`LICENSE`](LICENSE).
