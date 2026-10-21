# 💳 Credit Default Scorecard

**An interpretable probability-of-default scorecard, built the way a bank builds one: WOE/IV feature engineering, a logistic PD model, full validation (KS / Gini / PSI / Hosmer-Lemeshow), PDO score scaling to a points table, an applicant scorer with reason codes — then benchmarked against gradient boosting to price the interpretability honestly.**

![tests](https://github.com/esiihle/credit-default-scorecard/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## The headline result

A scorecard is judged on three things: does it rank, is it stable, and can you
explain it to the person you declined. This one is built to be checked on all
three.

| Metric | Test | Train | Threshold |
|---|---|---|---|
| Gini | **0.1997** | 0.4747 | ≥ 0.30 ❌ |
| AUC | 0.5998 | — | — |
| KS | 0.2059 | — | — |
| Train → test Gini drop | **0.2751** | | ≤ 0.15 ❌ |

**Two validation gates fail, and this README reports them rather than moving
the thresholds.** German Credit is 1,000 rows with a 30% bad rate — 300 events
across 11 characteristics — and a Gini drop of 0.2751 is the signature of a
model fitted on too little data, not of a modelling mistake. The thresholds
stay where a model-validation function would set them, and the model is
reported as failing them. Quietly relaxing a limit until a model passes is the
single most common way a validation process becomes theatre.

![score separation](docs/charts/score-separation.png)
![discrimination](docs/charts/discrimination-roc-ks.png)

### The scorecard itself

The artefact that makes the whole approach defensible: an applicant can be shown
exactly which attribute cost them how many points, and the arithmetic is
addition.

![the points table](docs/charts/points-table.png)

Scaled with **PDO 20, base score 600 at 50:1 odds** — so every 20 points halves
the odds of default, and 600 points means a 1.96% PD. `base_odds` is the lever
that moves the attainable range, not `base_score`: raising the base score shifts
the offset and the whole range moves with it, leaving the gap invariant.
Measured identical at 600, 520 and 500.

### What interpretability costs

![scorecard vs benchmark](docs/charts/scorecard-vs-benchmark.png)

| | Scorecard | Gradient boosting |
|---|---|---|
| Test Gini | 0.1997 | **0.2137** |
| Train Gini | 0.4747 | 0.8625 |
| Train → test drop | 0.2751 | **0.6487** |

The benchmark buys **+0.0141 Gini (+7.1% relative)** and pays **+0.3737** more
overfitting for it — and it cannot produce a reason code. On 1,000 rows that is
the honest answer: the interpretable model is not meaningfully worse, and it is
the one that can be defended to a regulator and explained to a customer.

### Calibration

![calibration](docs/charts/calibration.png)

Ranking is not enough. A card can order applicants perfectly and still predict
the wrong *level*, which matters the moment the PD is used for pricing or
provisioning rather than only for accept/decline.

---

## Fair lending: the finding this project is proudest of

The raw German Credit data contains `personal_status_sex`, which encodes **sex**.
It is loaded, because the data should stay faithful to its source. What happens
next was decided in two stages, and the first one was not good enough.

**18 October — the symptom.** The applicant scorer produced this reason code on
a declined application:

> *"you are in 'female: divorced / separated / married', which costs 18 points"*

That is an adverse-action notice citing a protected characteristic. A guard was
added: `reason_codes` raises `FairLendingError` rather than disclosing one.

**20 October — the actual fix.** The guard suppressed the *disclosure*. The
attribute was still in the model, still moving the score, still declining
people. It was removed from `schema.categorical_columns`, and the cost measured
rather than assumed:

| | With sex | Without |
|---|---|---|
| Test Gini | 0.2100 | 0.1997 |
| Test AUC | 0.6050 | 0.5998 |
| Test KS | 0.2255 | 0.2059 |

**About one Gini point**, for a model that can lawfully be used. Its standalone
IV was 0.0414 — "weak" on the conventional scale. The guard stays in place as a
second line of defence, and `config.yaml` records both the decision and its
price.

The lesson is the one worth taking to an interview: **suppressing a disclosure
is not the same as removing a variable.** The first makes the notice compliant;
only the second makes the decision compliant.

## The method

```
load → bin → WOE/IV → logistic PD → validate → scale → score → benchmark
```

1. **Load & target** — decode the A-codes to plain English at load time, define
   good/bad, split train/test with a fingerprinted manifest.
2. **Binning** — fine then coarse classing into monotonic risk bins, with a
   minimum bin size so no bin is 1% of the book.
3. **WOE / IV** — `WOE = ln(%good / %bad)`; positive means safer. IV selects
   features; suspiciously high IV (≥ 0.50) is **flagged for a human, never
   auto-dropped** — that is the signature of leakage.
4. **PD model** — logistic regression on WOE, with an events-per-variable guard
   (minimum 10), VIF, and a perfect-separation check.
5. **Validation** — KS, Gini/AUC, Hosmer-Lemeshow, PSI, rank ordering, each
   against a threshold set in config before the model was fitted.
6. **Scaling** — `Factor = PDO / ln2`, `Offset = BaseScore − Factor·ln(BaseOdds)`,
   verified against the model's own PD within a tolerance of 0.5 points *per
   characteristic* — rounding error scales with the number of characteristics,
   so a flat tolerance is wrong.
7. **Scorer** — PD, score, decision and reason codes for any applicant, with the
   fair-lending guard.
8. **Benchmark** — gradient boosting, to price the interpretability.

Full derivations in [`docs/methodology.md`](docs/methodology.md).

## Try it

```bash
git clone https://github.com/esiihle/credit-default-scorecard.git
cd credit-default-scorecard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # the full suite
python app/scorer.py --examples          # score three worked applicants
python app/scorer.py --table             # print the points table
python app/scorer.py                     # interactive: score your own
python app/scorer.py --batch applicants.csv --cutoff 580
```

For the real numbers, download
[`german.data`](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
into `data/raw/`, then:

```bash
python scripts/run_pipeline.py           # the full pipeline
python scripts/make_charts.py            # regenerates every chart above
pytest tests/test_docs.py                # checks this README against that run
```

`docs/charts/numbers.json` records the figures the charts were built from, and
`tests/test_docs.py` checks the tables above against it. A stale number in this
README fails the test suite.

## Bugs this project found, and what each one taught

Every one was invisible in the output. Each is now pinned by a regression test
whose docstring tells the story.

| Bug | Symptom | Fix |
|---|---|---|
| **Singular matrix on a small sample** | 14 features, 5 bads: coefficients of −21.9, NaN standard errors, VIF of 299 — and on other library versions, a crash | Events-per-variable guard, `LinAlgError` translation, and a separation check on the standard error (separation gives *huge* SEs, not NaNs — statsmodels returned coefficient −13.4 with SE 16,342) |
| **Flat verification tolerance** | A correct 11-characteristic card failed a 1.0-point check at 2.277 | Rounding error is bounded by 0.5 points **per characteristic**; tolerance now scales |
| **`base_score` treated as the lever** | Raising it to widen the attainable range did nothing — the gap was identical at 600, 520 and 500 | It shifts the offset, so the whole range moves. `base_odds` is the lever |
| **Sex in a reason code** | An adverse-action notice citing a protected characteristic | Guard, then removal from the model, with the cost measured |
| **Bundled sample swallowed by `.gitignore`** | `data/*` ignored the committed sample; a fresh clone crashed and 2 of 4 tests failed | Ignore `data/raw/*` and `data/processed/*` only |

## Why this project

It is a South African bank credit-risk role in miniature:

| This repo | The real-world task |
|---|---|
| WOE/IV binning | Standard scorecard feature engineering |
| Logistic PD + points table | The model form regulators expect |
| KS / Gini / PSI against pre-set thresholds | Model validation and monitoring |
| Reason codes | Adverse-action notices (NCA; ECOA/Reg B abroad) |
| Removing a protected characteristic and pricing it | Fair-lending review |
| Benchmark vs GBM | The interpretability conversation, with numbers |

## Repository structure

```
src/            load, binning, woe, model, validate, scaling, scorecard, benchmark
app/scorer.py   interactive applicant scorer with reason codes
tests/          the fragile numerics, the hand-computed WOE/IV anchors, every bug above
scripts/        run_pipeline.py (end to end) · make_charts.py (every chart here)
docs/           methodology.md · OVERVIEW.md · ROADMAP.md · charts/
```

## Dataset

**German Credit** (UCI, 1,000 rows). Clean, documented, runs in seconds — the
right choice for getting the methodology exactly right. **Lending Club is
deliberately avoided**: its public versions carry well-known target leakage, and
recognising that is the more mature move than scoring on it.

## Author

**Sihle Myeza** — BSc Computer Science (Wits), data analyst targeting
quantitative and credit-risk roles. Built in the open, one commit at a time.

## License

MIT — see [`LICENSE`](LICENSE).
