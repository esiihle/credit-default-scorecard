# Roadmap — Credit Default Scorecard

> **This is the project timeline: from inception to presentation.** It is the
> centrepiece of the initial commit — a public commitment to a schedule and a
> definition of done for every phase.

## Planning assumptions

- **Inception:** 28 September 2026 (documentation committed 1 Sep; the build
  itself started 28 Sep and the schedule is baselined from that date).
- **Cadence:** one commit per working day — the commit history is the evidence.
- **Effort:** ~1 hour per day, ~6 days per week → **~6 focused hours/week**.
- **Honesty note on small sessions:** short daily blocks carry a reload cost;
  estimates include a modest buffer. This project is well-trodden and the primary
  dataset (German Credit) is small, so the risk here is *methodology precision*,
  not runtime.
- Dates are targets, not promises. The commit history is the real record.

## Milestone summary

| Phase | Focus | Week | Target window | Hours |
|---|---|---|---|---|
| 0 | Foundations & EDA | 1 | Sep 28 – Oct 4 | 6 |
| 1 | Target definition & split | 2 | Oct 5 – Oct 11 | 6 |
| 2 | Binning / monotonic classing | 3 | Oct 12 – Oct 18 | 6 |
| 3 | WOE / IV & feature selection | 4 | Oct 19 – Oct 25 | 6 |
| 4 | Logistic PD model | 5 | Oct 26 – Nov 1 | 6 |
| 5 | Validation (KS/Gini/AUC/PSI) | 6 | Nov 2 – Nov 8 | 6 |
| 6 | Score scaling & points table | 7 | Nov 9 – Nov 15 | 6 |
| 7 | Interactive scorer + benchmark | 8 | Nov 16 – Nov 22 | 6 |
| 8 | Docs & presentation | 9 | Nov 23 – Nov 29 | 6 |
| 9* | *Optional:* SAS parallel build | 10–11 | Nov 30 – Dec 13 | 12 |

**🎯 Target presentation date (Python): 27 November 2026.**
**🎯 With optional SAS build: 11 December 2026.**

---

## Phase detail

### Phase 0 — Foundations & EDA (Week 1 · Sep 28–Oct 4)
Scaffold the repo, environment, config; load German Credit; first EDA.
- **Deliverables:** repo live; `pip install` works; EDA notebook (distributions,
  missingness, class balance).
- **Done when:** the dataset loads reproducibly and its shape and target balance
  are documented.

### Phase 1 — Target definition & split (Week 2 · Oct 5–11)
Define good/bad, build train/test (+ out-of-time slice where feasible).
- **Deliverables:** `load.py`; a documented target definition; frozen splits.
- **Done when:** the good/bad rule is explicit and defensible, and splits are
  reproducible from a seed.

### Phase 2 — Binning / monotonic classing (Week 3 · Oct 12–18)
Fine then coarse classing; enforce monotonic risk trends; special/missing bins.
- **Deliverables:** `binning.py`; per-feature bin tables with bad rates.
- **Done when:** each selected feature bins into a small, monotonic, interpretable
  set of buckets.

### Phase 3 — WOE / IV & feature selection (Week 4 · Oct 19–25)
WOE transform; Information Value; drop weak and investigate suspiciously strong.
- **Deliverables:** `woe.py`; an IV ranking table; the selected feature set.
- **Done when:** every model input is WOE-encoded and justified by IV, with
  leakage checks on high-IV variables.

### Phase 4 — Logistic PD model (Week 5 · Oct 26–Nov 1)
Fit logistic regression on WOE; check signs, VIF, significance.
- **Deliverables:** `model.py`; coefficient table with diagnostics.
- **Done when:** all signs are intuitive, multicollinearity is controlled, and
  the fit is stable.

### Phase 5 — Validation (Week 6 · Nov 2–8)
KS, Gini/AUC, ROC, calibration, PSI, rank-ordering.
- **Deliverables:** `validate.py`; a validation report with charts.
- **Done when:** discrimination, calibration and stability are all reported on
  held-out data.

### Phase 6 — Score scaling & points table (Week 7 · Nov 9–15)
PDO/factor/offset transform → the points table.
- **Deliverables:** `scaling.py`; the scorecard points table.
- **Done when:** any log-odds maps to a score, and the points table is complete
  and readable.

### Phase 7 — Interactive scorer + benchmark (Week 8 · Nov 16–22)
Applicant scorer with reason codes; gradient-boosted benchmark and comparison.
- **Deliverables:** `scorecard.py`, `app/scorer.py`, `benchmark.py`; a
  scorecard-vs-GBM comparison table and write-up.
- **Done when:** the scorer returns PD + score + reason codes for any applicant,
  and the accuracy-vs-interpretability trade-off is quantified.

### Phase 8 — Docs & presentation (Week 9 · Nov 23–29)
Finalise README with real charts; complete `methodology.md`; build the deck.
- **Deliverables:** results-populated README; methodology write-up; slide deck
  (problem → method → results → trade-off → limitations).
- **Done when:** the project can be presented as fluently as it can be run.
  **← Python presentation milestone, target 27 Nov 2026.**

### Phase 9* — Optional SAS parallel build (Weeks 10–11 · Nov 30–Dec 13)
Reimplement the scorecard in SAS (`PROC LOGISTIC`, WOE/IV in data steps).
- **Deliverables:** `sas/scorecard.sas`; a note reconciling SAS vs. Python output.
- **Done when:** both languages produce a matching scorecard, demonstrating
  fluency in the SA bank toolchain. **← With-SAS milestone, target 11 Dec 2026.**

---

## Parallel context

This project runs alongside **V8 (Quantitative Football Model)** at a matching ~1
hour/day. This scorecard is the smaller build and reaches its presentation
milestone first (~27 Nov 2026); V8 follows (~18 Dec 2026). The optional SAS phase
runs after both presentations. Both are baselined from 28 September 2026.

## Risks & adjustments

- **German Credit is small**, so overfitting is the real danger — hence the
  emphasis on out-of-time validation and PSI rather than chasing AUC.
- **SAS access** may not materialise; Phase 9 is explicitly optional and never
  blocks the Python presentation milestone.
- If time compresses, the interactive scorer and the benchmark are the last
  things cut — but the validated scorecard and its honest metrics never are.
