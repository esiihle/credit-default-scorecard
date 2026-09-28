# Progress Log — Credit Default Scorecard

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus
> `docs/ROADMAP.md`) and you know exactly where the project stands. Newest entry
> at the top.

**Current phase:** Phase 1 — Target definition & split (next)
**Overall status:** 🟢 Phase 0 scaffolding built; pipeline runs end-to-end; tests green.
**Next action:** In `src/load.py` (the `PHASE 1` marker), recode the target
(raw 1 = good, 2 = bad -> 0/1) and build a stratified train/test split.

---

## Open decisions
- [ ] **SAS build (Phase 9):** in or out? Depends on SAS OnDemand access.
- [ ] **Real data:** download the full German Credit set (1,000 rows) and add a
      documented header-mapping step; the bundled 24-row sample stands in for now.
- [ ] **Good/bad definition:** confirmed as raw target (2 = bad = default) — recode in Phase 1.

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
- [ ] Phase 8 — Docs & presentation
- [ ] Phase 9 — (optional) SAS parallel build

## How to run (current state)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py     # loads sample, prints EDA report + stage status
python app/scorer.py               # placeholder until Phase 7
pytest                             # 4 smoke tests, all green
```

---

## Session log

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
