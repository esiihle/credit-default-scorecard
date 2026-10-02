# Progress Log — Credit Default Scorecard

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus
> `docs/ROADMAP.md`) and you know exactly where the project stands. Newest entry
> at the top.

**Current phase:** Phase 0 complete (Sep 28 – Oct 4); Phase 1 opens 5 Oct
**Overall status:** 🟢 Pipeline runs end-to-end on the **real** 1,000-row UCI file;
tests green (7); exploration notebook written and run.
**Next action:** In `src/load.py` (the `PHASE 1` marker), recode the target
(raw 1 = good, 2 = bad -> 0/1) and build a stratified train/test split.
**Schedule:** baselined from 28 Sep 2026; target presentation **27 Nov 2026**
(Python), 11 Dec 2026 with the optional SAS build.
**Cadence:** one commit per working day.

---

## Open decisions
- [ ] **SAS build (Phase 9):** in or out? Depends on SAS OnDemand access.
- [x] **Real data:** done 30 Sep 2026 — UCI `german.data` (1,000 rows) loads via
      `src/load.py`, with the column order and A-code table applied at load time.
- [ ] **Good/bad definition:** confirmed as raw target (2 = bad = default) — recode in Phase 1.
- [ ] **`personal_status_sex`:** encodes sex, a protected attribute. Decide in
      Phase 3 whether it enters the model, and record the reasoning — a panel
      will ask.

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
python scripts/run_pipeline.py     # real data from data/raw if present, else the sample
python scripts/run_pipeline.py --sample   # force the committed sample
python scripts/make_sample.py      # rebuild the committed sample from the real file
python app/scorer.py               # placeholder until Phase 7
pytest                             # 7 smoke tests, all green
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

### 2026-10-01 — Exploration notebook
- Added `notebooks/01_exploration.ipynb`, importing from `src/` rather than
  duplicating loader logic; runs on the real file when present, the sample
  otherwise.
- Sections, each tied to the phase it justifies: target balance and good:bad
  odds (metric choice, Phase 5; reference odds, Phase 6); bad rate by level for
  every categorical (the raw material of WOE, Phase 3); **WOE and IV worked by
  hand for `checking_status`** as a fixture for Phase 3's code to reproduce; bad
  rate by quantile bin for every numeric, flagging non-monotonic features
  (Phase 2); data quality — duplicates, nulls, levels under 5% of book
  (`binning.min_bin_fraction`); bad rate by `personal_status_sex` with the
  fair-lending questions to answer in Phase 3.
- Added `ipykernel` to requirements.txt.

#### Findings from the real data — FILL IN after running the notebook
| Measurement | Value | Note |
|---|---|---|
| Overall bad rate | | expect ~30% on German Credit |
| Good:bad odds | | compare with `scaling.base_odds` (50) in config |
| Top 3 categorical features by bad-rate spread | | |
| Hand-computed IV, `checking_status` | | Phase 3 must reproduce this exactly |
| Numeric features NOT monotonic | | these are the Phase 2 to-do list |
| Thin levels (<5% of book) | | each needs merging in Phase 2 |

- **Next:** Phase 1 (opens 5 Oct) — target recode + stratified split.

### 2026-09-30 — Real data loading + fresh-clone fix + schedule re-baseline
- **Bug found and fixed:** `.gitignore` had `data/*`, which silently excluded the
  bundled sample from the repo. A fresh clone crashed on `run_pipeline.py` and
  2 of 4 tests failed. Now only `data/raw/` and `data/processed/` are ignored;
  `data/sample/` is committed on purpose.
- Removed the leftover dotless `gitignore` file (it was never doing anything).
- Settled the data source: UCI Statlog German Credit (id 144), `german.data`.
- `src/load.py`: added `RAW_COLUMN_ORDER` (the file has no header),
  `VALUE_LABELS` (the full A-code table), `decode_values()` — which raises on an
  unknown code instead of letting it become a junk bin — and
  `load_german_raw()`. The target is left raw (1/2); recoding belongs to Phase 1.
- `scripts/make_sample.py`: rebuilds the committed 24-row sample from the real
  file, stratified on the target with a fixed seed, so the sample is real data
  and anyone can regenerate it.
- `scripts/run_pipeline.py`: uses the real file when present, else the sample,
  printing the download link (`--sample` forces the sample).
- `docs/ROADMAP.md` re-baselined from 28 Sep; Python presentation now **27 Nov
  2026**. `docs/methodology.md` §1 filled in: source, the disputed code table
  (Grömping 2019), selection bias, vintage, and the protected-attribute flag.
- Tests: 7 green (added column-order/config drift, code decoding, unknown-code).
- **Next:** Phase 1 — target recode + stratified split.

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
