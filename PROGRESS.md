# Progress Log — Credit Default Scorecard

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus
> `docs/ROADMAP.md`) and you know exactly where the project stands. Newest entry
> at the top.

**Current phase:** Phase 1 — Target definition & split (Oct 5 – Oct 11), started early
**Overall status:** 🟢 Pipeline runs end-to-end on the **real** 1,000-row UCI file;
tests green (7); exploration notebook written and run.
**Next action:** Run the exploration notebook and fill in its findings table
(carried over from 1 Oct), then review the Phase 1 data-quality report and the
split manifest.
**Schedule:** baselined from 28 Sep 2026; target presentation **27 Nov 2026**
(Python), 11 Dec 2026 with the optional SAS build.
**Cadence:** one commit per working day.

---

## Open decisions
- [ ] **SAS build (Phase 9):** in or out? Depends on SAS OnDemand access.
- [x] **Real data:** done 30 Sep 2026 — UCI `german.data` (1,000 rows) loads via
      `src/load.py`, with the column order and A-code table applied at load time.
- [x] **Good/bad definition:** done 2 Oct 2026 — 1 = default (source code 2),
      0 = good. Documented in methodology §1.4 with its three limitations
      (no delinquency depth, accepted-applicants only, direction consistency).
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
pytest                             # 20 tests (smoke + Phase 1), all green
python scripts/run_pipeline.py --no-save  # run the checks without writing splits
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

### 2026-10-02 — Phase 1: target definition, validation, frozen split
- `src/load.py`: Phase 1 flow — `validate_raw` → `clean` → `define_target` →
  `to_analysis_table` → `split_train_test` → `save_splits`, orchestrated by
  `run_phase1`.
  - Structural faults raise `DataQualityError`: missing column, a target code
    that isn't 1 or 2, or a target with only one class.
  - Cleaning: numeric coercion, plausibility ranges for age / term / amount
    (all in `config.yaml`), exact duplicate rows dropped, every count reported.
  - `define_target`: **1 = default**, with the definition and its three
    limitations written into the docstring and methodology §1.4.
  - `applicant_id` added (APP_0001 …), since the source has no identifier.
  - 70/30 stratified split from the seed, written to `data/processed/` with
    `split_manifest.json` — seed, counts, bad rate per half, and a **fingerprint**
    hash of each half's IDs, so reproducibility is checkable, not asserted.
  - Guard rail: dropping more than `validation.max_dropped_fraction` (5%) of rows
    stops the run.
- `config.yaml`: new `validation` block; `split` block extended with
  `out_of_time: false` and the reason (no date column in this dataset).
- `scripts/run_pipeline.py`: runs Phase 1 and prints the quality report plus the
  frozen-split summary (`--no-save` to skip writing).
- `tests/test_load.py` (new): 13 behaviour tests — missing column, bad target
  code, single-class target, implausible age, duplicate rows, **recode
  direction**, split sizes, stratification, no train/test overlap, same-seed
  reproducibility, different-seed difference, end to end, and the drop-fraction
  guard. Suite now 20 green.
- **Not yet done:** the exploration notebook from 1 Oct still needs running and
  its findings table filling in.
- **Next:** Phase 2 — binning and monotonic coarse classing.

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
