# Progress Log — Credit Default Scorecard

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus `ROADMAP.md`)
> and you know exactly where the project stands. Newest entry at the top.

**Current phase:** Phase 0 — Foundations & EDA
**Overall status:** 🟢 Documentation committed; build not yet started.
**Next action:** Scaffold Phase 0 (requirements.txt, config.yaml, `src/` stubs,
data loader, EDA notebook).

---

## Open decisions
- [ ] **SAS build (Phase 9):** in or out? Depends on SAS OnDemand access.
- [ ] **Dataset scope:** German Credit confirmed as primary; Home Credit as a
      later stretch — confirm we're not touching it until Python build is done.
- [ ] **Good/bad definition:** to be fixed in Phase 1.

## Phase checklist
- [x] Documentation (OVERVIEW, README, ROADMAP, methodology, LICENSE, .gitignore)
- [ ] Phase 0 — Foundations & EDA
- [ ] Phase 1 — Target definition & split
- [ ] Phase 2 — Binning / monotonic classing
- [ ] Phase 3 — WOE / IV & feature selection
- [ ] Phase 4 — Logistic PD model
- [ ] Phase 5 — Validation (KS/Gini/AUC/PSI)
- [ ] Phase 6 — Score scaling & points table
- [ ] Phase 7 — Interactive scorer + benchmark
- [ ] Phase 8 — Docs & presentation
- [ ] Phase 9 — (optional) SAS parallel build

---

## Session log

### 2026-09-01 — Project inception
- Created repo and committed all documentation.
- Defined the full pipeline, roadmap, and methodology skeleton.
- **Next:** begin Phase 0 scaffolding.

<!--
Template for each new entry (copy this, newest at the top):

### YYYY-MM-DD — <short title>
- What I did this session:
- Decisions made:
- Anything broken / to revisit:
- Next action:
-->
