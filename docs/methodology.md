# Methodology — Credit Default Scorecard

> **Living document.** This is the mathematical spine of the scorecard. It is
> scaffolded now and filled in as each phase lands (see `ROADMAP.md`). Each
> section carries a **Status** tag so the reader always knows what is settled and
> what is pending. Metrics are reported only from held-out validation.

---

## 1. Data & target definition
**Status: 🟡 skeleton — finalised in Phase 1.**

- Primary dataset: **German Credit** (1,000 applicants, 20 attributes, binary
  good/bad outcome).
- The **good/bad target** and its rationale are stated here once fixed. Records
  are split into train / test, with an out-of-time slice where the data allows,
  to test stability rather than only fit.
- Notation: for a bin $b$ of a feature, let $g_b, n_b$ be the counts of goods and
  bads; $G, N$ the totals. Distributions $\%good_b = g_b / G$,
  $\%bad_b = n_b / N$.

## 2. Binning / classing
**Status: 🟡 skeleton — built in Phase 2.**

Each feature is **fine-classed** (many bins) then **coarse-classed** into a small
number of bins chosen so the bad rate is **monotonic** across the ordered bins.
Continuous and categorical features are handled, with explicit special bins for
missing / sentinel values. Monotonicity is what makes the final points table
stable and defensible to a regulator.

## 3. Weight of Evidence & Information Value
**Status: 🟡 skeleton — built in Phase 3.**

### 3.1 Weight of Evidence
Each bin is recoded to its log-odds of being good vs. bad:

$$\text{WOE}_b = \ln\!\left(\frac{\%good_b}{\%bad_b}\right)$$

WOE puts every feature on one monotonic, model-friendly scale and is what lets a
logistic model translate cleanly into a points table.

### 3.2 Information Value
Feature-level predictive strength:

$$\text{IV} = \sum_b \left(\%good_b - \%bad_b\right)\,\text{WOE}_b$$

Selection guide: $<0.02$ drop; $0.02\text{–}0.1$ weak; $0.1\text{–}0.3$ medium;
$0.3\text{–}0.5$ strong; $>0.5$ **suspiciously strong — check for leakage** before
trusting it.

## 4. PD model
**Status: 🟡 skeleton — built in Phase 4.**

Logistic regression on the WOE-transformed features:

$$\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{k} \beta_k \,\text{WOE}_k$$

where $p$ is the probability of default. Because inputs are WOE, the model is
linear in the log-odds — the property that makes §6 possible. Diagnostics:
coefficient **signs** must be intuitive, **VIF** controls multicollinearity, and
each variable must be significant and stable.

## 5. Validation
**Status: 🟡 skeleton — built in Phase 5.**

- **KS statistic** — maximum gap between the cumulative good and bad score
  distributions: $\text{KS} = \max_s \left| F_{good}(s) - F_{bad}(s) \right|$.
- **Gini / AUC** — rank-ordering power, with $\text{Gini} = 2\,\text{AUC} - 1$.
- **Calibration** — predicted PD vs. realised default rate (reliability curve).
- **PSI (Population Stability Index)** — drift between development and validation:
  $\text{PSI} = \sum_b (\%A_b - \%E_b)\,\ln(\%A_b / \%E_b)$, read with the usual
  $<0.1$ stable / $0.1\text{–}0.25$ shift / $>0.25$ major-shift bands.
- **Rank-ordering** — monotonic bad rate across score bands.

## 6. Score scaling
**Status: 🟡 skeleton — built in Phase 6.**

Model log-odds are mapped to a human-readable score with the standard
**PDO / factor / offset** transform:

$$\text{Score} = \text{Offset} + \text{Factor}\times \ln(\text{odds}),
\qquad \text{Factor} = \frac{\text{PDO}}{\ln 2}$$

where **PDO** is the points needed to double the odds. Distributing across the $n$
model variables gives the **points per bin**:

$$\text{Points}_k = -\left(\beta_k \,\text{WOE}_k + \frac{\beta_0}{n}\right)\times \text{Factor} + \frac{\text{Offset}}{n}$$

The full set of per-bin points **is the scorecard**.

## 7. Interactive scorer & reason codes
**Status: 🟡 skeleton — built in Phase 7.**

Given an applicant, the scorer returns their PD, their scaled score, and **reason
codes** — the bins contributing the largest negative points relative to a
reference (the adverse-action / explainability output a bank must provide). Built
to cover many applicant scenarios interactively, not a single demo row.

## 8. Benchmark & the trade-off
**Status: 🟡 skeleton — built in Phase 8.**

A gradient-boosted model (e.g. LightGBM) is trained on the same data and compared
to the scorecard on KS / Gini / AUC. The expected pattern — the GBM edges ahead on
raw discrimination while losing on interpretability, stability and auditability —
is quantified and discussed. This section is the core interview talking point:
*why a bank knowingly accepts a small accuracy cost for an explainable model.*

## References
- Siddiqi, N. (2017). *Intelligent Credit Scoring.* Wiley.
- Thomas, L., Crook, J. & Edelman, D. (2017). *Credit Scoring and Its
  Applications.* SIAM.
- Additional references added as the methodology is written up.
