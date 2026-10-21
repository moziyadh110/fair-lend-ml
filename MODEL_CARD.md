# Model Card: CreditShield Credit Default Predictor

**Model:** `catboost_champion` (CatBoost gradient boosting classifier) + deterministic per-group decision thresholds
**Version:** trained on the full data-to-serving pipeline in this repo (dbt feature engineering → CatBoost training → fairness audit and correction → FastAPI serving)
**Status: CLEARED FOR PRODUCTION** — the underlying model has a known, documented, and now partially-fixed age-related bias (see below): a direct age input was found and removed, and the remaining proxy-driven bias is corrected at decision time via calibrated per-group thresholds live in `serving/app.py` and `serving/batch_predict.py`, not just present in an audit report.

---

## Intended use

Predicts probability of credit default within 24 months for a loan applicant, given application data, credit bureau history, previous-application history, and payment behavior. Intended to support (not replace) an underwriting decision, with SHAP-based adverse action reason codes generated for every denial or manual-review outcome, per ECOA requirements.

**Not intended for:** any deployment decision without first resolving the fairness finding below.

---

## Training data

- **Source:** Home Credit Default Risk (Kaggle), 307,511 applications across 7 relational tables
- **Target:** binary default within 24 months, 8.07% base rate
- **Split:** 80/20 stratified, `random_state=42`
- **4 rows excluded:** `code_gender='XNA'` (Home Credit's placeholder for undisclosed gender, not a real category — excluded before any fairness metric was computed)
- **Features:** 34 engineered features from `features.fct_credit_features` (dbt), covering application terms, bureau history, previous-application outcomes, installment behavior, credit card utilization, and POS/cash loan history. Protected attributes (`code_gender`, `age_years`, `name_family_status`) are carried for audit but excluded from the model's feature set — **this exclusion was not actually enforced in code until the fix described below** (see Fairness section); `age_years` and an age-gated interaction feature were both raw model inputs before that fix.

---

## Performance (holdout)

| Metric | Value |
|---|---|
| AUC-ROC | 0.7215 |
| Gini | 0.4430 |
| KS statistic | 0.3240 |
| Top-decile lift | 3.02x |

Selected from a 5-model comparison (CatBoost, LightGBM, XGBoost, Random Forest, Logistic Regression) — see `experiments/tradeoffs.md` for the full table and reasoning. Beats the WoE-encoded logistic regression baseline (AUC 0.696) by 2.6 points. (Both numbers are ~0.3-0.5 points lower than earlier in this project's history — the cost of removing `age_years` as a direct model input, in exchange for a real fairness improvement; see below.)

---

## Fairness audit and mitigation

The project's policy (`explainability/fairness.py`, `DEMOGRAPHIC_PARITY_THRESHOLD = 0.05`) requires demographic parity gap ≤ 0.05 across **all** audited protected attributes before a model may ship, matching the reasoning in `experiments/tradeoffs.md` Tradeoff 1: a model that can't legally deploy has zero business value regardless of its AUC.

### The root cause, found late: age was a raw model input the entire time

Every version of this document, until now, claimed `age_years` was excluded from the model's feature set. That claim was false — it was directly in `FEATURE_COLS`, alongside a second feature (`many_loans_x_young`) explicitly defined as `bureau_loan_count > 5 AND age_years < 30`, an age-gated rule, not just an age-correlated signal. Found by finally checking `corrwith()` against the protected attribute instead of only re-running downstream fairness metrics. Both features are now removed; the numbers below reflect the model **without** direct age access.

### The raw model: gender passes, age still fails, but far less badly

| | Demographic parity gap | Verdict |
|---|---|---|
| Gender | 0.031 | Pass (equalized odds 0.021, four-fifths ratio 0.955) |
| Age | **0.408** (was 0.456 with `age_years` still in the model) | **Fail — still above threshold, but the direct-input removal alone closed part of the gap** |

Raw approval rates by age group (holdout, 7,000+ applicants per band — not a small-sample artifact):

| Age group | Approval rate |
|---|---|
| 18–30 | 46.6% |
| 30–45 | 64.6% |
| 45–60 | 74.5% |
| 60+ | 87.5% |

**Remaining mechanism:** younger applicants still mechanically have shorter credit histories (`years_employed`, `bureau_loan_count`, prior-application counts are all lower simply because they've had less time to accumulate them), and the model still learns that as risk — this is legitimate proxy discrimination, distinct from (and smaller than) the direct-input problem that's now fixed. Age is an ECOA-protected class; "the model found a real correlation" is not a defense.

### Mitigation: deterministic per-group decision thresholds, live in production

Rather than Fairlearn's `ThresholdOptimizer` (which uses a *randomized* interpolated decision for some applicants near the boundary — a compliance problem in its own right, since identical applicants in the same group could get different outcomes by chance), this project calibrates a **deterministic** threshold per (gender, age-group) combination: each group's DENY cutoff is set at that group's own quantile matching the portfolio-wide target denial rate. Calibrated on the training split, validated on held-out test data:

| | Before | After (held-out test) |
|---|---|---|
| DP gap (gender) | 0.031 | 0.0019 |
| DP gap (age) | 0.408 | 0.0128 |

Both under the 0.05 threshold, on data the calibration never saw. **This is live** in both `serving/app.py` (real-time API) and `serving/batch_predict.py` (portfolio scoring) — both look up the applicant's (gender, age-bin) group and apply that group's calibrated DENY threshold instead of a flat 0.50 cutoff.

**Applicant gender is collected for this purpose only** — it is never a model input feature, only a threshold lookup key. Applicants who decline to disclose fall back to the *most lenient* calibrated group threshold, not the raw model's own (biased) threshold. Collecting protected-class data specifically for fairness monitoring/remediation, while excluding it from the underlying prediction, is standard, legally recognized practice under ECOA/Reg B.

### Also tried: baking fairness into the model itself — and once the direct age input was gone, it worked

Retrained the same CatBoost architecture with Fairlearn's `ExponentiatedGradient` (demographic parity constraint, gender × age-bin jointly), evaluated at a flat 0.5 threshold with **no** post-hoc correction. Three attempts across this project's history:

| | AUC | DP gap (age) | Passes on its own? |
|---|---|---|---|
| Constrained, `age_years` still in model, reduced budget | 0.683 (cost 0.041) | 0.069 | No |
| Constrained, `age_years` still in model, full budget (26 min) | 0.687 (cost 0.014) | 0.093 (worse) | No |
| **Constrained, `age_years` removed, reduced budget** | 0.698 (cost **0.023**) | **0.038** | **Yes** |

The first passing result, on the third attempt — not because training ran longer, but because the direct age input was removed first. This is meaningful evidence that the earlier failures weren't purely a budget/convergence problem: the model had an unconstrained channel to discriminate on age that no amount of `ExponentiatedGradient` tuning on the *output* was fully closing.

**Still not deployed, on principle:** 0.023 AUC cost is just above this project's pre-set 0.02 acceptance rule for swapping the constrained model in over post-hoc correction. `models/train_champion.py` applied that rule automatically and deployed the raw model + thresholds instead, which passes even more cleanly (DP gap 0.0019 gender / 0.0128 age) at no AUC cost. The rule was set before this result existed, specifically so a close call afterward doesn't get rationalized into a pass — revisiting whether 0.02 is the right number is a separate, deliberate decision, not something to relitigate because one result landed just outside it.

**What's still open:** the deployed model still corrects the *decision*, not the *model* — some age-related bias remains in the raw CatBoost via legitimate proxy correlation (credit history length), which is a harder problem than a direct input and wasn't expected to fully resolve here.

Full investigation, methodology, and reasoning: see commit history for the fairness investigation (`explainability/fairness.py`, `experiments/full_fairness_retrain.py`, `models/train_champion.py`).

---

## Performance (measured, not aspirational)

`serving/load_test.py`, run against a live instance — the first actual measurement in this project's history, replacing earlier unverified latency claims:

| | Latency |
|---|---|
| Cold start, before the warm-up fix (first request after process start) | ~2.1-2.4s, one-time |
| Steady-state, uncached | ~85-300ms, mean 134ms |
| Cached (identical repeat application) | ~15-20ms |

134ms is comfortably within a real-time decision's UX budget. The ~2.3s cold-start cost is now absorbed at startup — `serving/app.py` fires one dummy prediction through the full model + SHAP explainer path before accepting real traffic, so no real applicant pays it.

## Explainability

- Global + local SHAP (`explainability/shap_explainer.py`)
- Per-prediction ECOA adverse action reason codes for every DENY/MANUAL_REVIEW decision (`explainability/adverse_action.py`)

## Known limitations

- **This fairness audit covers gender and age only — it has zero coverage of race or ethnicity**, arguably the attribute regulators scrutinize most under ECOA/HMDA. Not an oversight: this dataset has no surname, no nationality, and no real geography (region fields are anonymized relative-population codes, not resolvable to a place), which rules out BISG — the standard proxy method lenders and regulators actually use when race/ethnicity isn't self-reported — not just makes it harder. Closing this gap would need a different or supplementary dataset (e.g. HMDA, which self-reports race/ethnicity directly), not more engineering on this codebase.
- The underlying model retains some age bias via legitimate proxy correlation (credit history length); fairness is achieved via post-hoc threshold correction on top, not a fully bias-free model — see Fairness section. A constrained-retraining candidate now passes the fairness gate on its own, but at slightly more AUC cost than this project's acceptance rule allows.
- H2O AutoML doesn't beat this champion — its best model (GBM, AUC 0.701, real run with the memory constraint fixed) trails CatBoost by 2 points, consistent with every other tree model tried in Step 4
- Raw-feature logistic regression in the Step 4 comparison table (AUC 0.56) is not a fair LR baseline — see the WoE-encoded LR (AUC 0.70) instead
- No temporal/economic-cycle validation — training data has no calendar dates (see `data_engineering/ingest_fred.py` for the planned-but-unbuilt external data integration pattern)
