# Model Card: CreditShield Credit Default Predictor

**Model:** `catboost_champion` (CatBoost gradient boosting classifier) + deterministic per-age-group decision thresholds
**Version:** trained on the full data-to-serving pipeline in this repo (dbt feature engineering → CatBoost training → fairness audit and correction → FastAPI serving); MLflow registry `credit_default` v11, alias `champion`
**Status: CLEARED FOR PRODUCTION** — the underlying model has a known, documented, and now partially-fixed age-related bias (see below): a direct age input was found and removed, and the remaining proxy-driven bias is corrected at decision time via calibrated per-age-group thresholds live in `serving/app.py` and `serving/batch_predict.py`, not just present in an audit report. Gender plays no part in any decision.

---

## Intended use

Predicts probability of credit default within 24 months for a loan applicant, given application data, credit bureau history, previous-application history, and payment behavior. Intended to support (not replace) an underwriting decision, with SHAP-based adverse action reason codes generated for every denial or manual-review outcome, per ECOA requirements.

**Not intended for:** any deployment decision without first resolving the fairness finding below.

---

## Training data

- **Source:** Home Credit Default Risk (Kaggle), 307,511 applications across 7 relational tables
- **Target:** binary default within 24 months, 8.07% base rate
- **Split:** 80/20 stratified, `random_state=42`, on rows read in `sk_id_curr` order. Every number in this card comes from that fixed split. (Earlier versions read the table without an `ORDER BY`, so rebuilding the database silently changed the holdout set; the same data gave AUC 0.7215 in one row order and 0.7169 in another.)
- **4 rows excluded:** `code_gender='XNA'` (Home Credit's placeholder for undisclosed gender, not a real category — excluded before any fairness metric was computed)
- **Features:** 34 engineered features from `features.fct_credit_features` (dbt), covering application terms, bureau history, previous-application outcomes, installment behavior, credit card utilization, and POS/cash loan history. Protected attributes (`code_gender`, `age_years`, `name_family_status`) are carried for audit but excluded from the model's feature set — **this exclusion was not actually enforced in code until the fix described below** (see Fairness section); `age_years` and an age-gated interaction feature were both raw model inputs before that fix.

---

## Performance (holdout)

| Metric | Value |
|---|---|
| AUC-ROC | 0.7204 |
| Gini | 0.4409 |
| KS statistic | 0.3236 |
| Top-decile lift | 2.98x |

Selected from a 5-model comparison (CatBoost, LightGBM, XGBoost, Random Forest, Logistic Regression) — see `experiments/tradeoffs.md` for the full table and reasoning. Beats the WoE-encoded logistic regression baseline (AUC 0.696) by 2.4 points.

---

## Fairness audit and mitigation

The project's policy (`explainability/fairness.py`, `DEMOGRAPHIC_PARITY_THRESHOLD = 0.05`) requires every audited protected attribute to pass before a model may ship, matching the reasoning in `experiments/tradeoffs.md` Tradeoff 1: a model that can't legally deploy has zero business value regardless of its AUC.

- **Gender:** demographic parity gap ≤ 0.05.
- **Age (Reg B-aware):** demographic parity gap ≤ 0.05 across the under-62 groups (18–29, 30–44, 45–61), **and** the 62+ group must not be approved less often than any other group (within 2 standard errors of sampling noise). Regulation B (12 CFR 1002.6(b)(2)) lets a creditor favor applicants 62 or older but never disadvantage them because of age, so a gap that comes only from favoring 62+ is allowed; the all-ages gap is still reported below.

### The root cause, found late: age was a raw model input the entire time

Every version of this document, until now, claimed `age_years` was excluded from the model's feature set. That claim was false — it was directly in `FEATURE_COLS`, alongside a second feature (`many_loans_x_young`) explicitly defined as `bureau_loan_count > 5 AND age_years < 30`, an age-gated rule, not just an age-correlated signal. Found by finally checking `corrwith()` against the protected attribute instead of only re-running downstream fairness metrics. Both features are now removed; the numbers below reflect the model **without** direct age access.

### The raw model: gender passes, age fails

At a flat 0.5 cutoff, no correction:

| | Demographic parity gap | Verdict |
|---|---|---|
| Gender | 0.032 | Pass (equalized odds 0.023, four-fifths ratio 0.953) |
| Age, under-62 groups | **0.289** | **Fail** |
| Age, all four groups | 0.427 | (reported) |

Raw approval rates by age group (holdout; 4,800+ applicants in the smallest band — not a small-sample artifact):

| Age group | Approval rate |
|---|---|
| 18–29 | 46.5% |
| 30–44 | 64.4% |
| 45–61 | 75.4% |
| 62+ | 89.2% |

**Remaining mechanism:** younger applicants still mechanically have shorter credit histories (`years_employed`, `bureau_loan_count`, prior-application counts are all lower simply because they've had less time to accumulate them), and the model still learns that as risk — this is legitimate proxy discrimination, distinct from (and smaller than) the direct-input problem that's now fixed. Age is an ECOA-protected class; "the model found a real correlation" is not a defense.

### Mitigation: deterministic per-age-group decision thresholds, live in production

Rather than Fairlearn's `ThresholdOptimizer` (which uses a *randomized* interpolated decision for some applicants near the boundary — a compliance problem in its own right, since identical applicants in the same group could get different outcomes by chance), this project calibrates a **deterministic** DENY threshold per age group: each group's cutoff is set at that group's own quantile matching the portfolio-wide target denial rate. One override: **the 62+ group gets the most lenient cutoff of any group.** Equal-rate calibration alone would give 62+ the *strictest* cutoff (0.36), because the model scores older applicants as lower risk, and Reg B doesn't allow that.

| Age group | DENY threshold |
|---|---|
| 18–29 | 0.602 |
| 30–44 | 0.521 |
| 45–61 | 0.452 |
| 62+ | 0.602 (most lenient, by rule) |

Calibrated on the training split, validated on held-out test data:

| | Before (flat 0.5) | After (held-out test) |
|---|---|---|
| DP gap (gender) | 0.032 | **0.017** |
| DP gap (age, under 62) | 0.289 | **0.0049** |
| 62+ not disadvantaged | Yes | **Yes** |
| DP gap (age, all four groups) | 0.427 | 0.289 (62+ favored on purpose) |

Final approval rates: 67.4% (18–29), 67.9% (30–44), 67.4% (45–61), 96.3% (62+); 70.4% for women and 68.7% for men. The guarantee for applicants 62+ is the cutoff rule, enforced in code and by a unit test; the approval-rate check in the gate is only a backstop. Without the rule, the 62+ approval rate would land 1.4 points below the other groups, inside the 2-standard-error margin, so the outcome check alone would not have caught it. Every gated number is under the 0.05 threshold, on data the calibration never saw. **This is live** in both `serving/app.py` (real-time API) and `serving/batch_predict.py` (portfolio scoring) — both look up the applicant's age group and apply that group's calibrated DENY threshold instead of a flat 0.50 cutoff, and tests check that the two always agree.

**Gender is not used anywhere in the decision.** An earlier version keyed the thresholds on (gender, age group), with gender collected only as a lookup key. That was removed: setting a different cutoff by sex is disparate treatment under Reg B, whatever the intent. Gender is still audited offline, and the gender gap passes without it (0.017). The API no longer accepts a gender field, and a test checks that sending one never changes a decision. The same change fixed a fallback bug: the threshold for an unseen group was documented as the most lenient but was actually the strictest (DENY is `probability >= threshold`, and the code took the minimum).

### Also tried: baking fairness into the model itself

Retrained the same CatBoost architecture with Fairlearn's `ExponentiatedGradient` (demographic parity constraint, gender × age group jointly — gender is used here only as a training constraint, and this candidate isn't deployed), evaluated at a flat 0.5 threshold with **no** post-hoc correction:

| | AUC | DP gap (age) | Passes on its own? |
|---|---|---|---|
| Constrained, `age_years` still in model, reduced budget | 0.683 (cost 0.041) | 0.069 | No |
| Constrained, `age_years` still in model, full budget (26 min) | 0.687 (cost 0.014) | 0.093 (worse) | No |
| Constrained, `age_years` removed, reduced budget | 0.698 (cost 0.023) | 0.038 | Yes |
| **Current: same, on the fixed split, Reg B-aware gate** | **0.699 (cost 0.021)** | **0.042 under-62** (0.069 all ages); 62+ not disadvantaged | **Yes** |

The first three rows are historical (older age bins, all-ages gate, unordered split). Removing the direct age input is what made the constrained model pass: the model had an unconstrained channel to discriminate on age that no amount of `ExponentiatedGradient` tuning on the *output* was fully closing.

**Still not deployed, on principle:** 0.021 AUC cost is just above this project's pre-set 0.02 acceptance rule for swapping the constrained model in over post-hoc correction — by 0.001. `models/train_champion.py` applies that rule automatically and deployed the raw model + thresholds instead, which passes more cleanly (0.0049 under-62 gap) at no AUC cost. The rule was set before this result existed, specifically so a close call afterward doesn't get rationalized into a pass — revisiting whether 0.02 is the right number is a separate, deliberate decision, not something to relitigate because one result landed just outside it.

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
- The underlying model retains some age bias via legitimate proxy correlation (credit history length); fairness is achieved via post-hoc threshold correction on top, not a fully bias-free model — see Fairness section. A constrained-retraining candidate passes the fairness gate on its own, but at slightly more AUC cost than this project's acceptance rule allows.
- Applicants 62+ are approved at 96.3% vs. about 67–68% for every other age group. That's the deliberate result of never giving them a stricter cutoff than anyone else; it's allowed under Reg B, but it means this policy approves some higher-risk older applicants that a pure equal-rate policy would deny.
- The age-based thresholds are a fairness *remediation* that itself uses age. Reg B permits age in an empirically derived system as long as applicants 62+ aren't disadvantaged, which this design guarantees, but a real deployment would need compliance review of the approach.
- H2O AutoML doesn't beat this champion — its best model (GBM, AUC 0.701, real run with the memory constraint fixed; measured before the fixed-order split) trails CatBoost by about 2 points, consistent with every other tree model tried in Step 4
- Raw-feature logistic regression in the Step 4 comparison table (AUC 0.56) is not a fair LR baseline — see the WoE-encoded LR (AUC 0.70) instead
- No temporal/economic-cycle validation — training data has no calendar dates (see `data_engineering/ingest_fred.py` for the planned-but-unbuilt external data integration pattern)
