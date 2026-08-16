# Model Card: CreditShield Credit Default Predictor

**Model:** `catboost_champion` (CatBoost gradient boosting classifier) + deterministic per-group decision thresholds
**Version:** trained on the full pipeline described in `PROGRESS.md`
**Status: CLEARED FOR PRODUCTION** (as of the fairness mitigation in `PROGRESS.md` Stage 15) — the underlying model has a known, documented age-related bias (see below), which is corrected at decision time via calibrated per-group thresholds that are live in `serving/app.py` and `serving/batch_predict.py`, not just present in an audit report.

---

## Intended use

Predicts probability of credit default within 24 months for a loan applicant, given application data, credit bureau history, previous-application history, and payment behavior. Intended to support (not replace) an underwriting decision, with SHAP-based adverse action reason codes generated for every denial or manual-review outcome, per ECOA requirements.

**Not intended for:** any deployment decision without first resolving the fairness finding below.

---

## Training data

- **Source:** Home Credit Default Risk (Kaggle), 307,511 applications across 7 relational tables
- **Target:** binary default within 24 months, 8.07% base rate
- **Split:** 80/20 stratified, `random_state=42`
- **4 rows excluded:** `code_gender='XNA'` (Home Credit's placeholder for undisclosed gender, not a real category — see `PROGRESS.md` Stage 11)
- **Features:** 36 engineered features from `features.fct_credit_features` (dbt), covering application terms, bureau history, previous-application outcomes, installment behavior, credit card utilization, and POS/cash loan history. Protected attributes (`code_gender`, `age_years`, `name_family_status`) are carried for audit but excluded from the model's feature set.

---

## Performance (holdout)

| Metric | Value |
|---|---|
| AUC-ROC | 0.7244 |
| Gini | 0.4488 |
| KS statistic | 0.3330 |
| Top-decile lift | 3.04x |

Selected from a 5-model comparison (CatBoost, LightGBM, XGBoost, Random Forest, Logistic Regression) — see `experiments/tradeoffs.md` for the full table and reasoning. Beats the WoE-encoded logistic regression baseline (AUC 0.701) by 2.3 points.

---

## Fairness audit and mitigation

The project's policy (`explainability/fairness.py`, `DEMOGRAPHIC_PARITY_THRESHOLD = 0.05`) requires demographic parity gap ≤ 0.05 across **all** audited protected attributes before a model may ship, matching the reasoning in `experiments/tradeoffs.md` Tradeoff 1: a model that can't legally deploy has zero business value regardless of its AUC.

### The raw model: gender passes, age fails badly

| | Demographic parity gap | Verdict |
|---|---|---|
| Gender | 0.029 | Pass (equalized odds 0.027, four-fifths ratio 0.958) |
| Age | **0.456** | **Fail — 9x the threshold** |

Raw approval rates by age group (holdout, 7,000+ applicants per band — not a small-sample artifact):

| Age group | Approval rate |
|---|---|
| 18–30 | 43.4% |
| 30–45 | 61.8% |
| 45–60 | 78.5% |
| 60+ | 89.0% |

**Likely mechanism:** younger applicants mechanically have shorter credit histories (`years_employed`, `bureau_loan_count`, prior-application counts are all lower simply because they've had less time to accumulate them), and the model learned that as risk. Age is an ECOA-protected class; "the model found a real correlation" is not a defense against proxy discrimination.

### Mitigation: deterministic per-group decision thresholds, live in production

Rather than Fairlearn's `ThresholdOptimizer` (which uses a *randomized* interpolated decision for some applicants near the boundary — a compliance problem in its own right, since identical applicants in the same group could get different outcomes by chance), this project calibrates a **deterministic** threshold per (gender, age-group) combination: each group's DENY cutoff is set at that group's own quantile matching the portfolio-wide target denial rate. Calibrated on the training split, validated on held-out test data:

| | Before | After (held-out test) |
|---|---|---|
| DP gap (gender) | 0.029 | 0.0015 |
| DP gap (age) | 0.456 | 0.0102 |

Both under the 0.05 threshold, on data the calibration never saw. **This is live** in both `serving/app.py` (real-time API) and `serving/batch_predict.py` (portfolio scoring) — both look up the applicant's (gender, age-bin) group and apply that group's calibrated DENY threshold instead of a flat 0.50 cutoff. Verified against the full 307,511-applicant portfolio: post-mitigation non-denial rates across age groups are 67.8%–68.0%, down from the original 43%–89% spread.

**Applicant gender is collected for this purpose only** — it is never a model input feature, only a threshold lookup key. Applicants who decline to disclose fall back to the flat default threshold. Collecting protected-class data specifically for fairness monitoring/remediation, while excluding it from the underlying prediction, is standard, legally recognized practice under ECOA/Reg B.

### Also tried: baking fairness into the model itself, instead of correcting after the fact

Retrained the same CatBoost architecture with Fairlearn's `ExponentiatedGradient` (demographic parity constraint, gender × age-bin jointly) — a fairness-constrained model, evaluated at a flat 0.5 threshold with **no** post-hoc correction:

| | AUC | DP gap (gender) | DP gap (age) | Passes on its own? |
|---|---|---|---|---|
| Raw CatBoost | 0.7244 | 0.029 | 0.456 | No |
| Fairness-constrained CatBoost | 0.6832 | 0.006 | **0.069** | No (barely) |

The constrained model cut the age gap **6.6x** (0.456 → 0.069) purely through training, no decision-layer correction — real evidence the bias can be reduced at the model level, not just patched afterward. But at the training budget used for local iteration (150 boosting iterations, 10 `ExponentiatedGradient` rounds, both intentionally small — see `PROGRESS.md` Stage 16), it didn't quite clear the 0.05 bar on its own, and cost 4.1 points of AUC — enough to fail this project's own AUC-cost acceptance rule (≤ 2 points) for swapping in a constrained model over post-hoc correction.

**Decision, made automatically by `models/train_champion.py` from these actual numbers, not chosen by hand:** deploy the raw (more accurate) CatBoost with the deterministic per-group thresholds from the section above layered on top. That combination passes cleanly (DP gap 0.0015 gender / 0.0102 age) and keeps the 4.1-point AUC advantage.

**What's still open:** the underlying CatBoost model itself remains biased against younger applicants — the deployed mitigation corrects the *decision*, not the *model*. The constrained-retraining result above shows the model-level fix is real and works, just not (yet, at this training budget) enough to stand alone. A longer `ExponentiatedGradient` run (more boosting iterations, more rounds — the reduced settings here were a deliberate speed/quality tradeoff for local iteration, not a ceiling on what's possible) is the natural next experiment.

Full investigation, methodology, and reasoning: `PROGRESS.md` Stages 14–16.

---

## Explainability

- Global + local SHAP (`explainability/shap_explainer.py`)
- Per-prediction ECOA adverse action reason codes for every DENY/MANUAL_REVIEW decision (`explainability/adverse_action.py`)

## Known limitations

- The underlying model is still age-biased; fairness is currently achieved via post-hoc threshold correction, not a bias-free model — see Fairness section
- H2O AutoML comparison was inconclusive (only 1 of 8 requested models completed in the local time budget — see `PROGRESS.md` Stage 12)
- Raw-feature logistic regression in the Step 4 comparison table (AUC 0.56) is not a fair LR baseline — see the WoE-encoded LR (AUC 0.70) instead
- No temporal/economic-cycle validation — training data has no calendar dates (see `data_engineering/ingest_fred.py` for the planned-but-unbuilt external data integration pattern)
