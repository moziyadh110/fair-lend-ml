# Methodology

Technical description of how CreditShield's pipeline works end-to-end. For *why* each design choice was made over its alternatives, see `PROGRESS.md` — this document describes the mechanism, `PROGRESS.md` documents the reasoning.

## 1. Data ingestion

7 raw Home Credit tables (`application_train/test`, `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`) are loaded into Postgres schema `raw` by `data_engineering/load_to_postgres.py`, using `COPY FROM STDIN` in chunks rather than row-by-row `INSERT` — necessary at this data volume (~58.5M total rows, largest single file 723MB).

## 2. Feature engineering (dbt)

`data_engineering/dbt_project/` implements a three-layer architecture:
- **staging** (`stg_applications`): type cleanup, sentinel-value handling (e.g. `DAYS_EMPLOYED=365243` → NULL for "not employed"), derived age/tenure fields
- **intermediate** (5 models): per-`sk_id_curr` aggregation of each raw table — bureau history, previous applications, installment behavior, credit card behavior, POS/cash behavior
- **features** (`fct_credit_features`): final join of staging + all 5 intermediate models into one 307,511-row, 36-feature table, plus 3 derived risk-interaction features and 3 protected attributes carried for audit only (prefixed `_protected_`)

Schema names are forced to exactly `staging`/`intermediate`/`features` via a `generate_schema_name` macro override (dbt's default behavior prefixes custom schemas with the target schema otherwise).

## 3. Data validation

`data_engineering/validation/expectations.py` runs 6 checks against the feature table (null checks on keys, range checks on `age_years`/`credit_income_ratio`/`late_payment_rate`, target-balance sanity, row-count stability) — plain pandas, not Great Expectations (see `PROGRESS.md` Stage 6 for why).

## 4. Baseline model

`features/woe_encoder.py` implements Weight-of-Evidence encoding: numeric columns are quantile-binned (10 bins) unless low-cardinality (≤10 unique values), in which case they're treated as discrete categories directly — quantile-binning a binary flag collapses to one bin and silently destroys signal (see `PROGRESS.md` Stage 9). `models/woe_scorecard.py` converts the resulting logistic regression into a points-based scorecard (base score 600, PDO 20). `models/train_baseline.py` trains this end-to-end and reports Information Value per feature for selection.

## 5. Challenger model comparison

`experiments/model_zoo.py` defines 5 candidates; `experiments/compare.py` fits each once on an identical train/test split and reports AUC, Gini, KS, PR-AUC, F1, and fairness metrics per model, logged to MLflow (local file store). `experiments/run_comparison.py` is the entrypoint. `models/tuning/optuna_tuner.py` (Bayesian HPO, cross-validated objective) and `models/tuning/stability_analysis.py` (multi-seed variance check) support deeper tuning of the winner. `models/h2o_automl.py` runs H2O AutoML as a comparison point, exporting via `h2o.save_model` (MOJO export requires a paid license as of this build — see `PROGRESS.md` Stage 12).

## 6. Champion training and fairness gate

`models/train_champion.py`:
1. Trains CatBoost (the Step 4 winner) on the full training split
2. Runs `explainability/fairness.py`'s `full_fairness_audit` — demographic parity and equalized odds by gender, demographic parity by age group (4 bins: 18-30/30-45/45-60/60+), four-fifths rule by gender
3. If either protected attribute fails the 0.05 demographic-parity threshold, attempts post-hoc mitigation via Fairlearn's `ThresholdOptimizer` (per-group thresholds, jointly over gender × age group)
4. Exports `artifacts/champion_model.joblib` (the model, unmitigated — see below) and `artifacts/fairness_report.json` (full before/after audit, plus a `cleared_for_production` flag)

**Important:** the exported model is always the *unmitigated* CatBoost model. Threshold-based mitigation produces per-group decision thresholds, not a modified model — applying it correctly requires the serving layer to know each applicant's protected-attribute group at inference time and look up the right threshold, which `serving/app.py` does not yet implement. `cleared_for_production` reflects the *unmitigated* model's gate result, deliberately, so the flag can't be gamed by an audit that was never actually deployed. See `MODEL_CARD.md` for the current champion's actual (failing) status.

## 7. Explainability

`explainability/shap_explainer.py` wraps `shap.TreeExplainer` for global and per-prediction SHAP values. `explainability/adverse_action.py` maps the top-4 positive-SHAP (risk-increasing) features on any DENY/MANUAL_REVIEW prediction to human-readable ECOA reason codes via a fixed lookup table.

## 8. Serving

`serving/app.py` (FastAPI): loads `artifacts/champion_model.joblib` at startup, engineers the same derived features the dbt feature table computes (credit/income ratios, exposure, interaction flags), scores via the loaded model, converts probability to a 300-850 credit score (same PDO formula as the WoE scorecard), assigns a risk tier and decision (AUTO_APPROVE / APPROVE / MANUAL_REVIEW / DENY at 0.10/0.25/0.50 probability cutoffs), and generates adverse action reasons for non-approvals. Redis (`serving/cache.py`) caches identical-input predictions. `serving/batch_predict.py` reuses the same artifact for monthly portfolio scoring, writing to `serving.batch_scores` — guaranteeing real-time and batch predictions never diverge.

## 9. Monitoring

`monitoring/drift_detector.py`: PSI per feature (`< 0.10` stable, `0.10-0.25` investigate, `>= 0.25` retrain) plus an Evidently `DataDriftPreset` HTML report. `monitoring/alerting.py` evaluates current metric values against threshold rules in `configs/monitoring_config.yaml`. `monitoring/dashboard.py` (Streamlit) surfaces model performance, the fairness audit (including the age-gap finding), drift status, and pipeline health across 4 tabs.

## 10. Orchestration

`dags/credit_default_retrain.py` (Airflow): monthly schedule — dbt refresh → validation → drift check (branch) → conditional retrain → batch score. Task bodies call the functions described above directly; the DAG has not been executed against a live Airflow scheduler in this build (see `PROGRESS.md` open decisions).

## Reproducing this pipeline

```
docker compose up -d postgres redis
python data_engineering/load_to_postgres.py
cd data_engineering/dbt_project && dbt run && dbt test
python data_engineering/validation/expectations.py
python models/train_baseline.py
python experiments/run_comparison.py
python models/train_champion.py
python serving/batch_predict.py
uvicorn serving.app:app --reload   # or: streamlit run monitoring/dashboard.py
pytest tests/
```

Requires `.env` populated per `.env.example` and `kaggle.json` at `~/.kaggle/` (see `PROGRESS.md` Stage 2 for the dataset download steps).
