# CreditShield

Credit default prediction pipeline with fairness constraints: 7 relational tables → dbt feature engineering → WoE scorecard baseline → gradient-boosting challenger comparison → SHAP explainability → fairness-audited FastAPI serving → monitoring & orchestration.

**Current status: trained champion model is cleared for production**, with a documented age-related bias in the underlying model corrected via calibrated per-group decision thresholds (live in both the real-time API and batch scoring — not just an audit report). See `MODEL_CARD.md` for the full fairness investigation and mitigation before using anything here for a real decision.

## What's here

| Doc | What it's for |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Current champion model's performance, fairness status, and limitations — **read this first** |
| [`METHODOLOGY.md`](METHODOLOGY.md) | How the pipeline works, step by step, with the reproduce-it commands |
| `PROGRESS.md` (internal, not in this repo) | Every decision made and why, including the fairness investigation that found the age-bias issue |
| [`experiments/tradeoffs.md`](experiments/tradeoffs.md) | Interview-ready summary of the major engineering tradeoffs, with real numbers |

## Quick start

```bash
cp .env.example .env               # fill in FRED_API_KEY if using the FRED module; Postgres defaults are fine as-is
docker compose up -d postgres redis
pip install -r requirements.txt

python data_engineering/load_to_postgres.py
cd data_engineering/dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir . && cd ../..
python data_engineering/validation/expectations.py

python models/train_baseline.py        # WoE scorecard baseline
python experiments/run_comparison.py   # 5-model challenger comparison
python models/train_champion.py        # trains CatBoost champion + fairness audit

python serving/batch_predict.py        # score the full portfolio
uvicorn serving.app:app --reload       # real-time API on :8000
streamlit run monitoring/dashboard.py  # monitoring UI on :8501

docker compose up -d --build airflow   # optional: orchestration, see below
pytest tests/                          # 64 tests
python serving/load_test.py            # optional: needs the API running first
```

**Airflow (optional):** `docker compose up -d --build airflow` brings up a standalone Airflow instance (webserver+scheduler in one container) at `localhost:8080` (admin/admin), with the whole project mounted so `dags/credit_default_retrain.py`'s task callables run the real project code. Trigger manually with `docker exec <container> airflow dags trigger credit_default_retrain` — no need to wait for the monthly schedule.

**API auth:** `/predict`, `/batch-score`, and `/fairness-report` require an `X-API-Key` header matching `API_KEY` in `.env` (generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`). `/health` is intentionally public.

Requires a Kaggle account with `kaggle.json` at `~/.kaggle/` and the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition rules accepted — see `PROGRESS.md` Stage 2 for the exact steps if the download fails.

## Architecture

```
Postgres (raw) --dbt--> Postgres (staging/intermediate/features)
                              |
                    WoE baseline + 5-model comparison (MLflow)
                              |
                    CatBoost champion + fairness audit (Fairlearn)
                              |
                    MLflow Model Registry (`champion` alias)
                              |
              FastAPI + Redis (real-time)  |  batch_predict.py (portfolio)
                              |
              Streamlit dashboard  +  Airflow DAG (monthly retrain)
```

## Repository layout

- `data_engineering/` — Postgres loader, dbt project (staging/intermediate/features), validation, FRED economic-indicator ingestion
- `features/` — WoE encoder; PySpark/Dask scale supplement (payment-trend slope, utilization volatility — both real, both verified against the full dataset)
- `docker/spark_job.Dockerfile` — Java 8 + pinned pyarrow, the working way to run `spark_features.py`'s Arrow/pandas_udf path (native Windows Python can't — see `PROGRESS.md` Stage 22)
- `models/` — WoE scorecard, model comparison training entrypoints, Optuna/stability tuning, H2O AutoML
- `experiments/` — model zoo, comparison harness, tradeoffs writeup
- `evaluation/` — credit-risk metrics (KS, Gini, PSI, lift)
- `explainability/` — SHAP, ECOA adverse action codes, Fairlearn fairness audit + mitigation (post-hoc thresholds and constrained retraining, both tried)
- `serving/` — FastAPI app (API-key auth on decision endpoints), Redis cache, batch scoring, async load test
- `models/registry.py` — MLflow Model Registry: register, gate-and-promote to a `champion` alias, load in serving with a joblib fallback
- `.github/workflows/ci.yml` — pytest + `dbt parse` on every push/PR
- `monitoring/` — drift detection (PSI + Evidently), alerting, Streamlit dashboard
- `dags/` — Airflow retraining DAG, containerized (`airflow/`) and verified with a real end-to-end run
- `tests/` — 64 tests covering the above

## Known gaps

- The underlying model retains some age bias via legitimate proxy correlation (credit history length); fairness is achieved via calibrated per-group decision thresholds live in serving, not a fully bias-free model — see `MODEL_CARD.md`. A constrained-retraining candidate now passes the fairness gate on its own, but at slightly more AUC cost than this project's acceptance rule allows.
- Fairness coverage is gender + age only — no race/ethnicity, because this dataset has no surname/geography/nationality to support the standard proxy method (BISG). A dataset limitation, not something more engineering closes — see `PROGRESS.md` Stage 20.
- Applicants who decline to disclose gender get a "benefit of the doubt" fallback threshold (most lenient calibrated group), not the fairness correction proper — a deliberate policy choice, not an oversight.
- Full detail on every gap and why it's still open: `PROGRESS.md`, "Open decisions" section at the bottom
