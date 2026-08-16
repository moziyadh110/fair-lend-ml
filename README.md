# CreditShield

Credit default prediction pipeline with fairness constraints: 7 relational tables → dbt feature engineering → WoE scorecard baseline → gradient-boosting challenger comparison → SHAP explainability → fairness-audited FastAPI serving → monitoring & orchestration.

**Current status: trained champion model is cleared for production**, with a documented age-related bias in the underlying model corrected via calibrated per-group decision thresholds (live in both the real-time API and batch scoring — not just an audit report). See `MODEL_CARD.md` for the full fairness investigation and mitigation before using anything here for a real decision.

## What's here

| Doc | What it's for |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Current champion model's performance, fairness status, and limitations — **read this first** |
| [`METHODOLOGY.md`](METHODOLOGY.md) | How the pipeline works, step by step, with the reproduce-it commands |
| [`PROGRESS.md`](PROGRESS.md) | Every decision made and why, including the fairness investigation that found the age-bias issue |
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

pytest tests/                          # 34 tests
```

Requires a Kaggle account with `kaggle.json` at `~/.kaggle/` and the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition rules accepted — see `PROGRESS.md` Stage 2 for the exact steps if the download fails.

## Architecture

```
Postgres (raw) --dbt--> Postgres (staging/intermediate/features)
                              |
                    WoE baseline + 5-model comparison (MLflow)
                              |
                    CatBoost champion + fairness audit (Fairlearn)
                              |
              FastAPI + Redis (real-time)  |  batch_predict.py (portfolio)
                              |
              Streamlit dashboard  +  Airflow DAG (monthly retrain)
```

## Repository layout

- `data_engineering/` — Postgres loader, dbt project (staging/intermediate/features), validation
- `features/` — WoE encoder; PySpark/Dask scale supplement (not yet built)
- `models/` — WoE scorecard, model comparison training entrypoints, Optuna/stability tuning, H2O AutoML
- `experiments/` — model zoo, comparison harness, tradeoffs writeup
- `evaluation/` — credit-risk metrics (KS, Gini, PSI, lift)
- `explainability/` — SHAP, ECOA adverse action codes, Fairlearn fairness audit + mitigation
- `serving/` — FastAPI app, Redis cache, batch scoring
- `monitoring/` — drift detection (PSI + Evidently), alerting, Streamlit dashboard
- `dags/` — Airflow retraining DAG (written, not yet executed against a live scheduler)
- `tests/` — 34 tests covering the above

## Known gaps

- The underlying model is still age-biased; fairness is currently achieved via post-hoc per-group threshold correction, not a bias-free model — see `MODEL_CARD.md`
- `features/spark_features.py`, `features/dask_features.py`, `data_engineering/ingest_fred.py` are unimplemented stubs
- No Airflow scheduler has actually run the retraining DAG
- Full detail on every gap and why it's still open: `PROGRESS.md`, "Open decisions" section at the bottom
