# Production-Level Project Plans for Capital One Data Science Internship

## Two projects, one unified philosophy

Both projects follow the same principle: build them as if Capital One's engineering team will inherit your code on Monday. That means API endpoints with input validation, versioned models with rollback capability, automated retraining pipelines, monitoring for data drift and model decay, and documentation that a new hire can follow. A notebook that gets good AUC is a homework assignment. A system that runs reliably in production is what gets you hired.

---

# PROJECT 1: Credit Default Prediction Pipeline with Fairness Constraints

## Business problem

Predict whether a loan applicant will default within 24 months, while ensuring the model does not discriminate across protected demographic groups. This is Capital One's core business — every lending decision runs through a model like this, and every model must pass regulatory fairness audits (ECOA, Fair Lending Act). Building this shows you understand both the ML and the compliance side.

## Dataset

**Primary: Home Credit Default Risk** (Kaggle)

- 307,511 loan applications with 122 features
- 7 relational tables: application_train, application_test, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, credit_card_balance
- Binary target: 1 = default, 0 = no default (8.07% default rate — imbalanced)
- Download: `kaggle competitions download -c home-credit-default-risk`

Why this dataset specifically: the multi-table structure forces you to write SQL joins and engineer features across relational data — exactly what Capital One's internal data looks like. Single-table datasets don't demonstrate this skill.

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION SYSTEM                                     │
│                                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐ │
│  │  Data     │───▶│  Feature  │───▶│  Model   │───▶│  Serving  │───▶│  Monitor │ │
│  │  Pipeline │    │  Store    │    │  Training │    │  API      │    │  + Alerts│ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────┘    └──────────┘ │
│       │               │               │               │               │         │
│   PostgreSQL      Redis +          MLflow          FastAPI          Evidently    │
│   + dbt           PostgreSQL       + Optuna        + uvicorn       + Grafana    │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Fairness Layer: Fairlearn / AIF360 — runs at training AND serving      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Orchestration: Airflow — schedules retraining, drift checks, alerts    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Infrastructure: Docker Compose — everything runs with one command      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data engineering layer

### 1.1 Database setup (PostgreSQL)

Load all 7 Home Credit tables into PostgreSQL. This simulates the production environment where data lives in a relational database, not CSV files on your laptop.

```sql
CREATE SCHEMA raw;
CREATE SCHEMA features;
CREATE SCHEMA serving;

-- Load raw tables
CREATE TABLE raw.application_train (
    sk_id_curr INTEGER PRIMARY KEY,
    target INTEGER,
    name_contract_type VARCHAR(20),
    code_gender CHAR(1),
    flag_own_car CHAR(1),
    amt_income_total NUMERIC,
    amt_credit NUMERIC,
    amt_annuity NUMERIC,
    amt_goods_price NUMERIC,
    -- ... 122 columns total
    days_birth INTEGER,
    days_employed INTEGER,
    days_registration FLOAT,
    days_id_publish INTEGER
);

-- Repeat for bureau, bureau_balance, previous_application, etc.
```

Why PostgreSQL and not just Pandas: Capital One's data scientists query databases, not CSVs. Every feature engineering query you write in SQL is directly transferable to their tech stack. Interviewers will ask "how would you do this at scale?" and "I wrote it in SQL against PostgreSQL" is the right answer.

### 1.2 Feature engineering with dbt

Use dbt to build a layered transformation pipeline: staging (clean raw data) → intermediate (join tables) → features (ML-ready aggregates).

**Staging layer — clean and standardize:**

```sql
-- models/staging/stg_applications.sql
WITH raw AS (
    SELECT * FROM {{ source('raw', 'application_train') }}
)
SELECT
    sk_id_curr,
    target,
    name_contract_type,
    code_gender,
    CASE WHEN flag_own_car = 'Y' THEN 1 ELSE 0 END AS owns_car,
    CASE WHEN flag_own_realty = 'Y' THEN 1 ELSE 0 END AS owns_realty,
    amt_income_total,
    amt_credit,
    amt_annuity,
    amt_goods_price,
    ABS(days_birth) / 365.25 AS age_years,
    CASE
        WHEN days_employed = 365243 THEN NULL  -- sentinel value for unemployed
        ELSE ABS(days_employed) / 365.25
    END AS years_employed,
    cnt_children,
    name_education_type,
    name_family_status,
    occupation_type,
    organization_type
FROM raw
```

**Intermediate layer — join relational tables:**

```sql
-- models/intermediate/int_bureau_aggregated.sql
SELECT
    sk_id_curr,
    COUNT(*) AS bureau_loan_count,
    SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END) AS active_loans,
    SUM(CASE WHEN credit_active = 'Closed' THEN 1 ELSE 0 END) AS closed_loans,
    AVG(days_credit) AS avg_days_credit,
    AVG(credit_day_overdue) AS avg_days_overdue,
    MAX(credit_day_overdue) AS max_days_overdue,
    SUM(amt_credit_sum) AS total_bureau_credit,
    SUM(amt_credit_sum_debt) AS total_bureau_debt,
    AVG(amt_credit_sum_debt / NULLIF(amt_credit_sum, 0)) AS avg_debt_to_credit_ratio
FROM {{ source('raw', 'bureau') }}
GROUP BY sk_id_curr
```

```sql
-- models/intermediate/int_previous_app_aggregated.sql
SELECT
    sk_id_curr,
    COUNT(*) AS prev_app_count,
    SUM(CASE WHEN name_contract_status = 'Approved' THEN 1 ELSE 0 END) AS prev_approved,
    SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END) AS prev_refused,
    AVG(amt_application) AS avg_prev_application_amount,
    AVG(amt_credit) AS avg_prev_credit_amount,
    AVG(CASE WHEN amt_application > 0
         THEN amt_credit / amt_application ELSE NULL END) AS avg_credit_to_application_ratio,
    MAX(days_decision) AS most_recent_prev_app_days
FROM {{ source('raw', 'previous_application') }}
GROUP BY sk_id_curr
```

```sql
-- models/intermediate/int_installments_behavior.sql
SELECT
    sk_id_curr,
    COUNT(*) AS total_installments,
    SUM(CASE WHEN days_entry_payment > days_instalment THEN 1 ELSE 0 END) AS late_payments,
    AVG(CASE WHEN days_entry_payment > days_instalment
         THEN days_entry_payment - days_instalment ELSE 0 END) AS avg_days_late,
    SUM(amt_payment - amt_instalment) AS total_over_under_payment,
    STDDEV(amt_payment) AS payment_amount_volatility
FROM {{ source('raw', 'installments_payments') }}
GROUP BY sk_id_curr
```

**Feature layer — final ML-ready table:**

```sql
-- models/features/fct_credit_features.sql
SELECT
    a.sk_id_curr,
    a.target,

    -- Application features
    a.age_years,
    a.years_employed,
    a.amt_income_total,
    a.amt_credit,
    a.amt_annuity,
    a.amt_credit / NULLIF(a.amt_income_total, 0) AS credit_income_ratio,
    a.amt_annuity / NULLIF(a.amt_income_total, 0) AS annuity_income_ratio,
    a.amt_goods_price / NULLIF(a.amt_credit, 0) AS goods_to_credit_ratio,
    a.owns_car,
    a.owns_realty,
    a.cnt_children,

    -- Bureau features
    COALESCE(b.bureau_loan_count, 0) AS bureau_loan_count,
    COALESCE(b.active_loans, 0) AS active_loans,
    COALESCE(b.avg_days_overdue, 0) AS avg_days_overdue,
    COALESCE(b.max_days_overdue, 0) AS max_days_overdue,
    COALESCE(b.avg_debt_to_credit_ratio, 0) AS bureau_debt_ratio,

    -- Previous application features
    COALESCE(p.prev_app_count, 0) AS prev_app_count,
    COALESCE(p.prev_approved, 0) AS prev_approved,
    COALESCE(p.prev_refused, 0) AS prev_refused,
    COALESCE(p.avg_credit_to_application_ratio, 0) AS credit_application_ratio,

    -- Payment behavior features
    COALESCE(i.late_payments, 0) AS late_payments,
    COALESCE(i.avg_days_late, 0) AS avg_days_late,
    COALESCE(i.payment_amount_volatility, 0) AS payment_volatility,

    -- Derived risk signals
    COALESCE(b.active_loans, 0) * a.amt_credit AS exposure_at_risk,
    COALESCE(i.late_payments, 0) * 1.0 / NULLIF(COALESCE(i.total_installments, 1), 0) AS late_payment_rate,

    -- Fairness-sensitive features (kept for auditing, excluded from model input)
    a.code_gender AS _protected_gender,
    a.age_years AS _protected_age,
    a.name_family_status AS _protected_family_status

FROM {{ ref('stg_applications') }} a
LEFT JOIN {{ ref('int_bureau_aggregated') }} b ON a.sk_id_curr = b.sk_id_curr
LEFT JOIN {{ ref('int_previous_app_aggregated') }} p ON a.sk_id_curr = p.sk_id_curr
LEFT JOIN {{ ref('int_installments_behavior') }} i ON a.sk_id_curr = i.sk_id_curr
```

The protected features (prefixed with `_protected_`) are included in the table for fairness auditing but excluded from the model training feature set. This separation is critical for regulatory compliance — you need to measure disparate impact without using protected attributes as predictors.

### 1.3 Data validation

```python
# validation/expectations.py
import great_expectations as gx

def validate_feature_table(df):
    """Run after every dbt refresh before model training."""
    context = gx.get_context()
    validator = context.sources.pandas_default.read_dataframe(df)

    # No nulls in critical columns
    validator.expect_column_values_to_not_be_null("sk_id_curr")
    validator.expect_column_values_to_not_be_null("target")

    # Sensible ranges
    validator.expect_column_values_to_be_between("age_years", min_value=18, max_value=100)
    validator.expect_column_values_to_be_between("credit_income_ratio", min_value=0, max_value=200)
    validator.expect_column_values_to_be_between("late_payment_rate", min_value=0, max_value=1)

    # Target distribution hasn't shifted dramatically
    validator.expect_column_proportion_of_unique_values_to_be_between(
        "target", min_value=0.01, max_value=0.99
    )

    # Row count stability
    validator.expect_table_row_count_to_be_between(min_value=250000, max_value=350000)

    return validator.validate()
```

---

## Stage 2: ML engineering layer

### 2.1 Model training pipeline

```python
# training/train.py
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import mlflow
import joblib


class CreditDefaultTrainer:
    """End-to-end training pipeline for credit default prediction."""

    def __init__(self, feature_table_path: str, experiment_name: str = "credit_default"):
        self.df = pd.read_parquet(feature_table_path)
        self.experiment_name = experiment_name
        self.protected_cols = [c for c in self.df.columns if c.startswith("_protected_")]
        self.exclude_cols = self.protected_cols + ["sk_id_curr", "target"]
        self.feature_cols = [c for c in self.df.columns if c not in self.exclude_cols]

    def build_preprocessor(self):
        """Build sklearn preprocessing pipeline."""
        numeric_cols = self.df[self.feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df[self.feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", LabelEncoder()),  # In production use TargetEncoder or OrdinalEncoder
        ])

        return ColumnTransformer([
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ], remainder="drop")

    def objective(self, trial, X, y):
        """Optuna objective for XGBoost hyperparameter tuning."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params, eval_metric="auc", random_state=42)
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            pred = model.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], pred))

        return np.mean(scores)

    def train_with_tracking(self, n_trials: int = 50):
        """Full training loop with MLflow tracking and Optuna tuning."""
        mlflow.set_experiment(self.experiment_name)

        X = self.df[self.feature_cols]
        y = self.df["target"]
        protected = self.df[self.protected_cols]

        preprocessor = self.build_preprocessor()
        X_processed = preprocessor.fit_transform(X)

        # Hyperparameter optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_processed, y), n_trials=n_trials)

        # Train final model with best params
        best_params = study.best_params
        final_model = xgb.XGBClassifier(**best_params, eval_metric="auc", random_state=42)

        # Final train/test split for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            X_processed, y, protected, test_size=0.2, stratify=y, random_state=42
        )

        final_model.fit(X_train, y_train)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Log everything to MLflow
        with mlflow.start_run():
            mlflow.log_params(best_params)
            mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_proba))
            mlflow.log_metric("f1", f1_score(y_test, y_pred))

            # Fairness metrics
            fairness_results = self.compute_fairness_metrics(y_test, y_pred, y_pred_proba, prot_test)
            for metric_name, value in fairness_results.items():
                mlflow.log_metric(f"fairness_{metric_name}", value)

            # Save artifacts
            mlflow.xgboost.log_model(final_model, "model")
            joblib.dump(preprocessor, "preprocessor.joblib")
            mlflow.log_artifact("preprocessor.joblib")

        return final_model, preprocessor, fairness_results

    def compute_fairness_metrics(self, y_true, y_pred, y_pred_proba, protected_df):
        """Compute fairness metrics across protected groups."""
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
            MetricFrame,
        )

        results = {}

        # Gender fairness
        gender = protected_df["_protected_gender"].values
        results["demographic_parity_gender"] = demographic_parity_difference(
            y_true, y_pred, sensitive_features=gender
        )
        results["equalized_odds_gender"] = equalized_odds_difference(
            y_true, y_pred, sensitive_features=gender
        )

        # Age group fairness (binned)
        age_bins = pd.cut(
            protected_df["_protected_age"],
            bins=[18, 30, 45, 60, 100],
            labels=["18-30", "30-45", "45-60", "60+"]
        ).values
        results["demographic_parity_age"] = demographic_parity_difference(
            y_true, y_pred, sensitive_features=age_bins
        )

        # Group-level AUC
        metric_frame = MetricFrame(
            metrics=roc_auc_score,
            y_true=y_true,
            y_pred=y_pred_proba,
            sensitive_features=gender,
        )
        results["auc_disparity_gender"] = metric_frame.difference()

        return results
```

### 2.2 Fairness-constrained retraining

When fairness metrics exceed thresholds, automatically retrain with constraints:

```python
# training/fair_retrain.py
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def retrain_with_fairness_constraints(X_train, y_train, sensitive_features, constraint_type="dp"):
    """Retrain model with fairness constraints using Fairlearn."""

    if constraint_type == "dp":
        constraint = DemographicParity()
    elif constraint_type == "eo":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"Unknown constraint: {constraint_type}")

    # ExponentiatedGradient wraps any sklearn-compatible estimator
    base_estimator = LogisticRegression(max_iter=1000, solver="lbfgs")

    mitigated_model = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        max_iter=50,
    )

    mitigated_model.fit(
        X_train, y_train,
        sensitive_features=sensitive_features,
    )

    return mitigated_model


def threshold_optimization(y_true, y_pred_proba, sensitive_features):
    """Find group-specific thresholds that equalize FPR/FNR across groups."""
    from fairlearn.postprocessing import ThresholdOptimizer

    optimizer = ThresholdOptimizer(
        estimator=None,  # we pass predictions directly
        constraints="equalized_odds",
        prefit=True,
    )

    # ThresholdOptimizer adjusts decision thresholds per group
    # to achieve fairness without retraining the model
    return optimizer
```

### 2.3 Model comparison framework

Train and compare multiple model architectures:

```python
# training/compare_models.py

MODELS = {
    "xgboost": xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        eval_metric="auc", random_state=42
    ),
    "lightgbm": lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        random_state=42
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
    ),
}

# Each model gets logged to MLflow with AUC, F1, precision, recall,
# fairness metrics, training time, and feature importances.
# The comparison dashboard in MLflow lets you pick the best
# AUC-fairness tradeoff visually.
```

---

## Stage 3: Model serving layer (backend)

### 3.1 FastAPI prediction service

```python
# serving/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import joblib
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
import logging

app = FastAPI(title="Credit Default Prediction API", version="1.0.0")
logger = logging.getLogger(__name__)

# Load model and preprocessor on startup
MODEL = None
PREPROCESSOR = None
MODEL_VERSION = None

@app.on_event("startup")
async def load_model():
    global MODEL, PREPROCESSOR, MODEL_VERSION
    MODEL = mlflow.xgboost.load_model("models:/credit_default/Production")
    PREPROCESSOR = joblib.load("artifacts/preprocessor.joblib")
    MODEL_VERSION = "1.0.0"
    logger.info(f"Model v{MODEL_VERSION} loaded successfully")


class LoanApplication(BaseModel):
    """Input schema with validation — mirrors real loan application fields."""
    amt_income_total: float = Field(..., gt=0, description="Applicant annual income")
    amt_credit: float = Field(..., gt=0, description="Loan amount requested")
    amt_annuity: float = Field(..., gt=0, description="Monthly payment amount")
    amt_goods_price: Optional[float] = Field(None, gt=0)
    age_years: float = Field(..., ge=18, le=100)
    years_employed: Optional[float] = Field(None, ge=0)
    owns_car: int = Field(..., ge=0, le=1)
    owns_realty: int = Field(..., ge=0, le=1)
    cnt_children: int = Field(0, ge=0, le=20)
    bureau_loan_count: int = Field(0, ge=0)
    active_loans: int = Field(0, ge=0)
    avg_days_overdue: float = Field(0, ge=0)
    max_days_overdue: float = Field(0, ge=0)
    prev_app_count: int = Field(0, ge=0)
    prev_approved: int = Field(0, ge=0)
    prev_refused: int = Field(0, ge=0)
    late_payments: int = Field(0, ge=0)
    avg_days_late: float = Field(0, ge=0)
    payment_volatility: float = Field(0, ge=0)

    @validator("amt_credit")
    def credit_within_reason(cls, v, values):
        income = values.get("amt_income_total", 0)
        if income > 0 and v > income * 100:
            raise ValueError("Credit amount exceeds 100x income — likely data error")
        return v


class PredictionResponse(BaseModel):
    application_id: str
    default_probability: float
    risk_tier: str  # "low", "medium", "high", "very_high"
    credit_income_ratio: float
    model_version: str
    prediction_timestamp: str
    explanation: dict  # top contributing features


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """Score a loan application for default risk."""
    try:
        input_df = pd.DataFrame([application.dict()])

        # Engineer derived features server-side
        input_df["credit_income_ratio"] = input_df["amt_credit"] / input_df["amt_income_total"]
        input_df["annuity_income_ratio"] = input_df["amt_annuity"] / input_df["amt_income_total"]
        input_df["goods_to_credit_ratio"] = (
            input_df["amt_goods_price"].fillna(0) / input_df["amt_credit"]
        )
        input_df["bureau_debt_ratio"] = 0  # derived from bureau data in production
        input_df["late_payment_rate"] = (
            input_df["late_payments"] / input_df["prev_app_count"].replace(0, 1)
        )
        input_df["exposure_at_risk"] = input_df["active_loans"] * input_df["amt_credit"]

        # Preprocess and predict
        X = PREPROCESSOR.transform(input_df)
        probability = float(MODEL.predict_proba(X)[:, 1][0])

        # Assign risk tier
        if probability < 0.15:
            tier = "low"
        elif probability < 0.35:
            tier = "medium"
        elif probability < 0.60:
            tier = "high"
        else:
            tier = "very_high"

        # Feature importance explanation (top 5)
        import shap
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(X)
        feature_names = PREPROCESSOR.get_feature_names_out()
        top_features = sorted(
            zip(feature_names, shap_values[0]),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return PredictionResponse(
            application_id=f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            default_probability=round(probability, 4),
            risk_tier=tier,
            credit_income_ratio=round(input_df["credit_income_ratio"].iloc[0], 2),
            model_version=MODEL_VERSION,
            prediction_timestamp=datetime.now().isoformat(),
            explanation={name: round(float(val), 4) for name, val in top_features},
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": MODEL is not None,
    }


@app.get("/fairness-report")
async def fairness_report():
    """Return latest fairness audit results."""
    # In production, this reads from MLflow or a database
    return {
        "demographic_parity_gender": 0.032,
        "equalized_odds_gender": 0.041,
        "demographic_parity_age": 0.028,
        "threshold": 0.05,
        "status": "PASS",
        "last_audit": "2026-08-01T00:00:00Z",
    }
```

### 3.2 Batch prediction pipeline

For scoring existing customers in bulk (monthly portfolio review):

```python
# serving/batch_predict.py
import pandas as pd
from sqlalchemy import create_engine
import mlflow
import joblib


def batch_score_portfolio(connection_string: str, output_table: str):
    """Score all active customers and write results to database."""
    engine = create_engine(connection_string)

    # Pull latest features from feature table
    features_df = pd.read_sql("SELECT * FROM features.fct_credit_features", engine)

    # Load model
    model = mlflow.xgboost.load_model("models:/credit_default/Production")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    # Exclude protected and ID columns
    exclude = [c for c in features_df.columns if c.startswith("_protected_")] + ["sk_id_curr", "target"]
    feature_cols = [c for c in features_df.columns if c not in exclude]

    X = preprocessor.transform(features_df[feature_cols])
    probabilities = model.predict_proba(X)[:, 1]

    # Build output
    results = pd.DataFrame({
        "sk_id_curr": features_df["sk_id_curr"],
        "default_probability": probabilities,
        "risk_tier": pd.cut(probabilities, bins=[0, 0.15, 0.35, 0.60, 1.0],
                           labels=["low", "medium", "high", "very_high"]),
        "scored_at": pd.Timestamp.now(),
        "model_version": "1.0.0",
    })

    results.to_sql(output_table, engine, schema="serving", if_exists="replace", index=False)
    return len(results)
```

---

## Stage 4: MLOps layer

### 4.1 Model registry with MLflow

```python
# mlops/registry.py
import mlflow
from mlflow.tracking import MlflowClient


def promote_model(run_id: str, model_name: str = "credit_default"):
    """Promote a model to production after validation checks."""
    client = MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    # Validate before promoting
    run = client.get_run(run_id)
    auc = float(run.data.metrics.get("auc", 0))
    dp_gender = float(run.data.metrics.get("fairness_demographic_parity_gender", 1.0))

    if auc < 0.75:
        raise ValueError(f"AUC {auc} below minimum threshold 0.75")
    if dp_gender > 0.05:
        raise ValueError(f"Demographic parity gap {dp_gender} exceeds 0.05 threshold")

    # Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
    )

    # Archive previous production version
    for version in client.search_model_versions(f"name='{model_name}'"):
        if version.current_stage == "Production" and version.version != mv.version:
            client.transition_model_version_stage(
                name=model_name, version=version.version, stage="Archived"
            )
```

### 4.2 Data and model drift monitoring

```python
# mlops/monitoring.py
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset


def generate_drift_report(reference_df, current_df, output_path: str):
    """Compare current data distribution against training baseline."""

    column_mapping = ColumnMapping(
        target="target",
        prediction="default_probability",
        numerical_features=["amt_credit", "amt_income_total", "credit_income_ratio",
                           "bureau_loan_count", "late_payment_rate"],
        categorical_features=["name_contract_type"],
    )

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        ClassificationPreset(),
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report.save_html(output_path)
    return report


def check_drift_thresholds(report) -> dict:
    """Extract drift scores and flag if retraining is needed."""
    results = report.as_dict()

    alerts = {
        "data_drift_detected": False,
        "target_drift_detected": False,
        "retrain_recommended": False,
        "details": {},
    }

    # Check if more than 30% of features drifted
    for metric in results.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            drift_share = metric["result"].get("share_of_drifted_columns", 0)
            alerts["data_drift_detected"] = drift_share > 0.3
            alerts["details"]["drift_share"] = drift_share

    alerts["retrain_recommended"] = alerts["data_drift_detected"] or alerts["target_drift_detected"]
    return alerts
```

### 4.3 Automated retraining DAG

```python
# dags/credit_default_retrain.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "ml-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    "credit_default_retrain",
    default_args=default_args,
    schedule_interval="0 2 1 * *",  # Monthly, 2 AM on the 1st
    start_date=datetime(2026, 8, 1),
    catchup=False,
) as dag:

    refresh_features = PythonOperator(
        task_id="refresh_features",
        python_callable=run_dbt_refresh,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=run_great_expectations,
    )

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_and_branch,
        # Returns "retrain_model" if drift detected, "skip_retrain" otherwise
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=train_and_register,
    )

    run_fairness_audit = PythonOperator(
        task_id="run_fairness_audit",
        python_callable=audit_fairness,
        trigger_rule="none_failed",
    )

    promote_if_passed = PythonOperator(
        task_id="promote_if_passed",
        python_callable=promote_model,
    )

    skip_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=lambda: print("No drift detected, skipping retrain"),
    )

    batch_score = PythonOperator(
        task_id="batch_score_portfolio",
        python_callable=batch_score_portfolio,
        trigger_rule="none_failed",
    )

    (refresh_features >> validate_data >> check_drift
     >> [retrain_model, skip_retrain])
    retrain_model >> run_fairness_audit >> promote_if_passed >> batch_score
    skip_retrain >> batch_score
```

---

## Stage 5: Monitoring dashboard

Build a Streamlit dashboard with four tabs:

```python
# dashboard/app.py
import streamlit as st

st.set_page_config(page_title="Credit Default Model Monitor", layout="wide")

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Performance",
    "Fairness Audit",
    "Data Drift",
    "Pipeline Health"
])

with tab1:
    # AUC, precision, recall, F1 over time
    # Confusion matrix for current model
    # Prediction distribution histogram
    # Risk tier breakdown

with tab2:
    # Demographic parity by gender, age group
    # Equalized odds comparison
    # Group-level AUC comparison
    # Fairness threshold pass/fail indicator
    # Historical fairness trend

with tab3:
    # Feature distribution comparisons (training vs current)
    # Drift scores per feature
    # PSI (Population Stability Index) tracking
    # Alert log

with tab4:
    # Airflow DAG run history
    # dbt model refresh status
    # Data validation pass/fail rates
    # API latency and error rates
```

---

## Stage 6: Algorithm experimentation lab

This is where the project stops being a tutorial and becomes real data science. You train multiple models, compare them rigorously, and document why you chose what you chose. Interviewers at Capital One will ask "why XGBoost and not a neural net?" — this section gives you the answer backed by your own experiments.

### 6.1 Candidate algorithms

```python
# experiments/model_zoo.py
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn


CANDIDATE_MODELS = {

    # --- Gradient Boosting Family ---

    "xgboost": xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        eval_metric="auc", random_state=42
    ),

    "lightgbm": lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        random_state=42
    ),

    "catboost": CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        auto_class_weights="Balanced",
        verbose=0, random_seed=42
        # CatBoost handles categoricals natively — no encoding needed
    ),

    # --- Linear Model (baseline) ---

    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="saga", penalty="elasticnet", l1_ratio=0.5,
        random_state=42
    ),

    # --- Ensemble ---

    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ),
}


# --- PyTorch TabNet-style network ---

class CreditDefaultNet(nn.Module):
    """
    Tabular neural network with batch norm and residual connections.
    Tests whether deep learning adds value over gradient boosting
    for structured credit data (spoiler: usually it doesn't,
    but you need to prove it with your own data).
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- Stacking Ensemble ---
# Uses top 3 models as base learners with logistic regression as meta-learner

def build_stacking_ensemble(base_models_dict):
    """
    Stacking: train multiple models, use their predictions
    as features for a meta-learner.
    """
    estimators = [(name, model) for name, model in base_models_dict.items()
                  if name in ["xgboost", "lightgbm", "catboost"]]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
```

### 6.2 Rigorous comparison framework

```python
# experiments/compare.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, make_scorer
)
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import mlflow
import time


class ModelExperimentRunner:
    """
    Run every candidate model through the same evaluation pipeline.
    Log everything to MLflow for comparison.
    """

    def __init__(self, X, y, protected_features, feature_names):
        self.X = X
        self.y = y
        self.protected = protected_features
        self.feature_names = feature_names

    def run_all_experiments(self, models_dict):
        """Train and evaluate every model with identical CV splits."""

        mlflow.set_experiment("credit_default_model_comparison")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for model_name, model in models_dict.items():
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")

            with mlflow.start_run(run_name=model_name):
                fold_metrics = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
                    X_train, X_val = self.X[train_idx], self.X[val_idx]
                    y_train, y_val = self.y[train_idx], self.y[val_idx]
                    prot_val = self.protected.iloc[val_idx]

                    # Train with timing
                    start = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start

                    # Predict with timing
                    start = time.time()
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    inference_time = (time.time() - start) / len(X_val) * 1000  # ms per sample

                    y_pred = (y_pred_proba >= 0.5).astype(int)

                    fold_metrics.append({
                        "fold": fold,
                        "roc_auc": roc_auc_score(y_val, y_pred_proba),
                        "pr_auc": average_precision_score(y_val, y_pred_proba),
                        "f1": f1_score(y_val, y_pred),
                        "precision": precision_score(y_val, y_pred),
                        "recall": recall_score(y_val, y_pred),
                        "dp_gender": demographic_parity_difference(
                            y_val, y_pred,
                            sensitive_features=prot_val["_protected_gender"]
                        ),
                        "eo_gender": equalized_odds_difference(
                            y_val, y_pred,
                            sensitive_features=prot_val["_protected_gender"]
                        ),
                        "train_time_seconds": train_time,
                        "inference_ms_per_sample": inference_time,
                    })

                # Aggregate across folds
                metrics_df = pd.DataFrame(fold_metrics)
                summary = {
                    "model": model_name,
                    **{f"{col}_mean": metrics_df[col].mean()
                       for col in metrics_df.columns if col != "fold"},
                    **{f"{col}_std": metrics_df[col].std()
                       for col in metrics_df.columns if col != "fold"},
                }

                # Log to MLflow
                for key, value in summary.items():
                    if key != "model":
                        mlflow.log_metric(key, value)

                mlflow.log_param("model_type", model_name)
                results.append(summary)

        return pd.DataFrame(results)
```

### 6.3 What you'll likely find (and what to say in interviews)

Run the comparison and you'll get a table like this:

```
| Model               | AUC (mean±std)    | PR-AUC        | F1          | DP Gender | Train Time | Inference/ms |
|---------------------|-------------------|---------------|-------------|-----------|------------|-------------|
| logistic_regression | 0.76 ± 0.01       | 0.28 ± 0.02   | 0.31 ± 0.02 | 0.021     | 2.1s       | 0.01        |
| random_forest       | 0.83 ± 0.01       | 0.39 ± 0.02   | 0.42 ± 0.03 | 0.038     | 45s        | 0.15        |
| xgboost             | 0.87 ± 0.01       | 0.48 ± 0.02   | 0.51 ± 0.02 | 0.045     | 28s        | 0.08        |
| lightgbm            | 0.87 ± 0.01       | 0.47 ± 0.02   | 0.50 ± 0.02 | 0.042     | 12s        | 0.05        |
| catboost            | 0.86 ± 0.01       | 0.46 ± 0.02   | 0.49 ± 0.02 | 0.039     | 90s        | 0.10        |
| neural_net          | 0.84 ± 0.02       | 0.41 ± 0.03   | 0.44 ± 0.03 | 0.051     | 180s       | 0.03        |
| stacking_ensemble   | 0.88 ± 0.01       | 0.50 ± 0.02   | 0.53 ± 0.02 | 0.048     | 140s       | 0.25        |
```

**Interview-ready insights from this table:**

- "XGBoost and LightGBM were nearly identical in performance. I chose XGBoost because its regularization parameters gave me more control when I needed to trade off AUC for fairness, but LightGBM trained 2.3x faster — in a production setting with daily retraining, LightGBM might win on operational cost."

- "The neural net underperformed gradient boosting on this structured tabular data, which is consistent with recent benchmarks (Grinsztajn et al., 2022). I kept it in the comparison to demonstrate I tested the hypothesis rather than assuming it."

- "CatBoost handled categorical features natively without encoding, which simplified the preprocessing pipeline. Its AUC was slightly lower but its fairness metrics were better — suggesting the native categorical handling introduced less bias than one-hot encoding."

- "The stacking ensemble gained +1% AUC over single models but at 5x inference latency. For real-time scoring that's a meaningful tradeoff. I'd recommend stacking for batch scoring (monthly portfolio review) and single XGBoost for real-time API calls."

- "Logistic regression is the baseline every regulated model gets compared against. Regulators sometimes require that you can explain why a more complex model is justified — the 11-point AUC gap between logistic regression and XGBoost is that justification."

---

## Stage 7: Engineering tradeoffs (documented decisions)

Every production system involves tradeoffs. Documenting yours shows you think like an engineer, not just a data scientist. Capital One interviewers specifically probe this.

### Tradeoff 1: AUC vs Fairness

```
DECISION: Accept 0.87 AUC with fairness constraints instead of 0.89 AUC without them.

Option A — Maximize AUC (0.89)
  ✓ Better default prediction accuracy
  ✓ Lower false negative rate → fewer missed defaults
  ✗ Demographic parity gap of 0.08 (exceeds 0.05 regulatory threshold)
  ✗ Fails Fair Lending audit → model can't be deployed

Option B — Fairness-constrained (0.87) ← CHOSEN
  ✓ Demographic parity gap of 0.032 (passes regulatory threshold)
  ✓ Equalized odds gap of 0.041
  ✓ Deployable in production
  ✗ 2% AUC reduction → ~300 more missed defaults per year on 307K applications

WHY: A model that can't be deployed has zero business value. The 2% AUC
cost translates to roughly $X in additional defaults, while a Fair Lending
violation costs $10M+ in fines plus reputational damage. This isn't close.

INTERVIEW ANSWER: "I intentionally traded 2% AUC for regulatory
compliance. The unfair model was better at prediction but it was
doing so by using proxies for protected attributes — which means
it was accurate for the wrong reasons."
```

### Tradeoff 2: SQL (dbt) vs Python for feature engineering

```
DECISION: Use dbt/SQL for feature engineering instead of pure Python/Pandas.

Option A — Pure Python/Pandas
  ✓ More flexible for complex statistical features
  ✓ Easier to prototype and iterate
  ✓ Better support for ML-specific transforms (target encoding, binning)
  ✗ Doesn't scale past ~10M rows without Spark
  ✗ No built-in lineage tracking or documentation
  ✗ Hard to hand off to other engineers

Option B — dbt/SQL ← CHOSEN
  ✓ Scales to billions of rows in any SQL warehouse
  ✓ Built-in lineage tracking (dbt docs)
  ✓ Version-controlled transformations
  ✓ Testable (dbt tests for data quality)
  ✓ Directly transferable to Capital One's tech stack
  ✗ Some features awkward to express in SQL (rolling window percentiles)
  ✗ Requires SQL proficiency that not all DS teams have

HYBRID APPROACH: 80% of features in dbt/SQL, 20% (complex statistical
features like interaction terms, target encoding) in a Python post-
processing step. This gives you the best of both: scalability and
auditability from SQL, flexibility from Python.

INTERVIEW ANSWER: "I used dbt because Capital One operates at a scale
where Pandas breaks. My dbt models process 307K rows in 4 seconds —
the same Pandas pipeline took 45 seconds. More importantly, dbt gave
me free documentation, lineage tracking, and built-in testing, which
are non-negotiable in a regulated environment."
```

### Tradeoff 3: Real-time API vs batch-only scoring

```
DECISION: Build both real-time API AND batch pipeline.

Option A — Batch only
  ✓ Simpler architecture
  ✓ No latency requirements
  ✓ Easier to debug and monitor
  ✗ Can't score new loan applications in real time
  ✗ Stale predictions between batch runs

Option B — Real-time API only
  ✓ Instant predictions for new applications
  ✗ Can't efficiently score entire portfolio (307K predictions)
  ✗ API downtime = no predictions at all
  ✗ Higher infrastructure cost

Option C — Both ← CHOSEN
  ✓ Real-time API for new applications (FastAPI, <100ms)
  ✓ Batch pipeline for monthly portfolio review (Airflow, ~5 min)
  ✓ Same model, same preprocessor, two serving paths
  ✗ Two code paths to maintain
  ✗ Must ensure consistency between real-time and batch predictions

INTERVIEW ANSWER: "New loan applications need real-time scoring —
a customer won't wait 24 hours for a decision. But monthly portfolio
risk reviews need batch scoring of 300K+ customers, which would be
absurdly expensive through an API. I built both paths sharing the
same model artifact from MLflow, so they're always consistent."
```

### Tradeoff 4: Model complexity vs explainability

```
DECISION: Use XGBoost with SHAP instead of a simpler logistic regression.

Option A — Logistic Regression
  ✓ Inherently interpretable (coefficient = effect size)
  ✓ Regulators love it
  ✓ Fast training and inference
  ✗ AUC of 0.76 vs 0.87 — misses 40% more defaults
  ✗ Can't capture nonlinear interactions

Option B — XGBoost + SHAP ← CHOSEN
  ✓ 0.87 AUC — significantly fewer missed defaults
  ✓ SHAP provides per-prediction explanations
  ✓ Feature importance rankings for model governance
  ✗ Not inherently interpretable — requires post-hoc explanation
  ✗ SHAP computation adds ~50ms per prediction

HYBRID: Log both XGBoost and logistic regression predictions.
If they disagree on a borderline case, flag for human review.
The logistic regression acts as a "sanity check" on the complex model.

INTERVIEW ANSWER: "Regulators require explainability, not simplicity.
SHAP gives me the ability to say 'this applicant was denied because
their credit-to-income ratio of 8.5 contributed +0.23 to the default
probability, and their 4 late payments contributed +0.18.' That's
more informative than a logistic regression coefficient."
```

### Tradeoff 5: Great Expectations vs simpler validation

```
DECISION: Use Great Expectations despite setup overhead.

Option A — Simple assert statements
  ✓ Zero dependencies
  ✓ Fast to write
  ✗ No reporting or data docs
  ✗ No historical tracking of validation results
  ✗ Fails silently in production

Option B — Great Expectations ← CHOSEN
  ✓ Auto-generated data documentation
  ✓ HTML validation reports for stakeholders
  ✓ Historical pass/fail tracking
  ✓ Integrates with Airflow (fail DAG on validation failure)
  ✗ 30-minute setup overhead
  ✗ Learning curve for expectation suite syntax

INTERVIEW ANSWER: "The setup cost of Great Expectations is 30 minutes.
The cost of deploying a model trained on corrupted data is a regulatory
incident. In a bank, data validation isn't optional — it's the gate
that prevents garbage from reaching the model."
```

---

## Docker Compose — one command to run everything

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: credit_default
      POSTGRES_USER: pipeline
      POSTGRES_PASSWORD: pipeline123
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  mlflow:
    build: ./mlflow
    ports: ["5001:5000"]
    depends_on: [postgres]
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://pipeline:pipeline123@postgres/credit_default

  api:
    build: ./serving
    ports: ["8000:8000"]
    depends_on: [mlflow]

  dashboard:
    build: ./dashboard
    ports: ["8501:8501"]
    depends_on: [postgres, mlflow]

  airflow:
    build: ./airflow
    ports: ["8080:8080"]
    depends_on: [postgres]
    volumes: ["./dags:/opt/airflow/dags"]

volumes:
  pgdata:
```

---

## Project directory structure

```
credit-default-prediction/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── Makefile
│
├── data_engineering/
│   ├── load_to_postgres.py
│   ├── dbt_project/
│   │   ├── dbt_project.yml
│   │   ├── profiles.yml
│   │   ├── models/
│   │   │   ├── staging/
│   │   │   │   └── stg_applications.sql
│   │   │   ├── intermediate/
│   │   │   │   ├── int_bureau_aggregated.sql
│   │   │   │   ├── int_previous_app_aggregated.sql
│   │   │   │   └── int_installments_behavior.sql
│   │   │   └── features/
│   │   │       └── fct_credit_features.sql
│   │   └── tests/
│   │       └── schema.yml
│   └── validation/
│       └── expectations.py
│
├── training/
│   ├── train.py
│   ├── fair_retrain.py
│   ├── compare_models.py
│   └── configs/
│       └── hyperparams.yaml
│
├── serving/
│   ├── Dockerfile
│   ├── app.py                    # FastAPI
│   ├── batch_predict.py
│   └── schemas.py
│
├── mlops/
│   ├── registry.py
│   ├── monitoring.py
│   └── drift_report.py
│
├── dags/
│   └── credit_default_retrain.py
│
├── dashboard/
│   ├── Dockerfile
│   └── app.py                    # Streamlit
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_fairness.py
│   └── test_pipeline.py
│
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_feature_analysis.ipynb
    └── 03_fairness_analysis.ipynb
```

---
---
---
