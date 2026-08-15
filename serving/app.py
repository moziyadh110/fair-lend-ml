"""FastAPI serving layer: real-time scoring with Redis caching, SHAP-backed
adverse action reason codes, and optional champion-challenger shadow scoring.

Loads model + feature-column list from local joblib artifacts (produced by
`models/train_champion.py`) rather than MLflow's Model Registry stage
promotion (`models:/credit_default/Production`) that the plan sketches.
MLflow's registry needs a database-backed tracking store and a real
promotion workflow (Step 7's DAG) — wiring that up is out of scope for
getting the serving path itself working and testable today. Swapping the
loader to pull from the registry later is a localized change (see
PROGRESS.md).
"""
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException

from serving.auth import require_api_key
from serving.cache import build_client, cache_key, get_cached, set_cached
from serving.schemas import LoanApplication, PredictionResponse
from explainability.adverse_action import generate_adverse_action_reasons
from explainability.fairness import age_bin_labels, non_disclosure_threshold

load_dotenv()

app = FastAPI(title="CreditShield Risk API", version="2.0.0")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_VERSION = "2.0.0"

CHAMPION_MODEL = None
CHALLENGER_MODEL = None
FEATURE_COLS: list[str] = []
GROUP_THRESHOLDS: dict = {}
DEFAULT_THRESHOLD = 0.5
EXPLAINER = None
CACHE = None


def _load_champion():
    """MLflow Model Registry first (the `champion` alias — see
    models/registry.py), falling back to the local joblib artifact if the
    registry has nothing promoted yet (fresh clone, or a run that registered
    but didn't clear the promotion gate). Returns (model, feature_cols,
    group_thresholds, default_threshold) or (None, [], {}, 0.5)."""
    try:
        from models.registry import load_champion
        model, metadata = load_champion()
        logger.info("Loaded champion from MLflow Model Registry.")
        return model, metadata["feature_cols"], metadata["group_thresholds"], metadata["default_threshold"]
    except Exception as e:
        logger.info(f"MLflow registry load failed or nothing promoted yet ({e}); "
                     "falling back to artifacts/champion_model.joblib.")

    champion_path = ARTIFACTS_DIR / "champion_model.joblib"
    if not champion_path.exists():
        return None, [], {}, 0.5
    bundle = joblib.load(champion_path)
    return bundle["model"], bundle["feature_cols"], bundle.get("group_thresholds", {}), bundle.get("default_threshold", 0.5)


def _warm_up():
    """Pays the ~2.3s first-call cost (CatBoost + SHAP TreeExplainer
    background-dataset setup — measured in PROGRESS.md Stage 21) once at
    startup, on a dummy all-zero row, instead of making the first real
    applicant pay it. Best-effort: a warm-up failure shouldn't block startup
    or take the service down — the first real request just pays the cost
    that would otherwise have been absorbed here."""
    try:
        dummy_row = pd.DataFrame([{col: 0.0 for col in FEATURE_COLS}])
        CHAMPION_MODEL.predict_proba(dummy_row)
        EXPLAINER.shap_values(dummy_row)
        logger.info("Warm-up prediction complete.")
    except Exception as e:
        logger.warning(f"Warm-up prediction failed (non-fatal, startup continues): {e}")


@app.on_event("startup")
async def load_models():
    global CHAMPION_MODEL, CHALLENGER_MODEL, FEATURE_COLS, GROUP_THRESHOLDS, DEFAULT_THRESHOLD, EXPLAINER, CACHE

    CHAMPION_MODEL, FEATURE_COLS, GROUP_THRESHOLDS, DEFAULT_THRESHOLD = _load_champion()
    if CHAMPION_MODEL is not None:
        # SHAP's TreeExplainer needs one concrete tree model. If the deployed
        # champion is a FairEnsembleModel (fairness-constrained retraining —
        # see models/train_champion.py), explain via its single highest-weight
        # sub-model rather than the full weighted mixture. See PROGRESS.md
        # Stage 16 for why that's a documented approximation, not a silent one.
        explainer_model = getattr(CHAMPION_MODEL, "dominant_predictor", CHAMPION_MODEL)
        EXPLAINER = shap.TreeExplainer(explainer_model)
        _warm_up()

    challenger_path = ARTIFACTS_DIR / "challenger_model.joblib"
    if challenger_path.exists():
        CHALLENGER_MODEL = joblib.load(challenger_path)["model"]

    CACHE = build_client()
    logger.info(
        f"Champion loaded: {CHAMPION_MODEL is not None}. Challenger: {CHALLENGER_MODEL is not None}. "
        f"Fairness-calibrated group thresholds: {len(GROUP_THRESHOLDS)}"
    )


def _engineer_features(app_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([app_dict])
    df["credit_income_ratio"] = df["amt_credit"] / df["amt_income_total"]
    df["annuity_income_ratio"] = df["amt_annuity"] / df["amt_income_total"]
    df["goods_to_credit_ratio"] = df["amt_goods_price"].fillna(0) / df["amt_credit"]
    df["late_payment_rate"] = df["late_payments"] / df["prev_app_count"].replace(0, 1)
    df["exposure_at_risk"] = df["active_loans"] * df["amt_credit"]
    df["high_util_x_high_leverage"] = (
        (df["avg_cc_utilization"] > 0.8) & (df["credit_income_ratio"] > 5)
    ).astype(int)
    return df


def _credit_score(probability: float) -> int:
    factor = 20 / np.log(2)
    offset = 600 - factor * np.log(50)
    if 0 < probability < 1:
        odds = (1 - probability) / probability
        score = int(offset + factor * np.log(odds))
    else:
        score = 300 if probability >= 1 else 850
    return max(300, min(850, score))


def _deny_threshold(gender: str | None, age_years: float) -> float:
    """The fairness-calibrated DENY cutoff for this applicant's (gender, age
    bin) group. If the applicant didn't disclose gender, or no calibrated
    thresholds are loaded, or their group combination wasn't seen during
    calibration: fall back to the MOST LENIENT calibrated threshold, not the
    raw model's flat threshold — see explainability.fairness.
    non_disclosure_threshold's docstring for why (falling back to the flat
    threshold would quietly reintroduce the age bias non-disclosure exists
    to sidestep). Never below 0.25, so the MANUAL_REVIEW tier always has
    room below it regardless of calibration."""
    if not GROUP_THRESHOLDS:
        return DEFAULT_THRESHOLD
    fallback = non_disclosure_threshold(GROUP_THRESHOLDS, fallback=DEFAULT_THRESHOLD)
    if gender is None:
        return max(0.25, fallback)
    age_bin = age_bin_labels([age_years]).astype(str).iloc[0]
    return max(0.25, GROUP_THRESHOLDS.get((gender, age_bin), fallback))


def _risk_tier_and_decision(probability: float, deny_threshold: float) -> tuple[str, str]:
    if probability < 0.10:
        return "low", "AUTO_APPROVE"
    if probability < 0.25:
        return "medium", "APPROVE"
    if probability < deny_threshold:
        return "high", "MANUAL_REVIEW"
    return "very_high", "DENY"


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
async def predict(application: LoanApplication):
    if CHAMPION_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run models/train_champion.py first")

    try:
        app_dict = application.model_dump()
        key = cache_key(app_dict)
        cached = get_cached(CACHE, key) if CACHE else None
        if cached:
            return PredictionResponse(**cached)

        input_df = _engineer_features(app_dict)
        X = input_df.reindex(columns=FEATURE_COLS, fill_value=0)

        probability = float(CHAMPION_MODEL.predict_proba(X)[:, 1][0])
        credit_score = _credit_score(probability)
        deny_threshold = _deny_threshold(application.code_gender, application.age_years)
        tier, decision = _risk_tier_and_decision(probability, deny_threshold)

        reasons = []
        if decision in ("DENY", "MANUAL_REVIEW"):
            shap_values = EXPLAINER.shap_values(X)
            values = shap_values[0] if np.ndim(shap_values) == 2 else shap_values
            reasons = generate_adverse_action_reasons(values, FEATURE_COLS)

        if CHALLENGER_MODEL is not None and random.random() < 0.10:
            challenger_prob = float(CHALLENGER_MODEL.predict_proba(X)[:, 1][0])
            logger.info(f"Shadow score — champion={probability:.4f} challenger={challenger_prob:.4f}")

        response = PredictionResponse(
            application_id=f"APP-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            default_probability=round(probability, 4),
            credit_score=credit_score,
            risk_tier=tier,
            decision=decision,
            adverse_action_reasons=reasons,
            credit_income_ratio=round(float(input_df["credit_income_ratio"].iloc[0]), 2),
            model_version=MODEL_VERSION,
            prediction_timestamp=datetime.now().isoformat(),
        )

        if CACHE:
            set_cached(CACHE, key, response.model_dump())
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-score", dependencies=[Depends(require_api_key)])
async def batch_score(applications: list[LoanApplication]):
    # predict() is called as a plain Python function here, not through FastAPI's
    # routing — its own `dependencies=[Depends(require_api_key)]` doesn't run
    # in that path, so this endpoint needs (and has) its own auth check above.
    return [await predict(a) for a in applications]


@app.get("/health")
async def health():
    redis_ok = False
    if CACHE:
        try:
            redis_ok = CACHE.ping()
        except Exception:
            redis_ok = False
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "champion_loaded": CHAMPION_MODEL is not None,
        "challenger_loaded": CHALLENGER_MODEL is not None,
        "redis_connected": redis_ok,
        "fairness_thresholds_active": len(GROUP_THRESHOLDS) > 0,
    }


@app.get("/fairness-report", dependencies=[Depends(require_api_key)])
async def fairness_report():
    report_path = ARTIFACTS_DIR / "fairness_report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text())
