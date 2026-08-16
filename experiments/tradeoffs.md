# Engineering Tradeoffs — CreditShield

Real numbers from this build, not placeholders. Full reasoning and evidence for each is in `PROGRESS.md`; this is the interview-ready summary.

## Tradeoff 1: Model selection — CatBoost over XGBoost, LightGBM, Random Forest, and raw-feature Logistic Regression

| Model | AUC | Gini | KS | Train time |
|---|---|---|---|---|
| **CatBoost (champion)** | **0.7243** | **0.4487** | 0.3358 | 22.4s |
| LightGBM | 0.7202 | 0.4403 | 0.3315 | 6.2s |
| XGBoost | 0.7161 | 0.4322 | 0.3267 | 11.3s |
| Random Forest | 0.7058 | 0.4117 | 0.3026 | 54.5s |

**Chosen:** CatBoost. Best AUC/Gini/KS of the tree models, and native categorical handling meant no manual encoding step for `name_education_type`/`name_family_status`.

**Interview answer:** "CatBoost, LightGBM, and XGBoost were all within 1.2 points of AUC — the real differentiator was training time and categorical handling. LightGBM trained 3.6x faster than CatBoost for a 0.4-point AUC cost, which matters more for a daily-retrain cadence than a monthly one. I picked CatBoost here because monthly retraining doesn't make that speed difference operationally significant, and native categorical support simplified the pipeline."

## Tradeoff 2: WoE-encoded Logistic Regression baseline vs. raw-feature tree models

The Step 3 baseline (WoE-encoded logistic regression scorecard) scored **AUC 0.701, Gini 0.402** — competitive with Random Forest (0.706) and only 2.4 points of AUC behind the CatBoost champion (0.724), using a fully interpretable, regulator-familiar points-based scorecard.

**Interview answer:** "The WoE scorecard isn't a strawman baseline — it got within 2.4 AUC points of the best tree model while being fully interpretable via points-based scoring, no post-hoc explanation needed. That's the real tradeoff a bank makes: CatBoost buys 2.4 points of discrimination power at the cost of needing SHAP to explain any individual decision."

## Tradeoff 3: Fairness — investigate the metric, then fix the decision, then try fixing the model

**Step 1 — data quality first.** The champion model's raw gender demographic parity gap looked like a failure (0.191, threshold 0.05) — but that number included a 4-row `code_gender='XNA'` placeholder category (Home Credit's marker for undisclosed gender, not a real group) acting as a statistical outlier in a 3-way comparison. After excluding it and confirming stability across 5 model seeds, the real gender gap is **0.029** — a clean pass.

**Step 2 — but age was never checked, and failed badly.** The fairness gate only ever gated on gender; `dp_age` was being computed and ignored. Real number: **0.456** (approval rates 43% for 18-30 vs. 89% for 60+, on 7,000+ applicants per band — not noise).

**Step 3 — fix the decision.** Deterministic per-(gender, age-group) decision thresholds (not Fairlearn's `ThresholdOptimizer`, which randomizes decisions for some applicants near the boundary — a fair-lending problem in its own right). Calibrated on train, validated on held-out test: DP gap 0.456→0.0102 (age), 0.029→0.0015 (gender). Live in both the real-time API and batch scoring.

**Step 4 — try fixing the model itself, and let the numbers decide.** Retrained with `ExponentiatedGradient` fairness constraints baked in. Real result: age gap fell 6.6x (0.456→0.069) from training alone, but didn't clear 0.05 and cost 4.1 AUC points at the training budget used — so the system automatically kept the more accurate raw model with post-hoc correction rather than deploying the "architecturally cleaner" option that measurably underperformed.

**Interview answer:** "This went through four real iterations, not one. First I found the compliance metric itself was corrupted by a placeholder data category — fixed that before trusting any number. Then I found the gate was only checking gender and silently ignoring a much larger age disparity — fixed the gate itself. Then I built a deterministic, non-randomized per-group threshold correction and proved it works at portfolio scale, not just in a test split. Then I tried the 'proper' fix — baking fairness into the model via constrained retraining — and it genuinely helped, but didn't beat the post-hoc approach on this run's numbers, so I kept what actually won instead of what sounded better. Every one of those decisions came from measuring, not assuming."

## Tradeoff 4: H2O MOJO serving — dropped after a real licensing change

The plan called for H2O AutoML + MOJO export as a production serving path (single-digit ms latency). H2O-3's current open-source tier blocks MOJO export behind a paid "H2O-3 Secure" license — confirmed by hitting `H2OValueError: MOJO export requires H2O-3 Secure` directly, not assumed from documentation.

**Interview answer:** "H2O MOJO isn't a viable free serving path anymore — that changed since this architecture pattern became common advice. I still ran H2O AutoML as a third model family for the leaderboard comparison (using `h2o.save_model()`, which is still free), but it was never going to be the deployed model once CatBoost won the comparison, so losing MOJO didn't block anything real — it just means I can't repeat the old 'single-digit-ms MOJO' claim without a commercial license."

## Tradeoff 5: dbt/SQL vs. pandas for feature engineering

7 dbt models (1 staging, 5 intermediate, 1 feature table) built the full 307,511-row feature table with joins across all 7 raw tables in ~35 seconds end-to-end (`dbt run`), all version-controlled SQL with 6 passing dbt tests.

**Interview answer:** "I used dbt because Capital One operates at a scale where Pandas breaks — the layered staging/intermediate/features architecture also gave me free lineage tracking and testable transformations, which matter in a regulated environment. The tradeoff is dbt is worse than pandas for anything stateful across rows (rolling windows, target encoding) — the plan's PySpark/Dask supplement exists for exactly that 20%."

## Tradeoff 6: Local joblib artifact vs. MLflow Model Registry for serving

`serving/app.py` loads the champion model from a local `artifacts/champion_model.joblib` file rather than MLflow's Model Registry (`models:/credit_default/Production`).

**Interview answer:** "MLflow's Model Registry needs a database-backed tracking store and a real promotion workflow — both are things the Airflow retraining DAG should own, not something to build twice. I proved the serving architecture works end-to-end first with a direct artifact load; swapping that for a registry call is a one-line change once the promotion workflow exists, so I didn't want to block on infrastructure that doesn't have a caller yet."
