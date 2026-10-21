# Engineering Tradeoffs — CreditShield

Real numbers from this build, not placeholders.

## Tradeoff 1: Model selection — CatBoost over XGBoost, LightGBM, Random Forest, and raw-feature Logistic Regression

| Model | AUC | Gini | KS | Train time |
|---|---|---|---|---|
| **CatBoost (champion)** | **0.7243** | **0.4487** | 0.3358 | 22.4s |
| LightGBM | 0.7202 | 0.4403 | 0.3315 | 6.2s |
| XGBoost | 0.7161 | 0.4322 | 0.3267 | 11.3s |
| Random Forest | 0.7058 | 0.4117 | 0.3026 | 54.5s |

**Chosen:** CatBoost. Best AUC/Gini/KS of the tree models, and native categorical handling meant no manual encoding step for `name_education_type`/`name_family_status`.

CatBoost, LightGBM, and XGBoost all landed within 1.2 points of AUC, so training time and categorical handling were the real differentiators. LightGBM trained 3.6x faster than CatBoost for a 0.4-point AUC cost, which would matter more on a daily-retrain cadence than the monthly one this pipeline runs on — so the speed difference isn't operationally significant here, and CatBoost's native categorical support simplified the pipeline enough to make it the pick.

## Tradeoff 2: WoE-encoded Logistic Regression baseline vs. raw-feature tree models

The Step 3 baseline (WoE-encoded logistic regression scorecard) scored **AUC 0.701, Gini 0.402** — competitive with Random Forest (0.706) and only 2.4 points of AUC behind the CatBoost champion (0.724), using a fully interpretable, regulator-familiar points-based scorecard.

The WoE scorecard isn't a strawman baseline — it got within 2.4 AUC points of the best tree model while being fully interpretable via points-based scoring, with no post-hoc explanation needed. That's the real tradeoff here: CatBoost buys 2.4 points of discrimination power at the cost of needing SHAP to explain any individual decision.

## Tradeoff 3: Fairness — investigate the metric, then fix the decision, then try fixing the model

**Step 1 — data quality first.** The champion model's raw gender demographic parity gap looked like a failure (0.191, threshold 0.05) — but that number included a 4-row `code_gender='XNA'` placeholder category (Home Credit's marker for undisclosed gender, not a real group) acting as a statistical outlier in a 3-way comparison. After excluding it and confirming stability across 5 model seeds, the real gender gap is **0.029** — a clean pass.

**Step 2 — but age was never checked, and failed badly.** The fairness gate only ever gated on gender; `dp_age` was being computed and ignored. Real number: **0.456** (approval rates 43% for 18-30 vs. 89% for 60+, on 7,000+ applicants per band — not noise).

**Step 3 — fix the decision.** Deterministic per-(gender, age-group) decision thresholds (not Fairlearn's `ThresholdOptimizer`, which randomizes decisions for some applicants near the boundary — a fair-lending problem in its own right). Calibrated on train, validated on held-out test: DP gap 0.456→0.0102 (age), 0.029→0.0015 (gender). Live in both the real-time API and batch scoring.

**Step 4 — try fixing the model itself, and let the numbers decide.** Retrained with `ExponentiatedGradient` fairness constraints baked in, at two different budgets:

| | AUC cost vs. raw | DP gap (age) |
|---|---|---|
| Reduced budget (150 iter, 10 rounds) | 0.041 | 0.069 |
| Full budget (500 iter, 50 rounds, 26 min) | **0.014** | **0.093** |

More training capacity recovered almost all the AUC cost — but the age gap got *worse*, not better, at the fuller budget. Reported both numbers rather than only the flattering one: `ExponentiatedGradient` optimizes error subject to a constraint, it doesn't guarantee fairness improves monotonically with more compute. Neither run cleared the 0.05 threshold on its own, so the system kept the more accurate raw model with post-hoc correction both times — now backed by two independent real attempts, not one.

**Step 5 — check the model's own input list against the stated design, not just its outputs.** Four stages of fairness work (Steps 1-4 above) treated the age gap as something the model was *inferring* from correlated signals. A `corrwith()` check against the protected attribute — the check that should have happened before any of Steps 1-4 — found `age_years` was a raw model feature the whole time, directly contradicting this project's own documented design ("protected attributes... excluded from the model's feature set"). Removed it (plus a second feature explicitly gated on `age_years < 30`) and retrained everything:

| | Raw model `dp_age` | Constrained model `dp_age` | Constrained AUC cost |
|---|---|---|---|
| Before (age_years in model) | 0.456 | 0.069 / 0.093 (2 attempts, neither passed) | 0.041 / 0.014 |
| After (age_years removed) | **0.408** | **0.038 — passes on its own** | 0.023 |

The same reduced-budget `ExponentiatedGradient` configuration that failed twice now passes cleanly, for one reason: it was no longer fighting a model with an unconstrained direct channel to discriminate on age. Still not deployed over the post-hoc-corrected raw model — 0.023 is just above this project's pre-set 0.02 acceptance rule, and that rule doesn't get relaxed after the fact to fit one good result.

The fairness *metrics* got audited rigorously across four stages before the feature *list* did — checking whether the model's actual inputs matched what the documentation claimed. They didn't: age was a raw input the whole time, and that blind spot survived a lot of downstream rigor because nobody went back to check the premise. Once fixed, a fairness-constrained retrain that had failed twice passed on the first try, which is a good signal that the earlier failures were a scope problem, not a tuning problem. Overall this went through four real iterations, not one: first the compliance metric itself turned out to be corrupted by a placeholder data category, fixed before trusting any number; then the gate was found to be checking only gender while silently ignoring a much larger age disparity, fixed at the gate; then a deterministic, non-randomized per-group threshold correction proved out at portfolio scale, not just in a test split; then the "proper" fix — baking fairness into the model via constrained retraining — genuinely helped but didn't beat the post-hoc approach on this run's numbers, so the system kept what actually won instead of what sounded better.

## Tradeoff 4: H2O MOJO serving — dropped after a real licensing change

The plan called for H2O AutoML + MOJO export as a production serving path (single-digit ms latency). H2O-3's current open-source tier blocks MOJO export behind a paid "H2O-3 Secure" license — confirmed by hitting `H2OValueError: MOJO export requires H2O-3 Secure` directly, not assumed from documentation.

H2O MOJO isn't a viable free serving path anymore — that changed since this architecture pattern became common advice. H2O AutoML still ran as a third model family for the leaderboard comparison (using `h2o.save_model()`, which is still free), but it was never going to be the deployed model once CatBoost won the comparison, so losing MOJO didn't block anything real — it just means the old "single-digit-ms MOJO" claim doesn't hold without a commercial license.

## Tradeoff 5: dbt/SQL vs. pandas for feature engineering

7 dbt models (1 staging, 5 intermediate, 1 feature table) built the full 307,511-row feature table with joins across all 7 raw tables in ~35 seconds end-to-end (`dbt run`), all version-controlled SQL with 6 passing dbt tests.

dbt was chosen because a lending pipeline needs to operate at a scale where pandas breaks down — the layered staging/intermediate/features architecture also gives free lineage tracking and testable transformations, which matter in a regulated environment. The tradeoff is dbt is worse than pandas for anything stateful across rows (rolling windows, target encoding); the PySpark/Dask supplement exists for exactly that remaining slice of feature work.

## Tradeoff 6: Local joblib artifact vs. MLflow Model Registry for serving

`serving/app.py` loads the champion model from a local `artifacts/champion_model.joblib` file rather than MLflow's Model Registry (`models:/credit_default/Production`).

MLflow's Model Registry needs a database-backed tracking store and a real promotion workflow — both things the Airflow retraining DAG should own, not something to build twice. The serving architecture is proven end-to-end first with a direct artifact load; swapping that for a registry call is a one-line change once the promotion workflow exists, so it's not worth blocking on infrastructure that doesn't have a caller yet.
