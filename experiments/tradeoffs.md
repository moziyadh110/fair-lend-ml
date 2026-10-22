# Engineering Tradeoffs — CreditShield

Real numbers from this build, not placeholders.

## Tradeoff 1: Model selection — CatBoost over XGBoost, LightGBM, Random Forest, and raw-feature Logistic Regression

| Model | AUC | Gini | KS | Train time |
|---|---|---|---|---|
| **CatBoost (champion)** | **0.7204** | **0.4409** | 0.3236 | 19.8s |
| LightGBM | 0.7189 | 0.4378 | **0.3271** | 8.2s |
| XGBoost | 0.7114 | 0.4227 | 0.3129 | 9.9s |
| Random Forest | 0.7063 | 0.4125 | 0.3018 | 40.8s |

Measured on the fixed, `sk_id_curr`-ordered split (an earlier version of this table came from an unordered read whose holdout set changed whenever the database was rebuilt).

**Chosen:** CatBoost. Best AUC and Gini of the tree models, and native categorical handling meant no manual encoding step for `name_education_type`/`name_family_status`.

CatBoost, LightGBM, and XGBoost all landed within 0.9 points of AUC, so training time and categorical handling were the real differentiators. CatBoost and LightGBM are close to a tie: LightGBM trained 2.4x faster for a 0.15-point AUC cost, and actually has the slightly better KS. That speed would matter more on a daily-retrain cadence than the monthly one this pipeline runs on, so it isn't operationally significant here, and CatBoost's native categorical support simplified the pipeline enough to make it the pick.

## Tradeoff 2: WoE-encoded Logistic Regression baseline vs. raw-feature tree models

The Step 3 baseline (WoE-encoded logistic regression scorecard) scored **AUC 0.696, Gini 0.393** — close to Random Forest (0.706) and only 2.4 points of AUC behind the CatBoost champion (0.720), using a fully interpretable, regulator-familiar points-based scorecard.

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

**Step 6 — take gender out of the decision, and stop penalizing applicants 62+.** The Step 3 thresholds were keyed on (gender, age group), with gender collected only as a lookup key. However well-intended, a different cutoff by sex is disparate treatment under Regulation B, so the thresholds are now keyed on age group alone. That exposed a second problem: equalizing denial rates gives the lowest-risk group the strictest cutoff, so applicants 62+ were being held to a harsher standard than everyone else (0.36 vs. 0.45–0.60), which Reg B also doesn't allow. The 62+ group now always gets the most lenient cutoff.

Measured on real data, the 62+ rule conflicts with a plain "≤ 0.05 across every age group" gate:

| Decision rule | 62+ cutoff | Under-62 gap | 62+ approval | All-ages gap | Old gate | Reg B-aware gate |
|---|---|---|---|---|---|---|
| Age-only thresholds, no 62+ rule | 0.36 (strictest) | 0.0049 | 66.5% (others 67.4–67.9%) | 0.013 | Pass | Pass |
| Age-only thresholds + 62+ rule (deployed) | 0.60 (most lenient) | **0.0049** | **96.3%** | 0.289 | **Fail** | **Pass** |

The age gate was rewritten before retraining, for a legal reason rather than to rescue a number: parity is required among the under-62 groups, and the 62+ group must not be approved less often than any other (within 2 standard errors of noise). The all-ages gap is still reported.

The first row is the honest caveat. Without the 62+ rule, the 62+ group's approval rate comes out 1.4 points below the others, which is inside the noise margin, so the outcome check alone doesn't catch it (on an earlier, unordered split it did). The real protection is the cutoff rule itself: `calibrate_group_thresholds` always gives 62+ the most lenient cutoff, and a unit test enforces it. The approval-rate check is a backstop, not the guarantee. With gender removed, the gender gap still passes on its own (0.032 → 0.017). The same change fixed a fallback that was documented as "most lenient" but actually picked the strictest cutoff. And the constrained-retraining candidate from Step 5 still passes the new gate on its own (under-62 gap 0.042), but costs 0.021 AUC on the fixed split, 0.001 over the 0.02 rule, so it still isn't deployed.

## Tradeoff 4: H2O MOJO serving — dropped after a real licensing change

The plan called for H2O AutoML + MOJO export as a production serving path (single-digit ms latency). H2O-3's current open-source tier blocks MOJO export behind a paid "H2O-3 Secure" license — confirmed by hitting `H2OValueError: MOJO export requires H2O-3 Secure` directly, not assumed from documentation.

H2O MOJO isn't a viable free serving path anymore — that changed since this architecture pattern became common advice. H2O AutoML still ran as a third model family for the leaderboard comparison (using `h2o.save_model()`, which is still free), but it was never going to be the deployed model once CatBoost won the comparison, so losing MOJO didn't block anything real — it just means the old "single-digit-ms MOJO" claim doesn't hold without a commercial license.

## Tradeoff 5: dbt/SQL vs. pandas for feature engineering

7 dbt models (1 staging, 5 intermediate, 1 feature table) built the full 307,511-row feature table with joins across all 7 raw tables in ~35 seconds end-to-end (`dbt run`), all version-controlled SQL with 6 passing dbt tests.

dbt was chosen because a lending pipeline needs to operate at a scale where pandas breaks down — the layered staging/intermediate/features architecture also gives free lineage tracking and testable transformations, which matter in a regulated environment. The tradeoff is dbt is worse than pandas for anything stateful across rows (rolling windows, target encoding); the PySpark/Dask supplement exists for exactly that remaining slice of feature work.

## Tradeoff 6: MLflow Model Registry for serving, with a local artifact fallback

`models/train_champion.py` registers every trained champion in MLflow's Model Registry (`credit_default`) together with its serving metadata (feature list, calibrated thresholds) and promotes it to the `champion` alias only if it clears the gate: cleared for production by the fairness audit, and AUC ≥ 0.65 (`models/registry.py`). `serving/app.py` loads `models:/credit_default@champion` first and falls back to the local `artifacts/champion_model.joblib` only when the registry has nothing promoted, e.g. on a fresh clone.

The first version did the opposite: serving loaded the joblib file directly, on the reasoning that a registry needs a database-backed store and a real promotion workflow. That reasoning stopped holding once MLflow 3's default local backend turned out to be database-backed already, which was all the registry needed, so promotion moved into the training script instead of waiting for Airflow. The tradeoff that remains: every training run creates a new registry version (v11 at the time of writing), and the joblib fallback means a serving instance can run a model the registry never promoted, if someone copies an artifact in by hand.
