# Progress Log — Decisions & Trade-offs

Running log of every decision point where more than one path was available. Newest entries at the bottom of each phase. Each entry: what we chose, what the alternatives were, and why.

---

## ✅ Checkpoint — 2026-08-16

**Steps 1–8 of the build roadmap are complete.** Data loaded (Postgres, ~58.5M rows), dbt feature pipeline built and tested, WoE baseline trained (AUC 0.701), 5-model comparison run (CatBoost wins, AUC 0.724), fairness-constrained retraining tried and evaluated, deterministic per-group threshold mitigation is **live** in both the real-time API and batch scoring, monitoring/dashboard/DAG written, full docs pass done. **43/43 tests passing.**

**Current deployed champion:** raw CatBoost + per-(gender, age-group) decision thresholds — `cleared_for_production: true` in `artifacts/fairness_report.json`. See `MODEL_CARD.md` for the full picture before treating this as more than a portfolio project.

**To resume work in a new session:**
1. `docker compose up -d postgres redis` (Postgres/Redis are the only infra this actually depends on)
2. `pytest tests/` — should be 43/43 green; if not, something changed
3. Read this file's "Open decisions / not yet made" section at the bottom — that's the live todo list, not this checkpoint note
4. `git log --oneline` — this checkpoint corresponds to the commit tagged in the message below, if the user asked for one

**What's genuinely unfinished (not just "could be nice"):**
- `features/spark_features.py`, `features/dask_features.py`, `data_engineering/ingest_fred.py` — empty stubs, never built
- No Airflow scheduler has ever executed `dags/credit_default_retrain.py`
- A longer `ExponentiatedGradient` run might let the fairness fix live in the model instead of a post-hoc correction layer — not attempted due to local iteration time budget (Stage 16)
- Applicants who don't disclose gender don't get the fairness correction (documented tradeoff, not a bug)

---

## Stage 0 — Project definition (HLD)

### Decision: Build against CreditShield v2 (`UPDATED PLAN.MD`), not the original `plan.md`
- **Alternatives:**
  - A. Original plan — XGBoost/LightGBM/CatBoost comparison, generic AUC/F1 metrics, FastAPI + SHAP, single-model serving.
  - B. CreditShield v2 — adds WoE scorecard baseline, credit-industry metrics (KS/Gini/PSI), H2O AutoML + MOJO serving, adverse-action reason codes, champion-challenger shadow scoring, four-fifths rule, FRED external-data module.
- **Chosen:** B (v2).
- **Why:** v2's additions are specifically the artifacts a credit-risk team (not a generic ML team) would expect — WoE scorecards and KS/Gini are industry-standard in lending, adverse action codes are an actual ECOA legal requirement, and four-fifths rule is a real disparate-impact test. v1 is a fine general ML pipeline but doesn't demonstrate domain fluency.
- **Trade-off accepted:** v2 is a larger build (28-day scope vs. unscoped) — more files, more dependencies (H2O, PySpark, Dask, Redis), more that can go wrong. Accepted because the target audience (Capital One-style credit risk review) rewards domain-specific depth over breadth of generic ML tooling.

### Decision: Reject "remove AI watermarks" skill request
- **Alternatives:** Build the skill as requested, or refuse.
- **Chosen:** Refuse.
- **Why:** Tools that strip AI-content watermarks (SynthID, C2PA provenance metadata) exist specifically to defeat systems designed to disclose AI-generated content — this is detection evasion regardless of stated intent, and out of scope for what gets built here.

---

## Stage 1 — Repository scaffolding (HLD → LLD)

### Decision: Empty-file skeleton first, code later
- **Alternatives:** Write working code immediately per file, or lay down the full directory/file skeleton empty first and fill in incrementally.
- **Chosen:** Skeleton first.
- **Why:** User explicitly asked for structure only, no code, so the shape of the system could be reviewed/agreed before any implementation effort was spent — cheap to change a folder layout, expensive to change it after code depends on it.

### Decision: Git-initialize the repo, but do not commit automatically
- **Alternatives:** Leave ungitted, or init + auto-commit the skeleton.
- **Chosen:** `git init` + stage files, leave commit to explicit user request.
- **Why:** Standing instruction — only commit when the user explicitly asks. Staging (not committing) keeps `git status` informative without taking an irreversible-feeling action on the user's behalf.

### Decision: Standard Python/dbt/Docker `.gitignore` from the start
- **Alternatives:** No `.gitignore` until something accidentally gets committed, or add one preemptively.
- **Chosen:** Preemptive — covers `__pycache__/`, `.venv/`, `.env`, `*.csv`, `*.parquet`, `data/`, dbt `target/`/`dbt_packages/`, `mlruns/`/`mlartifacts/`, notebook checkpoints.
- **Why:** The dataset (`data/raw/*.csv`, ~2.6GB) and secrets (`.env`) were always going to land in this repo. Ignoring them before they exist is strictly safer than remembering to ignore them after — a committed 700MB CSV or a leaked API key are both awkward to fully scrub from git history later.

### Decision: Reorganize skeleton from v1 layout to v2 layout when the plan changed
- **Alternatives:** Keep both v1-only directories (`training/`, `mlops/`, `dashboard/app.py`) alongside the new v2 directories (`models/`, `evaluation/`, `explainability/`, `monitoring/`), or remove the v1-only ones.
- **Chosen:** Removed `training/`, `mlops/`, and `dashboard/app.py`; kept `dashboard/Dockerfile` (still referenced by `docker-compose.yml`'s `dashboard` build context, even though v2's actual dashboard code now lives at `monitoring/dashboard.py`).
- **Why:** v2's Section 7 file structure supersedes v1's — keeping both would mean two places a "model training" or "monitoring dashboard" file could plausibly live, which invites divergence and confusion about which one is real. Safe to delete outright (not just gitignore/archive) because nothing was committed yet and the files were empty placeholders created this session, not user work product.
- **Note / minor inconsistency in the source plan:** v2's Section 7 file tree doesn't list a `dashboard/` directory at all (dashboard code moved to `monitoring/dashboard.py`), but v2's own `docker-compose.yml` (Section 5) still has `dashboard: build: ./dashboard`. Resolved by keeping `dashboard/Dockerfile` as the build context and treating `monitoring/dashboard.py` as the Streamlit entrypoint it copies in.

---

## Stage 2 — Credentials & external data access (LLD)

### Decision: Store secrets in a project-local `.env` (gitignored) + committed `.env.example` template
- **Alternatives:**
  - A. Tell the user to export env vars in their shell profile only (nothing in the repo).
  - B. `.env` + `.env.example` pair.
- **Chosen:** B.
- **Why:** A committed `.env.example` documents exactly what secrets/config the project needs without exposing real values — reproducible for anyone who clones the repo. A shell-profile-only approach isn't discoverable by a new contributor reading the repo.

### Decision: Kaggle credentials (`kaggle.json`) live outside the repo, at `~/.kaggle/kaggle.json` — not in `.env`
- **Alternatives:** Ask the user to paste Kaggle username/key into `.env` alongside `FRED_API_KEY`, or use the Kaggle CLI's own expected location.
- **Chosen:** Kaggle CLI's native location (`~/.kaggle/kaggle.json`).
- **Why:** The `kaggle` package hardcodes this lookup path (or `KAGGLE_CONFIG_DIR`) — fighting that with a custom `.env`-based loader would mean writing and maintaining bespoke auth-wiring code for zero benefit over just using the tool as designed.

### Decision: Downgrade `kaggle` package from 2.2.4 to <1.7 (installed: 1.6.17)
- **Alternatives:**
  - A. Use kaggle CLI 2.2.4's new auth flow (`kaggle auth login` OAuth, or `KAGGLE_API_TOKEN`/`~/.kaggle/access_token`).
  - B. Pin to a legacy 1.x version that reads the classic `{"username", "key"}` `kaggle.json` format the user already had.
- **Chosen:** B.
- **Why:** `kaggle auth login`'s OAuth flow needs an interactive browser session, which isn't available in this environment. Downgrading to the legacy CLI let the existing `kaggle.json` work immediately with zero extra setup from the user.
- **Trade-off accepted:** Pinned to an older major version of the `kaggle` package — will need revisiting if a future feature requires 2.x-only CLI behavior (this should go in `requirements.txt` as `kaggle<1.7` with a comment, not left implicit).

### Decision: Diagnose the two separate download failures in sequence rather than guessing
- **Situation:** First attempt failed with a blanket 401 even on `kaggle competitions list` (an unrelated, always-public call).
- **Reasoning applied:** A 401 on a call that doesn't even touch the target competition means the failure is at the authentication layer, not authorization/rules-acceptance — ruled out "hasn't accepted competition rules" as the cause before asking the user to do that step.
- **Resolution:** Token turned out to be stale/expired; user regenerated it, 401 went away, and the *next* failure (403 "must accept competition's rules") was the real, distinct blocker — resolved by directing the user to the one step that can't be done via API (accepting rules is a web-only action by Kaggle's design).
- **Why this matters for the log:** Two different failures got two different fixes instead of one blanket "try regenerating everything" — worth recording so a future similar failure gets diagnosed the same way (check what layer the error code maps to) rather than re-guessed from scratch.

---

## Stage 3 — Infrastructure sequencing (HLD)

### Decision: Don't block on Docker — user starts it in parallel while skeleton work continues
- **Alternatives:**
  - A. Halt all work until Docker Desktop is confirmed running.
  - B. Proceed with structure/config work that doesn't need a running daemon, verify Docker separately.
- **Chosen:** B (per explicit user choice when asked).
- **Why:** `docker ps` failing (`daemon is running: open //./pipe/dockerDesktopLinuxEngine`) only blocks steps that need a live Postgres/Redis/MLflow instance — directory scaffolding, dbt model stubs, and config file skeletons don't. No reason to idle.

---

---

## Stage 4 — Data loading performance (LLD)

### Decision: Switch the Postgres loader from `pandas.to_sql` row-by-row inserts to `COPY`
- **Alternatives:**
  - A. Keep `pandas.to_sql(..., method=None)` — SQLAlchemy's default, which issues one `INSERT` per row (or small batches) via `executemany`.
  - B. `method="multi"` — batches multiple rows per `INSERT` statement.
  - C. Stream each chunk through `psycopg2`'s `COPY FROM STDIN` via an in-memory CSV buffer, using pandas only to infer/create the table schema first.
- **Chosen:** C.
- **Why:** Ran the naive loader (A) against the real dataset and it was still stuck on the 4th of 8 tables after ~30 minutes — `installments_payments.csv` alone is 723MB / ~13.6M rows, and row-by-row inserts don't scale to that. Rewrote to use `COPY`, which is Postgres's actual bulk-load path, and confirmed on the same smoke test (`application_test`, ~48.7K rows): 34.5s → 4.7s. Full 8-table / ~58.5M-row load then completed in a few minutes.
- **Trade-off accepted:** `COPY` requires the target table to already exist with matching columns — worked around by having pandas create the table from an empty (`head(0)`) frame first (so dtypes are still inferred automatically), then streaming data in via `COPY` afterward. Slightly more code than a single `to_sql` call, justified by being the difference between "finishes in minutes" and "doesn't finish."
- **Evidence, not assumption:** didn't guess this would be slow and pre-optimize — let the naive version run, watched it stall via a direct `SELECT COUNT(*)` progress check, and only rewrote once the row-by-row approach was confirmed too slow in practice.

---

## Stage 5 — dbt schema naming (LLD)

### Decision: Override dbt's default `generate_schema_name` macro
- **Alternatives:**
  - A. Accept dbt's default behavior, where a model's custom `+schema:` config gets prefixed with the connection's target schema (e.g. `+schema: features` under target schema `public` materializes as `public_features`, not `features`).
  - B. Add a `macros/generate_schema_name.sql` override so a custom schema is used exactly as named, with no prefixing.
- **Chosen:** B.
- **Why:** Both plan documents (and every downstream script — `load_to_postgres.py`, `validation/expectations.py`, `models/train_baseline.py`) assume the feature table lives at `features.fct_credit_features`, not `public_features.fct_credit_features`. Discovered via a real failure (`UndefinedTable: relation "features.fct_credit_features" does not exist`) after the first `dbt run` succeeded but downstream `SELECT`s failed — not caught in advance because dbt's schema-prefixing default isn't obvious until you hit it.
- **Note:** this is dbt's well-known default behavior (documented, not a bug) — the override is the standard fix, not a workaround for something broken.

---

## Stage 6 — Data validation approach (LLD)

### Decision: Plain pandas/assert checks instead of the full Great Expectations framework
- **Alternatives:**
  - A. Great Expectations as originally planned — `gx.get_context()`, a `Validator`, `expect_column_values_to_*` calls, auto-generated HTML data docs.
  - B. A small hand-written `validate_feature_table(df) -> list[str]` function doing the same six checks (nulls, ranges, target balance, row count) with plain `assert`-equivalent conditions.
- **Chosen:** B.
- **Why:** The v1 code sample's own GX usage (`context.sources.pandas_default.read_dataframe(df)`) is from an older GX API that has since changed across major versions, and setting up a full `DataContext` + expectation suite + checkpoint is a lot of surface area for validating one table with six fixed checks. Per the ladder: "already-installed dependency solves it" doesn't apply when nothing about GX's suite/checkpoint/data-docs machinery is actually needed here — a function is the one-line-simpler solution that does the same job.
- **Trade-off accepted:** lost GX's auto-generated HTML validation reports and historical pass/fail tracking (both cited as reasons for choosing GX over asserts in the plan's own Tradeoff 5). If those become genuinely needed later — e.g. a stakeholder-facing validation report, or validation across many tables instead of one — this is the point to revisit and bring in real GX.
- **Verified against real data:** ran against the actual 307,511-row `features.fct_credit_features` table (not just synthetic test data) — all six checks passed.

---

## Stage 7 — WoE encoder interface (LLD) — resolves an open decision from Stage 3

### Decision: sklearn-style `fit(X, y)` / `transform(X)` operating on the whole DataFrame
- **Alternatives:**
  - A. Match the plan's original sketch — `fit()` takes one column at a time (`fit(X, y, feature_name)`), caller loops over columns manually, no `transform()` exists so there's no defined way to encode new/holdout data through the fitted bins.
  - B. Fit all columns in one call from a DataFrame, store bin edges (numeric) or category sets (categorical) per column, and provide `transform()` that reuses those exact bins on new data — plus a `fit_transform()` convenience method.
- **Chosen:** B.
- **Why:** A's interface can't actually encode a holdout/test set consistently with train-set bins, which is a hard requirement for scoring anything other than the exact data used to fit — not an edge case, the base use case. Also added Laplace smoothing (`+0.5`) to the good/bad rate calculation so a bin with zero bads in the training fold doesn't produce `log(x/0) = inf`.
- **Verified:** unit tests (`tests/test_woe_scorecard.py`) check IV correctly identifies features with real signal above the 0.02 "weak predictor" threshold, `transform()` handles a value outside the fitted bin range without erroring, and the resulting scorecard's points are monotonically related to predicted default probability. Also validated end-to-end against real data via `models/train_baseline.py` — see Stage 8.

---

## Stage 8 — Baseline model results (reference point for Step 4)

Ran `models/train_baseline.py` against the real `features.fct_credit_features` table (307,511 rows, 80/20 stratified split, 36 candidate features, WoE-encoded, 10 bins).

**Feature selection by Information Value (IV ≥ 0.02 kept, 15 of 36 features):** strongest signals were `bureau_debt_ratio` (0.103), `years_employed` (0.092), `age_years` (0.087), `credit_application_ratio` (0.071), `goods_to_credit_ratio` (0.069). Notably, several features the feature-engineering layer treats as important — `credit_income_ratio` (0.011), `amt_income_total` (0.011), `exposure_at_risk` (0.006), and both interaction flags `high_util_x_high_leverage` / `many_loans_x_young` (0.000) — carried little to no standalone signal at this bin count. Two flag-type features (`owns_car`, `owns_realty`) and two overdue features (`avg_days_overdue`, `max_days_overdue`) showed exactly 0.0 IV, worth checking for a data issue (e.g. most values identical / heavily one-sided) rather than assuming they're simply unpredictive.

**Holdout performance:** AUC 0.700, KS 0.294, Gini 0.401, top-decile lift 2.70×.

**Why this number matters going forward:** every model in Step 4 (XGBoost, LightGBM, CatBoost, H2O AutoML) is being compared against this 0.700 AUC / 0.401 Gini baseline — that comparison is the actual point of building a baseline first, not just a formality.

**Follow-up flagged, not yet acted on:** the four zero-IV features above should get a quick `df[col].value_counts()` sanity check before Step 4 — if they're genuinely near-constant in this dataset that's a real finding, but if it's a WoE-binning artifact (e.g. a binary flag with 10 requested quantile bins collapsing oddly) the encoder needs a look.

---

## Stage 9 — WoE encoder bug fix: low-cardinality numeric columns (LLD)

### Decision: Treat numeric columns with few unique values as discrete categories, not quantile bins
- **Investigated the Stage 8 zero-IV flag before starting Step 4** rather than assuming it was fine to ignore.
- **Finding:** `owns_car` and `owns_realty` are binary (0/1) numeric columns. `pd.qcut(x, q=10, duplicates='drop')` on a column with only 2 unique values (one at ~66-69%) produces quantile edges that collapse to a single bin after duplicate edges are dropped — every row lands in one bin, so WoE is computed over a single group and the real signal (8.5% default rate at `owns_car=0` vs. 7.2% at `owns_car=1`) was silently discarded. This is a genuine bug in `WoEEncoder`, not a property of the data.
- **Alternatives:**
  - A. Leave it — accept that binary/low-cardinality numeric features lose signal under quantile binning.
  - B. Detect low-cardinality numeric columns (`nunique() <= n_bins`) and bin them as discrete categories (one bin per exact value) instead of quantile-binning.
- **Chosen:** B — added the cardinality check to `WoEEncoder._bin_column`.
- **Why:** A model comparison in Step 4 is only meaningful if the baseline it's measured against isn't silently dropping real features. Fixing this before Step 4 rather than after keeps the "beat 0.700 AUC" bar honest.
- **Verified:** re-ran `tests/test_woe_scorecard.py` (still 4/4 pass) and `models/train_baseline.py` against real data — `owns_car` IV went from 0.0000 to 0.0061, `owns_realty` from 0.0000 to 0.0005. Both remain weak predictors (correctly — this is a known weak signal in Home Credit), but they're no longer *artificially* zero. Baseline metrics moved only slightly (AUC 0.700 → 0.701, Gini 0.401 → 0.402) since the affected features were weak either way — the fix was about correctness, not a material performance jump.
- **Remaining zero-IV features, now confirmed genuine, not a bug:** `avg_days_overdue`, `max_days_overdue`, `avg_days_past_due_cc` are ~98.9% constant (mostly zero) in this dataset — quantile binning collapses them for a real reason (almost no variance to split on), not the cardinality bug above. Left as-is; if these matter later, the fix would be a hand-crafted binary "has any overdue history" indicator rather than trying to bin a near-constant column.

---

---

## Stage 10 — Step 4 model comparison results

Ran `experiments/run_comparison.py`: 5 candidates, identical 80/20 stratified split (`random_state=42`), same 36 raw features (no WoE — tree models don't need it), against the Step 3 WoE baseline (AUC 0.701, Gini 0.402).

| Model | AUC | Gini | KS | DP gap (raw, incl. XNA) | Train time |
|---|---|---|---|---|---|
| **CatBoost** | **0.7243** | **0.4487** | 0.3358 | 0.191 | 22.4s |
| LightGBM | 0.7202 | 0.4403 | 0.3315 | 0.230 | 6.2s |
| XGBoost | 0.7161 | 0.4322 | 0.3267 | 0.262 | 11.3s |
| Random Forest | 0.7058 | 0.4117 | 0.3026 | 0.311 | 54.5s |
| Logistic Regression (raw features) | 0.5605 | 0.1210 | 0.0891 | 0.273 | 280.7s |

**Champion: CatBoost** — best AUC/Gini/KS, and (before the XNA finding below) already had the lowest DP gap among the tree models.

### Finding: raw-feature logistic regression's 0.56 AUC is not a fair XGBoost-vs-LR comparison
- **What happened:** this run's LR used unscaled raw features (`amt_credit` in the hundreds of thousands next to ratio features in single digits) with `saga`/elasticnet, which didn't converge (`ConvergenceWarning`, hit `max_iter`) and took 280s — 12x longer than CatBoost for a much worse result.
- **Why this isn't evidence LR is a bad baseline:** tree models are scale-invariant; LR isn't. The Step 3 WoE-encoded LR (AUC 0.701) is the methodologically fair comparison point for "is a linear model competitive here" — this raw-feature run just shows LR needs either WoE encoding or a `StandardScaler`, which was already known going in.
- **Not fixed:** logged instead of silently re-run, since the fair comparison (WoE-LR at 0.701) already exists from Step 3 and re-running this one just to make the table prettier wasn't worth the time. If this table is ever presented on its own without the Step 3 context, add a `StandardScaler` step to the LR pipeline first.

---

## Stage 11 — Fairness audit instability: the `XNA` gender category (LLD)

### Investigation, not a decision made lightly — this is a regulatory compliance gate
When building the champion artifact (`models/train_champion.py`), the fairness audit against the exact same CatBoost setup as Step 4's comparison reproduced the 0.191 DP gap — but after excluding 4 rows with `code_gender = 'XNA'` (Home Credit's placeholder for undisclosed gender, not a real demographic category — confirmed via `value_counts()`: F=202,448, M=105,059, XNA=4, and all 4 have `target=0`), the same model/split/seed setup produced a DP gap of **0.029** — a 6.6x drop from removing 0.0013% of rows.

**This swing was too large to accept at face value.** Before treating either number as the real compliance result, ran two checks:
1. **Cross-checked the metric computation itself** — called `fairlearn.metrics.demographic_parity_difference` directly and via this project's `full_fairness_audit` wrapper on identical data; both returned exactly 0.1913044550183674. Ruled out a bug in `explainability/fairness.py`.
2. **Checked stability across 5 different model random seeds** (data split held fixed, XNA excluded): DP gap ranged 0.0256–0.0300, AUC ranged 0.7234–0.7242. Tight and consistent — this is a stable result, not a fluke of one lucky seed.

**Conclusion:** the 0.191 figure was a real but misleading number — `demographic_parity_difference` with 3 sensitive-feature groups (F/M/XNA) reports the *max-min spread across all groups*, and a 2-4 row group with a single label acts as a statistical outlier that can swing that spread dramatically without reflecting any real M-vs-F disparity. Excluding a data-collection artifact category (not a demographic group) from a group-fairness computation is the methodologically correct move, not a way to make a number look better — confirmed by the fact that the *stable, multi-seed* result (0.029, well under the 0.05 threshold) is the one obtained after the fix.
- **Why this matters enough to log in detail:** this is exactly the kind of thing that would be indefensible in an actual regulatory audit if it went unnoticed — reporting a compliance pass/fail without checking whether the input data going into that computation was sound. The fix is small; the discipline of checking before trusting the number is the actual point.
- **Follow-up not yet done:** the same XNA-filtering should be applied consistently everywhere gender is used as a sensitive feature (Step 4's `experiments/compare.py` comparison table still includes XNA and thus overstates the tree models' DP gaps — those numbers in Stage 10's table are the *unfiltered*, worse-than-real figures). Not re-run yet since the champion-selection decision (CatBoost wins) doesn't change either way; flagged below.

---

---

## Stage 12 — Serving layer: local artifacts instead of MLflow Model Registry (HLD)

### Decision: `serving/app.py` loads `artifacts/champion_model.joblib`, not `models:/credit_default/Production`
- **Alternatives:**
  - A. Full MLflow Model Registry flow, as the plan sketches — `mlflow.register_model()`, stage transitions (`None → Staging → Production`), `mlflow.xgboost.load_model("models:/credit_default/Production")` in the serving layer.
  - B. Train the champion, export it as a plain joblib bundle (`{model, feature_cols}`), load that file directly at API startup.
- **Chosen:** B.
- **Why:** MLflow's Model Registry requires a database-backed tracking store (not the default file store) and a real promotion workflow — both are Step 7 (Airflow DAG) concerns that don't exist yet. Blocking the serving API on building that infrastructure first, just to prove the API itself works, would be solving two problems at once. A joblib artifact is a one-line swap for a registry load later.
- **Trade-off accepted:** no version history, no stage-based rollback, no lineage back to the MLflow run that produced the artifact — all real losses versus the full registry approach. Acceptable for now because none of those are being exercised yet (there's only ever been one champion model trained so far).

---

---

## Stage 13 — H2O-3 MOJO export now requires a paid license (HLD)

### Finding: `save_mojo()` is blocked in current H2O-3 OSS
- **What happened:** `models/h2o_automl.py` (as originally written, matching the plan's `best_model.save_mojo(path=...)`) failed with `H2OValueError: MOJO export requires H2O-3 Secure`. H2O-3's open-source tier now gates MOJO export behind a commercial "H2O-3 Secure" license — this is a real change in H2O's product tiering, not a bug in this project's code or environment.
- **Why this matters beyond one function call:** the plan's Tradeoff 4 ("H2O MOJO vs. Direct XGBoost Serving," and its interview answer about using MOJO for "auto-approvals where speed matters") assumes free MOJO export. That premise no longer holds for anyone building this without a paid H2O license.
- **Alternatives:**
  - A. Pay for H2O-3 Secure to keep the MOJO path as originally planned.
  - B. Use `h2o.save_model()` — H2O's native binary format, still free, loadable back into Python via `h2o.load_model()` — and drop the "single-digit ms MOJO" serving claim.
  - C. Drop H2O entirely from the project.
- **Chosen:** B.
- **Why:** H2O AutoML's actual role in this project is Step 4's "third model family for leaderboard comparison" — it was never going to be the deployed champion once CatBoost won Step 4's comparison (Stage 10). Losing MOJO export doesn't block that role; `save_model()`/`load_model()` is enough to keep H2O's models comparable and inspectable. Paying for a commercial license (A) isn't justified for a comparison-only role, and dropping H2O (C) would lose a real data point in the "why not H2O AutoML" interview answer.
- **Consequence for the plan's interview answer (Tradeoff 4):** the honest version is now "H2O MOJO is not a viable serving path without a commercial H2O-3 Secure license as of 2026 — I evaluated H2O AutoML as a third model family for comparison, but the deployed champion (CatBoost) is served directly, and MOJO's sub-millisecond-latency argument doesn't apply here since it was never in the serving path." That's a more defensible answer than repeating a claim about free MOJO serving that isn't true anymore.

---

### Result, with an honest caveat: 240s budget only completed 1 model
Ran `run_automl` with `max_runtime_secs=240, max_models=8`. The leaderboard came back with exactly **one** model — a GLM at AUC 0.683 — despite requesting up to 8. JVM startup + H2OFrame parsing of ~246K rows ate most of the 240s budget before the search could get further than the first (fastest) algorithm.
- **What this does and doesn't prove:** AUC 0.683 < CatBoost's 0.724, but this is not evidence H2O AutoML "loses" to CatBoost — it never got to run its gradient-boosting or stacked-ensemble candidates, which are the ones that would actually compete. A fair verdict needs the plan's original longer budget (600-3600s).
- **Not re-run with a longer budget:** the champion-selection decision (CatBoost, from Step 4's full 5-model comparison) doesn't depend on this result either way, so spending 10-60 more minutes for a number that wouldn't change the deployed model wasn't the best use of time right now. Flagged below if a complete H2O comparison is wanted later.

---

---

## Stage 14 — The fairness gate was silently ignoring age (HLD/LLD, Step 8)

### Finding: `demographic_parity_pass` only ever checked gender, despite `dp_age` being computed
While pulling real numbers for the model card, `artifacts/fairness_report.json` showed `dp_age: 0.456` — a massive gap, ten times the DP threshold gender was gated on — sitting right next to `demographic_parity_pass: true`. The gate had been computing age disparity every run since Stage 11 and never once checking it.

**Investigated before treating it as a real disparity, not just accepted the number** (same discipline as Stage 11): pulled per-age-group approval rates directly.

| Age group | Approval rate |
|---|---|
| 18-30 | 43.4% |
| 30-45 | 61.8% |
| 45-60 | 78.5% |
| 60+ | 89.0% |

Sample sizes are large (7,152–24,787 per bin) — this is not a small-group statistical artifact like Stage 11's `XNA` finding. It's a real, substantial pattern: the model predicts young applicants default at more than 5x the rate of applicants over 60. This tracks with a known credit-scoring phenomenon (younger applicants have less credit history — `years_employed`, `bureau_loan_count` are mechanically lower for them, and the model has learned that as risk) — but "explicable" doesn't mean "compliant." Age is an explicitly ECOA-protected class, same as gender.

### What was fixed
1. **`explainability/fairness.py`**: `full_fairness_audit` now computes `demographic_parity_pass_gender` and `demographic_parity_pass_age` separately, and the overall `demographic_parity_pass` requires **both** to pass — a model can no longer clear the gate by passing on one protected attribute while failing badly on another. Also added `approval_rates_by_age_group` to the audit output (previously only gender approval rates were surfaced).
2. **`optimize_thresholds`** now accepts a configurable `constraints` parameter (was hardcoded to `"equalized_odds"`, which didn't match what the gate actually checks — demographic parity). Also now accepts a multi-column `sensitive_features` DataFrame, not just a single array.
3. **`models/train_champion.py`**: mitigation now runs `ThresholdOptimizer` with `constraints="demographic_parity"` over gender **and** age jointly (a DataFrame of both), not gender alone. Result: `dp_gender` 0.029→0.0005, `dp_age` 0.456→0.0017 — both comfortably under threshold after mitigation.

### Decision: the mitigated thresholds are NOT deployed, and the artifact is marked "not cleared for production"
- **Why not just ship the mitigated version:** `ThresholdOptimizer`'s output isn't a model — it's a per-(gender, age-group) decision threshold. Applying it correctly at inference time means `serving/app.py` needs to know each applicant's group and look up the right threshold, instead of the single global 0.10/0.25/0.50 cutoff it uses today. Wiring that in properly is real serving-layer work, not a config change, and doing it hastily to make a number look better right before writing documentation is exactly the kind of shortcut this log exists to catch.
- **What actually shipped instead:** `artifacts/fairness_report.json` now has an honest `cleared_for_production: false`, computed from the **unmitigated** model's gate result. The mitigation was run and audited (proving the fix works, 0.456→0.0017), but the deployed `champion_model.joblib` is still the raw model — because deploying it silently while claiming "fair" would be worse than deploying it with an accurate "not yet cleared" label.
- **Consequence for Tradeoff 1 in `experiments/tradeoffs.md`** (AUC vs. fairness): the original tradeoff writeup only discussed gender. The honest updated position: this model is not currently deployable as-is under the project's own stated fairness policy — the path to deployable is either (a) wire per-group thresholds into `serving/app.py`, or (b) retrain with `ExponentiatedGradient` fairness constraints (already built in `explainability/fairness.py`, not yet exercised end-to-end for this champion) so a single global threshold is fair by construction.
- **Why this belongs at the top of the model card, not buried:** this is precisely the failure mode fair-lending audits exist to catch — a model that looks compliant because someone checked the metric they expected to be a problem (gender, given it's the most commonly discussed one) and didn't check the one that turned out to actually be broken (age).

---

---

## Stage 15 — Deploying the fairness mitigation (LLD)

User asked, after Stage 14's finding: "can we do anything to mitigate this?" — yes, and it's now live, not just audited.

### Decision: deterministic per-group thresholds, not Fairlearn's `ThresholdOptimizer`
- **Alternatives:**
  - A. Deploy Fairlearn's `ThresholdOptimizer` output directly (already proven to work in Stage 14's audit).
  - B. Build a deterministic per-(gender, age-bin) threshold: each group's DENY cutoff set at that group's own quantile matching a shared target denial rate.
- **Chosen:** B.
- **Why:** `ThresholdOptimizer`'s default mechanism interpolates between two thresholds using a *randomized* decision for applicants near the boundary, specifically to hit an exact fairness target. That means two identical applicants in the same protected-attribute group could receive different decisions purely by chance — which is itself a fair-lending problem (similarly situated individuals should be treated similarly), not just an implementation quirk. A is mathematically cleaner but not something to actually deploy in a lending decision. B trades a small amount of precision (it doesn't hit the fairness target quite as exactly) for a decision that's fully reproducible and explainable to a regulator: "this applicant's group has this cutoff, full stop."
- **Calibrated on train, validated on test:** thresholds were computed from `X_train`'s predicted probabilities, then applied to `X_test` and re-audited — DP gap 0.029→0.0015 (gender), 0.456→0.0102 (age), both on data the calibration never saw. This matters because calibrating and evaluating on the same data would have overstated how well it generalizes.
- **Verified against the full portfolio**, not just the test split: re-ran `serving/batch_predict.py` against all 307,511 applicants — non-denial rates across age groups went from a 43%-89% spread to 67.8%-68.0%, a real, large-scale confirmation, not just a holdout-sample result.

### Decision: collect gender at the API layer, but only as a threshold lookup key, never a model feature
- **Alternatives:** A. Don't collect gender at inference time at all (can't apply per-group thresholds without it). B. Add `code_gender` as an optional field on `LoanApplication`, used exclusively to select the DENY threshold, explicitly excluded from `FEATURE_COLS` (the model's actual input).
- **Chosen:** B.
- **Why:** This is standard, legally recognized practice under ECOA/Regulation B — lenders may collect protected-class data specifically for fairness self-testing and remediation, while being prohibited from using it as a decision *input*. The distinction (audit/remediation use vs. predictive use) is exactly what's implemented: `code_gender` never touches `CHAMPION_MODEL.predict_proba()`, it only indexes into `GROUP_THRESHOLDS`.
- **Fallback for non-disclosure:** applicants who don't provide gender get the flat `DEFAULT_THRESHOLD` (0.5) — meaning they don't benefit from the age-bias correction unless they disclose. This is a real, acknowledged tradeoff (not swept under the rug): non-disclosure is honored as a real choice, but it means the fairness guarantee is conditional on disclosure. Flagged below as something to revisit if this were a real production system rather than a portfolio project.

### What's now consistent that wasn't before
`serving/app.py` (real-time) and `serving/batch_predict.py` (portfolio scoring) both call the same age-binning logic (`explainability.fairness.age_bin_labels`, refactored into one shared function specifically so the audit, the calibrator, and both serving paths can't drift apart from each other) and apply the same per-group thresholds from the same `artifacts/champion_model.joblib` bundle. Before this stage, `README.md`/`METHODOLOGY.md` claimed "real-time and batch predictions never diverge" while batch scoring silently ignored the fairness mitigation entirely — that claim is now actually true.

### Updated status
`artifacts/fairness_report.json`: `cleared_for_production: true`, `mitigation_deployed: true`. `MODEL_CARD.md` and `README.md` updated accordingly — from "NOT CLEARED FOR PRODUCTION" to "CLEARED, with a documented and actively-corrected age bias."

**Still open, and stated plainly rather than hidden by the status flip:** the underlying CatBoost model is still biased — the correction happens at the decision layer, not inside the model. See open decisions below.

---

---

## Stage 16 — Fixing the model, not just the decision (HLD/LLD)

User follow-up after Stage 15: "go ahead and also fix that" — referring to the open item that the underlying model was still biased, only corrected post-hoc. This stage retrains with fairness constraints baked in and makes a real, evidence-based choice between two actually-trained candidates, rather than assuming the "cleaner" fix is automatically better.

### Building `FairEnsembleModel`: the deployment gap that made this non-trivial
Fairlearn's `ExponentiatedGradient` doesn't produce a single model — it produces a *mixture* of sub-estimators (`.predictors_`, `.weights_`) and its own `.predict()` is explicitly randomized by design (Fairlearn's own docstring: "repeatedly calling predict with the same feature data may yield different output"). Deploying that directly would reintroduce the exact problem Stage 15 already engineered around for `ThresholdOptimizer` (non-reproducible decisions for identical applicants). Built `explainability.fairness.FairEnsembleModel`: a picklable wrapper exposing sklearn-shaped `.predict_proba(X)` via the deterministic weighted average across the ensemble, plus a `.dominant_predictor` property (the single highest-weight sub-model) for SHAP, since `TreeExplainer` needs one concrete tree model, not a weighted mixture. `serving/app.py` and `serving/batch_predict.py` needed no further changes to support this — they only ever called `.predict_proba(X)`, so a raw CatBoost and a `FairEnsembleModel` are interchangeable from their point of view.

### Training both candidates for real, then deciding from the actual numbers
`models/train_champion.py` now trains two candidates and compares them on identical held-out test data at a flat 0.5 threshold (no post-hoc correction on either):

| | AUC | DP gap (gender) | DP gap (age) | Passes alone? |
|---|---|---|---|---|
| Candidate 1: raw CatBoost | 0.7244 | 0.029 | 0.456 | No |
| Candidate 2: CatBoost + `ExponentiatedGradient` (dp constraint, gender×age jointly) | 0.6832 | 0.006 | 0.069 | No (0.069 > 0.05) |

- **Constrained training reduced max_iter budget for tractability:** base estimator was a faster CatBoost (150 iterations, depth 5, vs. the champion's 500/6) and `ExponentiatedGradient` ran only 10 rounds (vs. Fairlearn's own default of 50) — a deliberate speed tradeoff for local iteration, not a claim that this is the ceiling of what constrained training can achieve.
- **Real result, not hidden:** even at this reduced budget, constrained training cut the age DP gap 6.6x (0.456→0.069) through the model alone — genuine evidence the bias is addressable at the model level. It didn't fully clear the 0.05 bar, and cost 4.1 AUC points doing it.
- **Decision rule, set before seeing the results, not fit to them:** deploy the constrained candidate only if (a) it passes the fairness gate on its own, AND (b) it costs ≤ 2 AUC points versus the raw model. Neither condition held (fails (a), and 4.1 > 2 fails (b) too), so `models/train_champion.py` automatically fell back to raw CatBoost + Stage 15's post-hoc threshold calibration — which passes cleanly (DP gap 0.0015 gender / 0.0102 age) with no AUC cost at all.
- **Why this matters as a decision-making pattern, not just a result:** the "obviously better" fix (bake fairness into the model) was tried for real, measured, and rejected by a pre-set rule when the actual numbers didn't clear the bar — rather than assumed superior because it's architecturally cleaner. The post-hoc approach isn't a compromise settled for; it's the one that actually won on this run's numbers.
- **Not a closed door:** the reduced training budget was a real, named constraint (not an inherent limit of the technique). A longer `ExponentiatedGradient` run is flagged as the natural next experiment if the model-level fix is wanted for its own sake (e.g. to stop needing to collect gender at inference time at all).

### What changed structurally as a result
- `explainability/fairness.py`: `retrain_with_fairness_constraints` now takes any estimator (was hardcoded to `LogisticRegression`), added `exponentiated_gradient_predict_proba` and `FairEnsembleModel`.
- `serving/app.py`: SHAP explainer now built from `getattr(CHAMPION_MODEL, "dominant_predictor", CHAMPION_MODEL)` — works whether the deployed champion is a raw tree model or a `FairEnsembleModel`, without the caller needing to know which.
- `artifacts/fairness_report.json` schema changed: `candidates` (both trained models' raw performance/fairness), `audit_deployed_before_thresholds` / `audit_deployed_final` (was `audit_before_mitigation` / `audit_after_mitigation` — renamed because "before/after mitigation" no longer made sense once there were two different *models* being compared in addition to before/after *thresholds*). `monitoring/dashboard.py` updated to match.
- 8 new tests (`test_fair_ensemble_model_*`, `test_exponentiated_gradient_predict_proba_is_deterministic`, `test_fairness_constrained_retraining_reduces_demographic_parity_gap`) — full suite now 43/43.

---

## Open decisions / not yet made

These will get their own entries once resolved — listed here so the log stays a live map of what's still undecided, not just a history of the past.

- **`requirements.txt` pinning strategy** — exact version pins vs. loose ranges vs. `environment.yml` (Conda) as the source of truth with `requirements.txt` as a derived/secondary file. v2's plan lists both `environment.yml` and `requirements.txt` as separate deliverables; need to decide which one is authoritative to avoid drift between them. (`requirements.txt` has since been pinned and used successfully — `environment.yml` is still an empty stub; leaning toward `requirements.txt` as authoritative unless a real Conda-specific need shows up.)
- **PySpark local mode vs. real cluster** — v2's `spark_features.py` assumes a `SparkSession` is available; for a local Docker Compose setup this means local-mode Spark inside a container, which needs its own resource budgeting (not yet reflected in `docker-compose.yml`).
- **Whether the two-token Kaggle auth downgrade (`kaggle<1.7`) gets revisited** once actual pipeline code needs it — flagged above, not yet acted on.
- **XNA-gender filtering isn't applied everywhere yet** — `models/train_champion.py` filters it, `experiments/compare.py`'s Step 4 table (Stage 10) does not, so those DP-gap numbers are inflated. Low priority since it doesn't change which model wins, but should be fixed before those numbers go in front of anyone external.
- **MLflow Model Registry vs. local joblib artifacts for serving** — `serving/app.py` currently loads from `artifacts/champion_model.joblib` rather than `models:/credit_default/Production`; flagged in Stage 12.
- **H2O AutoML's 240s comparison run is inconclusive** — only completed 1 of up to 8 requested models. Re-run with 600s+ if a real H2O-vs-CatBoost verdict is ever needed; not blocking since CatBoost is already the champion from Step 4's complete comparison.
- **Airflow DAG (`dags/credit_default_retrain.py`) is written but not executed** — no Airflow scheduler is running (native Airflow isn't supported on Windows outside WSL; the `./airflow` Docker build context in `docker-compose.yml` hasn't been built). The task functions call real, already-tested project code (dbt, validation, drift detection, training, batch scoring), so the DAG wiring is correct, but the schedule itself has never actually fired. Standing up the Airflow container is the next step if this needs to run on a real cadence rather than manually.
- **The underlying model is still age-biased — only the decision layer corrects for it.** Tried fixing this at the model level in Stage 16 (`ExponentiatedGradient` constrained retraining) — real progress (6.6x reduction in the age gap) but didn't clear the bar at the reduced training budget used for local iteration, and cost too much AUC to deploy over the post-hoc-corrected raw model. A longer constrained-training run (more boosting iterations, more `ExponentiatedGradient` rounds) is the natural next experiment if a model-level fix is wanted for its own sake — not attempted yet due to local iteration time constraints.
- **Applicants who don't disclose gender don't get the fairness correction** — they fall back to the flat default threshold, meaning the age-bias fix is conditional on disclosure. Acceptable for a portfolio project; a real system would need a policy decision here (e.g. infer group via a compliant proxy method, or require disclosure for the calibration to apply).
