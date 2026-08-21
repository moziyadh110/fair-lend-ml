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

---

## Stage 17 — The improvement-list work: items #1–#5 (HLD/LLD)

User asked "what can we do to improve from last time," got a prioritized list back, then said "go one by one and complete all from 1-5." This stage covers all five, in order, including two real infrastructure blockers hit and fixed along the way.

### Item #1: full-budget `ExponentiatedGradient` run — real result, didn't change the conclusion
Ran the constrained retrain at the champion's own settings (500 iterations, up to 50 rounds — vs. the reduced 150/10 from Stage 16) via `experiments/full_fairness_retrain.py`. 26 minutes, 21 sub-estimators fitted (6 with nonzero weight).

| | Reduced budget (Stage 16) | Full budget |
|---|---|---|
| AUC cost vs. raw champion | 0.041 | **0.014** (now within the ≤0.02 rule) |
| DP gap (age) | 0.069 | **0.093** (worse, not better) |

**Non-monotonic, and reported honestly rather than picking whichever framing looked better.** More training capacity bought back almost all the AUC cost, but the age fairness gap got *worse*, not better — Fairlearn's reduction algorithm optimizes error subject to the constraint, not "improve fairness monotonically with more compute." Since `demographic_parity_pass` still fails at full budget too, the deployed system's decision (raw model + post-hoc thresholds) doesn't change — but this is now backed by two independent real attempts at different budgets, not one.

### Item #2: Airflow — built, deployed, actually executed a real DAG run
- `airflow/Dockerfile` (official `apache/airflow:2.9.2-python3.11` base + `airflow/requirements-airflow.txt` for the DAG's own dependencies) and a `docker-compose.yml` `airflow` service (whole project mounted at `/opt/airflow/project`, `command: standalone`, `POSTGRES_HOST=postgres` overriding the mounted `.env`'s `localhost` value since python-dotenv doesn't override already-set env vars).
- **Hit a real disk-full blocker mid-build** — Docker had accumulated 65GB of images and 48GB of build cache across *all* projects on this machine, not just this one, leaving 1.8GB free. `docker system prune` only reclaimed space inside Docker's own virtual disk (a known WSL2/Docker Desktop quirk — the backing `.vhdx` doesn't auto-shrink); actually freeing Windows disk space required the user to compact it manually, which they did.
- **Hit what looked like a hang, wasn't one** — after starting, the container spent ~10 minutes logging repeated `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'` warnings (a `cryptography`/`pyOpenSSL` version mismatch from our extra pip installs colliding with Airflow's own pinned providers — specifically breaking the SFTP/Snowflake provider imports, which this project doesn't use). Verified it wasn't actually stuck by checking `docker stats` (real CPU activity) and, decisively, `curl localhost:8080/health` returning `200` — the webserver was up the whole time; the log noise was cosmetic.
- **Real DAG execution, not just an import test:** unpaused and triggered `credit_default_retrain` via `airflow dags trigger`. Full run completed in ~2.5 minutes: `dbt_run` → `validate_data` → `check_drift` → branched to `skip_retrain` (correctly — no drift on a same-population reference/current split) → `batch_score_portfolio`. Every task `success`.
- **What this proves that wasn't proven before:** the DAG's Python callables aren't just syntactically valid, they actually import and execute the real project modules (dbt subprocess call, `data_engineering.validation`, `monitoring.drift_detector`, `serving.batch_predict`) inside a genuinely separate Airflow-managed process, end to end.

### Item #3: `spark_features.py` / `dask_features.py` — built with real features, hit three real environment bugs
Built two genuinely new, non-placeholder features: a per-applicant payment-trend slope via Spark (`groupBy().apply()` + `numpy.polyfit`, on `installments_payments.csv`) and a rolling credit-utilization-volatility feature via Dask (`credit_card_balance.csv`). Both are the "20% of features awkward in SQL" the plan calls for — a per-group regression slope and a per-group rolling statistic aren't natural dbt/SQL operations.

Three real bugs surfaced and fixed, in order:
1. **`pyspark==4.2.0` requires Java 11+; this machine has Java 8.** First symptom was a red herring (`Invalid maximum heap size: -Xmx4g` — this machine's JVM turned out to be a 32-bit HotSpot Client VM with a ~256MB default heap ceiling, so a 4g request couldn't even be represented). Reducing to `512m` didn't fix it — the real error underneath was `TypeError: 'JavaPackage' object is not callable` on `SparkSession` construction, the classic symptom of a Java/Spark version mismatch. **Fixed by pinning `pyspark==3.5.6`** (last major line supporting Java 8), not by chasing the misleading heap-size symptom further.
2. **Stale `PYSPARK_PYTHON` env var** on this machine pointed at a conda environment (`C:\Users\moham\anaconda3\envs\pyspark\python.exe`) that no longer exists — Spark's worker subprocess launch failed with `CreateProcess error=2`. Fixed by having `spark_features.py` set `PYSPARK_PYTHON`/`PYSPARK_DRIVER_PYTHON` to `sys.executable` itself, before importing `pyspark.sql` — makes the script self-contained regardless of whatever else is set in the host shell, rather than requiring the user to fix their environment.
3. **My own test had a sign error** — `test_slope_udf_detects_declining_payments` asserted `slope > 0` for a scenario where payments decline as `days_instalment` rises toward 0, which is a *negative* slope by construction. Caught by actually running the test rather than assuming the assertion was right because the function name matched the intent.

Real jobs run against real data: Dask computed `utilization_volatility` for 103,558 applicants; Spark's payment-trend slope job is running as of this stage (see next update if still in progress when this was written).

### Item #4: non-disclosure fairness policy — the old fallback was actively worse than doing nothing
Flagged in Stage 15/16 as an open item: applicants who don't disclose gender fell back to `DEFAULT_THRESHOLD` — the *raw model's own flat threshold*, i.e. exactly the one Stage 14 showed produces a 0.456 age gap. Non-disclosure was silently reintroducing the bias the whole calibration exists to remove.

**Fix:** `explainability.fairness.non_disclosure_threshold(thresholds)` returns the *most lenient* (lowest) threshold among all calibrated groups. An applicant who doesn't disclose is now judged at least as favorably as the best-treated disclosed group — non-disclosure can only help, never hurt, relative to the fairness-corrected population. This is a "benefit of the doubt" policy choice, not a mathematical fairness proof, and it has a real cost: it means the disclosed population's denial rate has to absorb slightly more to keep the portfolio-wide rate on target. Wired into `serving/app.py`'s `_deny_threshold`, `serving/batch_predict.py`'s `_deny_thresholds_per_row`, and `apply_group_thresholds`'s own default (was hardcoded `0.5`, now defaults to calling `non_disclosure_threshold` unless a caller explicitly overrides it).

### Item #5: XNA filter applied to `experiments/compare.py`'s comparison table
Flagged in Stage 11 as not-yet-done. Added the same `df[df["_protected_gender"] != "XNA"]` filter already used in `models/train_champion.py`. Re-ran the full 5-model comparison: every model's DP-gender gap dropped by roughly the same 6-10x factor Stage 11 found for CatBoost alone (e.g. CatBoost 0.191→0.029, XGBoost 0.262→0.028, Random Forest 0.311→0.032) — confirming the XNA-outlier effect wasn't specific to one model, it was inflating the whole table. Champion selection (CatBoost) doesn't change. `experiments/comparison_results.csv` now holds the corrected numbers; Stage 10's table above is left as the historical record of what was actually run at the time, per this file's own "don't rewrite history" convention.

---

---

## Stage 18 — The actual root cause: age was a raw model input the whole time (HLD)

User asked for a time estimate on closing the remaining gaps, ordered them "1→3→2→4→5→8". Item #1 (fix the core model bias) started with checking which features correlate with age before jumping to another `ExponentiatedGradient` run — and that check found something more direct than a proxy.

### Finding: `age_years` was a raw feature fed straight into the model
`models/train_baseline.py`'s `FEATURE_COLS` (shared by the WoE baseline and every challenger model, including the champion) included `age_years` directly. Every doc in this project — `MODEL_CARD.md`, `README.md`, `METHODOLOGY.md` — states "protected attributes (`code_gender`, `age_years`, `name_family_status`) are carried for audit but excluded from the model's feature set." That claim was false for `age_years` since the feature list was first written, through every stage of this project's fairness work (Stages 11, 14, 15, 16, and the Stage 17 improvement pass) — nobody had actually checked whether the stated exclusion was real. A `corrwith()` check against `_protected_age` surfaced it immediately: `age_years` correlated 1.0 with itself, meaning it was quite literally in the model's own input list.

**A second, related issue found the same way:** `many_loans_x_young`, one of the two engineered interaction features, is defined as `bureau_loan_count > 5 AND age_years < 30` — an explicit age-gated rule baked into a feature, not just a correlate of age. Its IV was 0.006 (essentially no signal), so excluding it cost nothing.

### Why this is a bigger problem than any proxy feature, and why it went unnoticed this long
All of Stages 14-17's fairness work treated the age bias as something the model was *inferring* from correlated signals (`years_employed`, `bureau_loan_count`, etc.) — a genuinely hard problem needing either post-hoc correction or constrained retraining, because the model was earning the disparity from real (if legally fraught) correlations. That framing was only half right: some of the disparity was the model directly discriminating on age, no inference involved. Every prior audit stage kept auditing the *output* (predictions, fairness metrics) without ever checking the *input list* against what the documentation claimed. The methodology was rigorous about verifying claims everywhere except the one place it mattered most — its own stated design principle.

### Fix and result
Removed both features from `FEATURE_COLS`. Retrained baseline and champion:

| | Before (age_years included) | After (excluded) |
|---|---|---|
| WoE baseline AUC | 0.701 | 0.696 (cost: 0.005) |
| Raw CatBoost AUC | 0.724 | 0.722 (cost: 0.003) |
| Raw CatBoost `dp_age` | 0.464 | **0.408** |

Small accuracy cost, meaningful fairness improvement on the raw model alone — but the more important result is what it unlocked for constrained retraining:

| | Stage 16 (age_years still in model) | Stage 18 (age_years removed) |
|---|---|---|
| Constrained-model `dp_age` | 0.069 / 0.093 (two attempts, neither passed) | **0.038 — passes on its own** |
| Constrained-model AUC cost | 0.041 / 0.014 | 0.023 |

**The same reduced-budget `ExponentiatedGradient` configuration that failed twice at Stage 16 now passes the fairness gate cleanly, without any post-hoc correction, once the direct age input was removed first.** This is strong evidence the earlier constrained-retraining failures weren't (only) a budget/convergence problem — they were fighting a harder problem than necessary, because the model had a direct, unconstrained channel to discriminate on age that no amount of post-processing on gender+age *combinations* was addressing at its root.

### Still deployed: raw model + post-hoc thresholds, not the constrained model
0.023 AUC cost is just above this project's own 0.02 acceptance rule for swapping in the constrained model, so `models/train_champion.py`'s existing decision logic (unchanged) still deploys the raw model with deterministic per-group thresholds on top — which now passes even more cleanly than before (`dp_gender` 0.0019, `dp_age` 0.0128). Not relaxing the 0.02 rule to fit this specific result — the rule was set before any of these numbers existed, precisely so a close call afterward doesn't get rationalized into a pass. Worth revisiting only as its own deliberate decision, not as a side effect of wanting this particular experiment to look better.

### Lesson for the process itself, not just the model
This is worth stating plainly: a fairness audit that never checks its own model's input list against its own stated exclusions has a blind spot no amount of downstream metric-checking can cover. Every future fairness-relevant feature change should include a `corrwith()` (or equivalent) pass against every protected attribute *and* a literal read of `FEATURE_COLS` against the documented exclusion list — not just a re-run of the audit metrics.

---

---

## Stage 19 — Real MLflow Model Registry, three unrelated environment bugs deep (LLD)

Item #4 from the improvement list. Builds `models/registry.py`: register the deployed champion, gate promotion on real metrics, alias it `champion`, and load it that way from serving — replacing the local-joblib-only approach from Stage 12 with the real thing this time.

### No Docker mlflow service needed after all
Stage 12 skipped building the plan's `mlflow` Docker service because it "needs a database-backed tracking store." Checking MLflow 3.x's actual default tracking URI found it's already `sqlite:///mlflow.db` — a real database, already registry-capable, with zero extra infrastructure. The Docker service from the original plan was never necessary for this project's scale; skipped as unneeded, not as deferred work.

### Aliases, not stages
MLflow deprecated stage-based promotion (`Production`/`Staging`) in favor of aliases (`set_registered_model_alias`) in 2.9+. Used aliases — `champion` — which also happens to be the exact vocabulary this project already uses everywhere else (`models/train_champion.py`, the whole champion/challenger framing in `serving/app.py`). Not a compromise; the modern API fit better than the deprecated one the plan sketched.

### Three unrelated environment bugs, found and fixed in sequence
1. **A broken, irrelevant `torch` install crashed `mlflow.sklearn.log_model`.** `torch` isn't a dependency of this project — some other tool on this machine left a broken install (missing VC++ redistributable DLLs) behind. `mlflow.sklearn.log_model` triggers scikit-learn's array-API-backend discovery, which does a `pkgutil` walk that imports every backend it knows about, including torch — and the broken DLL load crashed the whole walk with an unrecoverable `OSError`, not the `ImportError` the walk is designed to catch and skip past.
   - First fix attempt — `sys.modules["torch"] = None` — worked for scikit-learn's walk (which catches `ImportError`/`ModuleNotFoundError`) but broke a *different* mlflow function (`convert_metric_value_to_float_if_torch_tensor`) that just checks `"torch" in sys.modules`, which is `True` even for a poisoned `None` entry. Two different pieces of code checked "is torch present" two different ways, and one poisoning trick could only satisfy one of them.
   - Real fix: a `sys.meta_path` finder (`_BlockBrokenTorch`) that intercepts `import torch` and raises a clean `ModuleNotFoundError` before Python ever reaches the real, broken `torch/__init__.py`. Torch genuinely never appears in `sys.modules` under this approach, so every caller — regardless of which presence check it uses — sees the same, correct "not installed" answer.
2. **MLflow's `skops`-based serialization refused to save a `CatBoostClassifier`.** MLflow 3.x's sklearn flavor uses `skops` instead of raw pickle (a real security improvement — skops maintains an allowlist of trusted types rather than blindly unpickling anything). CatBoost isn't a standard sklearn type, so it was rejected by default. Fixed by passing `skops_trusted_types=["catboost.core.CatBoostClassifier", "explainability.fairness.FairEnsembleModel"]` explicitly — both of this project's possible champion types, trusted deliberately, not by disabling the check.
3. **The test suite itself silently overwrote the real production champion.** `tests/test_registry.py`'s tests registered dummy models under the exact same `credit_default` model name and `champion` alias used by the real `models/train_champion.py` registration — because both pointed at the same default `sqlite:///mlflow.db`. Running the full test suite after the real registration replaced the real CatBoost champion with a test `LogisticRegression` dummy, which then broke `serving/app.py` at startup (SHAP's `TreeExplainer` doesn't support `LogisticRegression`, so `test_serving.py` failed with a confusing, seemingly-unrelated error). Fixed with an `autouse` pytest fixture that points every test in that file at its own `tmp_path`-scoped SQLite database, so tests can never again collide with production registry state. Re-registered the real champion afterward and confirmed `test_serving.py` passes end-to-end using the model loaded through the registry, not just the joblib fallback.

### What's live now
`serving/app.py` tries `mlflow.sklearn.load_model("models:/credit_default@champion")` first (plus its serving metadata — feature columns, calibrated group thresholds — logged alongside it as a JSON artifact in the same run), falling back to the local joblib artifact only if nothing's been promoted yet. `models/train_champion.py` registers and conditionally promotes automatically at the end of every training run, gated on the same `cleared_for_production` flag and a minimum AUC — a model that fails the fairness gate gets registered (so it's not lost) but never gets the `champion` alias, so serving never picks it up.

Full test suite: 59/59 passing, including 4 new registry tests and confirmation that the real champion (not a test artifact) is what `serving/app.py` actually loads.

---

---

## Stage 20 — Item #5: race/ethnicity fairness coverage — a scope call, not a fix (HLD)

Checked before writing anything off: does the Home Credit dataset have *anything* usable for race/ethnicity fairness auditing, even indirectly?

**Checked the actual columns.** `application_train.csv`'s header has no surname or any name field (applicants are anonymized numeric IDs, `SK_ID_CURR`), no nationality or country-of-origin field, and no direct race/ethnicity field. The "region" columns (`REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT`, `REG_REGION_NOT_LIVE_REGION`, etc.) are anonymized relative-population and rating codes, not real geography — no ZIP code, no census tract, nothing a geocoder could resolve to an actual place.

**Why this rules out the standard method, not just makes it harder.** BISG (Bayesian Improved Surname Geocoding) — the method US regulators and lenders actually use to estimate race/ethnicity for fair-lending self-testing when it isn't self-reported — needs exactly the two things this dataset doesn't have: a surname to check against Census surname-race distributions, and a geography (tract/ZIP) to check against Census race-by-area data. Without either input, there's no defensible way to even estimate the missing attribute, let alone audit fairness against it.

**Decision: don't attempt it, and say so plainly rather than build something that looks like coverage but isn't.** A synthetic or fabricated proxy (e.g., randomly assigning race labels, or inferring from unrelated features) would produce numbers that look like a fairness audit without being one — actively worse than admitting the gap, since a fabricated "passing" audit could be mistaken for real assurance. The honest position, stated in `MODEL_CARD.md`: this model's fairness audit covers gender and age because those are the only protected attributes available in this dataset, and race/ethnicity — arguably the attribute regulators scrutinize most under ECOA/HMDA — has zero coverage here. That's a real, material limitation of *this dataset*, not something further engineering effort on this codebase can close.

**What would actually close it:** a different or supplementary dataset with real geography and (ideally) self-reported race/ethnicity — e.g., HMDA data, which self-reports race/ethnicity directly and is the actual dataset regulators use for this kind of analysis. Out of scope for this project's current data source, flagged here so it isn't quietly forgotten.

---

---

## Stage 21 — Item #8: load testing — first real latency numbers for this API

Nothing had ever actually measured `/predict`'s latency before — the dashboard/README's speed claims (going back to the original plan's "single-digit ms MOJO" framing, itself abandoned in Stage 12/13) were aspirational, not measured. Built `serving/load_test.py` (async `httpx`, no new dependency) and ran it for real against a live `uvicorn` instance.

### First result was confusing, investigated rather than reported as-is
500 requests / 50 concurrency, identical payload: p50 507ms, p95 785ms, p99 845ms, 0 errors. Since every request used the exact same application, and Redis caches identical inputs, this looked wrong — most of those 500 requests should have been near-instant cache hits.

**Diagnosed with two cheap, targeted checks instead of writing up the confusing number:**
1. **5 sequential identical requests:** 1st = 2105ms, 2nd-5th = 15-22ms. Confirms caching genuinely works (~100x speedup) — the load test's number was misleading, not the caching.
2. **20 requests with varied payloads** (defeats caching, isolates true per-request cost): 1st = 2398ms, remaining 19 averaged **134ms**.

**Explanation:** the original 50-concurrency test created a "thundering herd" — 50 requests racing on the *same, still-empty* cache key simultaneously at test start, so a meaningful fraction of the 500 requests hit the slow uncached path (each ~100-300ms, occasionally the ~2s cold-start cost), pulling the percentiles up even though the *majority* of individual requests, run cleanly, are fast.

### Real numbers, going forward
| | Latency |
|---|---|
| Cold start (first inference ever, after process start) | ~2.1-2.4s (one-time) |
| Steady-state, uncached | ~85-300ms, mean 134ms |
| Cached (identical repeat application) | ~15-20ms |

**134ms uncached is fine for a real-time lending decision** (well under typical UX latency budgets), and the cache gives a large win for the realistic case of a user resubmitting or an operator re-querying the same application. **The ~2.3s cold-start cost is a real operational consideration** — the first request after a deploy or restart pays it — and a production rollout would want a warm-up health check that fires one dummy prediction before accepting real traffic, so the first real customer never eats that cost. Not implemented here (out of scope for a load-test finding), flagged as the concrete follow-up.

**Why this matters beyond the numbers:** the dashboard and README have been asserting latency characteristics since early in this project without ever measuring them. This is the first actual evidence backing (or correcting) those claims — and it took investigating a confusing initial result rather than accepting it, same discipline as every fairness-metric investigation in this log.

---

---

## Stage 22 — Closing out the remaining flagged items (LLD)

User asked to close the remaining gaps from Stage 21's list. Covers what actually needed doing (and one deliberate non-change):

### Cold-start warm-up — implemented
`serving/app.py` now fires one dummy all-zero prediction (through both the model and the SHAP explainer) at the end of the startup event, wrapped in a try/except so a warm-up failure can't take the service down. Verified via `tests/test_serving.py`'s existing suite (11/11 still pass with the added startup work).

### H2O AutoML re-run — real memory constraint found and fixed
The original 240s/1-model result wasn't just a time-budget problem: `h2o.init()` was defaulting to ~247MB free memory and 0 allowed cores on this host (same practical-ceiling pattern already found for Spark's JVM). GBM failed outright with "tree model will not fit in the driver node's memory." Added an explicit `max_mem_size` parameter (default `1500m`) to `models/h2o_automl.py`'s `run_automl`. Re-ran at 600s/1500m: GBM completed this time (AUC 0.701) — a real, defensible comparison point now, still below the CatBoost champion (0.7215) and every other tree model from Step 4. Verdict: H2O AutoML doesn't beat purpose-tuned gradient boosting here, and that's now backed by an actual multi-model leaderboard, not a single degenerate GLM run.

### Alerting notification sink — implemented and wired into the DAG
`monitoring/alerting.py` gained `send_alerts()` (POSTs to a Slack incoming webhook via `requests` — no new dependency) and `format_alert_message()`. No-ops safely if `SLACK_WEBHOOK_URL` isn't configured, so it's safe to wire in everywhere without requiring every environment to have a real webhook. Wired into `dags/credit_default_retrain.py`'s `check_and_branch`: a drift-triggered retrain now also fires a real alert. Confirmed the DAG still loads with no import errors via `airflow dags list` against the live container. 4 new tests (message formatting, no-op-without-webhook, and a mocked-webhook POST verification).

### `environment.yml` — removed, not filled in
The open decision already leaned toward `requirements.txt` as authoritative; the file itself was still an empty 0-byte stub. Deleted it rather than either leaving a confusing empty file or investing in a second, parallel dependency spec nothing uses.

### Kaggle CLI pin (`kaggle<1.7`) — confirmed, not changed
Nothing in this project currently needs the 2.x CLI's features, and the pin has worked reliably since Stage 2. Revisiting it now would be effort spent on a non-problem. Closing this open item with "confirmed no action needed" rather than leaving it open indefinitely or forcing an unnecessary upgrade.

### The 0.02 AUC-cost acceptance threshold — confirmed, not changed
Deliberately revisited (not just left alone by default) now that a constrained candidate exists at 0.023, just above it. The reasoning: this project's own foundational tradeoff framing (`experiments/tradeoffs.md` Tradeoff 1, itself echoing the original plan's worked example of accepting a ~2-point AUC cost for regulatory compliance) already establishes "roughly 2 AUC points" as the deliberate line between an acceptable and unacceptable fairness-accuracy tradeoff — independent of any specific model result. 0.023 sitting just barely above 0.02 is exactly the scenario this rule exists to resolve without relitigating: close enough to look tempting to wave through, which is precisely why the rule was fixed *before* any result existed. Kept at 0.02. If this gets revisited for real, it should be from a fresh cost-benefit analysis (e.g. an actual dollar estimate of 0.02 AUC's worth in missed defaults vs. the value of a bias-free model), not from one experiment landing close to the line.

### Spark job on Linux — six real failures deep, then a genuine success
Building a one-off Linux container (`python:3.11-slim` + JRE, `docker run --rm` joined to the project's Docker network) to actually run `spark_features.py` outside this Windows host, in pursuit of closing the item flagged in Stage 17. Six distinct, real failures, each diagnosed and fixed in turn rather than guessed past:

1. **Forgot `pyarrow` in the Dockerfile's pip install** — `pandas_udf` needs it explicitly; plain Spark operations don't. Added it, unpinned.
2. **Unpinned `pyarrow` grabbed 25.0.1** — same version-incompatibility class already found and fixed on Windows (pinned to 18.1.0 there), just never applied to this Dockerfile. Pinned here too. Didn't fix it alone (see below) but was still a real, necessary fix.
3. **`openjdk-17-jre-headless` isn't packaged for this Debian release** — only 21 and 25 are. Reverted to `default-jre-headless` (21).
4. **The actual root cause: Java 21's module system blocks Arrow's reflective `sun.misc.Unsafe` access**, even with `pyarrow==18.1.0` pinned. Tried `--add-opens=java.base/java.nio=ALL-UNNAMED` two different ways: via `spark.driver.extraJavaOptions` (SparkConf) — didn't work, and a check confirmed why: that config path never reaches local mode's driver JVM, since py4j launches it *before* Spark reads its own config. Then via the `JDK_JAVA_OPTIONS` environment variable — the standard mechanism every JVM 9+ process reads automatically — verified working on a bare `java -version` call (`NOTE: Picked up JDK_JAVA_OPTIONS`), but **still didn't reach the actual JVM py4j spawns for Spark** (0 occurrences of that message in the real job's logs, checked directly rather than assumed). Two real, verified attempts, both genuine dead ends — not documented as "should have worked," documented as "confirmed via direct test not to."
5. **Switched strategy: match the one JVM version confirmed not to hit this restriction (Java 8, already used successfully for H2O) instead of chasing more flags for Java 17+.** No `openjdk-8` package available via this image's apt repos either — switched base image entirely to `eclipse-temurin:8-jre-jammy` (Java 8, officially maintained). **This is what actually fixed the Arrow/pandas_udf failure** — the job's computation succeeded completely for the first time, reaching the final `pdf.to_sql()` write step.
6. **One more real bug on the way to the finish line:** the write to Postgres failed with `could not translate host name ... : Name or service not known`. Root cause, found by testing DNS resolution directly rather than assuming a transient network blip: this project's directory name is long enough that the Docker Compose container name (`creditdefaultpredictionpipelinewithfairnessconstraints-postgres-1`, 67 characters) **exceeds DNS's 63-character label limit** — confirmed via a direct `UnicodeError: label too long`. Fixed by using Docker Compose's short service-name alias (`postgres`) instead of the long container name, which resolved immediately.

**Final, verified result:** `python features/spark_features.py` run against the real, full 13.6M-row `installments_payments.csv` inside the Java-8 Linux container — 339,587 applicants scored, loaded into `features.payment_trend_features`, confirmed by direct query afterward. The working Dockerfile is kept permanently at `docker/spark_job.Dockerfile` (not deleted as scratch work), and `features/spark_features.py`'s docstring now documents the real working path instead of "known limitation, untested on Linux."

**Why six real failures in a row is worth logging in full, not condensing to "fixed it":** every one of them was a genuine, different root cause (a missing dependency, a version mismatch, an unavailable package, a confirmed-dead-end JVM flag mechanism — twice — a JVM version incompatibility, and a DNS label-length limit). Compressing this to "eventually got it working" would hide that the fix required actually falsifying two plausible-sounding hypotheses (the `--add-opens` delivery mechanisms) with direct evidence before finding the one that worked, which is the same discipline this whole log has tried to apply to every claim in this project, model performance included.

---

## Open decisions / not yet made

These will get their own entries once resolved — listed here so the log stays a live map of what's still undecided, not just a history of the past.

- **`requirements.txt` pinning strategy** — confirmed authoritative in Stage 22 (`environment.yml` removed as an empty, unused stub). Closed.
- **Whether the two-token Kaggle auth downgrade (`kaggle<1.7`) gets revisited** — confirmed in Stage 22: nothing currently needs the 2.x CLI, pin stays. Closed, not open.
- **The underlying model is still (somewhat) age-biased — the decision layer remains the thing making this deployable.** Stage 18 found and removed the real root cause (`age_years` was a raw model input the whole time, contradicting this project's own docs) and the constrained-retraining candidate finally passes the fairness gate on its own (`dp_age` 0.038) for the first time across three attempts — but at 0.023 AUC cost, just above this project's 0.02 acceptance rule, so the deployed system still uses the raw model + post-hoc thresholds rather than the constrained one. Revisited deliberately in Stage 22: the 0.02 rule stays, because it traces back to this project's own foundational tradeoff framing rather than being picked to fit this one result. Still genuinely open in the sense that the underlying model itself remains imperfect — closed as a *decision*, not as a *limitation*.
- **Race/ethnicity fairness coverage is out of scope for this dataset, not unresolved** (Stage 20) — Home Credit has no surname, nationality, or real geography, which rules out BISG (the standard proxy method). Closing this would need a different/supplementary dataset (e.g. HMDA), not more work on this codebase. Listed here for visibility, not as a task to pick up.
