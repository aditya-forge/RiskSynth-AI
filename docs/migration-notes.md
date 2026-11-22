# Migration: synth-bank-es -> RiskSynth AI (schema 2.0.0)

RiskSynth AI repurposes the INE-calibrated synthetic banking pipeline to generate
**client risk profiles for robo-advisory platforms**. This document records what
changed in the configuration, why, and what code work remains.

- Pre-migration configs: `archive/configs_pre_risksynth/`
- Feature specification (source of truth): `configs/data_config.yaml` -> `wealth_profile_schema`, `edge_case_profiles`
- Models, constraints, thresholds: `configs/model_config.yaml`

**Status:** only the configuration has changed. No Python code reads the new sections yet (see Phase 1–4 below).
Existing code paths are unaffected: `client_generator.py` only reads
`synthetic_generation.methods.<method>`, which was not modified.

---

## 1. Data audit findings (blocking)

Verified against `data/processed/*.parquet` on 2026-11-21. Recorded under
`data_config.yaml -> ine_source_mapping.known_issues`.

| ID | Problem in `ine.py` | Effect on RiskSynth |
|---|---|---|
| ISSUE-01 | ECV `PB140` is mapped to `edad`, but it is **year of birth** (1938–2007). `combined_base.parquet` mixes birth years (ECV) with 5-year age bands (EPA) in one column. | All age-driven features (goal, horizon, risk tolerance) are wrong until fixed. |
| ISSUE-02 | The substring match `'IMPEXAC'` maps an **income** column to `gasto_total` (median = −9, a missing-value code). | Must not be used as expenses. |
| ISSUE-03 | `gasto_monetario` (`GASTMON`) is grossed up to population level (median ~ 16.5M). | Divide by `FACTOR` -> median ~ €22.8k per household per year. |
| ISSUE-04 | ECV `HY020` (household disposable income) is in the household file and is never merged. | No ECV household income is available. |
| ISSUE-05 | The surveys are stacked with `concat`, not joined, so no row has income, expenses and demographics together. | Compute the surplus within EPF, then fuse at generation time. |

---

## 2. New features

Full generation logic is in `data_config.yaml -> wealth_profile_schema.features`.
**[E]** = measured on EPF 2024. **[A]** = design assumption that needs calibration.

| Feature | Type / range | Derivation |
|---|---|---|
| `investable_surplus_eur` | float, €/yr, >= 0 | `max(0, net_income − GASTMON/FACTOR − 12·(CAPITMENS+INTERESMENS) − consumer debt)`. Net income = `12·IMPEXAC`, or the `INTERIN` bracket median when missing. **[E]** |
| `raw_surplus_eur` *(companion)* | float, signed | The value before the floor. Kept because 40.9% of EPF households are negative. **[E]** |
| `primary_investment_goal` | Capital Preservation / Retirement / Wealth Growth / Income | P(goal \| age band) table, adjusted when surplus = 0 or age >= 67. **[A]** |
| `investment_horizon_years` + `investment_horizon_bucket` | 0.5–50 y; Short < 3, Medium 3–<7, Long >= 7 | Conditional on goal (Retirement = 67 − age); capped at 90 − age. |
| `risk_tolerance_score` + `risk_tolerance_category` | int 1–10; Conservative 1–3, Moderate 4–6, Aggressive 7–8, Ultra-Aggressive 9–10 | Latent linear model (age, income, surplus ratio, education, employment, children, goal, horizon) + noise, then quantile-mapped to the target mix 40/35/18/7 %. **[A]** |
| `max_acceptable_drawdown_pct` | 5–40 % | Drawn within the category band (5–10 / 10–20 / 20–30 / 30–40), positioned by score. |
| `liquidity_need_pct` | 0–100 % | Beta distribution by goal + adjustments (unemployed, temporary contract, short horizon, children, no buffer). >= 50 = liquidity-constrained. **[A]** |
| `esg_preference_level` | None / Low / High | Ordinal logit (age, education, income) mapped to 40/35/25 %. **[A]** |

Supporting fields: `risk_capacity_score` (MiFID II ability to bear losses: additive
points plus hard caps), `effective_risk_score = min(tolerance, capacity)`,
`debt_service_to_income`, `liquid_savings_eur`, `profile_inconsistency_flag`.

**Empirical anchors (EPF 2024, n = 19,333 households):** net income P50 €26.4k, P90 €48.0k.
Raw surplus P25 −€5.3k, P50 €2.7k, P75 €10.7k. Median surplus rises with age of the household head
(18–34: €0.9k -> 65+: €4.4k). 20.6% of households pay a mortgage, with DSTI P90 = 0.364.

> **Known bias:** the EPF under-reports income relative to its expenditure diary, which inflates
> the negative-surplus share. `income_calibration_factor` (default 1.0) is the hook to correct it
> against ECV HY020 or AEAT data.

---

## 3. Edge-case profiles

Specified in `data_config.yaml -> edge_case_profiles`. The total target is 8% of rows.

- Rows that naturally match a predicate are tagged `natural`.
- Any shortfall against a profile's target is injected by conditional sampling with rejection and tagged `injected`.
- Injected rows are **excluded from fidelity tests** but included in constraint tests and ML tests.

| ID | Profile | Predicate (summary) | Share | Expected advisor behaviour |
|---|---|---|---|---|
| EC01 | High income, low liquidity | income >= P90 and liquid savings < 3 months of expenses | 1.0% | Build the emergency fund first; effective risk <= 6 |
| EC02 | Young, high risk, low capital | age 18–30, score >= 8, savings < €5k, surplus < €2.4k | 1.0% | Aggressive allocation OK through small recurring contributions; check product minimums |
| EC03 | Near-retiree, aggressive growth | age 58–66, Growth/Retirement goal, Aggressive+, horizon <= 9 | 1.0% | Sequence-of-returns warning; glide path; effective risk <= 7 |
| EC04 | Negative-surplus investor | raw surplus < 0 and score >= 5 | 1.0% | No recurring contributions; effective risk <= 5 |
| EC05 | Over-leveraged risk seeker | DSTI >= P90 (~ 0.35) and score >= 7 | 1.0% | Downgrade; pay down debt first; effective risk <= 5 |
| EC06 | Retiree income, high liquidity | age >= 67, Income goal, liquidity >= 50% | 0.75% | Cash and short-duration bonds; effective risk <= 6 |
| EC07 | Short horizon, aggressive | Short bucket and Aggressive+ | 0.75% | Horizon overrides tolerance; effective risk <= 6 |
| EC08 | Inconsistent questionnaire | Ultra-Aggressive with drawdown <= 10% | 0.5% | Flag for re-profiling (the only profile allowed to break CR02) |
| EC09 | Asset-rich, cash-poor | age >= 60, home value >= P90 (~ €300k), surplus = 0 | 1.0% | Illiquid wealth is not capacity; effective risk <= 5 |

Each `test_assertion` follows directly from a named capacity cap
(`CAP_NO_SURPLUS`, `CAP_HIGH_DSTI`, `CAP_NO_BUFFER`, `CAP_SHORT_HORIZON`,
`CAP_HIGH_LIQUIDITY`, `CAP_NEAR_RETIREMENT`). The assertions therefore test the generator's
consistency; they are not hopes about model behaviour.

---

## 4. Configuration changes by file

### `configs/data_config.yaml`
- **Added** `project` (name, `schema_version: 2.0.0`, reference year).
- **Added** `ine_source_mapping`: verified INE column semantics, the INTERIN bracket table (bounds and imputation medians measured from IMPEXAC), and `known_issues`.
- **Extended** `data_quality.validation_rules` with 13 rules covering not-null, ranges, allowed values and score/category and years/bucket consistency. Added the surplus columns to outlier detection.
- **Extended** `transformations`: categorical/ordinal encodings, scaling (log1p + robust scaling for surplus), and derived-feature formulas (including the age fix).
- **Deprecated** `financial_profiles` (3-level). It is kept read-only and superseded by the 4-level schema.
- **Added** `compliance.mifid2_suitability`: maps the suitability dimensions to fields and defines the effective-risk rule.
- **Added** `wealth_profile_schema` and `edge_case_profiles`.

### `configs/model_config.yaml`
- **Added** `schema_version` and a 3-stage `synthetic_generation.pipeline`:
  1. SDV learns the INE base.
  2. Rules derive the wealth features.
  3. Edge cases are injected.

  Why hybrid: identities such as `investable = max(0, raw)` must hold exactly, and GANs only approximate them.
- **Added** `sdv_metadata_overrides` for the new columns.
- **Extended** `constraints.business_rules` (CR01–CR07 plus the effective-risk identity), `statistical_constraints` (EPF quantile targets with tolerances, categorical targets), `dependency_constraints` (Spearman sign/magnitude checks), and `regulatory_compliance` (MiFID II).
- **Extended** `post_processing`: rounding, clipping, allowed values, and `recompute_after_clipping`.
- **Added** `profiling_models`:
  - risk-category classifier with a leakage guard and a QWK >= 0.50 acceptance threshold
  - surplus regression
  - goal classifier

  `scoring_models` is now marked `status: legacy`.
- **Extended** `feature_engineering` and `evaluation_config` with explicit thresholds:
  - KS D <= 0.10
  - Wasserstein grading as in `statistical_tests.py`
  - JS <= 0.05 (stricter than the code's "good" grade of < 0.1)
  - ±2pp category shares
  - zero hard-rule violations
  - ±0.5pp edge-case shares
  - 100% assertion pass rate
  - DCR / k-anonymity >= 5

---

## 5. Remaining migration plan (code)

| Phase | Work | Files |
|---|---|---|
| 1 - Fix ingestion *(2026-11-21)* | Done; see "Phase 1 log" below. ISSUE-04 is blocked on data: HY020 needs the ECV_Th raw file. | `src/risksynth/ingest/ine.py` |
| 2 - Feature engine *(2026-11-21)* | Done; see "Phase 2 log" below. | `src/risksynth/engine/{wealth_profile,consistency,stage2}.py` |
| 3 - Edge cases *(2026-11-21)* | Done; see "Phase 3 log" below. | `src/risksynth/engine/edge_cases.py` |
| 4 - Integration & UI *(2026-11-22)* | Done; see "Phase 4 log" below. | generators, `src/risksynth/scoring/robo_advisor.py`, `src/risksynth/evaluation/fidelity.py`, `app/streamlit_app.py` |
| 5 - Tests & release *(2026-11-22)* | Done; see "Phase 5 log" below. | `tests/`, packaging, `README.md` |
| 6 - Calibrate | Replace the **[A]** priors with Banco de España EFF 2022 (risk attitude, financial assets); set `income_calibration_factor`. | configs |

### Phase 1 log (2026-11-21)

- **Config-driven processing.** `ine.py` reads three new `data_config.yaml` sections:
  - `ine_source_mapping`: layout, join keys, columns, sentinels, income brackets
  - `harmonization`: sexo, CCAA->NUTS2, education, employment, contract
  - `combined_base`

  It applies the `data_quality.validation_rules` and logs violations.
- **Source.** `data/raw` is absent, so the processor rebuilt from `data/interim/legacy_v1_processed/`, a frozen copy of the schema-1.x parquet files that also serves as the backup. It restores the raw INE names first. Use `--source raw` once the microdata is in place.
- **Age.** ECV `edad = PB110 − PB140` (17–86; 86 = top-coded cohort). EPA bands become midpoints, with the 65+ band set to the ECV weighted median age of 75. `edad_min`, `edad_max` and `edad_precision` are explicit.
- **Expenditure.** `gasto_total = GASTOT/FACTOR` (median €30.3k) and `gasto_monetario = GASTMON/FACTOR` (median €22.8k). −9 becomes NaN across EPF. The former "gasto_total" was `IMPEXACSP` (the main earner's income) and is kept under its raw name.
- **Income.** EPF `net_income_annual_eur` (median €26.4k) comes from IMPEXAC, or the INTERIN bracket median when missing. The ECV P->H->D join on `PB030 // 100` is implemented and tested on a fixture. `renta_disponible` stays null until `ECV_Th`/`ECV_Td` are supplied; meanwhile `renta_neta_personal_eur` (sum of PY*N components, median €15.6k) is a person-level proxy, not HY020.
- **combined_base.** A schema-aligned stack of independent samples with `peso` / `peso_norm` and `fuente` as stratum, filtered to ages 18–100: 170,307 rows (previously 191,863 incl. children and birth years). `sexo` is now 1/2 everywhere (EPA/EPF used 6 for female) and `ccaa` is NUTS2, which `generate_synthetic.py` expects.
- **Downstream fix.** `generate_synthetic.py` now falls back to `ingresos_trabajo` when `renta_disponible` is null.

### Phase 2 log (2026-11-21)

**Modules** (in `src/risksynth/engine/`):
- `wealth_profile.py`: input resolution, the 7 features, surplus, liquid savings, and risk capacity with caps.
- `consistency.py`: CR01–CR07 plus three MiFID II checks (effective risk, capacity current, ESG collected); validates, then repairs in config order.
- `stage2.py`: orchestrator, report, and a dry-run CLI (`python -m risksynth.engine.stage2 --n 20000`).

All parameters come from `data_config.yaml`. Term and cap names are validated against the code.

**Input resolution.** Rows without a household-finance block (ECV, EPA) receive a weighted EPF donor household from the same 10-year age band and employment group (hot-deck, with coarser fallback cells). The source is recorded in `household_finance_source`. HY020 is used only when observed and never imputed. ECV person-level income is not netted against household spending.

**Liquid savings.** Generator account balances are used when available. Otherwise the `[ASSUMPTION]` months-of-expenses model applies, with the source recorded in `liquid_savings_source`.

**Audit columns.**
- `cr_violations_raw`: rules broken by the raw draws
- `cr_repaired`: rules repaired
- `capacity_caps_applied`: capacity caps that fired
- `profile_inconsistency_flag`: true if any rule is still broken after enforcement (only exempt rows can stay broken)

**Integration.**
- `generate_synthetic.generate_synthetic_data(..., wealth_stage=True)` resolves base rows, builds the payload from household net income, and runs stage 2 using the generated balances and personal loans. It writes `<file>_wealth_report.json`.
- `ClientGenerator.generate(apply_wealth_stage=True)` runs stage 2 on the SDV samples; `save()` embeds the report in `metadata.json`. SDV and loguru are now lazy/optional imports.
- `main()` trains on `build_stage1_table()` (EPF households).

**Dry run on `combined_base` (170,307 rows), raw -> enforced:**
- CR02: 942 -> 0 (drawdown noise outside band ± 2pp). All other rules and checks: 0 -> 0.
- Validator stress test (independently shuffled columns): every CR violated before, 0 after.

**Known gaps:**
- (a) Liquid-savings levels differ widely between the assumption model (median €4.5k) and generator balances (median €41k). `CAP_NO_BUFFER` therefore fires 55% vs rarely. Calibrate both against EFF 2022.
- (b) Generator mortgages are still random (30%) and independent of EPF mortgage status. The wealth features use only the EPF mortgage plus generator personal loans, so the credit columns `tiene_hipoteca` / `cuota_mensual_total` can disagree with `debt_service_annual_eur`.
- (c) 88.6% of dry-run rows carry donor (EPF) finances, so the wealth block's distribution is EPF's.

### Phase 3 log (2026-11-21)

**Semantics.** Natural predicate matches overshoot 8% (EC04 alone matches ~14% of rows), so two concepts are kept apart:
- `edge_case_matches`: every profile a row satisfies, for stratifying test suites.
- `edge_case_id`: a quota-controlled cohort of `round(target_share × N)` rows per profile, about 8% in total. It can be overridden with `run_stage2(edge_case_share=…)` or `--edge-case-share`.

**How each quota is filled.**
- Natural matches come first. Shortfalls are injected by mutating the fewest fields of non-cohort rows, with rejection: the injected row must satisfy its predicate and every rule except its `relaxed_rules`.
- Profiles with the scarcest natural supply are filled first.
- Injection prefers rows that match no other profile, and rows are changed in place (N is unchanged).

**Columns.**
- Cohort: `edge_case_id`, `edge_case_name`, `edge_case_origin` (natural / injected / none), `is_edge_case`
- Expected behaviour: `expected_robo_behavior`, `expected_capacity_override`, `edge_case_capacity_cap`, `edge_case_assertion`, `edge_case_assertion_pass`
- Audit: `edge_case_mutated_fields`, `edge_case_matches`
- Product feasibility: `flag_minimum_investment`, based on the `[ASSUMPTION]` robo-advisor minimums of €1,000 initial and €50/month

**Exemptions.** EC08 rows are exempt from CR02 only. After injection, `consistency.enforce` runs again with those exemptions, so any other violation is still repaired, including on EC08 rows. `violations_exempted` counts the intentional breaks, and `profile_inconsistency_flag` is True exactly on EC08 rows. Capacity and effective risk are recomputed after injection, and each capped profile records its named cap (`capacity_cap`).

**Results (full combined base, 170,307 rows).**
- Cohort is 8.00% (13,624 rows: 10,762 natural, 2,862 injected), with no shortfalls.
- Every predicate holds for 100% of its cohort rows, and every named cap fires on 100% of its profile's rows.
- After enforcement: 0 non-exempt violations; CR02 exempted on 852 rows (= EC08).
- Generator (n = 10,000) and ClientGenerator (n = 12,000) runs give the same picture.

**Open finding.** EC02's assertion (`Long horizon or flag_minimum_investment`) fails for 1.1% of dry-run members (94–97.5% pass in the generator runs). Those rows are young, aggressive clients with Medium horizons who can afford the product minimums. The predicate does not require the long horizon that the recommendation assumes. **Decision needed:** tighten the predicate (e.g. `investment_horizon_years >= 7`) or relax the assertion.

### Phase 4 log (2026-11-22)

**EC02 rule.** The predicate now also requires `investment_horizon_years >= 7` (the injection candidate filter does too). The assertion passes at 100% on the full combined base (1,703 natural rows) and in generator runs, and every acceptance criterion passes on the full base.

**Generator alignment** (`generate_synthetic.py`). When a row carries EPF household finances:
- Payload spending follows `household_expenses_annual_eur`, without the artificial €1,000–15,000 income clamp.
- A mortgage exists exactly when the EPF debt service is above 0, with that monthly payment (previously a random 30%).
- Liquid savings are the positive account balances only.

After stage 3, `reconcile_credit_fields()` rescales the account balances on rows whose `liquid_savings_eur` was changed by injection. It then recomputes `balance_cuentas`, `colchon_meses`, `sin_colchon`, `mayor_estable` and the default target for those rows. Its report (`credit_reconciliation`) shows 0 debt mismatches, and mortgages match EPF on 100% of rows. The default rate stays at about 5–6% in auto mode.

**Encoding.** Every `print`, `input` and log line in the generators, `statistical_tests.py` and `ine.py` is now ASCII (`[OK]`, `->`, accents transliterated). The generator entry point also sets `errors="replace"` on stdout/stderr. The interactive CLI runs piped on cp1252 with exit code 0 and an all-ASCII log. `statistical_tests.py` now writes its HTML report as UTF-8.

**New modules.**
- `src/risksynth/scoring/robo_advisor.py`: a transparent model-portfolio engine (`[ASSUMPTION]` parameters in `model_config.yaml -> robo_advisor_simulator`). It produces three allocations per client (tolerance only -> capacity and liquidity -> drawdown guard) plus the advisor actions.
- `src/risksynth/evaluation/fidelity.py`: KS / Wasserstein (normalised by the reference p1–p99 range) / Jensen–Shannon tests against INE; target-mix, Spearman-correlation and dependency checks; CR01–CR07 and MiFID II checks with exempt counts; a self-contained HTML report. It needs no loguru, seaborn or SDV.

**`client_generator.py`.** Adds a `bootstrap` method (weighted resampling of EPF households; no SDV needed) and optional `generate(wealth_config=…, edge_case_share=…)`.

**Streamlit app** (`app/streamlit_app.py`, theme in `.streamlit/config.toml`; the old app is in `archive/`).
- Four tabs: Overview, Generator, Advisory simulator, Fidelity & compliance.
- Charts use Altair with palettes checked by the dataviz validator: an ordinal blue ramp for risk tiers, categorical slots 1–3, and a Lab-interpolated diverging blue–grey–red scale.
- Verified with `streamlit.testing` (all tabs, generator runs, validation errors, every stress-profile filter, dataset switching) and with a live server plus rendered screenshots.

**Known caveats.**
- The ESG label `"None"` becomes NaN under `pd.read_csv` defaults. The app reads CSVs with explicit NA tokens; downstream consumers should do the same, or use the Parquet outputs.
- Rule-based runs inside the app are sequential and capped at 5,000 rows. App runs also leave the generator's own files in `data/synthetic/`.
- SDV is not installed in this Python 3.14 environment, so the SDV option is disabled in the UI.

### Phase 5 log (2026-11-22)

**Test suite.** `pytest tests/` gives **99 passed in ~40 s** on Python 3.14 / pandas 3.0, with no skips or warnings. It consolidates the Phase 1–4 verification scripts.
- `test_data_processing.py` (18): mapping, age, top-coding, P->H->D join, sentinels, FACTOR weighting, income imputation, harmonisation, processed-parquet invariants.
- `test_wealth_engine.py` (18): surplus identity and calibration factor, boundaries, exact target mixes, capacity points and every named cap, CR01–CR07 validate/repair, idempotence.
- `test_edge_cases.py` (38): 8% total, per-profile quotas, 100% predicate and assertion pass (EC02 included), caps, EC08 per-rule exemption, share override, allocation caps.
- `test_generators_alignment.py` (23): EPF-aligned mortgages and debt, credit reconciliation, bootstrap sampler, SDV absence, CSV `"None"` round trip, cp1252-safe output including a piped CLI subprocess.
- `test_app/streamlit_app.py` (2): all tabs render; form validation.

Slow end-to-end tests are marked `slow`. Fixtures never write to `data/processed/` or `data/synthetic/`; generator outputs go to tmp dirs via the new optional `RISKSYNTH_OUTPUT_DIR`.

**Code.** `src/risksynth/io.py` adds `read_profiles()` / `CSV_NA_VALUES` (keeps ESG `"None"`), now used by the app. `generate_synthetic.py` honours `RISKSYNTH_OUTPUT_DIR`.

**Packaging.**
- `pyproject.toml` is the single source of metadata: `risksynth-ai` 2.0.0, core deps (pandas, numpy, scipy, pyarrow, pyyaml), extras `app` / `sdv` / `legacy-eval` / `dev`, pytest config.
- Package discovery is configured in `pyproject.toml`; an editable install resolves (see the layout change below).
- `setup.py` is now a shim; `requirements.txt` was refreshed (unused scikit-learn, plotly, ctgan, dotenv, tqdm, joblib dropped).
- `.env.example` documents only variables the code reads.

**Repository.**
- New `.gitignore` covers caches, `.env`, `data/raw/`, `data/interim/`, `data/synthetic/`, `logs/`, `reports/` and `*.pkl`; `data/processed/*.parquet` and the configs stay tracked.
- The pre-migration configs moved to `archive/configs_pre_risksynth/`.
- Empty placeholder test packages and stray caches were removed.
- `README.md` was rewritten; `CONTEXT.md` is marked superseded.

**Open for the owner.**
- ~~`pyproject.toml` declares MIT, but there is no `LICENSE` file.~~ Resolved: the project is now proprietary (all rights reserved), see `LICENSE`.
- `data/interim/legacy_v1_processed/` (the rebuild source when raw microdata is absent) is now git-ignored, so keep a copy outside git if you need to re-run `risksynth.ingest.ine --source legacy`.

### Repository layout (2026-11-22)

Before the first commit the code moved from six generic top-level packages under `src/` (`data`, `synthetic`, `scoring`, `evaluation`, `utils`, `visualization`) into one namespaced package, `src/risksynth/`. The old layout clashed with common package names once installed and relied on `sys.path` edits in every module.

| Old path | New path |
|---|---|
| `src/data/processing/process_data.py` | `src/risksynth/ingest/ine.py` |
| `src/synthetic/constraints/*.py` | `src/risksynth/engine/*.py` |
| `src/synthetic/generators/*.py` | `src/risksynth/generators/*.py` |
| `src/scoring/robo_advisor.py` | `src/risksynth/scoring/robo_advisor.py` |
| `src/evaluation/*.py` | `src/risksynth/evaluation/*.py` |
| `src/utils/io.py` | `src/risksynth/io.py` |
| `streamlit_app.py` | `app/streamlit_app.py` |

Repository paths are resolved in one place (`src/risksynth/paths.py`). The CLIs are entry points (`risksynth-ingest`, `risksynth-generate`, `risksynth-profiles`) or `python -m` modules. Empty placeholder packages were dropped. The original notebooks and hand-over notes moved to `archive/`. File paths in the phase logs above have been updated to the new layout.

## 6. Open decisions

1. **Unit of analysis.** The surplus is household-level (EPF), while ECV/EPA are person-level. The current choice is the household head.
2. **Retirement age 67.** A parameter: the reduced age of 65 applies to long contribution careers.
3. **[A] priors.** These are placeholders. Do not publish distributions derived from them as statistics about the Spanish population.
4. **Rename the package** (`synth-bank-es` -> `risksynth-ai`) in `pyproject.toml` / `setup.py` / README: deferred, not done.
