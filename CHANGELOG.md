# Changelog

## 2.0.0 - 2026-11-22

First release as RiskSynth AI. The project started as synth-bank-es, a synthetic
credit-risk dataset generator, and was reworked into a robo-advisory profile generator.

### Added
- Wealth profile engine: investable surplus, goal, horizon, risk tolerance, maximum
  drawdown, liquidity need, ESG preference and MiFID II risk capacity with named caps.
- Consistency rules CR01-CR07 with validation and repair, and an audit trail per row.
- Edge-case cohort EC01-EC09 (8% of rows by default) with expected advisor behaviour.
- Bootstrap sampler, so profiles can be generated without SDV.
- Advisory simulator and a fidelity/compliance report with HTML export.
- Streamlit app with overview, generator, simulator and evaluation tabs.
- Test suite (99 tests) and CI.

### Changed
- INE ingestion fixed: age from year of birth, `-9` missing codes, survey weighting of
  household spending, harmonised sex and region codes, household income join.
- Generator mortgages, spending and balance-derived credit fields now follow the EPF
  household data instead of fixed rules.
- Code reorganised into a single `risksynth` package with console entry points.
- Console output is plain ASCII, so piped runs on Windows (cp1252) no longer crash.

### Known gaps
- Behavioural parameters tagged `[ASSUMPTION]` still need calibration against the
  Banco de España EFF 2022 survey.
- Household disposable income (HY020) stays empty until the ECV household files are added.
