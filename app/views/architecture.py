# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Solution architecture: pipeline, output schema, rules and the edge-case catalogue."""

import html

import pandas as pd
import streamlit as st

from portal.icons import icon
from portal.ui import html_table, section
from risksynth.engine import wealth_profile

cfg = wealth_profile.load_config()
schema = cfg["wealth_profile_schema"]

section(
    "Solution architecture",
    "From public INE microdata to tested client profiles. Every stage is configured in "
    "<code>configs/data_config.yaml</code> and <code>configs/model_config.yaml</code>.",
    eyebrow="How it works",
    first=True,
)

STEPS = [
    (
        "01",
        "INE microdata",
        "ECV 2024 living conditions, EPF 2024 household budgets, EPA 2025 T3 labour force.",
        "data/raw/",
    ),
    (
        "02",
        "Ingestion",
        "Age from birth year, -9 missing codes, survey weights, NUTS2 regions and harmonised codes.",
        "risksynth.ingest.ine",
    ),
    (
        "03",
        "Stage 1 engine",
        "Rule-based banking payloads, bootstrap sampler, or SDV generative models.",
        "risksynth.generators",
    ),
    (
        "04",
        "Suitability profile",
        "Surplus, goal, horizon, tolerance, drawdown, liquidity, ESG and risk capacity.",
        "engine.wealth_profile",
    ),
    (
        "05",
        "Rules",
        "CR01-CR07 consistency rules and MiFID II checks: validate, repair, audit trail per row.",
        "engine.consistency",
    ),
    (
        "06",
        "Edge cases",
        "EC01-EC09 stress cohort, natural matches first, minimal injection for the rest.",
        "engine.edge_cases",
    ),
    (
        "07",
        "Outputs",
        "Advisory simulator, fidelity and compliance report, Parquet, CSV and JSON exports.",
        "scoring, evaluation",
    ),
]
STEP_ICONS = ["database", "filter", "layers", "user", "shield", "alert", "file"]
cells = "".join(
    f'<div class="rs-step"><div class="top">{icon(ic, 18)}<span class="n">{n}</span></div>'
    f'<div class="t">{t}</div><div class="d">{d}</div><div class="m">{m}</div></div>'
    for ic, (n, t, d, m) in zip(STEP_ICONS, STEPS)
)
st.markdown(f'<div class="rs rs-grid rs-flow">{cells}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rs rs-note" style="margin-top:8px">'
    "Stages 04 to 06 are shared by every engine (<code>risksynth.engine.stage2.run_stage2</code>), "
    "so all outputs carry the same schema and rules."
    "</div>",
    unsafe_allow_html=True,
)

section("Output schema", "Main columns of every generated client profile.")
SCHEMA = pd.DataFrame(
    [
        (
            "Demographics",
            "INE",
            "edad, sexo, ccaa (NUTS2), nivel_estudios, situacion_laboral, n_miembros, dependent_children",
        ),
        (
            "Household finances",
            "INE EPF",
            "net_income_annual_eur, household_expenses_annual_eur, debt_service_annual_eur, "
            "main_residence_value_eur, household_finance_source",
        ),
        (
            "Surplus",
            "Derived",
            "raw_surplus_eur, investable_surplus_eur, surplus_ratio, debt_service_to_income, "
            "liquid_savings_eur",
        ),
        (
            "Suitability",
            "Derived",
            "risk_tolerance_score / category, investment_horizon_years / bucket, "
            "max_acceptable_drawdown_pct, liquidity_need_pct, esg_preference_level, primary_investment_goal",
        ),
        (
            "Risk capacity",
            "MiFID II",
            "risk_capacity_score, capacity_caps_applied, effective_risk_score, flag_minimum_investment",
        ),
        ("Audit", "Rules", "cr_violations_raw, cr_repaired, profile_inconsistency_flag"),
        (
            "Edge cases",
            "Stage 3",
            "edge_case_id, edge_case_matches, edge_case_origin, expected_robo_behavior, "
            "edge_case_assertion_pass",
        ),
    ],
    columns=["Group", "Source", "Columns"],
)
st.markdown(html_table(SCHEMA), unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    section("Consistency rules", "Validated on every profile and repaired in the configured order.")
    rules = pd.DataFrame(
        [(r["id"], r["rule"]) for r in schema["consistency_rules"]], columns=["Rule", "Definition"]
    )
    st.markdown(html_table(rules, code_cols=("Definition",)), unsafe_allow_html=True)
with right:
    section("Risk capacity caps", "Hard limits on capacity, applied after the points score.")
    caps = schema["supporting_features"]["risk_capacity_score"]["caps"]
    cap_tbl = pd.DataFrame(
        [(c["id"], c["when"], c["max"]) for c in caps], columns=["Cap", "Condition", "Max"]
    )
    st.markdown(html_table(cap_tbl, code_cols=("Condition",)), unsafe_allow_html=True)

section(
    "Edge-case catalogue",
    "Each archetype has a quota and an explicit assertion about the expected advisor behaviour.",
)
ec = pd.DataFrame(
    [
        (
            q["id"],
            q["name"].replace("_", " ").title(),
            f"{100 * q['target_share']:.2f}%",
            q["capacity_cap"] or "-",
            q["expected_advisory_outcome"]["recommendation"],
        )
        for q in cfg["edge_case_profiles"]["profiles"]
    ],
    columns=["ID", "Archetype", "Share", "Capacity cap", "Expected behaviour"],
)
st.markdown(html_table(ec), unsafe_allow_html=True)

section("Calibration status")
st.markdown(
    '<div class="rs rs-grid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">'
    '<div class="rs-pillar">'
    '<div class="k rs-mono">EMPIRICAL</div>'
    '<div class="t">Measured on INE microdata</div>'
    '<div class="d">'
    "Income and surplus quantiles, the 40.9% share of households spending more than they earn, "
    "mortgage debt-to-income ratios, income bracket medians."
    "</div>"
    "</div>"
    '<div class="rs-pillar">'
    '<div class="k rs-mono">ASSUMPTION</div>'
    '<div class="t">Design priors, calibration pending</div>'
    '<div class="d">'
    "Risk-tier mix, ESG mix, goals by age, tolerance model coefficients, liquidity priors and "
    "simulator portfolios. To be calibrated against the Banco de Espa&ntilde;a EFF 2022 survey "
    "before production use."
    "</div>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="rs rs-note" style="margin-top:8px">Schema version '
    f'{html.escape(str(cfg["project"]["schema_version"]))}.</div>',
    unsafe_allow_html=True,
)
