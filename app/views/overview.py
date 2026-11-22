# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Product overview: hero, value pillars and the live population view."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from portal import PAGES
from portal.data import active_dataset, baseline_records
from portal.icons import icon
from portal.theme import TIER_ORDER, palette
from portal.ui import EC_SHORT, eur, kpi_grid, pct, section, show
from risksynth.engine import wealth_profile

p = palette()
cfg = wealth_profile.load_config()
n_rules = len(cfg["wealth_profile_schema"]["consistency_rules"])

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
hero, spec = st.columns([1.8, 1], gap="large", vertical_alignment="top")
with hero:
    st.markdown(
        '<div class="rs rs-hero">'
        '<div class="eyebrow">RiskSynth AI &middot; Synthetic data for wealth management</div>'
        '<h1>Synthetic client risk profiles for robo-advisory and wealth management</h1>'
        '<p class="lead">Synthetic client profiles built from Spanish household survey microdata (INE ECV, EPF and '
        'EPA). Use them to test portfolio allocation models and to check how MiFID II risk-capacity rules handle '
        'difficult clients, from over-leveraged investors to contradictory questionnaires.</p></div>',
        unsafe_allow_html=True, anchors=False)
    with st.container(key="hero_ctas", horizontal=True, gap=None):
        with st.container(key="hero_primary", width="content"):
            st.page_link(PAGES["generator"], label="Launch Workbench")
        with st.container(key="hero_secondary", width="content"):
            st.page_link(PAGES["architecture"], label="View the architecture")
with spec:
    rows = [("Data sources", "INE ECV 2024, EPF 2024, EPA 2025 T3"),
            ("Suitability model", "MiFID II tolerance vs capacity"),
            ("Rules enforced", f"CR01-CR0{n_rules} and 3 MiFID II checks"),
            ("Stress cohort", "9 archetypes, 8% of rows by default"),
            ("Engines", "Rule-based, bootstrap sampler, SDV (optional)"),
            ("Outputs", "Parquet, CSV, JSON metadata, HTML report")]
    st.markdown('<div class="rs rs-spec-card"><div class="title">Specification</div><table class="rs-spec">'
                + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table></div>",
                unsafe_allow_html=True, anchors=False)

# Stats bar: every figure is computed or read from config, not typed in
st.markdown(
    '<div class="rs rs-statbar">'
    f'<div><div class="v">{baseline_records():,}</div><div class="l">baseline INE microdata records</div></div>'
    f'<div><div class="v">{len(cfg["edge_case_profiles"]["profiles"])}</div><div class="l">edge-case archetypes (EC01-EC09)</div></div>'
    f'<div><div class="v">{n_rules} + 3</div><div class="l">consistency rules and MiFID II checks on every profile</div></div>'
    '<div><div class="v">0</div><div class="l">real individuals in any generated record</div></div>'
    '</div>', unsafe_allow_html=True)

# Pillars
PILLARS = [
    ("database", "Built on real survey data",
     "Income, spending, mortgage payments and employment come from INE microdata, cleaned up and weighted to the "
     "Spanish population. Nothing is invented from scratch."),
    ("shield", "Tolerance and capacity kept apart",
     "What a client says they can stomach and what they can actually afford are scored separately. The recommended "
     "risk is the lower of the two, which is what MiFID II expects."),
    ("alert", "Hard cases on purpose",
     "Every dataset includes nine kinds of difficult client, each with a written expectation of what a good advisor "
     "should do. The simulator shows the de-risking step by step."),
]
st.markdown('<div class="rs rs-grid rs-pillars">' + "".join(
    f'<div class="rs-pillar"><div class="icon">{icon(ic, 20)}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
    for ic, t, d in PILLARS) + "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Live population
# --------------------------------------------------------------------------- #
section("Live population", "Key figures and distributions for the active dataset. Generate or switch datasets at any time.",
        eyebrow="Dataset")
loaded = active_dataset("overview")
if loaded:
    df, _, _ = loaded
    target = cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]["target_distribution"]
    has_ec = "edge_case_id" in df
    target_ec = 100 * cfg["edge_case_profiles"]["total_target_share"]
    share = 100 * df["is_edge_case"].astype(bool).mean() if has_ec else float("nan")
    kpi_grid([
        ("Clients", f"{len(df):,}", "in the active dataset"),
        ("Edge-case cohort", pct(share, 2) if has_ec else "n/a",
         f"{share - target_ec:+.2f} pp vs {target_ec:.0f}% target" if has_ec else "no stage-3 columns"),
        ("Median surplus", eur(df["investable_surplus_eur"].median()), "investable, per year"),
        ("Spending > income", pct(100 * (df["raw_surplus_eur"] < 0).mean()), "of households, incl. debt"),
        ("Capacity-bound", pct(100 * (df["effective_risk_score"] < df["risk_tolerance_score"]).mean()),
         "effective risk below stated"),
        ("Re-profile flags", f"{int(df.get('profile_inconsistency_flag', pd.Series(False)).astype(bool).sum()):,}",
         "contradictory answers"),
    ])
    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        mix = (100 * df["risk_tolerance_category"].astype("string").value_counts(normalize=True)) \
            .reindex(TIER_ORDER).fillna(0).rename("actual").to_frame()
        mix["target"] = [100 * target[t] for t in TIER_ORDER]
        mix["diff"] = mix["actual"] - mix["target"]
        mix["label_x"] = mix[["actual", "target"]].max(axis=1)
        mix = mix.reset_index(names="tier")
        base = alt.Chart(mix).encode(y=alt.Y("tier:N", sort=TIER_ORDER, title=None))
        bars = base.mark_bar(height=22).encode(
            x=alt.X("actual:Q", title="% of clients", scale=alt.Scale(domain=[0, max(45, mix.actual.max() + 5)])),
            color=alt.Color("tier:N", scale=alt.Scale(domain=TIER_ORDER, range=p["tiers"]), legend=None),
            tooltip=[alt.Tooltip("tier:N", title="Tier"), alt.Tooltip("actual:Q", format=".2f", title="Actual %"),
                     alt.Tooltip("target:Q", format=".1f", title="Target %"), alt.Tooltip("diff:Q", format="+.2f", title="Diff pp")])
        ticks = base.mark_tick(color=p["ink"], thickness=2, size=30).encode(x="target:Q", tooltip=["tier", "target"])
        labels = base.mark_text(align="left", dx=8, color=p["ink"], fontSize=11).encode(
            x="label_x:Q", text=alt.Text("actual:Q", format=".1f"))
        show((bars + ticks + labels).properties(title=alt.Title(
            "Risk tolerance tiers", subtitle="Bars: actual share. Ticks: configured target mix")), height=290)
    with c2, st.container(border=True):
        s = df["raw_surplus_eur"].dropna()
        lo, hi = s.quantile([0.01, 0.99])
        counts, edges = np.histogram(s.clip(lo, hi), bins=40)
        hist = pd.DataFrame({"from": edges[:-1], "to": edges[1:], "clients": counts})
        bars = alt.Chart(hist).mark_bar(color=p["series"][0], binSpacing=1).encode(
            x=alt.X("from:Q", bin="binned", title="Raw surplus, EUR per year (p1-p99)", axis=alt.Axis(format="~s")),
            x2="to:Q", y=alt.Y("clients:Q", title="Clients"),
            tooltip=[alt.Tooltip("from:Q", format=",.0f", title="From EUR"), alt.Tooltip("to:Q", format=",.0f", title="To EUR"),
                     alt.Tooltip("clients:Q", format=",", title="Clients")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=p["ink"], strokeDash=[4, 3]).encode(x="x:Q")
        show((bars + zero).properties(title=alt.Title(
            "Household surplus distribution",
            subtitle=f"{100 * (s < 0).mean():.1f}% below zero: investable surplus is floored at EUR 0 (dashed line)")),
            height=290)

    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True):
        if has_ec:
            cohort = df[df["is_edge_case"].astype(bool)]
            tab = cohort.groupby(["edge_case_id", "edge_case_origin"]).size().rename("rows").reset_index()
            tab["share"] = 100 * tab["rows"] / len(df)
            tab["profile"] = tab["edge_case_id"] + "  " + tab["edge_case_id"].map(EC_SHORT)
            order = sorted(tab["profile"].unique())
            targets = pd.DataFrame([{"profile": f"{q['id']}  {EC_SHORT[q['id']]}", "target": 100 * q["target_share"]}
                                    for q in cfg["edge_case_profiles"]["profiles"]])
            bars = alt.Chart(tab).mark_bar(stroke=p["surface"], strokeWidth=2).encode(
                y=alt.Y("profile:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("share:Q", stack="zero", title="% of all clients"),
                color=alt.Color("edge_case_origin:N", title="Origin",
                                scale=alt.Scale(domain=["natural", "injected"], range=p["series"][:2])),
                order=alt.Order("edge_case_origin:N", sort="descending"),
                tooltip=["edge_case_id", "edge_case_origin", alt.Tooltip("rows:Q", format=","),
                         alt.Tooltip("share:Q", format=".2f", title="% of clients")])
            ticks = alt.Chart(targets).mark_tick(color=p["ink"], thickness=2, size=26).encode(
                y=alt.Y("profile:N", sort=order), x="target:Q", tooltip=["profile", "target"])
            show((bars + ticks).properties(title=alt.Title(
                "Edge-case stress cohort", subtitle="Natural matches fill each quota first; injection tops up. Ticks: target")),
                height=440)
        else:
            st.info("This dataset has no edge-case columns.")
    with c4, st.container(border=True):
        grid = df.groupby(["risk_tolerance_score", "effective_risk_score"]).size().rename("clients").reset_index()
        heat = alt.Chart(grid).mark_rect(stroke=p["surface"], strokeWidth=1).encode(
            x=alt.X("risk_tolerance_score:O", title="Stated tolerance (1-10)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("effective_risk_score:O", title="Effective risk (1-10)", sort="descending"),
            color=alt.Color("clients:Q", title="Clients", scale=alt.Scale(range=p["seq"], type="sqrt")),
            tooltip=[alt.Tooltip("risk_tolerance_score:O", title="Stated"),
                     alt.Tooltip("effective_risk_score:O", title="Effective"), alt.Tooltip("clients:Q", format=",")])
        show(heat.properties(title=alt.Title(
            "Capacity vs stated tolerance",
            subtitle="Cells below the diagonal: capacity caps the risk a client asked for")), height=440)

    with st.expander("Data tables"):
        t1, t2 = st.columns(2)
        t1.dataframe(df["risk_tolerance_category"].value_counts().rename("clients"), width="stretch")
        if has_ec:
            t2.dataframe(df.groupby("edge_case_id").agg(
                rows=("edge_case_id", "size"),
                injected=("edge_case_origin", lambda x: int((x == "injected").sum()))), width="stretch")
