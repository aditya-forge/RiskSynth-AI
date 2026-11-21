"""Product overview: hero, value pillars and the live population view."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from portal import PAGES
from portal.data import active_dataset, baseline_records
from portal.theme import TIER_ORDER, palette
from portal.ui import EC_SHORT, eur, kpi_grid, pct, section, show
from risksynth.engine import wealth_profile

p = palette()
cfg = wealth_profile.load_config()
n_rules = len(cfg["wealth_profile_schema"]["consistency_rules"])

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
hero, spec = st.columns([1.75, 1], gap="large", vertical_alignment="center")
with hero:
    st.markdown(
        '<div class="rs" style="padding:34px 0 8px 0">'
        '<div class="rs-eyebrow">RiskSynth AI &middot; Synthetic data for wealth management</div>'
        '<div class="rs-h1">Synthetic client risk profiles for robo-advisory and wealth management</div>'
        '<div class="rs-lead">Privacy-safe client data grounded in INE statistical microdata (ECV, EPF, EPA). '
        'Test portfolio-allocation models and stress-test MiFID II loss-bearing capacity across '
        'edge-case cohorts EC01 to EC09.</div></div>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1.1, 2.2])
    with b1.container(key="cta_hero"):
        st.page_link(PAGES["generator"], label="Launch Workbench", width="stretch")
    with b2.container(key="secondary_hero"):
        st.page_link(PAGES["architecture"], label="View the architecture", width="stretch")
with spec:
    st.markdown(
        '<div class="rs" style="border:1px solid var(--rs-line);background:var(--rs-panel);padding:18px 20px">'
        '<div class="rs-label" style="margin-top:0">Specification</div><table class="rs-spec">'
        '<tr><td>Data sources</td><td>INE ECV 2024, EPF 2024, EPA 2025 T3</td></tr>'
        '<tr><td>Suitability model</td><td>MiFID II tolerance vs capacity</td></tr>'
        f'<tr><td>Rules enforced</td><td>CR01-CR0{n_rules} and 3 MiFID II checks</td></tr>'
        '<tr><td>Stress cohort</td><td>9 archetypes, 8% of rows by default</td></tr>'
        '<tr><td>Engines</td><td>Rule-based, bootstrap sampler, SDV (optional)</td></tr>'
        '<tr><td>Outputs</td><td>Parquet, CSV, JSON metadata, HTML report</td></tr>'
        '<tr><td>License</td><td>MIT</td></tr></table></div>', unsafe_allow_html=True)

st.markdown(
    '<style>.st-key-cta_hero [data-testid="stPageLink"] a { background:var(--rs-accent); border:1px solid var(--rs-accent);'
    ' border-radius:0; justify-content:center; padding:8px 14px; }'
    '.st-key-cta_hero [data-testid="stPageLink"] p { color:#fff !important; font-weight:600; }'
    '.st-key-secondary_hero [data-testid="stPageLink"] a { border:1px solid var(--rs-rule); border-radius:0;'
    ' justify-content:center; padding:8px 14px; background:transparent; }'
    '.st-key-secondary_hero [data-testid="stPageLink"] p { color:var(--rs-ink) !important; font-weight:500; }</style>',
    unsafe_allow_html=True)

# Stats ribbon: every figure is computed or read from config, not typed in
st.markdown(
    '<div class="rs rs-grid rs-stats" style="margin-top:28px">'
    f'<div class="rs-stat"><div class="v">{baseline_records():,}</div><div class="l">baseline INE microdata records</div></div>'
    f'<div class="rs-stat"><div class="v">{len(cfg["edge_case_profiles"]["profiles"])}</div><div class="l">edge-case archetypes (EC01-EC09)</div></div>'
    f'<div class="rs-stat"><div class="v">{n_rules} + 3</div><div class="l">consistency rules and MiFID II checks on every profile</div></div>'
    '<div class="rs-stat"><div class="v">0</div><div class="l">real individuals in any generated record</div></div>'
    '</div>', unsafe_allow_html=True)

# Pillars
st.markdown(
    '<div class="rs rs-grid rs-pillars" style="margin-top:28px">'
    '<div class="rs-pillar"><div class="k rs-mono">01</div><div class="t">Microdata grounded</div>'
    '<div class="d">Household income, spending, mortgage debt and employment come from INE survey microdata, '
    'harmonised across ECV, EPF and EPA and weighted to the Spanish population.</div></div>'
    '<div class="rs-pillar"><div class="k rs-mono">02</div><div class="t">MiFID II suitability</div>'
    '<div class="d">Two separate engines: stated risk tolerance from the questionnaire, and objective '
    'loss-bearing capacity from surplus, debt, horizon and liquidity. Effective risk is the lower of the two.</div></div>'
    '<div class="rs-pillar"><div class="k rs-mono">03</div><div class="t">Edge-case stress testing</div>'
    '<div class="d">A controlled 8% cohort of nine archetypes, each with an explicit assertion about the '
    'expected advisor behaviour, plus an allocation simulator that shows the de-risking step by step.</div></div>'
    '</div>', unsafe_allow_html=True)

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
        ("Median investable surplus", eur(df["investable_surplus_eur"].median()), "per year"),
        ("Negative raw surplus", pct(100 * (df["raw_surplus_eur"] < 0).mean()), "spending and debt above income"),
        ("Capacity-bound", pct(100 * (df["effective_risk_score"] < df["risk_tolerance_score"]).mean()),
         "effective risk below stated"),
        ("Re-profiling flags", f"{int(df.get('profile_inconsistency_flag', pd.Series(False)).astype(bool).sum()):,}",
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
