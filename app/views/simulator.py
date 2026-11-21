"""Robo-advisory simulator: allocation per client and protection per stress profile."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from portal.data import active_dataset, advisory
from portal.theme import palette
from portal.ui import EC_SHORT, eur, kpi_grid, pct, section, show
from risksynth.scoring import robo_advisor

p = palette()
SLEEVES = ["Equities", "Fixed income", "Cash"]

section("Robo-advisory simulator",
        "Feed synthetic clients into a model-portfolio engine and see how each suitability protection changes the "
        "recommendation: stated tolerance only, then risk capacity and liquidity, then a drawdown limit.",
        eyebrow="Stress testing", first=True)

loaded = active_dataset("simulator")
if loaded:
    df, path, mtime = loaded
    rec = advisory(path, mtime)
    has_ec = "edge_case_id" in df
    groups = ["All clients"]
    if has_ec:
        groups += ["Base population"] + [f"{e}  {EC_SHORT[e]}" for e in sorted(df["edge_case_id"].dropna().unique())]

    def _pick(prefix: str) -> None:
        match = next((g for g in groups if g.startswith(prefix)), None)
        if match:
            st.session_state["sim_group"] = match
            st.session_state.pop("sim_client", None)

    if st.session_state.get("sim_group") not in groups:
        st.session_state["sim_group"] = groups[0]
    a, b, c, d, e = st.columns([1.5, 1.8, 0.9, 1.05, 1.05], vertical_alignment="bottom")
    group = a.selectbox("Stress profile", groups, key="sim_group")
    c.button("Random client", width="stretch",
             on_click=lambda: st.session_state.update(sim_client="random"))
    d.button("EC05 over-leveraged", width="stretch", disabled="EC05" not in "".join(groups),
             on_click=_pick, args=("EC05",))
    e.button("EC08 inconsistent", width="stretch", disabled="EC08" not in "".join(groups),
             on_click=_pick, args=("EC08",))

    if group == "All clients":
        pool = df.index
    elif group == "Base population":
        pool = df.index[df["edge_case_id"].isna()]
    else:
        pool = df.index[df["edge_case_id"].eq(group.split()[0]).fillna(False)]
    options = list(pool[:400])
    if st.session_state.get("sim_client") == "random":
        st.session_state["sim_client"] = int(np.random.default_rng().choice(options))
    current = st.session_state.get("sim_client")
    client = b.selectbox("Client", options, index=options.index(current) if current in options else 0,
                         format_func=lambda i: (f"#{i}  age {int(df.at[i, 'edad'])}  {df.at[i, 'risk_tolerance_category']}  "
                                                f"{df.at[i, 'edge_case_id'] if has_ec and pd.notna(df.at[i, 'edge_case_id']) else ''}"))
    row, r = df.loc[client], rec.loc[client]

    left, right = st.columns([1, 1.55], gap="large")
    with left, st.container(border=True):
        st.markdown('<div class="rs rs-label" style="margin-top:0">Client profile</div>', unsafe_allow_html=True)
        facts = {
            "Age / goal": f"{int(row['edad'])} / {row['primary_investment_goal']}",
            "Horizon": f"{row['investment_horizon_years']:.1f} y ({row['investment_horizon_bucket']})",
            "Stated tolerance": f"{int(row['risk_tolerance_score'])} ({row['risk_tolerance_category']})",
            "Risk capacity": f"{int(row['risk_capacity_score'])}",
            "Effective risk": f"{int(row['effective_risk_score'])}",
            "Max drawdown accepted": pct(row["max_acceptable_drawdown_pct"], 0),
            "Liquidity need": pct(row["liquidity_need_pct"], 0),
            "Investable surplus": eur(row["investable_surplus_eur"]) + " / y",
            "Liquid savings": eur(row["liquid_savings_eur"]),
            "Debt service / income": pct(100 * (row["debt_service_to_income"] if pd.notna(row["debt_service_to_income"]) else 0)),
            "ESG preference": str(row["esg_preference_level"]),
            "Capacity caps": str(row.get("capacity_caps_applied")) if pd.notna(row.get("capacity_caps_applied")) else "none",
        }
        st.markdown('<dl class="rs rs-facts">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts.items())
                    + "</dl>", unsafe_allow_html=True)
        if has_ec and pd.notna(row.get("edge_case_id")):
            verdict = "assertion holds" if bool(row.get("edge_case_assertion_pass")) else "assertion fails"
            st.markdown(f'<div class="rs rs-expect"><b>{row["edge_case_id"]} {EC_SHORT[row["edge_case_id"]]}</b> '
                        f'({row["edge_case_origin"]})<br>Expected behaviour: {row["expected_robo_behavior"]}<br>'
                        f'<span class="rs-note">Test <code>{row["edge_case_assertion"]}</code>: {verdict}</span></div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="rs rs-label">Advisor actions</div>', unsafe_allow_html=True)
        raised = [lbl for col, lbl in robo_advisor.ACTION_LABELS.items() if bool(r[col])]
        if bool(r["drawdown_guard_applied"]):
            raised.append("Equity cut to keep the stressed loss within the accepted drawdown")
        st.markdown("".join(f'<div class="rs rs-action">{t}</div>' for t in raised) if raised else
                    '<div class="rs rs-action ok">No suitability actions: the tolerance-based allocation stands</div>',
                    unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            steps = [("1  Tolerance only", "naive"), ("2  + Capacity and liquidity", "capacity"), ("3  + Drawdown limit", "rec")]
            rows = [{"step": label, "sleeve": s, "pct": float(v), "order": i}
                    for label, pre in steps
                    for i, (s, v) in enumerate(zip(SLEEVES, [r[f"{pre}_equity_pct"], r[f"{pre}_fixed_income_pct"],
                                                             r[f"{pre}_cash_pct"]]))]
            alloc = pd.DataFrame(rows)
            sort = [s for s, _ in steps]
            bars = alt.Chart(alloc).mark_bar(height=30, stroke=p["surface"], strokeWidth=2).encode(
                y=alt.Y("step:N", title=None, sort=sort, axis=alt.Axis(labelLimit=220)),
                x=alt.X("pct:Q", stack="zero", title="% of portfolio", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("sleeve:N", title=None, scale=alt.Scale(domain=SLEEVES, range=p["series"])),
                order=alt.Order("order:Q"), tooltip=["step", "sleeve", alt.Tooltip("pct:Q", format=".1f", title="%")])
            text = alt.Chart(alloc[alloc["pct"] >= 7]).mark_text(color="white", fontSize=11, fontWeight=600).encode(
                y=alt.Y("step:N", sort=sort), x=alt.X("pct:Q", stack="zero", bandPosition=0.5),
                detail="sleeve:N", order=alt.Order("order:Q"), text=alt.Text("pct:Q", format=".0f"))
            show((bars + text).properties(title=alt.Title(
                "Recommended allocation", subtitle="Each row adds one suitability protection")), height=250)
        kpi_grid([
            ("Stressed loss", pct(r["rec_stress_loss_pct"]), f"limit {row['max_acceptable_drawdown_pct']:.0f}%"),
            ("Tolerance-only loss", pct(r["naive_stress_loss_pct"]), "same stress scenario"),
            ("Equity cut", f"{r['equity_reduction_pp']:.1f} pp",
             f"capacity {r['equity_cut_by_capacity_pp']:.1f}, limit {r['equity_cut_by_drawdown_pp']:.1f}"),
            ("ESG-screened", pct(r["rec_esg_pct"]), f"preference {row['esg_preference_level']}"),
        ])

    section("Protection by stress profile",
            "Mean equity share per archetype under each rule. The gap between the dots is the equity the suitability "
            "layer removes.")
    key = df["edge_case_id"].fillna("Base") if has_ec else pd.Series("All", index=df.index)
    agg = pd.concat([key.rename("group"), rec], axis=1).groupby("group").agg(
        naive=("naive_equity_pct", "mean"), capacity=("capacity_equity_pct", "mean"),
        recommended=("rec_equity_pct", "mean"), clients=("rec_equity_pct", "size"),
        reprofile=("action_reprofile", "mean"), capacity_bound=("action_capacity_bound", "mean"),
        no_contrib=("action_no_new_contributions", "mean"), guard=("drawdown_guard_applied", "mean")).reset_index()
    agg["label"] = agg["group"].map(lambda g: f"{g}  {EC_SHORT[g]}" if g in EC_SHORT else f"{g} population")
    long = agg.melt(id_vars=["label"], value_vars=["naive", "capacity", "recommended"], var_name="allocation",
                    value_name="equity")
    names = {"naive": "Tolerance only (1)", "capacity": "Capacity (2)", "recommended": "Recommended (3)"}
    long["allocation"] = long["allocation"].map(names)
    order = list(agg.sort_values("group")["label"])
    with st.container(border=True):
        rule = alt.Chart(agg).mark_rule(color=p["neutral"], strokeWidth=2).encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)), x="recommended:Q", x2="naive:Q")
        dots = alt.Chart(long).mark_circle(size=110, opacity=1, stroke=p["surface"], strokeWidth=2).encode(
            y=alt.Y("label:N", sort=order), x=alt.X("equity:Q", title="Mean equity share, % of portfolio"),
            color=alt.Color("allocation:N", title=None, scale=alt.Scale(
                domain=list(names.values()), range=[p["series"][1], p["series"][2], p["series"][0]])),
            tooltip=["label", "allocation", alt.Tooltip("equity:Q", format=".1f", title="Equity %")])
        show((rule + dots).properties(title=alt.Title(
            "Mean equity share per stress profile",
            subtitle="EC08 is driven to bonds and cash by its 5-10% drawdown limit; EC05 by its debt-service cap")),
            height=440)
    st.dataframe(agg[["group", "clients", "naive", "capacity", "recommended", "capacity_bound", "guard", "no_contrib",
                      "reprofile"]], hide_index=True, width="stretch", column_config={
        "group": "Profile", "clients": st.column_config.NumberColumn("Clients", format="%d"),
        "naive": st.column_config.NumberColumn("Equity, tolerance only %", format="%.1f"),
        "capacity": st.column_config.NumberColumn("Equity, capacity %", format="%.1f"),
        "recommended": st.column_config.NumberColumn("Equity, recommended %", format="%.1f"),
        "capacity_bound": st.column_config.NumberColumn("Capacity-bound", format="percent"),
        "guard": st.column_config.NumberColumn("Drawdown limit applied", format="percent"),
        "no_contrib": st.column_config.NumberColumn("No contributions", format="percent"),
        "reprofile": st.column_config.NumberColumn("Re-profile", format="percent")})
    st.caption("Model portfolios are illustrative ([ASSUMPTION] parameters in model_config.yaml, robo_advisor_simulator). "
               "Stress losses: equities -50%, fixed income -10%, cash 0%.")
