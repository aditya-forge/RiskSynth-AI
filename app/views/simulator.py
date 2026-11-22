"""Robo-advisory simulator: allocation per client and protection per stress profile."""
import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from portal.data import active_dataset, advisory, edge_scorecard
from portal.icons import icon
from portal.theme import palette
from portal.ui import EC_SHORT, pct, scorecard_html, section, show
from risksynth.scoring import counterfactual

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
    edge = df["edge_case_id"].notna() if has_ec else pd.Series(False, index=df.index)
    limit = df["max_acceptable_drawdown_pct"].astype(float)
    caps = df.get("capacity_caps_applied", pd.Series(pd.NA, index=df.index)).astype("string").fillna("")

    # Test cohorts: the slices a robo-advisor has to get right, side by side
    COHORTS = {
        "Baseline profiles": ~edge,
        "Edge cases EC01-EC09": edge,
        "Capacity-constrained": rec["action_capacity_bound"].fillna(False).astype(bool),
        "Re-profiling": rec["action_reprofile"].fillna(False).astype(bool),
    }
    section("Test cohorts",
            "How each slice of the synthetic book stresses an allocation model: how often MiFID II capacity binds, "
            "how far a tolerance-only portfolio would fall against what clients accept, and whether the "
            "recommended portfolio stays inside the limit.")
    rows = []
    for name, m in COHORTS.items():
        n = int(m.sum())
        if n == 0:
            rows.append((name, 0) + ("-",) * 8)
            continue
        breach = 100 * (rec.loc[m, "rec_stress_loss_pct"] > limit[m] + 0.05).mean()
        rows.append((name, n,
                     pct(100 * rec.loc[m, "action_capacity_bound"].fillna(False).astype(bool).mean()),
                     pct(100 * (caps[m] != "").mean()),
                     pct(100 * rec.loc[m, "action_reprofile"].fillna(False).astype(bool).mean()),
                     pct(limit[m].mean()),
                     (pct(rec.loc[m, "naive_stress_loss_pct"].mean()), "bad"),
                     pct(rec.loc[m, "rec_stress_loss_pct"].mean()),
                     (pct(breach), "good" if breach == 0 else "bad"),
                     f"{rec.loc[m, 'equity_reduction_pp'].mean():.1f} pp"))
    heads = ["Cohort", "Clients", "Capacity-bound", "Capacity caps", "Re-profile", "Accepted drawdown",
             "Stressed loss, tolerance only", "Stressed loss, recommended", "Over limit after advice", "Equity cut"]

    def _td(v, i):
        text, cls = v if isinstance(v, tuple) else (v, "")
        text = f"{text:,}" if isinstance(text, int) else text
        inner = f'<span class="{cls}">{text}</span>' if cls else text
        return f'<td class="{"num" if i else "name"}">{inner}</td>'
    st.markdown('<div class="rs rs-scroll"><table class="rs-table rs-cohorts"><thead><tr>'
                + "".join(f'<th class="{"num" if i else "name"}">{h}</th>' for i, h in enumerate(heads))
                + "</tr></thead><tbody>"
                + "".join("<tr>" + "".join(_td(v, i) for i, v in enumerate(r)) + "</tr>" for r in rows)
                + "</tbody></table></div>", unsafe_allow_html=True)
    st.caption("Capacity-bound: effective risk below stated tolerance. Capacity caps: at least one MiFID II capacity "
               "cap (CAP_*) applied. Accepted drawdown and stressed losses are cohort means; stress scenario equities "
               "-50%, fixed income -10%, cash 0%. No real client data is involved at any step.")

    section("Counterfactual client inspector",
            "Pick a client, including any EC01-EC09 edge case, then change its inputs. Risk capacity, the rule checks, "
            "the suitability status and the portfolio are recomputed with the same engine code the generator uses.")
    groups = ["All clients"] + [c for c, m in COHORTS.items() if m.any()]
    if has_ec:
        groups += [f"{e}  {EC_SHORT[e]}" for e in sorted(df["edge_case_id"].dropna().unique())]

    def _pick(name: str) -> None:
        st.session_state["sim_group"] = name
        st.session_state.pop("sim_client", None)

    if st.session_state.get("sim_group") not in groups:
        st.session_state["sim_group"] = groups[0]
    with st.container(horizontal=True, key="sim_presets", gap="small"):
        for name, m in COHORTS.items():
            st.button(name, disabled=not m.any(), on_click=_pick, args=(name,),
                      type="primary" if st.session_state["sim_group"] == name else "secondary", key=f"cohort_{name}")
        st.button("Random client", key="sim_random", on_click=lambda: st.session_state.update(sim_client="random"))
    a, b = st.columns([1.5, 2.8], vertical_alignment="bottom")
    group = a.selectbox("Cohort or archetype", groups, key="sim_group")

    if group == "All clients":
        pool = df.index
    elif group in COHORTS:
        pool = df.index[COHORTS[group]]
    else:
        pool = df.index[df["edge_case_id"].eq(group.split()[0]).fillna(False)]
    options = list(pool[:400])
    if st.session_state.get("sim_client") == "random":
        st.session_state["sim_client"] = int(np.random.default_rng().choice(options))
    current = st.session_state.get("sim_client")
    client = b.selectbox("Client", options, index=options.index(current) if current in options else 0,
                         format_func=lambda i: (f"#{i}  age {int(df.at[i, 'edad'])}  {df.at[i, 'risk_tolerance_category']}  "
                                                f"{df.at[i, 'edge_case_id'] if has_ec and pd.notna(df.at[i, 'edge_case_id']) else ''}"))
    row = df.loc[client]

    # Counterfactual inputs: keyed by client, so picking another client starts from its own values
    goals = sorted(df["primary_investment_goal"].dropna().astype(str).unique())
    INPUTS = {   # column: (kind, min, max, step)
        "edad": ("int", 18, 100, 1), "net_income_annual_eur": ("eur", 0, 500_000, 1_000),
        "household_expenses_annual_eur": ("eur", 0, 500_000, 1_000),
        "total_debt_service_annual_eur": ("eur", 0, 200_000, 500), "liquid_savings_eur": ("eur", 0, 2_000_000, 1_000),
        "risk_tolerance_score": ("slider", 1, 10, 1), "investment_horizon_years": ("float", 0.5, 60.0, 0.5),
        "max_acceptable_drawdown_pct": ("slider", 0, 60, 1), "liquidity_need_pct": ("slider", 0, 100, 1),
        "primary_investment_goal": ("goal", None, None, None),
    }

    def _original(col):
        v = row[col]
        if col == "primary_investment_goal":
            return str(v)
        if INPUTS[col][0] in ("int", "slider"):
            return int(round(float(v)))
        return float(round(float(0 if pd.isna(v) else v), 1))

    def _reset() -> None:
        for col in INPUTS:
            st.session_state.pop(f"cf_{client}_{col}", None)

    left, right = st.columns([1, 1.55], gap="large")
    with left, st.container(border=True):
        st.markdown('<div class="rs rs-label" style="margin-top:0">What if: adjust the inputs</div>',
                    unsafe_allow_html=True)
        values = {}
        c1, c2 = st.columns(2)
        for i, (col, (kind, lo, hi, step)) in enumerate(INPUTS.items()):
            key, label, orig = f"cf_{client}_{col}", counterfactual.EDITABLE[col], _original(col)
            box = c1 if i % 2 == 0 else c2
            if kind == "goal":
                values[col] = box.selectbox(label, goals, index=goals.index(orig) if orig in goals else 0, key=key)
            elif kind == "slider":
                values[col] = box.slider(label, lo, hi, min(max(orig, lo), hi), step, key=key)
            elif kind == "int":
                values[col] = box.number_input(label, lo, hi, min(max(orig, lo), hi), step, key=key)
            else:
                values[col] = box.number_input(label, float(lo), float(hi), min(max(orig, float(lo)), float(hi)),
                                               float(step), key=key)
        changes = {c: v for c, v in values.items() if v != _original(c)}
        st.button("Reset to the generated profile", on_click=_reset, disabled=not changes, key=f"cf_reset_{client}")

    cf = counterfactual.reassess(row, changes)
    r, prof = cf["recommendation"], cf["profile"].iloc[0]
    with right:
        tone = {"Suitable": "good", "Suitable with restrictions": "warn", "Re-profile": "bad"}[cf["status"]]
        moved = f"{len(changes)} input{'s' if len(changes) != 1 else ''} changed" if changes else "as generated"

        def _delta(before, after):
            return f"{before} &rarr; {after}" if before != after else str(after)
        st.markdown(
            f'<div class="rs rs-verdict {tone}"><div><div class="k">Suitability status</div>'
            f'<div class="v">{cf["status"]}</div><div class="n">{moved}</div></div>'
            f'<div><div class="k">Risk capacity</div><div class="v">'
            f'{_delta(int(row["risk_capacity_score"]), cf["capacity_score"])}</div>'
            f'<div class="n">{cf["capacity_points"]} points before caps</div></div>'
            f'<div><div class="k">Effective risk</div><div class="v">'
            f'{_delta(int(row["effective_risk_score"]), cf["effective_risk"])}</div>'
            f'<div class="n">stated {int(prof["risk_tolerance_score"])} ({cf["tolerance_category"]})</div></div>'
            f'<div><div class="k">Stressed loss</div><div class="v">{pct(r["rec_stress_loss_pct"])}</div>'
            f'<div class="n">limit {prof["max_acceptable_drawdown_pct"]:.0f}%, tolerance only '
            f'{pct(r["naive_stress_loss_pct"])}</div></div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            steps = [("1  Tolerance only", "naive"), ("2  + Capacity and liquidity", "capacity"),
                     ("3  + Drawdown limit", "rec")]
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
                "Recommended allocation", subtitle="Each row adds one suitability protection")), height=260)

    def _state(item):
        if not item["violated"]:
            return '<span class="rs-pass">pass</span>'
        if item.get("exempt"):
            return '<span class="rs-muted">relaxed for this edge case</span>'
        return '<span class="rs-fail">violated</span>'

    checks, actions = st.columns([1.15, 1], gap="large")
    with checks:
        st.markdown('<div class="rs rs-label" style="margin-top:0">CR01-CR07 and MiFID II checks</div>',
                    unsafe_allow_html=True)
        body = "".join(f'<tr><td class="id">{x["rule"].replace("MIFID_", "MiFID ")}</td>'
                       f'<td><code>{html.escape(x["text"])}</code></td>'
                       f'<td class="st">{_state(x)}</td></tr>' for x in cf["rules"] + cf["mifid"])
        st.markdown(f'<table class="rs rs-table rs-checks"><tbody>{body}</tbody></table>', unsafe_allow_html=True)
    with actions:
        caps = ", ".join(cf["caps"]) if cf["caps"] else "none"
        st.markdown(f'<div class="rs rs-label" style="margin-top:0">Capacity caps</div>'
                    f'<div class="rs rs-note" style="margin-bottom:8px"><code>{caps}</code></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="rs rs-label">Advisor actions</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs">' + ("".join(f'<div class="rs-action">{icon("flag", 14)}<span>{t}</span></div>'
                                                  for t in cf["actions"]) if cf["actions"] else
                    f'<div class="rs-action ok">{icon("check", 14)}<span>No suitability actions: the '
                    'tolerance-based allocation stands</span></div>') + '</div>', unsafe_allow_html=True)
        ea = cf["edge_case_assertion"]
        if ea:
            verdict = '<span class="rs-pass">holds</span>' if ea["holds"] else '<span class="rs-fail">fails</span>'
            st.markdown(f'<div class="rs rs-expect" style="margin-top:12px"><b>{ea["id"]} {EC_SHORT[ea["id"]]}</b>'
                        f'<br>Expected behaviour: {html.escape(ea["expected"])}<br>'
                        f'<span class="rs-note">Test <code>{html.escape(ea["test"])}</code>: {verdict}</span></div>',
                        unsafe_allow_html=True)

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
    section("Edge-case suitability scorecard",
            "One row per archetype: does the written assertion hold, does the expected capacity cap fire, and does "
            "the recommended portfolio stay inside the accepted drawdown. A cohort passes when every client does.")
    st.markdown(scorecard_html(edge_scorecard(path, mtime)), unsafe_allow_html=True)
    st.caption("Model portfolios are illustrative ([ASSUMPTION] parameters in model_config.yaml, robo_advisor_simulator). "
               "Stress losses: equities -50%, fixed income -10%, cash 0%.")
