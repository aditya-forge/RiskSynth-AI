"""Statistical fidelity against INE and the rule / MiFID II compliance audit."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from portal.data import active_dataset, edge_scorecard, fidelity_results, rel
from portal.theme import palette
from portal.ui import SHORT, heatmap, kpi_grid, scorecard_html, section, show
from risksynth.evaluation import fidelity

p = palette()

section("Compliance and statistical fidelity",
        "How closely the synthetic population matches the INE microdata it was built from, and whether every profile "
        "passes the consistency rules and MiFID II checks.", eyebrow="Audit", first=True)


def status_table(frame: pd.DataFrame, **config) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "status": st.column_config.TextColumn("Status", help="PASS / FAIL against model_config.yaml thresholds"),
        **config})


def scorecard(frame: pd.DataFrame, value: str, threshold: float, title: str, subtitle: str, fmt: str):
    data = frame.assign(label=frame["column"].map(lambda c: SHORT.get(c, c)))
    order = list(data.sort_values(value, ascending=False)["label"])
    bars = alt.Chart(data).mark_bar(height=14, color=p["series"][0]).encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X(f"{value}:Q", title=None, scale=alt.Scale(domain=[0, max(threshold * 1.25, float(data[value].max()) * 1.1)])),
        tooltip=[alt.Tooltip("column:N", title="Column"), alt.Tooltip(f"{value}:Q", format=fmt), "status"])
    text = alt.Chart(data).mark_text(align="left", dx=5, fontSize=10.5, color=p["ink2"]).encode(
        y=alt.Y("label:N", sort=order), x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=fmt))
    limit = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color=p["ink"], strokeDash=[4, 3]).encode(x="x:Q")
    return (bars + text + limit).properties(title=alt.Title(title, subtitle=subtitle))


loaded = active_dataset("compliance")
if loaded:
    df, path, mtime = loaded
    with st.spinner("Running fidelity and compliance tests"):
        res = fidelity_results(path, mtime)
    thr = fidelity.load_model_config()["evaluation_config"]["thresholds"]
    ref = ("INE persons base (ECV, EPF, EPA) for demographics and EPF households for finances"
           if res["reference_kind"] == "persons" else "INE EPF households")
    st.markdown(f'<div class="rs rs-note" style="margin:-4px 0 12px 0">Reference: {ref}. Fidelity tests use '
                f'{res["rows_base_population"]:,} base-population rows; {res["rows_injected_excluded"]:,} injected '
                'edge cases are excluded. Rule checks use all rows.</div>', unsafe_allow_html=True)

    def passed(frame):
        return f"{int((frame['status'] == 'PASS').sum())} / {len(frame)}" if len(frame) else "n/a"
    comp = res["compliance"]
    kpi_grid([
        ("Numeric tests", passed(res["numeric"]), "KS D and Wasserstein"),
        ("Categorical tests", passed(res["categorical"]), "Jensen-Shannon"),
        ("Target mixes", passed(res["target_mix"]), "risk tier and ESG"),
        ("Correlation drift", f"{res['correlation']['mean_abs_diff']:.3f}",
         f"mean |rho diff|, {res['correlation']['status']}"),
        ("Unexpected violations", f"{int(comp['unexpected'].sum()):,}", "CR01-CR07 and MiFID II"),
        ("Intentional exemptions", f"{int(comp['exempted'].sum()):,}", "EC08 drawdown contradiction"),
    ])

    section("Distribution fidelity", "Per-column distance between synthetic and INE distributions. "
            "Bars past the dashed line fail.")
    s1, s2 = st.columns(2, gap="medium")
    num = res["numeric"]
    with s1, st.container(border=True):
        show(scorecard(num, "ks_d", thr["numerical"]["ks_statistic_max"], "Kolmogorov-Smirnov D",
                       f"Maximum CDF gap; limit {thr['numerical']['ks_statistic_max']}", ".3f"), height=300)
    with s2, st.container(border=True):
        show(scorecard(num, "wasserstein_norm", thr["numerical"]["wasserstein_normalized"]["acceptable"],
                       "Wasserstein distance (normalised)",
                       f"Earth mover's distance over the p1-p99 range; limit "
                       f"{thr['numerical']['wasserstein_normalized']['acceptable']}", ".3f"), height=300)
    with st.expander("Test tables"):
        status_table(num, ks_p_value=st.column_config.NumberColumn("KS p-value", format="%.2e"))
        t1, t2 = st.columns(2, gap="large")
        with t1:
            status_table(res["categorical"])
        with t2:
            status_table(res["target_mix"])

    section("Correlation structure", "Spearman correlations. Injected edge cases are excluded from both matrices.")
    h1, h2 = st.columns(2, gap="medium")
    corr = res["correlation"]
    with h1, st.container(border=True):
        show(heatmap(corr["difference"], "Synthetic minus INE",
                     f"Household finance and demographics. Mean |diff| {corr['mean_abs_diff']}, max {corr['max_abs_diff']}",
                     0.3), height=450)
    with h2, st.container(border=True):
        show(heatmap(res["wealth_correlation"], "Profile feature correlations", "Synthetic base population", 1.0),
             height=450)
    status_table(res["dependencies"])

    section("Rule and suitability audit", "Violations after repair. Exempted rows are intentional edge-case traits.")
    status_table(comp, definition=st.column_config.TextColumn("Rule", width="large"))

    section("Edge-case suitability scorecard",
            "Pass/fail per EC01-EC09 archetype: the written assertion, the expected capacity cap, and the stressed "
            "loss of the recommended portfolio against the client's accepted drawdown.")
    st.markdown(scorecard_html(edge_scorecard(path, mtime)), unsafe_allow_html=True)

    section("Evaluation report", "A self-contained HTML report with every table on this page.")
    if st.button("Build HTML report"):
        out = fidelity.save_report(res, rel(Path(path)))
        st.session_state["report_path"] = str(out)
    rp = st.session_state.get("report_path")
    if rp and Path(rp).exists():
        st.download_button("Download HTML report", Path(rp).read_bytes(), Path(rp).name, "text/html")
        st.caption(f"Saved to reports/evaluation_reports/{Path(rp).name}")
