"""Reusable portal components: brand, header, footer, KPI grid, charts."""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from portal import GITHUB_URL
from portal.icons import icon
from portal.theme import mode, palette

EC_SHORT = {"EC01": "High income, low liquidity", "EC02": "Young, high risk, low capital",
            "EC03": "Near-retiree, aggressive", "EC04": "Negative surplus investor",
            "EC05": "Over-leveraged risk seeker", "EC06": "Retiree, income, high liquidity",
            "EC07": "Short horizon, aggressive", "EC08": "Inconsistent questionnaire",
            "EC09": "Asset-rich, cash-poor"}


def logo_svg(size: int = 28) -> str:
    """Emblem: chart axes with an efficient-frontier curve through three portfolio nodes."""
    p = palette()
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" '
            f'role="img" aria-label="RiskSynth AI">'
            f'<rect x="0.75" y="0.75" width="26.5" height="26.5" stroke="{p["rule"]}" stroke-width="1.5"/>'
            f'<path d="M6 5.5V22H22.5" stroke="{p["ink"]}" stroke-width="1.6" stroke-linecap="square"/>'
            f'<path d="M7.5 20C10 13 14 9.5 21.5 7.5" stroke="{p["accent"]}" stroke-width="1.6"/>'
            f'<rect x="8.6" y="15.1" width="3" height="3" fill="{p["accent"]}"/>'
            f'<rect x="13.1" y="9.9" width="3" height="3" fill="{p["ink"]}"/>'
            f'<rect x="18.6" y="6.7" width="3" height="3" fill="{p["accent"]}"/>'
            f'</svg>')


def brand_html() -> str:
    return f'<div class="rs rs-brand">{logo_svg(28)}<span class="word">RiskSynth <b>AI</b></span></div>'


NAV_LABELS = {"overview": "Product Overview", "architecture": "Solution Architecture",
              "generator": "Live Generator", "simulator": "Advisory Simulator",
              "compliance": "Compliance & Fidelity"}


def theme_switch() -> str:
    """White / Dark switch: two icon buttons with labels for screen readers and tooltips.

    A link rather than a widget, because the browser applies the theme when the page loads.
    """
    on = mode()
    return ('<div class="rs rs-theme" role="group" aria-label="Colour theme">'
            f'<a href="?embed_options=light_theme" target="_self" class="{"on" if on == "light" else ""}" '
            f'title="White theme" aria-label="White theme" aria-pressed="{str(on == "light").lower()}">{icon("sun", 15)}</a>'
            f'<a href="?embed_options=dark_theme" target="_self" class="{"on" if on == "dark" else ""}" '
            f'title="Dark theme" aria-label="Dark theme" aria-pressed="{str(on == "dark").lower()}">{icon("moon", 15)}</a>'
            '</div>')


def header(pages: dict, current) -> None:
    """Top navigation bar as one flex row: brand | page links + GitHub | theme switch + call to action.

    Built from horizontal containers so every item sizes to its content; below 1180px the links
    move to a second, horizontally scrollable row.
    """
    active = next((k for k, page in pages.items() if page.title == current.title), "overview")
    st.markdown(f"<style>.st-key-nav_{active} [data-testid='stPageLink'] a {{ border-bottom-color: "
                f"{palette()['accent']} !important; }} .st-key-nav_{active} [data-testid='stPageLink'] p "
                f"{{ color: {palette()['heading']} !important; }}</style>", unsafe_allow_html=True)
    with st.container(key="navbar", horizontal=True, vertical_alignment="center",
                      horizontal_alignment="distribute", gap=None):
        st.markdown(brand_html(), unsafe_allow_html=True, width="content")
        with st.container(key="navlinks", horizontal=True, vertical_alignment="center", width="content", gap=None):
            for key, label in NAV_LABELS.items():
                with st.container(key=f"nav_{key}", width="content"):
                    st.page_link(pages[key], label=label)
            # wrapped in a div: inline HTML would otherwise get a <p> with a bottom margin
            st.markdown(f'<div class="rs"><a class="rs-navext" href="{GITHUB_URL}" target="_blank" rel="noopener">'
                        f'{icon("github", 15)}GitHub</a></div>', unsafe_allow_html=True, width="content")
        with st.container(key="navtools", horizontal=True, vertical_alignment="center", width="content", gap=None):
            st.markdown(theme_switch(), unsafe_allow_html=True, width="content")
            with st.container(key="cta", width="content"):
                st.page_link(pages["generator"], label="Launch Workbench")


def footer() -> None:
    why = [
        ("code", "The full source, configs and INE processing code are public, so nothing about how a profile "
                 "is built is hidden."),
        ("terminal", "Every rule and edge case is covered by a pytest suite you can run on your own machine."),
        ("refresh", "GitHub Actions runs the tests on every push to main, on Python 3.11 and 3.12."),
        ("search", "Reviewers can trace the INE microdata handling, the MiFID II suitability rules and the "
                   "EC01-EC09 edge-case profiles line by line."),
    ]
    items = "".join(f"<li>{icon(name, 15)}<span>{text}</span></li>" for name, text in why)
    st.markdown(
        '<div class="rs rs-footer">'
        f'<div>{brand_html()}<p style="margin-top:14px">Synthetic client risk profiles for building and testing '
        'robo-advisors and suitability checks. Built on Spanish household survey microdata.</p></div>'
        f'<div><div class="h">Why the code is on GitHub</div><ul>{items}</ul>'
        '<p class="assume">Parameters marked [ASSUMPTION] are design choices, not measurements. Calibrate them '
        'against the Banco de Espa&ntilde;a EFF 2022 survey before any production use.</p>'
        f'<a class="repo" href="{GITHUB_URL}" target="_blank" rel="noopener">{icon("github", 15)}'
        'aditya-forge/RiskSynth-AI</a></div>'
        '<div><div class="h">Data</div><p>INE Encuesta de Condiciones de Vida 2024</p>'
        '<p>INE Encuesta de Presupuestos Familiares 2024</p><p>INE Encuesta de Poblaci&oacute;n Activa 2025 T3</p>'
        '<p>No real individuals appear in any generated record.</p></div>'
        '<div class="legal"><span>Synthetic data for development and testing. Not calibrated to any real client '
        'book and not investment advice.</span><span>RiskSynth AI 2.0.0</span></div>'
        '</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "", eyebrow: str = "", first: bool = False) -> None:
    eb = f'<div class="rs-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    sub = f'<div class="rs-sub">{subtitle}</div>' if subtitle else ""
    cls = "rs" if first else "rs rs-section"
    st.markdown(f'<div class="{cls}">{eb}<div class="rs-h2">{html.escape(title)}</div>{sub}</div>',
                unsafe_allow_html=True)


def kpi_grid(items: list[tuple[str, str, str]]) -> None:
    """Hairline-divided KPI cells: (label, value, note)."""
    cells = "".join(f'<div class="rs-kpi"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(v)}</div>'
                    f'<div class="n">{html.escape(n)}</div></div>' for l, v, n in items)
    st.markdown(f'<div class="rs rs-grid rs-kpis">{cells}</div>', unsafe_allow_html=True)


def html_table(frame: pd.DataFrame, code_cols: tuple = ()) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, r in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = html.escape(str(r[c]))
            cells.append(f"<td><code>{v}</code></td>" if c in code_cols else f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="rs rs-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def eur(x: float) -> str:
    return f"EUR {x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


# --------------------------------------------------------------------------- #
# Charts (Altair): thin marks, recessive axes, tooltips on every mark
# --------------------------------------------------------------------------- #
def styled(chart, height: int = 260):
    p = palette()
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=p["ink2"], titleColor=p["ink2"], gridColor=p["grid"], domainColor=p["axis"],
                            tickColor=p["axis"], labelFontSize=11, titleFontSize=11, titleFontWeight=500,
                            labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")
            .configure_legend(labelColor=p["ink"], titleColor=p["ink2"], labelFontSize=11, titleFontSize=11,
                              orient="top", direction="horizontal", symbolType="square", labelFont="IBM Plex Sans")
            .configure_title(fontSize=13, fontWeight=600, color=p["ink"], anchor="start", subtitleColor=p["ink2"],
                             subtitleFontSize=11.5, font="IBM Plex Sans", subtitleFont="IBM Plex Sans"))


def show(chart, height: int = 260) -> None:
    st.altair_chart(styled(chart, height), width="stretch")


SHORT = {"net_income_annual_eur": "Net income", "household_expenses_annual_eur": "Spending",
         "debt_service_annual_eur": "Debt service", "household_surplus_eur": "Surplus",
         "main_residence_value_eur": "Home value", "edad": "Age", "n_miembros": "Household size",
         "dependent_children": "Children", "raw_surplus_eur": "Raw surplus", "liquid_savings_eur": "Liquid savings",
         "risk_tolerance_score": "Tolerance", "investment_horizon_years": "Horizon",
         "max_acceptable_drawdown_pct": "Max drawdown", "liquidity_need_pct": "Liquidity need",
         "risk_capacity_score": "Capacity", "effective_risk_score": "Effective risk"}


def heatmap(matrix: pd.DataFrame, title: str, subtitle: str, domain: float) -> alt.LayerChart:
    """Diverging matrix with a neutral grey midpoint; tooltips keep the full column names."""
    p = palette()
    m = matrix.copy()
    m.index.name = "row"
    long = m.reset_index().melt(id_vars="row", var_name="col", value_name="value")
    long["row_label"] = long["row"].map(lambda c: SHORT.get(c, c))
    long["col_label"] = long["col"].map(lambda c: SHORT.get(c, c))
    long["strong"] = long["value"].abs() >= 0.6 * domain
    order = [SHORT.get(c, c) for c in matrix.columns]
    base = alt.Chart(long).encode(
        x=alt.X("col_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelLimit=140)),
        y=alt.Y("row_label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)))
    rect = base.mark_rect(stroke=p["surface"], strokeWidth=1).encode(
        # Lab interpolation keeps the midpoint neutral (the default HCL drifts through pink)
        color=alt.Color("value:Q", title=None, scale=alt.Scale(domain=[-domain, 0, domain], range=p["div"],
                                                               interpolate="lab", clamp=True)),
        tooltip=[alt.Tooltip("row:N", title="Row"), alt.Tooltip("col:N", title="Column"),
                 alt.Tooltip("value:Q", format="+.3f", title="Value")])
    text = base.mark_text(fontSize=9.5).encode(       # white on saturated cells, theme ink elsewhere
        text=alt.Text("value:Q", format=".2f"),
        color=alt.condition("datum.strong", alt.value("white"), alt.value(p["ink"])))
    return (rect + text).properties(title=alt.Title(title, subtitle=subtitle))
