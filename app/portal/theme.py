"""Colours, type and global CSS for the portal, for both the White and the Dark theme.

The theme is chosen in the header (White / Dark). The choice travels in the URL as
Streamlit's ``embed_options=light_theme|dark_theme`` option, which switches Streamlit's own
widgets (inputs, tables, buttons) to the matching [theme.light] / [theme.dark] section of
.streamlit/config.toml. This module reads the same option, so the portal CSS and every
chart switch together with the widgets.

Chart palettes were checked for colour-blind separation and contrast against each
background. Data series never use the UI accent or the neutral grey.
"""
from __future__ import annotations

import streamlit as st

TIER_ORDER = ["Conservative", "Moderate", "Aggressive", "Ultra-Aggressive"]

THEMES = {
    "dark": {
        "series": ["#2a8ae6", "#e0701e", "#16a085"],            # blue, amber, emerald (validated on #050608 and #0d1016)
        "tiers": ["#184f95", "#2a78d6", "#5598e7", "#9ec5f4"],  # ordinal ramp, brighter = riskier
        "seq": ["#0f3a70", "#b7d3f6"],
        "div": ["#e66767", "#1f242d", "#2a8ae6"],               # red, neutral grey midpoint, blue
        "surface": "#050608", "panel": "#0d1016", "nav": "#090b0f",
        "heading": "#eef2f7", "ink": "#d4dbe5", "ink2": "#8b95a5", "ink3": "#5f6878",
        "line": "#1a1f29", "rule": "#232936", "grid": "#141821", "axis": "#232936",
        "eyebrow": "#58a6ff", "accent": "#1e88e5", "button": "#1976d2", "button_hover": "#1565c0",
        "neutral": "#6e7681", "good": "#3fb950", "bad": "#f85149",
    },
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],            # validated on #ffffff
        "tiers": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],  # ordinal ramp, darker = riskier
        "seq": ["#dbe9fb", "#0d366b"],
        "div": ["#e34948", "#eef0f2", "#2a78d6"],
        "surface": "#ffffff", "panel": "#f6f8fa", "nav": "#ffffff",
        "heading": "#0d1117", "ink": "#1f2328", "ink2": "#57606a", "ink3": "#6e7781",
        "line": "#d8dee4", "rule": "#d0d7de", "grid": "#eaeef2", "axis": "#d0d7de",
        "eyebrow": "#0969da", "accent": "#1e88e5", "button": "#1976d2", "button_hover": "#1565c0",
        "neutral": "#8c959f", "good": "#1a7f37", "bad": "#cf222e",
    },
}


def mode() -> str:
    """'light' or 'dark'. The URL option set by the header switch wins; then the browser's theme."""
    options = ",".join(st.query_params.get_all("embed_options"))
    if "light_theme" in options:
        return "light"
    if "dark_theme" in options:
        return "dark"
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return "light" if kind == "light" else "dark"


def palette() -> dict:
    return THEMES[mode()]


def inject_css() -> None:
    p = palette()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=JetBrains+Mono:wght@500;700&display=swap');

:root {{ --rs-surface:{p['surface']}; --rs-panel:{p['panel']}; --rs-nav:{p['nav']}; --rs-heading:{p['heading']};
         --rs-ink:{p['ink']}; --rs-ink2:{p['ink2']}; --rs-ink3:{p['ink3']}; --rs-line:{p['line']}; --rs-rule:{p['rule']};
         --rs-eyebrow:{p['eyebrow']}; --rs-accent:{p['accent']}; --rs-button:{p['button']};
         --rs-button-hover:{p['button_hover']}; --rs-good:{p['good']}; --rs-bad:{p['bad']}; }}

/* Streamlit chrome: the portal draws its own header; no heading anchor icons */
header[data-testid="stHeader"], div[data-testid="stHeader"] {{ background:transparent; display:none; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stHeaderActionElements"] {{ display:none !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top:0 !important; padding-bottom:88px !important; max-width:1440px; }}
/* Style-only markdown blocks take no space (otherwise each adds a gap above the navbar) */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style:only-child),
[data-testid="stElementContainer"]:has(style):not(:has(.rs)) {{ display:none; }}

.rs, .rs * {{ font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; }}
.rs-mono {{ font-family:'IBM Plex Mono', ui-monospace, Consolas, monospace !important; }}
.rs svg.ic {{ flex:none; display:inline-block; vertical-align:-0.18em; }}
.rs a {{ text-decoration:none; }}

/* Header: one 64px flex row; the background bleeds to the viewport edges */
.st-key-navbar {{ min-height:64px; padding:0 16px; background:var(--rs-nav); position:sticky; top:0; z-index:1000;
                  box-shadow:0 0 0 100vmax var(--rs-nav), 0 1px 0 100vmax var(--rs-line); clip-path:inset(0 -100vmax -1px);
                  margin-bottom:8px; flex-wrap:wrap; row-gap:4px; }}
.st-key-navbar [data-testid="stMarkdownContainer"], .st-key-navbar [data-testid="stMarkdown"] {{ margin:0 !important; }}
.rs-brand {{ display:flex; align-items:center; gap:10px; white-space:nowrap; }}
.rs-brand svg {{ width:28px; height:28px; flex:none; display:block; }}
.rs-brand .word {{ font-size:20px; font-weight:600; letter-spacing:-0.01em; color:var(--rs-heading); line-height:1; }}
.rs-brand .word b {{ color:var(--rs-accent); font-weight:600; }}
.st-key-navlinks {{ gap:24px !important; flex-wrap:wrap; }}
.st-key-navlinks [data-testid="stPageLink"], .st-key-navlinks [data-testid="stPageLink"] > div, .st-key-navlinks [data-testid="stPageLink"] a,
.st-key-navlinks [data-testid="stElementContainer"], .st-key-navlinks [data-testid="stMarkdownContainer"] {{ margin:0 !important; padding:0 !important; }}
.st-key-navbar .st-key-navlinks [data-testid="stPageLink"] a, .rs-navext {{ box-sizing:border-box; height:64px; padding:0 !important;
                  display:flex; align-items:center; border-radius:0; background:transparent !important;
                  border-top:2px solid transparent; border-bottom:2px solid transparent; }}
.st-key-navbar [data-testid="stPageLink"] a:hover {{ border-bottom-color:var(--rs-rule); }}
.st-key-navbar [data-testid="stPageLink"] p {{ font-size:14px; font-weight:500; color:var(--rs-ink2); white-space:nowrap; margin:0; }}
.rs-navext {{ gap:6px; font-size:14px; font-weight:500; color:var(--rs-ink2) !important; white-space:nowrap;
              text-decoration:none !important; }}
.rs-navext:visited {{ color:var(--rs-ink2) !important; }}
.rs-navext:hover {{ color:var(--rs-ink) !important; border-bottom-color:var(--rs-rule); }}
.st-key-navtools {{ gap:12px !important; }}
.rs-theme {{ display:inline-flex; border:1px solid var(--rs-rule); border-radius:2px; overflow:hidden; height:34px; }}
.rs-theme a {{ display:inline-flex; align-items:center; justify-content:center; width:34px; color:var(--rs-ink2) !important;
               text-decoration:none !important; }}
.rs-theme a + a {{ border-left:1px solid var(--rs-rule); }}
.rs-theme a.on {{ background:var(--rs-panel); color:var(--rs-heading) !important; }}
.rs-theme a:hover {{ color:var(--rs-heading) !important; }}
.st-key-cta [data-testid="stPageLink"] a {{ background:var(--rs-button) !important; border:none !important; border-radius:2px;
                  height:36px; padding:0 16px; display:flex; align-items:center; justify-content:center; }}
.st-key-cta [data-testid="stPageLink"] p {{ color:#ffffff !important; font-size:13px; font-weight:600; white-space:nowrap; }}
.st-key-cta [data-testid="stPageLink"] a:hover {{ background:var(--rs-button-hover) !important; }}

/* Narrow screens: row 1 brand + tools, row 2 links as one scrollable strip */
@media (max-width: 1180px) {{
  .st-key-navbar {{ padding:8px 12px 0 12px; }}
  /* Streamlit wraps each keyed container in a stLayoutWrapper: that wrapper is the flex item */
  [data-testid="stLayoutWrapper"]:has(> .st-key-navtools) {{ order:2; }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-navlinks) {{ order:3; flex:1 0 100%; width:100% !important;
                      min-width:0; overflow:hidden; }}
  .st-key-navlinks {{ width:100% !important; flex-wrap:nowrap !important; overflow-x:auto; scrollbar-width:none; gap:20px !important; }}
  .st-key-navlinks::-webkit-scrollbar {{ display:none; }}
  .st-key-navbar .st-key-navlinks [data-testid="stPageLink"] a, .rs-navext {{ height:42px; }}
}}

/* Hero */
.rs-hero {{ padding:40px 0 0 0; }}
.rs-hero .eyebrow {{ font-size:12px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:var(--rs-eyebrow); }}
.rs-hero h1 {{ font-size:38px; line-height:1.2; font-weight:600; color:var(--rs-heading); margin:12px 0 16px 0; padding:0;
               letter-spacing:-0.015em; overflow-wrap:break-word; }}
.rs-hero .lead {{ font-size:15px; line-height:1.6; color:var(--rs-ink2); max-width:600px; margin:0 0 24px 0; }}
.st-key-hero_ctas {{ gap:12px !important; flex-wrap:wrap; }}
.st-key-hero_ctas [data-testid="stPageLink"] a {{ height:40px; padding:0 18px; border-radius:2px; display:flex;
                  align-items:center; justify-content:center; }}
.st-key-hero_primary [data-testid="stPageLink"] a {{ background:var(--rs-button) !important; border:1px solid var(--rs-button) !important; }}
.st-key-hero_primary [data-testid="stPageLink"] a:hover {{ background:var(--rs-button-hover) !important; }}
.st-key-hero_primary [data-testid="stPageLink"] p {{ color:#ffffff !important; font-size:14px; font-weight:600; }}
.st-key-hero_secondary [data-testid="stPageLink"] a {{ background:transparent !important; border:1px solid var(--rs-rule) !important; }}
.st-key-hero_secondary [data-testid="stPageLink"] a:hover {{ border-color:var(--rs-ink2) !important; }}
.st-key-hero_secondary [data-testid="stPageLink"] p {{ color:var(--rs-ink) !important; font-size:14px; font-weight:500; }}

/* Specification card */
.rs-spec-card {{ background:var(--rs-panel); border:1px solid var(--rs-rule); border-radius:2px; padding:20px; margin-top:40px; }}
.rs-spec-card .title {{ font-size:11px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:var(--rs-ink2);
                        margin-bottom:6px; }}
table.rs-spec, table.rs-spec tr, table.rs-spec td {{ border:none !important; background:transparent !important; }}
table.rs-spec {{ width:100%; border-collapse:collapse; margin:0; table-layout:fixed; }}
table.rs-spec td {{ border-bottom:1px solid var(--rs-line) !important; padding:8px 0 !important; font-size:13px;
                    vertical-align:top; color:var(--rs-ink); overflow-wrap:break-word; }}
table.rs-spec tr:last-child td {{ border-bottom:none !important; }}
table.rs-spec td:first-child {{ color:var(--rs-ink2); width:40%; padding-right:12px !important; }}

/* Stats bar: one connected bar */
.rs-statbar {{ background:var(--rs-panel); border:1px solid var(--rs-rule); border-radius:2px; display:grid;
               grid-template-columns:repeat(4, minmax(0, 1fr)); margin-top:32px; overflow:hidden; }}
.rs-statbar > div {{ padding:20px 24px; border-right:1px solid var(--rs-line); min-width:0; }}
.rs-statbar > div:last-child {{ border-right:none; }}
.rs-statbar .v {{ font-family:'JetBrains Mono', ui-monospace, monospace !important; font-size:30px; font-weight:500;
                  color:var(--rs-heading); line-height:1.15; font-variant-numeric:tabular-nums; }}
.rs-statbar .l {{ font-size:12px; color:var(--rs-ink2); margin-top:4px; line-height:1.4; }}

/* Section scaffolding */
.rs-eyebrow {{ font-size:11.5px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--rs-eyebrow); }}
.rs-h2 {{ font-size:22px; font-weight:600; letter-spacing:-0.01em; color:var(--rs-heading); margin:0 0 4px 0; }}
.rs-section {{ border-top:1px solid var(--rs-line); padding-top:22px; margin-top:26px; }}
.rs-sub {{ font-size:13.5px; color:var(--rs-ink2); margin-bottom:14px; line-height:1.55; }}
.rs-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--rs-ink2); font-weight:600; margin:16px 0 6px 0; }}
.rs-note {{ color:var(--rs-ink2); font-size:12.5px; line-height:1.5; }}

/* Grids with hairline dividers: 1px gap on a line-coloured background */
.rs-grid {{ display:grid; gap:1px; background:var(--rs-line); border:1px solid var(--rs-rule); border-radius:2px; overflow:hidden; }}
.rs-grid > div {{ background:var(--rs-surface); min-width:0; }}
.rs-kpis {{ grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); }}
.rs-kpi {{ padding:14px 16px 12px 16px; }}
.rs-kpi .l {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--rs-ink2); font-weight:600; }}
.rs-kpi .v {{ font-family:'JetBrains Mono', ui-monospace, monospace !important; font-size:24px; font-weight:500; color:var(--rs-heading); font-variant-numeric:tabular-nums; margin-top:4px; line-height:1.25; }}
.rs-kpi .n {{ font-size:12px; color:var(--rs-ink2); margin-top:2px; min-height:16px; }}
.rs-pillars {{ grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); margin-top:28px; }}
.rs-pillar {{ padding:22px 22px 24px 22px; }}
/* Scenario presets (generator) and cohort scorecard (simulator) */
.rs-preset .t {{ display:flex; align-items:center; gap:8px; font-size:14px; font-weight:600; color:var(--rs-heading); }}
.rs-preset .t .ic {{ color:var(--rs-eyebrow); }}
.rs-preset .d {{ font-size:13px; line-height:1.5; color:var(--rs-ink2); margin:6px 0 4px 0; min-height:39px; }}
.rs-preset.on .t {{ color:var(--rs-eyebrow); }}
.rs-scroll {{ overflow-x:auto; }}
.rs-pass {{ color:var(--rs-good); font-weight:600; }}
.rs-fail {{ color:var(--rs-bad); font-weight:600; }}
.rs-muted {{ color:var(--rs-ink3); }}
.rs-scorecard td code {{ white-space:normal; font-size:11.5px; }}
.rs-scorecard td .sub {{ margin-top:3px; font-size:11.5px; color:var(--rs-ink3); white-space:normal; }}
.rs-scorecard td.name .sub {{ max-width:420px; }}
.rs-scorecard td.num .sub {{ white-space:nowrap; }}
.rs-checks td {{ padding:7px 10px; }}
.rs-checks td.id {{ font-family:'IBM Plex Mono', monospace; font-size:12px; white-space:nowrap; color:var(--rs-heading); }}
.rs-verdict > div:first-child .v {{ font-family:inherit !important; font-weight:600; font-size:17px; }}
.rs-checks td.st {{ text-align:right; white-space:nowrap; font-size:12.5px; }}
.rs-verdict {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); border:1px solid var(--rs-line);
               border-left-width:3px; background:var(--rs-panel); margin-bottom:16px; border-radius:2px; }}
.rs-verdict > div {{ padding:12px 14px; border-left:1px solid var(--rs-line); min-width:0; }}
.rs-verdict > div:first-child {{ border-left:none; }}
.rs-verdict .k {{ font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--rs-ink2); }}
.rs-verdict .v {{ font-family:'JetBrains Mono', ui-monospace, monospace !important; font-size:18px; font-weight:500;
                  color:var(--rs-heading); margin-top:4px; }}
.rs-verdict .n {{ font-size:12px; color:var(--rs-ink2); margin-top:2px; }}
.rs-verdict.good {{ border-left-color:var(--rs-good); }} .rs-verdict.good > div:first-child .v {{ color:var(--rs-good); }}
.rs-verdict.warn {{ border-left-color:var(--rs-accent); }} .rs-verdict.warn > div:first-child .v {{ color:var(--rs-eyebrow); }}
.rs-verdict.bad {{ border-left-color:var(--rs-bad); }} .rs-verdict.bad > div:first-child .v {{ color:var(--rs-bad); }}
@media (max-width: 900px) {{ .rs-verdict {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
  .rs-verdict > div:nth-child(3) {{ border-left:none; }} .rs-verdict > div:nth-child(n+3) {{ border-top:1px solid var(--rs-line); }} }}
.rs-cohorts td.num, .rs-cohorts th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.rs-cohorts th.num {{ white-space:normal; min-width:92px; }}
.rs-cohorts .name {{ white-space:nowrap; color:var(--rs-heading); }}
.rs-cohorts td .bad {{ color:var(--rs-bad); }}
.rs-cohorts td .good {{ color:var(--rs-good); }}
.rs-pillar .icon {{ width:36px; height:36px; border:1px solid var(--rs-rule); border-radius:2px; display:flex;
                    align-items:center; justify-content:center; color:var(--rs-accent); }}
.rs-pillar .t {{ font-size:16px; font-weight:600; color:var(--rs-heading); margin:14px 0 8px 0; }}
.rs-pillar .d {{ font-size:14px; line-height:1.6; color:var(--rs-ink2); }}

/* Tables */
table.rs-table {{ width:100%; border-collapse:collapse; font-size:13px; border:1px solid var(--rs-line); }}
table.rs-table th {{ text-align:left; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                     color:var(--rs-ink2); background:var(--rs-panel); padding:9px 12px; border-bottom:1px solid var(--rs-line); }}
table.rs-table td {{ padding:9px 12px; border-bottom:1px solid var(--rs-line); color:var(--rs-ink); vertical-align:top;
                     overflow-wrap:anywhere; }}
table.rs-table td:first-child {{ white-space:nowrap; overflow-wrap:normal; }}
table.rs-table td code, .rs code {{ font-family:'IBM Plex Mono', monospace; font-size:12px; color:var(--rs-ink);
                     background:var(--rs-panel); border:1px solid var(--rs-line); padding:0 4px; border-radius:2px; }}

/* Architecture flow */
.rs-flow {{ grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); }}
.rs-step {{ padding:16px 16px 18px 16px; }}
.rs-step .top {{ display:flex; align-items:center; justify-content:space-between; color:var(--rs-accent); }}
.rs-step .n {{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; color:var(--rs-ink3); }}
.rs-step .t {{ font-size:14.5px; font-weight:600; color:var(--rs-heading); margin:10px 0 6px 0; }}
.rs-step .d {{ font-size:12.5px; line-height:1.5; color:var(--rs-ink2); }}
.rs-step .m {{ font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--rs-ink3); margin-top:10px; overflow-wrap:anywhere; }}

/* Simulator */
dl.rs-facts {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 18px; font-size:13.5px; margin:0; }}
dl.rs-facts dt {{ color:var(--rs-ink2); }}
dl.rs-facts dd {{ margin:0; font-weight:500; color:var(--rs-ink); font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
.rs-action {{ display:flex; gap:8px; align-items:flex-start; padding:6px 0; font-size:13.5px; color:var(--rs-ink);
              border-bottom:1px solid var(--rs-line); }}
.rs-action:last-child {{ border-bottom:none; }}
.rs-action .ic {{ color:var(--rs-bad); margin-top:2px; }}
.rs-action.ok .ic {{ color:var(--rs-good); }}
.rs-expect {{ border:1px solid var(--rs-line); background:var(--rs-panel); padding:12px 14px; font-size:13.5px; border-radius:2px;
              margin-top:12px; color:var(--rs-ink); line-height:1.55; }}

/* Footer: two short rows. The bottom padding of the page keeps it clear of Streamlit Cloud's
   floating "Manage app" button. */
.rs-footer {{ border-top:1px solid var(--rs-line); margin-top:44px; padding-top:18px; font-size:13px; color:var(--rs-ink2); }}
.rs-footer .row {{ display:flex; align-items:flex-start; gap:12px 32px; flex-wrap:wrap; }}
.rs-footer .why {{ flex:1 1 520px; line-height:1.6; margin:0; max-width:880px; }}
.rs-footer .why a {{ color:var(--rs-eyebrow) !important; white-space:nowrap; }}
.rs-footer .why a:hover {{ text-decoration:underline !important; }}
.rs-footer .why .ic {{ margin-right:4px; }}
.rs-footer .legal {{ margin-top:14px; padding-top:12px; border-top:1px solid var(--rs-line); font-size:12px;
                     color:var(--rs-ink3); line-height:1.5; padding-right:180px; }}
@media (max-width: 640px) {{ .rs-footer .legal {{ padding-right:0; }} }}

/* Widgets */
[data-testid="stForm"], [data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:2px !important; }}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{ border-radius:2px; }}

@media (max-width: 480px) {{ .st-key-cta {{ display:none; }} }}
@media (max-width: 900px) {{
  .rs-statbar {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
  .rs-statbar > div:nth-child(2n) {{ border-right:none; }}
  .rs-statbar > div:nth-child(-n+2) {{ border-bottom:1px solid var(--rs-line); }}
  .rs-hero h1 {{ font-size:30px; }}
}}
@media (max-width: 520px) {{
  .rs-statbar {{ grid-template-columns:1fr; }}
  .rs-statbar > div {{ border-right:none; border-bottom:1px solid var(--rs-line); }}
  .rs-statbar > div:last-child {{ border-bottom:none; }}
  .rs-statbar .v {{ font-size:28px; }}
  .rs-hero h1 {{ font-size:26px; }}
  .rs-spec-card {{ margin-top:8px; }}
  .rs-pillars {{ grid-template-columns:1fr; }}
}}
</style>
""", unsafe_allow_html=True)
