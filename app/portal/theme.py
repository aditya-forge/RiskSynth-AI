"""Colours, type and global CSS for the portal.

Chart palettes were checked for colour-blind separation and contrast against each
background (dark #0e1117, light #ffffff). Data series never use the UI accent or the
neutral grey; those are reserved for controls and non-data marks.
"""
from __future__ import annotations

import streamlit as st

TIER_ORDER = ["Conservative", "Moderate", "Aggressive", "Ultra-Aggressive"]

THEMES = {
    "dark": {
        "series": ["#2a8ae6", "#e0701e", "#16a085"],            # blue, amber, emerald (validated set)
        "tiers": ["#184f95", "#2a78d6", "#5598e7", "#9ec5f4"],  # ordinal ramp, brighter = riskier
        "seq": ["#104281", "#b7d3f6"],
        "div": ["#e66767", "#2d333b", "#2a8ae6"],               # red, neutral grey midpoint, blue
        "ink": "#e6edf3", "ink2": "#8b949e", "ink3": "#6e7681",
        "grid": "#21262d", "axis": "#30363d", "line": "#21262d", "rule": "#30363d",
        "neutral": "#757575", "surface": "#0e1117", "panel": "#161b22", "accent": "#1e88e5",
        "good": "#3fb950", "bad": "#f85149",
    },
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "tiers": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "seq": ["#cde2fb", "#0d366b"],
        "div": ["#e34948", "#f0efec", "#2a78d6"],
        "ink": "#161616", "ink2": "#5c5c5c", "ink3": "#8a8a8a",
        "grid": "#ececec", "axis": "#cfcfcf", "line": "#e4e4e4", "rule": "#d6d6d6",
        "neutral": "#9a9a9a", "surface": "#ffffff", "panel": "#f6f7f9", "accent": "#1e88e5",
        "good": "#0b6b0b", "bad": "#b42318",
    },
}


def palette() -> dict:
    """Palette for the viewer's theme; dark is the configured base."""
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return THEMES["light" if kind == "light" else "dark"]


def inject_css() -> None:
    p = palette()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=JetBrains+Mono:wght@500;700&display=swap');

:root {{ --rs-ink:{p['ink']}; --rs-ink2:{p['ink2']}; --rs-line:{p['line']}; --rs-rule:{p['rule']};
         --rs-panel:{p['panel']}; --rs-surface:{p['surface']}; --rs-accent:{p['accent']}; }}

/* Streamlit chrome: the portal draws its own header */
header[data-testid="stHeader"], div[data-testid="stHeader"] {{ background:transparent; display:none; }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top:0 !important; padding-bottom:2rem; max-width:1360px; }}

.rs, .rs * {{ font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; }}
.rs-mono {{ font-family: 'IBM Plex Mono', ui-monospace, Consolas, monospace !important; }}

/* Header: one 64px flex row; the background bleeds to the viewport edges */
.st-key-navbar {{ min-height:64px; padding:0 16px; background:#0b0e14; position:sticky; top:0; z-index:1000;
                  box-shadow:0 0 0 100vmax #0b0e14, 0 1px 0 100vmax #21262d; clip-path:inset(0 -100vmax -1px);
                  margin-bottom:8px; flex-wrap:wrap; row-gap:4px; }}
.rs-brand {{ display:flex; align-items:center; gap:10px; white-space:nowrap; }}
.rs-brand svg {{ width:28px; height:28px; flex:none; display:block; }}
.rs-brand .word {{ font-size:20px; font-weight:700; letter-spacing:-0.01em; color:#ffffff; line-height:1; }}
.rs-brand .word b {{ color:#1e88e5; font-weight:700; }}
.st-key-navlinks {{ gap:24px !important; flex-wrap:wrap; }}
.st-key-navbar [data-testid="stPageLink"] a {{ border-radius:0; padding:20px 0 18px 0; background:transparent !important;
                  border-bottom:2px solid transparent; }}
.st-key-navbar [data-testid="stPageLink"] a:hover {{ border-bottom-color:#30363d; }}
.st-key-navbar [data-testid="stPageLink"] p {{ font-size:14px; font-weight:500; color:#8b949e; white-space:nowrap; margin:0; }}
.st-key-cta [data-testid="stPageLink"] a {{ background:#1e88e5 !important; border:none !important; border-radius:4px;
                  height:36px; line-height:36px; padding:0 16px; display:flex; align-items:center; justify-content:center; }}
.st-key-cta [data-testid="stPageLink"] p {{ color:#ffffff !important; font-size:13px; font-weight:600; white-space:nowrap; }}
.st-key-cta [data-testid="stPageLink"] a:hover {{ background:#1976d2 !important; border-bottom:none; }}

/* Style-only markdown blocks take no space (otherwise each adds a flex gap above the navbar) */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style:only-child),
[data-testid="stElementContainer"]:has(style):not(:has(.rs)) {{ display:none; }}
.st-key-navbar [data-testid="stMarkdownContainer"], .st-key-navbar [data-testid="stMarkdown"] {{ margin:0 !important; }}
.st-key-navbar [data-testid="stMarkdownContainer"] > div {{ margin:0; }}

/* Narrow screens: row 1 brand + CTA, row 2 links as one scrollable strip */
@media (max-width: 1100px) {{
  .st-key-navbar {{ padding:8px 12px 0 12px; }}
  /* Streamlit wraps each keyed container in a stLayoutWrapper: that wrapper is the flex item */
  [data-testid="stLayoutWrapper"]:has(> .st-key-cta) {{ order:2; }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-navlinks) {{ order:3; flex:1 0 100%; width:100% !important;
                      min-width:0; overflow:hidden; }}
  .st-key-navlinks {{ width:100% !important; flex-wrap:nowrap !important; overflow-x:auto;
                      scrollbar-width:none; gap:20px !important; }}
  .st-key-navlinks::-webkit-scrollbar {{ display:none; }}
  .st-key-navbar [data-testid="stPageLink"] a {{ padding:10px 0 8px 0; }}
}}

/* Hero */
.rs-hero {{ padding:40px 0 0 0; }}
.rs-hero .eyebrow {{ font-size:12px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#58a6ff; }}
.rs-hero h1 {{ font-size:38px; line-height:1.2; font-weight:700; color:#f0f6fc; margin:12px 0 16px 0; padding:0;
               letter-spacing:-0.015em; }}
.rs-hero .lead {{ font-size:15px; line-height:1.6; color:#8b949e; max-width:580px; margin:0 0 24px 0; }}
.st-key-hero_ctas {{ gap:12px !important; }}
.st-key-hero_ctas [data-testid="stPageLink"] a {{ height:40px; padding:0 18px; border-radius:4px; display:flex;
                  align-items:center; justify-content:center; }}
.st-key-hero_primary [data-testid="stPageLink"] a {{ background:#1e88e5 !important; border:1px solid #1e88e5 !important; }}
.st-key-hero_primary [data-testid="stPageLink"] p {{ color:#ffffff !important; font-size:14px; font-weight:600; }}
.st-key-hero_secondary [data-testid="stPageLink"] a {{ background:transparent !important; border:1px solid #30363d !important; }}
.st-key-hero_secondary [data-testid="stPageLink"] a:hover {{ border-color:#8b949e !important; }}
.st-key-hero_secondary [data-testid="stPageLink"] p {{ color:#e6edf3 !important; font-size:14px; font-weight:500; }}

/* Specification card */
.rs-spec-card {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:20px; margin-top:40px; }}
.rs-spec-card .title {{ font-size:11px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#8b949e;
                        margin-bottom:6px; }}
table.rs-spec, table.rs-spec tr, table.rs-spec td {{ border:none !important; background:transparent !important; }}
table.rs-spec {{ width:100%; border-collapse:collapse; margin:0; }}
table.rs-spec td {{ border-bottom:1px solid #21262d !important; padding:8px 0 !important; font-size:13px;
                    vertical-align:top; color:#c9d1d9; }}
table.rs-spec tr:last-child td {{ border-bottom:none !important; }}
table.rs-spec td:first-child {{ color:#8b949e; width:40%; padding-right:12px !important; }}

/* Stats ribbon: one connected bar */
.rs-statbar {{ background:#161b22; border:1px solid #30363d; border-radius:6px; display:grid;
               grid-template-columns:repeat(4, 1fr); margin-top:32px; overflow:hidden; }}
.rs-statbar > div {{ padding:20px 24px; border-right:1px solid #21262d; }}
.rs-statbar > div:last-child {{ border-right:none; }}
.rs-statbar .v {{ font-family:'JetBrains Mono', ui-monospace, monospace !important; font-size:32px; font-weight:700;
                  color:#f0f6fc; line-height:1.15; font-variant-numeric:tabular-nums; }}
.rs-statbar .l {{ font-size:12px; color:#8b949e; margin-top:4px; line-height:1.4; }}
@media (max-width: 900px) {{
  .rs-statbar {{ grid-template-columns:repeat(2, 1fr); }}
  .rs-statbar > div:nth-child(2n) {{ border-right:none; }}
  .rs-statbar > div:nth-child(-n+2) {{ border-bottom:1px solid #21262d; }}
  .rs-hero h1 {{ font-size:30px; }}
}}
@media (max-width: 520px) {{
  .rs-statbar {{ grid-template-columns:1fr; }}
  .rs-statbar > div {{ border-right:none; border-bottom:1px solid #21262d; }}
  .rs-statbar > div:last-child {{ border-bottom:none; }}
  .rs-hero h1 {{ font-size:26px; }}
}}

/* Section scaffolding */
.rs-eyebrow {{ font-size:11.5px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--rs-accent); }}
.rs-h1 {{ font-size:40px; line-height:1.12; font-weight:600; letter-spacing:-0.02em; color:var(--rs-ink); margin:10px 0 14px 0; }}
.rs-h2 {{ font-size:22px; font-weight:600; letter-spacing:-0.01em; color:var(--rs-ink); margin:0 0 4px 0; }}
.rs-lead {{ font-size:16px; line-height:1.6; color:var(--rs-ink2); max-width:820px; }}
.rs-section {{ border-top:1px solid var(--rs-line); padding-top:22px; margin-top:26px; }}
.rs-sub {{ font-size:13.5px; color:var(--rs-ink2); margin-bottom:14px; }}
.rs-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--rs-ink2); font-weight:600; margin:16px 0 6px 0; }}
.rs-note {{ color:var(--rs-ink2); font-size:12.5px; }}

/* Grids with hairline dividers: 1px gap on a line-coloured background */
.rs-grid {{ display:grid; gap:1px; background:var(--rs-line); border:1px solid var(--rs-rule); border-radius:6px; overflow:hidden; }}
.rs-grid > div {{ background:var(--rs-surface); }}
.rs-kpis {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
.rs-kpi {{ padding:14px 16px 12px 16px; }}
.rs-kpi .l {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--rs-ink2); font-weight:600; }}
.rs-kpi .v {{ font-size:26px; font-weight:500; color:var(--rs-ink); font-variant-numeric:tabular-nums; margin-top:4px; line-height:1.25; }}
.rs-kpi .n {{ font-size:12px; color:var(--rs-ink2); margin-top:2px; min-height:16px; }}
.rs-stats {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
.rs-stat {{ padding:18px 20px; }}
.rs-stat .v {{ font-size:30px; font-weight:500; color:var(--rs-ink); font-variant-numeric:tabular-nums; }}
.rs-stat .l {{ font-size:13px; color:var(--rs-ink2); margin-top:2px; }}
.rs-pillars {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
.rs-pillar {{ padding:22px 22px 24px 22px; }}
.rs-pillar .k {{ font-size:12px; color:var(--rs-accent); letter-spacing:.06em; }}
.rs-pillar .t {{ font-size:17px; font-weight:600; color:var(--rs-ink); margin:8px 0 8px 0; }}
.rs-pillar .d {{ font-size:14px; line-height:1.55; color:var(--rs-ink2); }}

/* Spec sheet / tables */
table.rs-spec {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
table.rs-spec td {{ border-bottom:1px solid var(--rs-line); padding:9px 0; vertical-align:top; color:var(--rs-ink); }}
table.rs-spec td:first-child {{ color:var(--rs-ink2); width:42%; padding-right:12px; }}
table.rs-table {{ width:100%; border-collapse:collapse; font-size:13px; border:1px solid var(--rs-line); }}
table.rs-table th {{ text-align:left; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                     color:var(--rs-ink2); background:var(--rs-panel); padding:9px 12px; border-bottom:1px solid var(--rs-line); }}
table.rs-table td {{ padding:9px 12px; border-bottom:1px solid var(--rs-line); color:var(--rs-ink); vertical-align:top; }}
table.rs-table td code, .rs code {{ font-family:'IBM Plex Mono', monospace; font-size:12px; color:var(--rs-ink);
                     background:var(--rs-panel); border:1px solid var(--rs-line); padding:0 4px; border-radius:0; }}

/* Architecture flow */
.rs-flow {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.rs-step {{ padding:16px 16px 18px 16px; position:relative; }}
.rs-step .n {{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; color:var(--rs-accent); }}
.rs-step .t {{ font-size:14.5px; font-weight:600; color:var(--rs-ink); margin:6px 0 6px 0; }}
.rs-step .d {{ font-size:12.5px; line-height:1.5; color:var(--rs-ink2); }}
.rs-step .m {{ font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--rs-ink3); margin-top:10px; }}

/* Simulator */
dl.rs-facts {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 18px; font-size:13.5px; margin:0; }}
dl.rs-facts dt {{ color:var(--rs-ink2); }}
dl.rs-facts dd {{ margin:0; font-weight:500; color:var(--rs-ink); font-variant-numeric:tabular-nums; }}
.rs-action {{ border-left:2px solid {p['bad']}; padding:4px 10px; margin:6px 0; font-size:13.5px; color:var(--rs-ink); }}
.rs-action.ok {{ border-left-color:{p['good']}; }}
.rs-expect {{ border:1px solid var(--rs-line); background:var(--rs-panel); padding:12px 14px; font-size:13.5px;
              margin-top:12px; color:var(--rs-ink); line-height:1.55; }}

/* Footer */
.rs-footer {{ border-top:1px solid var(--rs-line); margin-top:40px; padding-top:16px; display:flex; flex-wrap:wrap;
              gap:10px 40px; justify-content:space-between; font-size:12.5px; color:var(--rs-ink2); }}
.rs-footer a {{ color:var(--rs-ink2); text-decoration:none; border-bottom:1px solid var(--rs-rule); }}

/* Widgets: sharp edges everywhere */
[data-testid="stForm"], [data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:0 !important; }}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{ border-radius:0; }}
</style>
""", unsafe_allow_html=True)
