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
        "grid": "#21262d", "axis": "#30363d", "line": "#23272e", "rule": "#333842",
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{ --rs-ink:{p['ink']}; --rs-ink2:{p['ink2']}; --rs-line:{p['line']}; --rs-rule:{p['rule']};
         --rs-panel:{p['panel']}; --rs-surface:{p['surface']}; --rs-accent:{p['accent']}; }}

/* Streamlit chrome: the portal draws its own header */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none; }}
.block-container {{ padding-top: 0.75rem; padding-bottom: 2rem; max-width: 1360px; }}

.rs, .rs * {{ font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; }}
.rs-mono {{ font-family: 'IBM Plex Mono', ui-monospace, Consolas, monospace !important; }}

/* Header */
.st-key-navbar {{ border-bottom: 1px solid var(--rs-line); padding: 6px 0 8px 0; margin-bottom: 6px; }}
.rs-brand {{ display:flex; align-items:center; gap:10px; text-decoration:none; }}
.rs-brand .word {{ font-size:18px; font-weight:600; letter-spacing:-0.01em; color:var(--rs-ink); }}
.rs-brand .word b {{ color:var(--rs-accent); font-weight:600; }}
.rs-brand .tag {{ font-size:11px; color:var(--rs-ink2); border-left:1px solid var(--rs-rule); padding-left:10px;
                  letter-spacing:.02em; }}
.st-key-navbar [data-testid="stPageLink"] a {{ border-radius:0; padding:6px 4px; background:transparent;
                  border-bottom:2px solid transparent; }}
.st-key-navbar [data-testid="stPageLink"] a:hover {{ background:transparent; border-bottom-color:var(--rs-rule); }}
.st-key-navbar [data-testid="stPageLink"] p {{ font-size:13.5px; font-weight:500; color:var(--rs-ink2); white-space:nowrap; }}
.st-key-cta [data-testid="stPageLink"] a {{ background:var(--rs-accent) !important; border:1px solid var(--rs-accent) !important;
                  border-radius:0; padding:7px 14px; justify-content:center; }}
.st-key-cta [data-testid="stPageLink"] p {{ color:#ffffff !important; font-weight:600; }}
.st-key-cta [data-testid="stPageLink"] a:hover {{ filter:brightness(1.1); }}

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
.rs-grid {{ display:grid; gap:1px; background:var(--rs-line); border:1px solid var(--rs-line); }}
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
