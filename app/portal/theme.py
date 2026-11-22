# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
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

from pathlib import Path

import streamlit as st

TIER_ORDER = ["Conservative", "Moderate", "Aggressive", "Ultra-Aggressive"]

THEMES = {
    "dark": {
        "series": [
            "#2a8ae6",
            "#e0701e",
            "#16a085",
        ],  # blue, amber, emerald (validated on #050608 and #0d1016)
        "tiers": ["#184f95", "#2a78d6", "#5598e7", "#9ec5f4"],  # ordinal ramp, brighter = riskier
        "seq": ["#0f3a70", "#b7d3f6"],
        "div": ["#e66767", "#1f242d", "#2a8ae6"],  # red, neutral grey midpoint, blue
        "surface": "#050608",
        "panel": "#0d1016",
        "nav": "#090b0f",
        "heading": "#eef2f7",
        "ink": "#d4dbe5",
        "ink2": "#8b95a5",
        "ink3": "#5f6878",
        "line": "#1a1f29",
        "rule": "#232936",
        "grid": "#141821",
        "axis": "#232936",
        "eyebrow": "#58a6ff",
        "accent": "#1e88e5",
        "button": "#1976d2",
        "button_hover": "#1565c0",
        "neutral": "#6e7681",
        "good": "#3fb950",
        "bad": "#f85149",
    },
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],  # validated on #ffffff
        "tiers": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],  # ordinal ramp, darker = riskier
        "seq": ["#dbe9fb", "#0d366b"],
        "div": ["#e34948", "#eef0f2", "#2a78d6"],
        "surface": "#ffffff",
        "panel": "#f6f8fa",
        "nav": "#ffffff",
        "heading": "#0d1117",
        "ink": "#1f2328",
        "ink2": "#57606a",
        "ink3": "#6e7781",
        "line": "#d8dee4",
        "rule": "#d0d7de",
        "grid": "#eaeef2",
        "axis": "#d0d7de",
        "eyebrow": "#0969da",
        "accent": "#1e88e5",
        "button": "#1976d2",
        "button_hover": "#1565c0",
        "neutral": "#8c959f",
        "good": "#1a7f37",
        "bad": "#cf222e",
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


# Theme colour -> CSS variable used throughout styles.css
CSS_TOKENS = {
    "surface": "--rs-surface",
    "panel": "--rs-panel",
    "nav": "--rs-nav",
    "heading": "--rs-heading",
    "ink": "--rs-ink",
    "ink2": "--rs-ink2",
    "ink3": "--rs-ink3",
    "line": "--rs-line",
    "rule": "--rs-rule",
    "eyebrow": "--rs-eyebrow",
    "accent": "--rs-accent",
    "button": "--rs-button",
    "button_hover": "--rs-button-hover",
    "good": "--rs-good",
    "bad": "--rs-bad",
}
STYLESHEET = Path(__file__).with_name("styles.css")


def inject_css() -> None:
    """Write the active theme's colours as CSS variables, followed by the shared stylesheet."""
    p = palette()
    tokens = "\n".join(f"  {var}: {p[key]};" for key, var in CSS_TOKENS.items())
    css = STYLESHEET.read_text(encoding="utf-8")
    # @import is only honoured at the top of a stylesheet, so it goes before the :root block
    imports = "\n".join(line for line in css.splitlines() if line.startswith("@import"))
    rules = "\n".join(line for line in css.splitlines() if not line.startswith("@import"))
    st.markdown(
        f"<style>\n{imports}\n\n:root {{\n{tokens}\n}}\n{rules}</style>",
        unsafe_allow_html=True,
    )
