"""
RiskSynth AI product portal (Streamlit).

Pages (app/views/):
  overview       product overview, live population KPIs and distributions
  architecture   pipeline, output schema, rules and edge-case catalogue
  generator      live profile generation and exports
  simulator      robo-advisory allocation stress tests
  compliance     statistical fidelity, rule audit and HTML report

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

import portal  # noqa: E402,F401  (puts src/ on the path when the package is not installed)
from portal import PAGES  # noqa: E402
from portal.theme import inject_css  # noqa: E402
from portal.ui import footer, header  # noqa: E402

st.set_page_config(page_title="RiskSynth AI", page_icon=str(APP_DIR / "assets" / "favicon.png"), layout="wide")
inject_css()

PAGES.update({
    "overview": st.Page("views/overview.py", title="Product Overview", url_path="overview", default=True),
    "architecture": st.Page("views/architecture.py", title="Solution Architecture", url_path="architecture"),
    "generator": st.Page("views/generator.py", title="Live Generator", url_path="generator"),
    "simulator": st.Page("views/simulator.py", title="Advisory Simulator", url_path="simulator"),
    "compliance": st.Page("views/compliance.py", title="Compliance & Fidelity", url_path="compliance"),
})
current = st.navigation(list(PAGES.values()), position="hidden")
header(PAGES, current)
current.run()
footer()
