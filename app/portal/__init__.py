# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Shared building blocks for the RiskSynth AI Streamlit portal."""

import sys
from pathlib import Path

try:
    import risksynth  # noqa: F401  (installed with: pip install -e .)
except ImportError:  # running from a plain checkout, e.g. Streamlit Community Cloud
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GITHUB_URL = "https://github.com/aditya-forge/RiskSynth-AI"

# Page objects registered by streamlit_app.py on every run, for links between pages
PAGES: dict = {}
