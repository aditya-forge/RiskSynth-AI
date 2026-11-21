"""Streamlit UI smoke test: every tab renders without exceptions (headless AppTest)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_app_renders_all_tabs(require_processed):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert [t.label for t in at.tabs] == ["Overview", "Generator", "Advisory simulator", "Fidelity & compliance"]


@pytest.mark.slow
def test_generator_form_validates_risk_mix(require_processed):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    [w for w in at.number_input if w.label == "Conservative"][0].set_value(90.0)
    [b for b in at.button if b.label == "Generate profiles"][0].click()
    at.run()
    assert not at.exception
    assert any("must add up to 100%" in e.value for e in at.error)
