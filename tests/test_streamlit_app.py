"""Streamlit portal smoke tests (headless AppTest): header, every page, first-visit demo flow."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["views/architecture.py", "views/generator.py", "views/simulator.py", "views/compliance.py"]


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


@pytest.mark.slow
def test_overview_renders_with_header(require_processed):
    at = _app()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "<svg" in html and "RiskSynth" in html, "brand logo and wordmark in the header"
    assert "Synthetic client risk profiles for robo-advisory and wealth management" in html
    assert "rs-statbar" in html and "rs-spec-card" in html


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(require_processed, page):
    at = _app()
    at.switch_page(page)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.mark.slow
def test_generator_form_validates_risk_mix(require_processed):
    at = _app()
    at.switch_page("views/generator.py")
    at.run()
    [w for w in at.number_input if w.label == "Conservative"][0].set_value(90.0)
    [b for b in at.button if b.label == "Generate profiles"][0].click()
    at.run()
    assert not at.exception
    assert any("must add up to 100%" in e.value for e in at.error)


@pytest.mark.slow
def test_generator_presets_fill_the_form(require_processed):
    at = _app()
    at.switch_page("views/generator.py")
    at.run()
    [b for b in at.button if b.label == "Use this scenario"][0].click()  # first unselected preset: edge-case stress
    at.run()
    assert not at.exception
    assert at.session_state["gen_edge"] == 15.0
    assert [w for w in at.slider if w.label == "Edge-case share"][0].value == 15.0


@pytest.mark.slow
def test_generated_dataset_feeds_simulator_and_compliance(require_processed):
    at = _app()
    at.switch_page("views/generator.py")
    at.run()
    [w for w in at.number_input if w.label == "Sample size"][0].set_value(1000)
    [b for b in at.button if b.label == "Generate profiles"][0].click()
    at.run()
    run_dir = Path(at.session_state["last_run"])
    try:
        assert not at.exception and (run_dir / "profiles.parquet").exists()
        assert [b.label for b in at.get("download_button")] == ["Download CSV", "Download Parquet",
                                                                "Download JSON metadata", "Download run contract"]
        from risksynth import contract
        c = contract.load(run_dir / "run_contract.json")
        assert c["parameters"]["rows"] == 1000 and c["assumptions"]
        assert contract.verify(c, pd.read_parquet(run_dir / "profiles.parquet")) == []
        at.switch_page("views/simulator.py")
        at.run()
        html = " ".join(m.value for m in at.markdown)
        assert "rs-cohorts" in html and "Edge cases EC01-EC09" in html, "cohort scorecard"
        for label in ["Edge cases EC01-EC09", "Capacity-constrained", "Baseline profiles"]:
            [b for b in at.button if b.label == label][0].click()
            at.run()
            assert not at.exception
            assert at.session_state["sim_group"] == label
        # counterfactual: push the stated tolerance to 10 and the accepted drawdown to 1%
        [w for w in at.slider if w.label.startswith("Stated risk tolerance")][0].set_value(10)
        [w for w in at.slider if w.label.startswith("Accepted drawdown")][0].set_value(1)
        at.run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "rs-verdict" in html and "2 inputs changed" in html and "rs-scorecard" in html
        at.switch_page("views/compliance.py")
        at.run()
        assert not at.exception
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
