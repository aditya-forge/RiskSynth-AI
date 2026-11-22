# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Shared fixtures for the RiskSynth AI test suite.

Expensive pipeline runs are session-scoped and reused across test modules. Nothing here
writes to data/processed/ or data/synthetic/: generator outputs go to pytest tmp dirs.
Tests that need the versioned INE parquet files skip cleanly when they are absent.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]      # src/ is on the path via pyproject [tool.pytest]

PROCESSED = ROOT / "data" / "processed"
SAMPLE_ROWS = 20_000          # 0.5% EC08 quota -> 100 rows, so every profile clears 50 rows


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs (generator CLI, Streamlit app)")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_cfg() -> dict:
    from risksynth.engine import wealth_profile
    return wealth_profile.load_config()


@pytest.fixture(scope="session")
def model_cfg() -> dict:
    cfg: dict = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    return cfg


@pytest.fixture(scope="session")
def require_processed():
    missing = [f for f in ["combined_base.parquet", "epf_processed.parquet"] if not (PROCESSED / f).exists()]
    if missing:
        pytest.skip(f"processed INE parquet files missing: {missing}")


@pytest.fixture(scope="session")
def combined_sample(require_processed) -> pd.DataFrame:
    base = pd.read_parquet(PROCESSED / "combined_base.parquet")
    return base.sample(n=SAMPLE_ROWS, random_state=7).reset_index(drop=True)


@pytest.fixture(scope="session")
def stage2_only(combined_sample, data_cfg):
    """Stage 2 without the edge-case cohort: target distributions must be exact here."""
    from risksynth.engine.stage2 import run_stage2
    return run_stage2(combined_sample, seed=1042, cfg=data_cfg, edge_cases=False)


@pytest.fixture(scope="session")
def stage3(combined_sample, data_cfg):
    """Stages 2 + 3 (edge-case cohort at the configured 8%)."""
    from risksynth.engine.stage2 import run_stage2
    return run_stage2(combined_sample, seed=1042, cfg=data_cfg)


@pytest.fixture(scope="session")
def generator_run(require_processed, tmp_path_factory, data_cfg):
    """Rule-based generator end to end (n=600, in-process), outputs redirected to a tmp dir."""
    from risksynth.generators import generate_synthetic as gs
    out_dir = tmp_path_factory.mktemp("generator_out")
    original = gs.OUTPUT_DIR
    gs.OUTPUT_DIR = out_dir
    try:
        base = pd.read_parquet(PROCESSED / "combined_base.parquet")
        df = gs.generate_synthetic_data(600, base, n_jobs=1, batch_size=600, seed=21, wealth_cfg=copy.deepcopy(data_cfg))
    finally:
        gs.OUTPUT_DIR = original
    report = next(out_dir.glob("*_wealth_report.json"))
    return df, report, out_dir


def frame(**columns) -> pd.DataFrame:
    """Small helper used by tests to build typed frames inline."""
    return pd.DataFrame(columns)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
