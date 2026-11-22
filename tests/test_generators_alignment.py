# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Generator alignment with EPF data, stage-1 samplers, CSV reading and console encoding."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risksynth.generators import client_generator as cg
from risksynth.generators import generate_synthetic as gs
from risksynth.engine import wealth_profile as wp
from risksynth.io import CSV_NA_VALUES, read_profiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gen(generator_run):
    return generator_run[0]


@pytest.fixture(scope="module")
def gen_report(generator_run):
    return json.loads(generator_run[1].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Mortgage and debt alignment with EPF
# --------------------------------------------------------------------------- #
def test_mortgage_presence_follows_epf_debt(gen):
    has_epf = gen["debt_service_annual_eur"].notna()
    assert has_epf.all(), "every generator row carries a resolved household block"
    assert (gen["tiene_hipoteca"] == (gen["debt_service_annual_eur"] > 0).astype(int)).all()


def test_loan_book_matches_wealth_debt_service(gen, gen_report):
    expected = gen["debt_service_annual_eur"] + gen["consumer_debt_service_annual_eur"]
    assert np.allclose(12 * gen["cuota_mensual_total"], expected, atol=12)
    assert gen_report["credit_reconciliation"]["debt_mismatch_rows"] == 0
    assert gen_report["credit_reconciliation"]["mortgage_matches_epf_pct"] == 100.0


def test_payload_follows_epf_income_and_spending(gen):
    assert gen["ingresos_anuales"].corr(gen["net_income_annual_eur"]) > 0.99
    assert gen["gastos_anuales"].corr(gen["household_expenses_annual_eur"]) > 0.99
    assert (
        gen["tasa_ahorro"] < 0
    ).mean() > 0.2, "EPF-aligned spending yields negative saving for many households"


@pytest.mark.parametrize("mortgage_monthly, expect", [(0.0, False), (650.0, True)])
def test_gen_accounts_mortgage_modes(mortgage_monthly, expect):
    random.seed(1)
    accounts = gs.gen_accounts(2500, n_cuentas=1, gasto_mensual=2000, mortgage_monthly=mortgage_monthly)
    loans = [l for l in accounts[0]["loans"] if l["type"] == "mortgage"]
    assert bool(loans) is expect
    if expect:
        assert loans[0]["monthly_payment"] == 650.0


def test_gen_accounts_legacy_random_mortgage_still_available():
    random.seed(0)
    shares = [
        any(l["type"] == "mortgage" for l in gs.gen_accounts(2500, n_cuentas=1)[0]["loans"])
        for _ in range(400)
    ]
    assert 0.2 < np.mean(shares) < 0.4, "without EPF data the legacy 30% rule applies"


# --------------------------------------------------------------------------- #
# Balance-derived credit fields after edge-case injection
# --------------------------------------------------------------------------- #
def test_colchon_meses_and_sin_colchon_are_consistent(gen):
    monthly = (gen["gastos_anuales"] / 12).clip(lower=1)
    assert np.allclose(gen["colchon_meses"], (gen["balance_cuentas"] / monthly).round(2), atol=0.011)
    assert (gen["sin_colchon"] == (gen["balance_cuentas"] < monthly).astype(int)).all()


def test_liquid_savings_reconciled_with_accounts(gen, gen_report):
    assert np.allclose(gen["liquid_savings_eur"], gen["saldo_total_cuentas"], atol=0.5)
    rebalanced = gen["credit_fields_reconciled"]
    assert rebalanced.sum() == gen_report["credit_reconciliation"]["rows_rebalanced"]
    mutated = gen["edge_case_mutated_fields"].fillna("").str.contains("liquid_savings_eur")
    assert (rebalanced == mutated).all(), "exactly the rows whose savings stage 3 lowered are rebalanced"
    assert gen["default"].isin([0, 1]).all() and gen["prob_default"].between(0, 1).all()


def test_generator_rows_pass_all_rules(gen_report):
    assert sum(gen_report["violations_enforced"].values()) == 0
    assert all(p["assertion_pass_pct"] == 100.0 for p in gen_report["edge_cases"]["profiles"].values())


# --------------------------------------------------------------------------- #
# Stage-1 samplers
# --------------------------------------------------------------------------- #
def test_bootstrap_sampler_end_to_end(require_processed, tmp_path):
    g = cg.ClientGenerator(n_samples=1200, method="bootstrap", seed=3)
    g.fit(wp.build_stage1_table())
    df = g.generate()
    assert len(df) == 1200
    assert (df["household_finance_source"] == "observed").all()
    assert g.wealth_report["edge_cases"]["achieved_total_share_pct"] == pytest.approx(8.0, abs=0.1)
    paths = g.save(str(tmp_path / "run"), df)
    assert all(Path(p).exists() for p in paths.values())
    meta = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
    assert meta["wealth_stage"]["violations_enforced"] and meta["method"] == "bootstrap"
    reloaded = cg.ClientGenerator.from_pretrained(paths["model"])
    reloaded.n_samples = 50
    assert len(reloaded.synthesizer.sample(50)) == 50


def test_wealth_overrides_reach_stage2(require_processed, data_cfg):
    import copy

    cfg = copy.deepcopy(data_cfg)
    gen_cfg = cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]
    gen_cfg["target_distribution"] = {
        "Conservative": 0.25,
        "Moderate": 0.25,
        "Aggressive": 0.25,
        "Ultra-Aggressive": 0.25,
    }
    g = cg.ClientGenerator(n_samples=2000, method="bootstrap", seed=4)
    g.fit(wp.build_stage1_table())
    df = g.generate(wealth_config=cfg, edge_case_share=0.0)
    assert "edge_case_id" in df and not df["is_edge_case"].any()
    shares = df["risk_tolerance_category"].value_counts(normalize=True)
    assert np.allclose(shares.values, 0.25, atol=0.01)


def test_sdv_methods_degrade_gracefully():
    if cg.sdv_available():
        pytest.skip("SDV installed: covered by SDV's own training path")
    g = cg.ClientGenerator(n_samples=10, method="copula", seed=1)
    with pytest.raises(ImportError):
        g.fit(pd.DataFrame({"edad": [30, 40], "situacion_laboral": ["ocupado", "jubilado"]}))


# --------------------------------------------------------------------------- #
# CSV round trip: ESG "None" is a category, not a missing value
# --------------------------------------------------------------------------- #
def test_csv_reader_keeps_esg_none(tmp_path):
    df = pd.DataFrame({"esg_preference_level": ["None", "Low", "High"], "x": [1.0, np.nan, 3.0]})
    path = tmp_path / "profiles.csv"
    df.to_csv(path, index=False)
    assert pd.read_csv(path)["esg_preference_level"].isna().sum() == 1, "pandas defaults lose 'None'"
    back = read_profiles(path)
    assert back["esg_preference_level"].tolist() == ["None", "Low", "High"]
    assert back["x"].isna().tolist() == [False, True, False], "genuine blanks are still missing"
    assert "None" not in CSV_NA_VALUES


# --------------------------------------------------------------------------- #
# Console safety (cp1252)
# --------------------------------------------------------------------------- #
OUTPUT_CALL = re.compile(r"\b(print|input|logger\.\w+|log\.\w+)\(")
SCANNED = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src" / "risksynth").rglob("*.py"))


@pytest.mark.parametrize("rel_path", SCANNED)
def test_console_output_lines_are_ascii(rel_path):
    lines = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    bad = [i + 1 for i, line in enumerate(lines) if OUTPUT_CALL.search(line) and not line.isascii()]
    assert not bad, f"non-ASCII print/input/log lines in {rel_path}: {bad}"


def test_generator_stdout_is_cp1252_safe(require_processed, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gs, "OUTPUT_DIR", tmp_path)
    base = pd.read_parquet(ROOT / "data" / "processed" / "combined_base.parquet").sample(5000, random_state=1)
    gs.generate_synthetic_data(150, base, n_jobs=1, batch_size=150, seed=2)
    out = capsys.readouterr().out
    assert out.isascii()
    out.encode("cp1252")


@pytest.mark.slow
def test_cli_runs_piped_on_cp1252(require_processed, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env.update(
        RISKSYNTH_OUTPUT_DIR=str(tmp_path),
        PYTHONIOENCODING="cp1252",
        PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])),
    )
    proc = subprocess.run(
        [sys.executable, "-m", "risksynth.generators.generate_synthetic"],
        input=b"300\n\n1\n300\n",
        capture_output=True,
        env=env,
        timeout=600,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr.decode("cp1252", "replace")[-2000:]
    assert proc.stdout.isascii() and b"Traceback" not in proc.stderr
    assert list(tmp_path.glob("synthetic_300_auto_*.parquet")), "outputs honour RISKSYNTH_OUTPUT_DIR"
