# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Edge-case cohort EC01-EC09, EC08 rule exemption and advisory allocation caps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risksynth.scoring import robo_advisor
from risksynth.engine import consistency
from risksynth.engine import edge_cases as ec
from risksynth.engine.stage2 import run_stage2

EC_IDS = [f"EC0{i}" for i in range(1, 10)]


@pytest.fixture(scope="module")
def profiles(stage3):
    return stage3[0]


@pytest.fixture(scope="module")
def edge(stage3):
    return stage3[1]["edge_cases"]


# --------------------------------------------------------------------------- #
# Cohort size and composition
# --------------------------------------------------------------------------- #
def test_total_share_hits_target(edge, profiles, data_cfg):
    target = 100 * data_cfg["edge_case_profiles"]["total_target_share"]
    assert edge["achieved_total_share_pct"] == pytest.approx(target, abs=0.01)
    assert profiles["is_edge_case"].mean() == pytest.approx(target / 100, abs=1e-4)


@pytest.mark.parametrize("pid", EC_IDS)
def test_profile_quota_is_filled(edge, data_cfg, pid):
    p = edge["profiles"][pid]
    target = next(x["target_share"] for x in data_cfg["edge_case_profiles"]["profiles"] if x["id"] == pid)
    assert p["shortfall"] == 0
    assert p["share_pct"] == pytest.approx(100 * target, abs=0.01)
    assert p["rows"] >= 50, "model_config min_rows_per_profile"


@pytest.mark.parametrize("pid", EC_IDS)
def test_predicate_and_assertion_hold_for_every_member(edge, pid):
    p = edge["profiles"][pid]
    assert p["predicate_holds_pct"] == 100.0
    assert p["assertion_pass_pct"] == 100.0, f"{pid}: {p['assertion']}"


def test_acceptance_criteria_all_pass(edge):
    assert all(v["pass"] for v in edge["acceptance"].values()), edge["acceptance"]


@pytest.mark.parametrize("pid", [p for p in EC_IDS if p not in ("EC02", "EC08")])
def test_named_capacity_cap_fires_for_profile(edge, profiles, pid):
    cap = edge["profiles"][pid]["capacity_cap"]
    rows = profiles["edge_case_id"].eq(pid).fillna(False)
    assert profiles.loc[rows, "capacity_caps_applied"].str.contains(cap).all(), f"{pid} -> {cap}"


def test_ec02_requires_long_horizon(data_cfg, profiles):
    ec02 = next(p for p in data_cfg["edge_case_profiles"]["profiles"] if p["id"] == "EC02")
    assert "investment_horizon_years >= 7" in ec02["predicate"]
    rows = profiles["edge_case_id"].eq("EC02").fillna(False)
    assert (profiles.loc[rows, "investment_horizon_bucket"] == "Long").all()


def test_tags_and_origins(profiles):
    for col in [
        "edge_case_id",
        "edge_case_matches",
        "edge_case_origin",
        "is_edge_case",
        "expected_robo_behavior",
        "expected_capacity_override",
        "edge_case_assertion_pass",
        "edge_case_mutated_fields",
    ]:
        assert col in profiles, col
    assert (profiles["is_edge_case"] == profiles["edge_case_id"].notna()).all()
    assert set(profiles["edge_case_origin"].unique()) <= {"natural", "injected", "none"}
    natural = profiles["edge_case_origin"] == "natural"
    assert profiles.loc[natural, "edge_case_mutated_fields"].isna().all(), "natural members are never mutated"
    injected = profiles["edge_case_origin"] == "injected"
    assert profiles.loc[injected, "edge_case_mutated_fields"].notna().all()


def test_ec08_never_occurs_naturally_and_is_fully_injected(edge):
    assert edge["profiles"]["EC08"]["natural_prevalence_pct"] == 0
    assert edge["profiles"]["EC08"]["injected"] == edge["profiles"]["EC08"]["rows"]


# --------------------------------------------------------------------------- #
# EC08 exemption precision
# --------------------------------------------------------------------------- #
def test_ec08_cr02_is_exempt_and_flagged(stage3, profiles):
    report = stage3[1]
    ec08 = profiles["edge_case_id"].eq("EC08").fillna(False)
    assert report["violations_exempted"]["CR02"] == int(ec08.sum())
    assert sum(report["violations_enforced"].values()) == 0
    assert (profiles["profile_inconsistency_flag"] == ec08).all()


def test_exemption_is_per_rule_and_per_row(profiles, data_cfg):
    t = profiles.copy()
    e8 = t.index[t["edge_case_id"].eq("EC08").fillna(False)][:15]
    other = t.index[t["edge_case_id"].isna()][:15]
    t.loc[e8, "risk_tolerance_category"] = "Conservative"  # CR01 on EC08: must be repaired
    t.loc[other, "risk_tolerance_score"] = 1
    t.loc[other, "risk_tolerance_category"] = "Conservative"
    t.loc[other, "max_acceptable_drawdown_pct"] = 40.0  # CR02 off-cohort: must be repaired
    fixed, repairs = consistency.enforce(t, data_cfg, np.random.default_rng(0), ec.exemptions(t, data_cfg))
    m = consistency.rule_masks(fixed, data_cfg)
    assert not m.loc[e8, "CR01"].any()
    assert m.loc[e8, "CR02"].all(), "EC08's intentional contradiction survives"
    assert not m.loc[other, "CR02"].any()
    assert repairs["CR01"] >= 15
    assert (
        fixed["effective_risk_score"]
        == np.minimum(fixed["risk_tolerance_score"], fixed["risk_capacity_score"])
    ).all()


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def test_share_override(combined_sample, data_cfg):
    sub = combined_sample.head(10_000)
    _, rep = run_stage2(sub, seed=5, cfg=data_cfg, edge_case_share=0.04)
    assert rep["edge_cases"]["achieved_total_share_pct"] == pytest.approx(4.0, abs=0.02)


def test_edge_cases_can_be_disabled(stage2_only):
    df, rep = stage2_only
    assert "edge_case_id" not in df and rep["edge_cases"] is None


# --------------------------------------------------------------------------- #
# Advisory allocation caps (robo-advisor simulator)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def advice(profiles):
    return robo_advisor.recommend(profiles)


def test_allocations_sum_to_100(advice):
    for pre in ["naive", "capacity", "rec"]:
        total = advice[f"{pre}_equity_pct"] + advice[f"{pre}_fixed_income_pct"] + advice[f"{pre}_cash_pct"]
        assert np.allclose(total, 100, atol=0.11), pre


def test_stressed_loss_never_exceeds_accepted_drawdown(advice, profiles):
    assert (advice["rec_stress_loss_pct"] <= profiles["max_acceptable_drawdown_pct"] + 0.11).all()


def test_protections_only_remove_risk(advice):
    assert (advice["capacity_equity_pct"] <= advice["naive_equity_pct"] + 0.11).all()
    assert (advice["rec_equity_pct"] <= advice["capacity_equity_pct"] + 0.11).all()


def test_stress_profiles_are_protected(advice, profiles):
    by = pd.concat([profiles["edge_case_id"].fillna("base"), advice], axis=1).groupby("edge_case_id")
    equity = by[["naive_equity_pct", "rec_equity_pct"]].mean()
    assert equity.loc["EC05", "rec_equity_pct"] < equity.loc["EC05", "naive_equity_pct"] - 20
    assert (
        equity.loc["EC08", "rec_equity_pct"] < 5
    ), "EC08's 5-10% drawdown limit forces a defensive portfolio"
    ec08 = profiles["edge_case_id"].eq("EC08").fillna(False)
    assert advice.loc[ec08, "action_reprofile"].all()
    ec04 = profiles["edge_case_id"].eq("EC04").fillna(False)
    assert advice.loc[ec04, "action_no_new_contributions"].all()
