# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Counterfactual re-assessment, the EC01-EC09 scorecard and run contracts."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from risksynth import contract
from risksynth.scoring import counterfactual, robo_advisor, scorecard

EC_IDS = [f"EC0{i}" for i in range(1, 10)]


@pytest.fixture(scope="module")
def profiles(stage3):
    return stage3[0]


def _first(profiles, ec_id):
    return profiles.loc[profiles["edge_case_id"].eq(ec_id).fillna(False)].iloc[0]


# --------------------------------------------------------------------------- #
# Counterfactual
# --------------------------------------------------------------------------- #
def test_unchanged_profile_reproduces_stored_capacity(profiles, data_cfg):
    sample = profiles.sample(200, random_state=3)
    for _, row in sample.iterrows():
        cf = counterfactual.reassess(row, {}, data_cfg)
        assert cf["capacity_score"] == row["risk_capacity_score"]
        assert cf["effective_risk"] == row["effective_risk_score"]


@pytest.mark.parametrize("ec_id", EC_IDS)
def test_unchanged_edge_case_keeps_its_assertion(profiles, data_cfg, ec_id):
    cf = counterfactual.reassess(_first(profiles, ec_id), {}, data_cfg)
    assert cf["edge_case_assertion"]["id"] == ec_id
    assert cf["edge_case_assertion"]["holds"]


def test_ec08_drawdown_contradiction_is_relaxed_but_flags_reprofile(profiles, data_cfg):
    cf = counterfactual.reassess(_first(profiles, "EC08"), {}, data_cfg)
    cr02 = next(r for r in cf["rules"] if r["rule"] == "CR02")
    assert cr02["violated"] and cr02["exempt"]
    assert "CR02" not in cf["blocking_rules"]
    assert cf["status"] == "Re-profile" and cf["recommendation"]["action_reprofile"]


def test_losing_all_surplus_caps_capacity(profiles, data_cfg):
    base = profiles[profiles["edge_case_id"].isna() & profiles["capacity_caps_applied"].isna()].iloc[0]
    cf = counterfactual.reassess(base, {"household_expenses_annual_eur": float(base["net_income_annual_eur"]) * 2},
                                 data_cfg)
    assert "CAP_NO_SURPLUS" in cf["caps"]
    assert cf["profile"]["investable_surplus_eur"].iloc[0] == 0
    assert cf["effective_risk"] <= 5


def test_contradictory_drawdown_breaks_cr02_for_a_base_client(profiles, data_cfg):
    base = profiles[profiles["edge_case_id"].isna()].iloc[0]
    cf = counterfactual.reassess(base, {"risk_tolerance_score": 10, "max_acceptable_drawdown_pct": 1}, data_cfg)
    assert "CR02" in cf["blocking_rules"] and cf["status"] == "Re-profile"
    assert cf["tolerance_category"] == "Ultra-Aggressive"
    assert cf["recommendation"]["rec_stress_loss_pct"] <= 1 + 0.05


def test_unknown_input_is_rejected(profiles, data_cfg):
    with pytest.raises(KeyError):
        counterfactual.reassess(profiles.iloc[0], {"risk_capacity_score": 10}, data_cfg)


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def test_scorecard_covers_every_archetype_and_passes(profiles, data_cfg):
    card = scorecard.edge_case_scorecard(profiles, robo_advisor.recommend(profiles), data_cfg)
    assert list(card["id"]) == EC_IDS
    assert (card["clients"] > 0).all()
    assert (card["status"] == "PASS").all(), card[card["status"] != "PASS"].to_dict("records")
    assert (card["over_limit_after_advice_pct"] == 0).all()
    capped = card[card["capacity_cap"].notna()]
    assert (capped["cap_fired_pct"] > 0).all()


def test_scorecard_fails_a_broken_assertion(profiles, data_cfg):
    broken = profiles.copy()
    m = broken["edge_case_id"].eq("EC05").fillna(False)
    broken.loc[m, "effective_risk_score"] = 9          # EC05 asserts effective_risk_score <= 5
    card = scorecard.edge_case_scorecard(broken, cfg=data_cfg).set_index("id")
    assert card.at["EC05", "status"] == "FAIL" and card.at["EC05", "assertion_pass_pct"] == 0


def test_scorecard_without_cohort_is_empty(profiles, data_cfg):
    assert scorecard.edge_case_scorecard(profiles.drop(columns="edge_case_id"), cfg=data_cfg).empty


# --------------------------------------------------------------------------- #
# Run contract
# --------------------------------------------------------------------------- #
def test_assumptions_are_found_with_their_keys():
    found = contract.assumptions()
    keys = {a["key"] for a in found}
    assert "wealth_profile_schema.features.risk_tolerance_score.generation.target_distribution" in keys
    assert "robo_advisor_simulator" in keys
    assert all(a["key"] for a in found)
    assert not any((a["note"] or "").startswith("=") for a in found)


def test_frame_hash_ignores_row_index_but_not_values():
    a = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    assert contract.frame_sha256(a) == contract.frame_sha256(a.set_axis([5, 6]))
    assert contract.frame_sha256(a) != contract.frame_sha256(a.assign(x=[1.0, 2.5]))


def test_bootstrap_run_is_reproduced_from_its_contract(require_processed, data_cfg, tmp_path):
    mix = {"Conservative": 40.0, "Moderate": 35.0, "Aggressive": 18.0, "Ultra-Aggressive": 7.0}
    out, report, _ = contract.generate("Bootstrap sampler", 600, 11, 0.08, mix, cfg=data_cfg)
    c = contract.build(method="Bootstrap sampler", stage1_model="bootstrap", sdv_model="copula", rows=600, seed=11,
                       edge_case_share=0.08, risk_mix_pct=mix, report=report, frame=out, cfg=data_cfg)
    path = tmp_path / "run_contract.json"
    path.write_text(json.dumps(c, default=str), encoding="utf-8")
    c = contract.load(path)

    assert c["parameters"]["seed"] == 11 and c["parameters"]["target_risk_mix_pct"] == mix
    assert set(c["cohort_proportions"]["profiles"]) == set(EC_IDS)
    assert set(c["rule_checks"]["consistency_rules"]) == {f"CR0{i}" for i in range(1, 8)}
    assert c["rule_checks"]["mifid_checks"] and c["assumptions"]
    assert "EFF 2022" in c["calibration_note"]

    out.to_parquet(tmp_path / "profiles.parquet", index=False)
    assert contract.verify(c, pd.read_parquet(tmp_path / "profiles.parquet")) == []
    assert contract.verify(c, contract.reproduce(c, cfg=data_cfg)) == []
    assert contract.verify(c, out.iloc[:-1]) != []
