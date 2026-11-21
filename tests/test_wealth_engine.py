"""Wealth feature engine, MiFID II risk capacity and consistency rules CR01-CR07."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from risksynth.engine import consistency
from risksynth.engine import wealth_profile as wp
from risksynth.engine.stage2 import run_stage2


@pytest.fixture(scope="module")
def profiles(stage2_only):
    return stage2_only[0]


@pytest.fixture(scope="module")
def report(stage2_only):
    return stage2_only[1]


# --------------------------------------------------------------------------- #
# Surplus
# --------------------------------------------------------------------------- #
def test_investable_surplus_is_floored_raw_surplus(profiles):
    assert (profiles["investable_surplus_eur"] == profiles["raw_surplus_eur"].clip(0, 500_000)).all()
    assert profiles["investable_surplus_eur"].min() == 0
    assert (profiles["raw_surplus_eur"] < 0).any(), "signed surplus must be retained, not floored"


def test_negative_surplus_share_matches_epf(profiles):
    share = (profiles["raw_surplus_eur"] < 0).mean()
    assert 0.37 <= share <= 0.45, f"EPF 2024 weighted share is 0.409, got {share:.3f}"


def test_surplus_formula_and_income_calibration_factor(data_cfg):
    df = pd.DataFrame({"net_income_annual_eur": [30000.0, 20000.0], "household_expenses_annual_eur": [25000.0, 21000.0],
                       "debt_service_annual_eur": [3000.0, 0.0], "consumer_debt_service_annual_eur": [1200.0, np.nan]})
    out = wp.derive_surplus(df.copy(), data_cfg)
    assert out["raw_surplus_eur"].tolist() == [800.0, -1000.0]
    assert out["investable_surplus_eur"].tolist() == [800.0, 0.0]
    assert out["total_debt_service_annual_eur"].tolist() == [4200.0, 0.0]
    cfg = copy.deepcopy(data_cfg)
    cfg["wealth_profile_schema"]["features"]["investable_surplus_eur"]["income_calibration_factor"] = 1.2
    calibrated = wp.derive_surplus(df.copy(), cfg)
    assert calibrated["raw_surplus_eur"].tolist() == [6800.0, 3000.0]


def test_hy020_is_never_fabricated(profiles):
    assert profiles["renta_disponible"].isna().all()
    assert set(profiles["household_finance_source"].unique()) <= {"observed", "donor_epf"}
    epf_rows = profiles["fuente"] == "EPF"
    assert (profiles.loc[epf_rows, "household_finance_source"] == "observed").mean() > 0.99


# --------------------------------------------------------------------------- #
# Boundaries and target distributions
# --------------------------------------------------------------------------- #
def test_horizon_bucket_boundaries(data_cfg):
    buckets = data_cfg["wealth_profile_schema"]["features"]["investment_horizon_years"]["buckets"]
    years = pd.Series([0.5, 2.9, 3.0, 6.9, 7.0, 30.0])
    assert wp.bucket_from_years(years, buckets).tolist() == ["Short", "Short", "Medium", "Medium", "Long", "Long"]


def test_score_category_cutoffs(data_cfg):
    cats = data_cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["categories"]
    out = wp.category_from_score(pd.Series(range(1, 11)), cats).tolist()
    assert out == ["Conservative"] * 3 + ["Moderate"] * 3 + ["Aggressive"] * 2 + ["Ultra-Aggressive"] * 2


def test_age_bands_are_contiguous():
    out = wp.age_band(pd.Series([17, 34, 34.5, 35, 64.9, 65, 99]), ["18-34", "35-49", "50-64", "65+"])
    assert out.tolist() == ["18-34", "18-34", "18-34", "35-49", "50-64", "65+", "65+"]


@pytest.mark.parametrize("column, path", [
    ("risk_tolerance_category", "risk_tolerance_score"),
    ("esg_preference_level", "esg_preference_level"),
])
def test_target_distributions_are_exact(profiles, data_cfg, column, path):
    target = data_cfg["wealth_profile_schema"]["features"][path]["generation"]["target_distribution"]
    actual = profiles[column].astype("string").value_counts(normalize=True)
    for category, share in target.items():
        assert abs(actual.get(str(category), 0) - share) < 0.002, f"{column}={category}"


def test_feature_ranges(profiles):
    assert profiles["risk_tolerance_score"].between(1, 10).all()
    assert profiles["max_acceptable_drawdown_pct"].between(5, 40).all()
    assert profiles["liquidity_need_pct"].between(0, 100).all()
    assert profiles["investment_horizon_years"].between(0.5, 50).all()
    assert profiles["primary_investment_goal"].isin(["Capital Preservation", "Retirement", "Wealth Growth", "Income"]).all()


def test_expected_dependencies(profiles, model_cfg):
    for c in model_cfg["constraints"]["dependency_constraints"]:
        a, b = c["pair"]
        rho = profiles[a].corr(profiles[b], method="spearman")
        assert c.get("min_rho", -1) <= rho <= c.get("max_rho", 1), f"{a} ~ {b}: {rho:.3f}"


# --------------------------------------------------------------------------- #
# Risk capacity (MiFID II) and named caps
# --------------------------------------------------------------------------- #
def test_effective_risk_is_min_of_tolerance_and_capacity(profiles):
    assert (profiles["effective_risk_score"]
            == np.minimum(profiles["risk_tolerance_score"], profiles["risk_capacity_score"])).all()
    assert profiles["risk_capacity_score"].between(1, 10).all()


def test_every_named_capacity_cap_binds(profiles, data_cfg):
    caps = data_cfg["wealth_profile_schema"]["supporting_features"]["risk_capacity_score"]["caps"]
    for cap in caps:
        hit = wp.evaluate(profiles, cap["when"])
        assert hit.any(), f"{cap['id']} never fires on the sample"
        assert (profiles.loc[hit, "risk_capacity_score"] <= cap["max"]).all(), cap["id"]
        assert profiles.loc[hit, "capacity_caps_applied"].str.contains(cap["id"]).all(), cap["id"]


def test_capacity_points_scheme(data_cfg):
    cap_cfg = data_cfg["wealth_profile_schema"]["supporting_features"]["risk_capacity_score"]
    assert wp.points(pd.Series([-0.1, 0.05, 0.2, 0.5]), cap_cfg["points"]["surplus_ratio"]).tolist() == [0, 1, 2, 3]
    assert wp.points(pd.Series([0.4, 0.25, 0.1]), cap_cfg["points"]["debt_service_to_income"]).tolist() == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Consistency rules CR01-CR07 and MiFID checks
# --------------------------------------------------------------------------- #
def test_no_violations_after_enforcement(report, profiles, data_cfg):
    assert sum(report["violations_enforced"].values()) == 0
    masks = consistency.rule_masks(profiles, data_cfg)
    assert not masks.any().any()
    assert not profiles["profile_inconsistency_flag"].any()


def test_raw_draws_and_repairs_are_auditable(report, profiles):
    assert report["violations_raw"]["CR02"] > 0, "drawdown noise produces some raw CR02 breaks"
    assert report["repairs_applied"]["CR02"] == report["violations_raw"]["CR02"]
    raw = profiles["cr_violations_raw"].notna()
    assert (profiles.loc[raw, "cr_repaired"].notna()).all()


def test_enforce_repairs_every_rule_on_shuffled_columns(profiles, data_cfg):
    rng = np.random.default_rng(3)
    noisy = profiles.sample(4000, random_state=3).reset_index(drop=True)
    for col in ["risk_tolerance_category", "investment_horizon_bucket", "primary_investment_goal",
                "investment_horizon_years", "max_acceptable_drawdown_pct", "investable_surplus_eur"]:
        noisy[col] = rng.permutation(noisy[col].to_numpy())
    before = consistency.rule_masks(noisy, data_cfg)
    for rule in ["CR01", "CR02", "CR03", "CR04", "CR06", "CR07"]:
        assert before[rule].any(), f"shuffle should break {rule}"
    fixed, _ = consistency.enforce(noisy, data_cfg, rng)
    assert not consistency.rule_masks(fixed, data_cfg).any().any()


def test_stage2_is_idempotent_on_household_block(profiles, data_cfg):
    sub = profiles.head(2000).drop(columns=["liquid_savings_source"])
    again, _ = run_stage2(sub, seed=99, cfg=data_cfg, edge_cases=False)
    assert (again["household_finance_source"] == sub["household_finance_source"]).all()
    assert np.allclose(again["raw_surplus_eur"], sub["raw_surplus_eur"])
