"""
Counterfactual re-assessment of one client profile.

Change a few inputs (age, income, expenses, debt, savings, stated tolerance, horizon,
accepted drawdown, liquidity need, goal) and recompute everything that depends on them
with the same engine code the generator uses: surplus, risk capacity and caps, effective
risk, the CR01-CR07 and MiFID II checks, the edge-case assertion and the model portfolio.

Nothing is repaired here. The point is to see which rules a changed profile would break,
not to fix it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risksynth.engine import consistency
from risksynth.engine import wealth_profile as wp
from risksynth.scoring import robo_advisor

# Inputs a reviewer may change, with the label used in the app
EDITABLE = {
    "edad": "Age",
    "net_income_annual_eur": "Net household income, EUR / year",
    "household_expenses_annual_eur": "Household expenses, EUR / year",
    "total_debt_service_annual_eur": "Debt service, EUR / year",
    "liquid_savings_eur": "Liquid savings, EUR",
    "risk_tolerance_score": "Stated risk tolerance (1-10)",
    "investment_horizon_years": "Investment horizon, years",
    "max_acceptable_drawdown_pct": "Accepted drawdown, %",
    "liquidity_need_pct": "Liquidity need, %",
    "primary_investment_goal": "Primary goal",
}
MIFID_CHECKS = {
    "MIFID_EFFECTIVE_RISK": "effective risk == min(stated tolerance, risk capacity)",
    "MIFID_CAPACITY_CURRENT": "risk capacity matches a fresh computation from current inputs",
    "MIFID_ESG_COLLECTED": "ESG preference recorded",
}


def _edge_case(cfg: dict, ec_id) -> dict | None:
    if ec_id is None or (isinstance(ec_id, float) and np.isnan(ec_id)) or pd.isna(ec_id):
        return None
    return next((p for p in cfg["edge_case_profiles"]["profiles"] if p["id"] == ec_id), None)


def reassess(row: pd.Series, changes: dict | None = None, cfg: dict | None = None,
             params: dict | None = None) -> dict:
    """Recompute the suitability picture for `row` with `changes` applied.

    Returns a dict with the recomputed profile (one-row frame), per-rule results, the
    edge-case assertion (if the row belongs to EC01-EC09), the model portfolio and an
    overall status: "Suitable", "Suitable with restrictions" or "Re-profile".
    """
    cfg = cfg or wp.load_config()
    unknown = set(changes or {}) - set(EDITABLE)
    if unknown:
        raise KeyError(f"not editable: {sorted(unknown)}")
    s = wp.schema(cfg)
    df = row.to_frame().T.reset_index(drop=True).infer_objects()
    for col, value in (changes or {}).items():
        df[col] = value
    if "total_debt_service_annual_eur" in (changes or {}):
        # derive_surplus rebuilds the total from its parts: put the whole amount in the main part
        df["debt_service_annual_eur"] = float(changes["total_debt_service_annual_eur"])
        if "consumer_debt_service_annual_eur" in df:
            df["consumer_debt_service_annual_eur"] = 0.0

    # Derived fields, in the generator's dependency order
    df = wp.derive_surplus(df, cfg)
    feat = s["features"]
    df["risk_tolerance_score"] = df["risk_tolerance_score"].astype(int)
    df["risk_tolerance_category"] = wp.category_from_score(df["risk_tolerance_score"],
                                                           feat["risk_tolerance_score"]["categories"])
    horizon = feat["investment_horizon_years"]
    df[horizon["bucket_column"]] = wp.bucket_from_years(df["investment_horizon_years"].astype(float),
                                                        horizon["buckets"])
    df = wp.compute_capacity(df, cfg)

    ec = _edge_case(cfg, row.get("edge_case_id"))
    relaxed = set(ec["relaxed_rules"]) if ec else set()
    masks = consistency.rule_masks(df, cfg).iloc[0]
    rules = [{"rule": rid, "text": r["rule"], "violated": bool(masks[rid]), "exempt": rid in relaxed}
             for rid, r in ((r["id"], r) for r in s["consistency_rules"])]
    mifid = [{"rule": rid, "text": text, "violated": bool(masks[rid])} for rid, text in MIFID_CHECKS.items()]
    df["profile_inconsistency_flag"] = bool(masks[consistency.CR_IDS].any())

    rec = robo_advisor.recommend(df, params).iloc[0]
    actions = [label for col, label in robo_advisor.ACTION_LABELS.items() if bool(rec[col])]
    if bool(rec["drawdown_guard_applied"]):
        actions.append("Equity cut to keep the stressed loss within the accepted drawdown")

    assertion = None
    if ec:
        test = ec["expected_advisory_outcome"]["test_assertion"]
        assertion = {"id": ec["id"], "test": test,
                     "expected": ec["expected_advisory_outcome"]["recommendation"],
                     "holds": bool(wp.evaluate(df, test).iloc[0])}

    blocking = [r["rule"] for r in rules + mifid if r["violated"] and not r.get("exempt")]
    if blocking or df["profile_inconsistency_flag"].iloc[0]:
        status = "Re-profile"
    elif actions:
        status = "Suitable with restrictions"
    else:
        status = "Suitable"
    caps = df["capacity_caps_applied"].iloc[0]
    return {
        "profile": df,
        "status": status,
        "blocking_rules": blocking,
        "rules": rules,
        "mifid": mifid,
        "caps": [] if pd.isna(caps) else str(caps).split("|"),
        "capacity_points": int(df["risk_capacity_points"].iloc[0]),
        "capacity_score": int(df["risk_capacity_score"].iloc[0]),
        "effective_risk": int(df["effective_risk_score"].iloc[0]),
        "tolerance_category": str(df["risk_tolerance_category"].iloc[0]),
        "recommendation": rec,
        "actions": actions,
        "edge_case_assertion": assertion,
    }
