# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Edge-case suitability scorecard: one row per EC01-EC09 archetype.

For each archetype it checks what a robo-advisor test suite would check: does the
written assertion hold, does the expected capacity cap fire, and does the recommended
portfolio keep the stressed loss inside the client's accepted drawdown. A cohort passes
when every client passes the assertion and none ends up over the limit.
"""

from __future__ import annotations

import pandas as pd

from risksynth.engine import wealth_profile as wp
from risksynth.scoring import robo_advisor

COLUMNS = [
    "id",
    "archetype",
    "clients",
    "assertion",
    "assertion_pass_pct",
    "capacity_cap",
    "cap_fired_pct",
    "accepted_drawdown_pct",
    "stress_loss_tolerance_only_pct",
    "stress_loss_recommended_pct",
    "over_limit_after_advice_pct",
    "equity_cut_pp",
    "expected_behaviour",
    "status",
]


def edge_case_scorecard(
    df: pd.DataFrame, rec: pd.DataFrame | None = None, cfg: dict | None = None
) -> pd.DataFrame:
    """Per-archetype pass/fail and stress outcomes.

    Pass `rec` (robo_advisor.recommend(df)) if it is already computed, to avoid recomputing it.
    """
    cfg = cfg or wp.load_config()
    if "edge_case_id" not in df:
        return pd.DataFrame(columns=COLUMNS)
    rec = robo_advisor.recommend(df) if rec is None else rec
    limit = df["max_acceptable_drawdown_pct"].astype(float)
    caps = df.get("capacity_caps_applied", pd.Series(pd.NA, index=df.index)).astype("string").fillna("")
    rows = []
    for p in cfg["edge_case_profiles"]["profiles"]:
        m = df["edge_case_id"].eq(p["id"]).fillna(False).astype(bool)
        n = int(m.sum())
        outcome = p["expected_advisory_outcome"]
        if n == 0:
            rows.append(
                {
                    "id": p["id"],
                    "archetype": p["name"].replace("_", " ").title(),
                    "clients": 0,
                    "assertion": outcome["test_assertion"],
                    "capacity_cap": p["capacity_cap"],
                    "expected_behaviour": outcome["recommendation"],
                    "status": "NO DATA",
                }
            )
            continue
        held = wp.evaluate(df.loc[m], outcome["test_assertion"]).astype(bool)
        over = rec.loc[m, "rec_stress_loss_pct"] > limit[m] + 0.05
        cap = p["capacity_cap"]
        rows.append(
            {
                "id": p["id"],
                "archetype": p["name"].replace("_", " ").title(),
                "clients": n,
                "assertion": outcome["test_assertion"],
                "assertion_pass_pct": round(100 * held.mean(), 1),
                "capacity_cap": cap,
                "cap_fired_pct": (
                    round(100 * caps[m].str.contains(cap, regex=False).mean(), 1) if cap else None
                ),
                "accepted_drawdown_pct": round(limit[m].mean(), 1),
                "stress_loss_tolerance_only_pct": round(rec.loc[m, "naive_stress_loss_pct"].mean(), 1),
                "stress_loss_recommended_pct": round(rec.loc[m, "rec_stress_loss_pct"].mean(), 1),
                "over_limit_after_advice_pct": round(100 * over.mean(), 1),
                "equity_cut_pp": round(rec.loc[m, "equity_reduction_pp"].mean(), 1),
                "expected_behaviour": outcome["recommendation"],
                "status": "PASS" if held.all() and not over.any() else "FAIL",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)
