# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Transparent model-portfolio engine for testing RiskSynth AI profiles.

Three allocations per client, so each protection is visible on its own:
  naive        stated risk tolerance only (what a questionnaire-only robo-advisor does)
  capacity     effective risk = min(tolerance, capacity) plus a cash sleeve for the stated
               liquidity need (MiFID II ability to bear losses)
  recommended  capacity + a drawdown guard so the stressed loss never exceeds
               max_acceptable_drawdown_pct (the "more conservative signal" for EC08)

Plus the advisor actions a suitability engine should raise (re-profiling, contribution
freeze, product minimums, sequence-of-returns warning).
Parameters: configs/model_config.yaml -> robo_advisor_simulator ([ASSUMPTION], illustrative).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from risksynth.paths import MODEL_CONFIG


def load_params(path: Path = MODEL_CONFIG) -> dict:
    params: dict = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["robo_advisor_simulator"]
    return params


def _allocate(score: pd.Series, cash: pd.Series, params: dict) -> tuple[pd.Series, pd.Series]:
    curve = np.array(params["equity_pct_by_score"], dtype=float)
    equity_share = curve[score.clip(1, 10).astype(int).to_numpy() - 1] / 100
    equity = equity_share * (100 - cash)
    return equity, 100 - cash - equity


def stressed_loss(equity, fixed_income, cash, params) -> pd.Series:
    loss = params["stress_loss"]
    return equity * loss["equity"] + fixed_income * loss["fixed_income"] + cash * loss["cash"]


def recommend(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Allocation (% of portfolio) and advisor actions for each client row."""
    p = params or load_params()
    c = p["cash"]
    out = pd.DataFrame(index=df.index)
    loss = p["stress_loss"]

    # Naive: stated tolerance, minimum cash only
    naive_cash = pd.Series(float(c["min_pct"]), index=df.index)
    out["naive_equity_pct"], out["naive_fixed_income_pct"] = _allocate(df["risk_tolerance_score"], naive_cash, p)
    out["naive_cash_pct"] = naive_cash
    out["naive_stress_loss_pct"] = stressed_loss(out["naive_equity_pct"], out["naive_fixed_income_pct"],
                                                 naive_cash, p)

    # Step 2 - capacity: effective risk (min of tolerance and capacity) + liquidity sleeve
    cash = (c["liquidity_need_factor"] * df["liquidity_need_pct"]).clip(c["min_pct"], c["max_pct"])
    equity, fixed_income = _allocate(df["effective_risk_score"], cash, p)
    out["capacity_equity_pct"] = equity.round(1)
    out["capacity_fixed_income_pct"] = fixed_income.round(1)
    out["capacity_cash_pct"] = cash.round(1)

    # Step 3 - drawdown guard: stressed loss must not exceed max_acceptable_drawdown_pct
    limit = df["max_acceptable_drawdown_pct"].astype(float)
    over = stressed_loss(equity, fixed_income, cash, p) > limit + 1e-9
    # 1) cut equity (into fixed income) until loss == limit
    max_equity = ((limit - loss["fixed_income"] * (100 - cash)) / (loss["equity"] - loss["fixed_income"])).clip(lower=0)
    equity = equity.where(~over, np.minimum(equity, max_equity))
    fixed_income = 100 - cash - equity
    # 2) if fixed income alone still breaches the limit, move the excess to cash
    still = stressed_loss(equity, fixed_income, cash, p) > limit + 1e-9
    max_fi = (limit - loss["equity"] * equity) / loss["fixed_income"]
    fixed_income = fixed_income.where(~still, np.minimum(fixed_income, max_fi.clip(lower=0)))
    cash = 100 - equity - fixed_income

    esg_share = df["esg_preference_level"].astype("string").map(
        {k: float(v) for k, v in p["esg_equity_share"].items()}).astype(float).fillna(0.0)
    out["rec_equity_pct"] = equity.round(1)
    out["rec_fixed_income_pct"] = fixed_income.round(1)
    out["rec_cash_pct"] = (100 - out["rec_equity_pct"] - out["rec_fixed_income_pct"]).round(1)
    out["rec_esg_pct"] = (equity * esg_share).round(1)
    out["rec_stress_loss_pct"] = stressed_loss(out["rec_equity_pct"], out["rec_fixed_income_pct"],
                                               out["rec_cash_pct"], p).round(1)
    out["drawdown_guard_applied"] = over
    out["equity_reduction_pp"] = (out["naive_equity_pct"] - out["rec_equity_pct"]).round(1)
    out["equity_cut_by_capacity_pp"] = (out["naive_equity_pct"] - out["capacity_equity_pct"]).round(1)
    out["equity_cut_by_drawdown_pp"] = (out["capacity_equity_pct"] - out["rec_equity_pct"]).round(1)

    caps = df.get("capacity_caps_applied", pd.Series(pd.NA, index=df.index)).astype("string").fillna("")
    out["action_reprofile"] = df.get("profile_inconsistency_flag", False)
    out["action_capacity_bound"] = df["effective_risk_score"] < df["risk_tolerance_score"]
    out["action_no_new_contributions"] = df["investable_surplus_eur"] <= 0
    out["action_below_minimum"] = df.get("flag_minimum_investment", False)
    out["action_sequence_risk_warning"] = caps.str.contains("CAP_NEAR_RETIREMENT")
    out["action_emergency_fund_first"] = caps.str.contains("CAP_NO_BUFFER")
    return out


ACTION_LABELS = {
    "action_reprofile": "Re-profile: questionnaire answers contradict each other",
    "action_capacity_bound": "Risk capped by capacity (effective < stated tolerance)",
    "action_no_new_contributions": "No recurring contributions: no investable surplus",
    "action_below_minimum": "Below product minimum (initial or monthly)",
    "action_sequence_risk_warning": "Sequence-of-returns warning: near retirement",
    "action_emergency_fund_first": "Build a 3-month emergency fund first",
}
