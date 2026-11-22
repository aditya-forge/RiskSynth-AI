# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Consistency rules CR01-CR07 and MiFID II suitability checks for wealth profiles.

validate()  -> per-rule violation masks and counts (nothing is changed)
enforce()   -> repairs violations in the configured order, then recomputes risk
               capacity and effective risk so the MiFID identities hold

Rules come from data_config.yaml (wealth_profile_schema.consistency_rules / .enforcement)
and compliance.mifid2_suitability. `exempt` maps rule ids to row masks that may legitimately
break a rule (edge-case scenarios: EC08 relaxes CR02).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .wealth_profile import (bucket_from_years, category_from_score, compute_capacity,
                             sample_goal, sample_horizon, schema)

log = logging.getLogger(__name__)
CR_IDS = ["CR01", "CR02", "CR03", "CR04", "CR05", "CR06", "CR07"]


def _ctx(cfg: dict) -> dict:
    s = schema(cfg)
    f = s["features"]
    dd = f["max_acceptable_drawdown_pct"]
    return {
        "s": s,
        "categories": f["risk_tolerance_score"]["categories"],
        "buckets": f["investment_horizon_years"]["buckets"],
        "bands": dd["generation"]["bands"],
        "band_tol": dd["generation"]["band_tolerance_pct"],
        "dd_range": dd["range"],
        "surplus_clip": f["investable_surplus_eur"]["clip"],
        "preservation_max": s["enforcement"]["preservation_horizon_max_years"],
    }


def _band(df: pd.DataFrame, c: dict, key: str) -> pd.Series:
    return df["risk_tolerance_category"].map({k: v[key] for k, v in c["bands"].items()}).astype(float)


def _horizon_cap(df: pd.DataFrame, c: dict) -> pd.Series:
    return (c["s"]["life_horizon_cap_age"] - df["edad"]).clip(lower=0.5)


def rule_masks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Boolean violation mask per CR rule and MiFID check (True = violated)."""
    c = _ctx(cfg)
    s = c["s"]
    goal, years = df["primary_investment_goal"], df["investment_horizon_years"]
    expected_cat = category_from_score(df["risk_tolerance_score"], c["categories"])
    expected_bucket = bucket_from_years(years, c["buckets"])
    dd = df["max_acceptable_drawdown_pct"]
    lo, hi = c["surplus_clip"]
    recomputed = compute_capacity(df.copy(), cfg)
    masks = {
        "CR01": df["risk_tolerance_category"].astype("string").ne(expected_cat).fillna(True),
        "CR02": (dd < _band(df, c, "min") - c["band_tol"]) | (dd > _band(df, c, "max") + c["band_tol"]),
        "CR03": df["investment_horizon_bucket"].astype("string").ne(expected_bucket).fillna(True),
        "CR04": goal.eq("Retirement") & (df["edad"] >= s["retirement_age"]),
        "CR05": years > s["life_horizon_cap_age"] - df["edad"] + 1e-9,
        "CR06": (df["investable_surplus_eur"] - df["raw_surplus_eur"].clip(lo, hi)).abs() > 0.5,
        "CR07": goal.eq("Capital Preservation") & (years > c["preservation_max"]),
        "MIFID_EFFECTIVE_RISK": df["effective_risk_score"].ne(
            np.minimum(df["risk_tolerance_score"], df["risk_capacity_score"])),
        "MIFID_CAPACITY_CURRENT": df["risk_capacity_score"].ne(recomputed["risk_capacity_score"]),
        "MIFID_ESG_COLLECTED": df["esg_preference_level"].isna(),
    }
    return pd.DataFrame(masks, index=df.index).fillna(True).astype(bool)


def validate(df: pd.DataFrame, cfg: dict, exempt: dict | None = None,
             stage: str = "") -> tuple[pd.DataFrame, dict]:
    """Violation masks (exempt rows cleared) and counts; logs one line per rule."""
    masks = rule_masks(df, cfg)
    for rule, mask in (exempt or {}).items():
        masks[rule] &= ~mask.reindex(df.index, fill_value=False)
    counts = {rule: int(masks[rule].sum()) for rule in masks}
    for rule, n in counts.items():
        log.log(logging.WARNING if n else logging.INFO, "[%s] %-22s %7d violations (%.2f%%)",
                stage, rule, n, 100 * n / max(len(df), 1))
    return masks, counts


def enforce(df: pd.DataFrame, cfg: dict, rng: np.random.Generator,
            exempt: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Repair CR violations in config order; returns the repaired frame and repair counts."""
    c = _ctx(cfg)
    df = df.copy()
    # Append to repairs recorded by an earlier pass (stage 2 before stage 3)
    previous = df["cr_repaired"].astype("string").fillna("") if "cr_repaired" in df else ""
    repaired = pd.Series(previous, index=df.index, dtype="string")
    counts = {}
    for rule in c["s"]["enforcement"]["order"]:
        mask = rule_masks(df, cfg)[rule]
        if exempt and rule in exempt:
            mask &= ~exempt[rule].reindex(df.index, fill_value=False)
        counts[rule] = int(mask.sum())
        if not mask.any():
            continue
        _REPAIRS[rule](df, mask, c, cfg, rng)
        repaired[mask] = repaired[mask] + "|" + rule
    df = compute_capacity(df, cfg)            # capacity/effective risk depend on repaired fields
    df["cr_repaired"] = repaired.str.lstrip("|").replace("", pd.NA)
    return df, counts


def _repair_cr01(df, m, c, cfg, rng):
    df.loc[m, "risk_tolerance_category"] = category_from_score(df.loc[m, "risk_tolerance_score"],
                                                               c["categories"])


def _repair_cr02(df, m, c, cfg, rng):
    lo = (_band(df.loc[m], c, "min") - c["band_tol"]).clip(lower=c["dd_range"][0])
    hi = (_band(df.loc[m], c, "max") + c["band_tol"]).clip(upper=c["dd_range"][1])
    df.loc[m, "max_acceptable_drawdown_pct"] = df.loc[m, "max_acceptable_drawdown_pct"].clip(lo, hi)


def _repair_cr03(df, m, c, cfg, rng):
    df.loc[m, "investment_horizon_bucket"] = bucket_from_years(
        df.loc[m, "investment_horizon_years"], c["buckets"])


def _repair_cr04(df, m, c, cfg, rng):
    # The goal adjustments already zero Retirement at/after retirement age.
    sub = df.loc[m].copy()
    sub["primary_investment_goal"] = sample_goal(sub, cfg, rng)
    df.loc[m, "primary_investment_goal"] = sub["primary_investment_goal"]
    df.loc[m, "investment_horizon_years"] = sample_horizon(sub, cfg, rng)
    _repair_cr03(df, m, c, cfg, rng)


def _repair_cr05(df, m, c, cfg, rng):
    df.loc[m, "investment_horizon_years"] = np.minimum(df.loc[m, "investment_horizon_years"],
                                                       _horizon_cap(df.loc[m], c))
    _repair_cr03(df, m, c, cfg, rng)


def _repair_cr06(df, m, c, cfg, rng):
    lo, hi = c["surplus_clip"]
    df.loc[m, "investable_surplus_eur"] = df.loc[m, "raw_surplus_eur"].clip(lo, hi)


def _repair_cr07(df, m, c, cfg, rng):
    df.loc[m, "investment_horizon_years"] = df.loc[m, "investment_horizon_years"].clip(
        upper=c["preservation_max"])
    _repair_cr03(df, m, c, cfg, rng)


_REPAIRS = {"CR01": _repair_cr01, "CR02": _repair_cr02, "CR03": _repair_cr03, "CR04": _repair_cr04,
            "CR05": _repair_cr05, "CR06": _repair_cr06, "CR07": _repair_cr07}


def exempted_counts(df: pd.DataFrame, cfg: dict, exempt: dict | None) -> dict:
    """Per rule: rows that violate it but are exempt (intentional edge-case traits)."""
    if not exempt:
        return {}
    masks = rule_masks(df, cfg)
    return {rule: int((masks[rule] & m.reindex(df.index, fill_value=False)).sum())
            for rule, m in exempt.items()}


def violation_labels(masks: pd.DataFrame, rules: list[str] = CR_IDS) -> pd.Series:
    """Pipe-joined ids of the rules each row violates (NA when none)."""
    m = masks[rules]
    labels = m.apply(lambda col: np.where(col, col.name + "|", "")).sum(axis=1).str.rstrip("|")
    return labels.replace("", pd.NA).astype("string")
