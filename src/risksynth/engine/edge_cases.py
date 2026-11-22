# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Stage 3 of the RiskSynth AI pipeline: edge-case stress-test cohort (EC01-EC09).

Spec: configs/data_config.yaml -> edge_case_profiles. Two concepts:

  edge_case_matches  every profile whose predicate a row satisfies (natural prevalence)
  edge_case_id       the stress-test cohort: round(target_share * N) rows per profile
                     (total ~ total_target_share). Natural matches fill each quota first;
                     any shortfall is injected by mutating the fewest fields of non-cohort
                     rows (conditional resampling + rejection). Rows are changed in place,
                     never appended, so N is unchanged.

Injected rows must satisfy their predicate and every consistency rule except the
profile's `relaxed_rules` (EC08 relaxes CR02). Those exemptions are handed to
consistency.enforce() so intentional traits survive while any other violation is still
repaired. Fidelity tests should exclude edge_case_origin == "injected".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import consistency
from . import wealth_profile as wp

log = logging.getLogger(__name__)

# Columns an injection recipe may change; used to record edge_case_mutated_fields per row.
MUTABLE = [
    "risk_tolerance_score",
    "risk_tolerance_category",
    "max_acceptable_drawdown_pct",
    "primary_investment_goal",
    "investment_horizon_years",
    "investment_horizon_bucket",
    "liquidity_need_pct",
    "liquid_savings_eur",
    "main_residence_value_eur",
]


def spec(cfg: dict) -> dict:
    return cfg["edge_case_profiles"]


# --------------------------------------------------------------------------- #
# Thresholds and predicates
# --------------------------------------------------------------------------- #
def resolve_thresholds(df: pd.DataFrame, cfg: dict) -> dict:
    """Percentile thresholds from the base population, or config fallbacks for small runs."""
    ec = spec(cfg)
    use_percentiles = len(df) >= ec["min_base_population_for_percentiles"]
    out = {}
    for p in ec["profiles"]:
        values = {}
        for name, t in p.get("thresholds", {}).items():
            s = df[t["of"]]
            if "among" in t:
                s = s[wp.evaluate(df, t["among"])]
            ok = use_percentiles and s.notna().any()
            values[name] = float(s.quantile(t["percentile"] / 100)) if ok else float(t["fallback"])
        out[p["id"]] = values
    return out


def predicate_masks(df: pd.DataFrame, cfg: dict, thresholds: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            p["id"]: wp.evaluate(df, p["predicate"].format(**thresholds[p["id"]]))
            for p in spec(cfg)["profiles"]
        },
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# Injection recipes: candidate filter + minimal mutation
# --------------------------------------------------------------------------- #
def _raise_score(sub, rng, cfg, lo, hi=10, redraw_drawdown=True):
    """Raise risk tolerance into [lo, hi] where it is below lo; keep category/drawdown coherent."""
    low = sub["risk_tolerance_score"] < lo
    sub.loc[low, "risk_tolerance_score"] = rng.integers(lo, hi + 1, low.sum())
    cats = wp.schema(cfg)["features"]["risk_tolerance_score"]["categories"]
    sub["risk_tolerance_category"] = wp.category_from_score(sub["risk_tolerance_score"], cats)
    if redraw_drawdown:
        sub.loc[low, "max_acceptable_drawdown_pct"] = wp.derive_drawdown(sub, cfg, rng)[low]
    return sub


def _rederive_liquidity(sub, rng, cfg, rows):
    if rows.any():
        sub.loc[rows, "liquidity_need_pct"] = wp.derive_liquidity_need(sub, cfg, rng)[rows]
    return sub


def _rebucket(sub, cfg):
    h = wp.schema(cfg)["features"]["investment_horizon_years"]
    sub[h["bucket_column"]] = wp.bucket_from_years(sub["investment_horizon_years"], h["buckets"])
    return sub


def _ec01(sub, rng, cfg, th):
    monthly = sub["household_expenses_annual_eur"] / 12
    high = sub["liquid_savings_eur"] >= 3 * monthly
    sub.loc[high, "liquid_savings_eur"] = (rng.uniform(0.25, 2.75, len(sub)) * monthly)[high].round(-1)
    return _rederive_liquidity(sub, rng, cfg, high)


def _ec02(sub, rng, cfg, th):
    sub = _raise_score(sub, rng, cfg, 8)
    rich = sub["liquid_savings_eur"] >= 5000
    sub.loc[rich, "liquid_savings_eur"] = pd.Series(rng.uniform(300, 4900, len(sub)), index=sub.index)[
        rich
    ].round(-1)
    return _rederive_liquidity(sub, rng, cfg, rich)


def _ec03(sub, rng, cfg, th):
    s = wp.schema(cfg)
    goal = sub["primary_investment_goal"]
    regoal = ~goal.isin(["Wealth Growth", "Retirement"])
    sub.loc[regoal, "primary_investment_goal"] = rng.choice(["Wealth Growth", "Retirement"], regoal.sum())
    redraw = regoal | (sub["investment_horizon_years"] > 9)
    retire = sub["primary_investment_goal"] == "Retirement"
    years = np.where(
        retire,
        np.maximum(1, s["retirement_age"] - sub["edad"]) + rng.normal(0, 1, len(sub)),
        rng.uniform(2, 9, len(sub)),
    )
    sub.loc[redraw, "investment_horizon_years"] = (
        pd.Series(years, index=sub.index)[redraw].clip(0.5, 9).round(1)
    )
    sub = _rebucket(sub, cfg)
    sub = _raise_score(sub, rng, cfg, 7)
    return _rederive_liquidity(sub, rng, cfg, redraw)


def _ec06(sub, rng, cfg, th):
    s = wp.schema(cfg)
    regoal = sub["primary_investment_goal"] != "Income"
    sub.loc[regoal, "primary_investment_goal"] = "Income"
    cap = (s["life_horizon_cap_age"] - sub["edad"]).clip(lower=0.5)
    years = pd.Series(rng.triangular(2, 5, 12, len(sub)), index=sub.index).clip(upper=cap).round(1)
    sub.loc[regoal, "investment_horizon_years"] = years[regoal]
    sub = _rebucket(sub, cfg)
    low = sub["liquidity_need_pct"] < 50
    sub.loc[low, "liquidity_need_pct"] = pd.Series(rng.uniform(50, 90, len(sub)), index=sub.index)[
        low
    ].round()
    return sub


def _ec08(sub, rng, cfg, th):
    sub = _raise_score(sub, rng, cfg, 9, redraw_drawdown=False)
    sub["max_acceptable_drawdown_pct"] = rng.integers(5, 11, len(sub)).astype(float)  # contradicts the band
    return sub


def _ec09(sub, rng, cfg, th):
    p90 = th["home_value_p90"]
    low = sub["main_residence_value_eur"] < p90
    sub.loc[low, "main_residence_value_eur"] = pd.Series(
        rng.uniform(p90, 2.5 * p90, len(sub)), index=sub.index
    )[low].round(-3)
    return sub


RECIPES = {  # id -> (candidate filter, mutation). Filters may use the profile's thresholds.
    "EC01": ("net_income_annual_eur >= {income_p90}", _ec01),
    "EC02": (
        "edad >= 18 and edad <= 30 and investable_surplus_eur < 2400 and investment_horizon_years >= 7",
        _ec02,
    ),
    "EC03": ("edad >= 58 and edad <= 66", _ec03),
    "EC04": ("raw_surplus_eur < 0", lambda sub, rng, cfg, th: _raise_score(sub, rng, cfg, 5)),
    "EC05": (
        "debt_service_to_income >= {dsti_high}",
        lambda sub, rng, cfg, th: _raise_score(sub, rng, cfg, 7),
    ),
    "EC06": ("edad >= 67", _ec06),
    "EC07": (
        "investment_horizon_bucket == 'Short'",
        lambda sub, rng, cfg, th: _raise_score(sub, rng, cfg, 7),
    ),
    "EC08": ("edad >= 0", _ec08),
    "EC09": ("edad >= 60 and investable_surplus_eur == 0 and main_residence_value_eur > 0", _ec09),
}


# --------------------------------------------------------------------------- #
# Cohort construction
# --------------------------------------------------------------------------- #
def build_cohort(
    df: pd.DataFrame, cfg: dict, rng: np.random.Generator, total_share: float | None = None
) -> tuple[pd.DataFrame, dict, dict]:
    """Assign the stress-test cohort; returns (df, thresholds, per-profile fill stats)."""
    ec = spec(cfg)
    profiles = ec["profiles"]
    missing = {p["id"] for p in profiles} - set(RECIPES)
    if missing:
        raise KeyError(f"Edge-case profiles without an injection recipe: {sorted(missing)}")
    scale = 1.0 if total_share is None else total_share / ec["total_target_share"]
    n = len(df)
    df = df.copy()
    thresholds = resolve_thresholds(df, cfg)
    natural = predicate_masks(df, cfg, thresholds)
    any_natural = natural.any(axis=1)

    primary = pd.Series(pd.NA, index=df.index, dtype="string")
    origin = pd.Series("none", index=df.index, dtype="string")
    mutated = pd.Series(pd.NA, index=df.index, dtype="string")
    quotas = {p["id"]: int(round(p["target_share"] * scale * n)) for p in profiles}
    order = sorted(profiles, key=lambda p: natural[p["id"]].sum() / max(quotas[p["id"]], 1))
    max_attempts = ec["injection"]["max_rejection_attempts"]
    stats = {}

    for p in order:
        pid, quota, th = p["id"], quotas[p["id"]], thresholds[p["id"]]
        free = primary.isna()
        pool = natural.index[natural[pid] & free].to_numpy()
        take = rng.choice(pool, size=min(quota, len(pool)), replace=False) if quota else pool[:0]
        primary[take], origin[take] = pid, "natural"
        need = quota - len(take)

        hard = [r for r in consistency.CR_IDS if r not in p["relaxed_rules"]]
        candidate_expr, mutate = RECIPES[pid]
        predicate = p["predicate"].format(**th)
        injected, attempts = 0, 0
        while need > 0 and attempts < max_attempts:
            attempts += 1
            eligible = primary.isna() & wp.evaluate(df, candidate_expr.format(**th))
            if not eligible.any():
                break
            preferred = df.index[eligible & ~any_natural].to_numpy()  # keep other natural matches intact
            fallback = df.index[eligible & any_natural].to_numpy()
            size = min(int(eligible.sum()), max(2 * need, 10))
            batch = np.concatenate([rng.permutation(preferred), rng.permutation(fallback)])[:size]
            sub = wp.compute_capacity(mutate(df.loc[batch].copy(), rng, cfg, th), cfg)
            ok = wp.evaluate(sub, predicate) & ~consistency.rule_masks(sub, cfg)[hard].any(axis=1)
            accepted = sub.index[ok][:need]
            if len(accepted):
                before, after = df.loc[accepted, MUTABLE], sub.loc[accepted, MUTABLE]
                changed = (before.astype("string") != after.astype("string")).fillna(True) & ~(
                    before.isna() & after.isna()
                )
                mutated[accepted] = changed.apply(lambda r: "|".join(c for c in MUTABLE if r[c]), axis=1)
                df.loc[accepted, sub.columns] = sub.loc[accepted, sub.columns]
                primary[accepted], origin[accepted] = pid, "injected"
                need -= len(accepted)
                injected += len(accepted)
        if need > 0:
            log.warning(
                "%s: shortfall %d of quota %d (natural %d, injected %d, %d attempts)",
                pid,
                need,
                quota,
                len(take),
                injected,
                attempts,
            )
        stats[pid] = {
            "quota": quota,
            "natural_available": int(natural[pid].sum()),
            "natural_selected": int(len(take)),
            "injected": injected,
            "shortfall": int(need),
            "rejection_attempts": attempts,
        }

    tags = ec["tag_columns"]
    by_id = {p["id"]: p for p in profiles}
    df[tags["primary"]] = primary
    df[tags["origin"]] = origin
    df[tags["flag"]] = primary.notna()
    df["edge_case_name"] = primary.map({k: v["name"] for k, v in by_id.items()}).astype("string")
    df[tags["behaviour"]] = primary.map(
        {k: v["expected_advisory_outcome"]["recommendation"] for k, v in by_id.items()}
    ).astype("string")
    df["expected_capacity_override"] = primary.map(
        {k: v["expected_advisory_outcome"]["capacity_override"] for k, v in by_id.items()}
    ).astype("boolean")
    df["edge_case_capacity_cap"] = primary.map({k: v["capacity_cap"] for k, v in by_id.items()}).astype(
        "string"
    )
    df["edge_case_assertion"] = primary.map(
        {k: v["expected_advisory_outcome"]["test_assertion"] for k, v in by_id.items()}
    ).astype("string")
    df["edge_case_mutated_fields"] = mutated
    return df, thresholds, stats


def exemptions(df: pd.DataFrame, cfg: dict) -> dict:
    """Rule id -> mask of cohort rows allowed to violate it (profile `relaxed_rules`)."""
    primary = df[spec(cfg)["tag_columns"]["primary"]]
    exempt: dict = {}
    for p in spec(cfg)["profiles"]:
        for rule in p["relaxed_rules"]:
            exempt[rule] = exempt.get(rule, pd.Series(False, index=df.index)) | primary.eq(p["id"]).fillna(
                False
            )
    return exempt


def label_and_validate(
    df: pd.DataFrame, cfg: dict, thresholds: dict, fill_stats: dict, acceptance: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Final tags (matches, assertion results) and the per-profile report. Run after enforcement."""
    ec = spec(cfg)
    tags = ec["tag_columns"]
    n = len(df)
    matches = predicate_masks(df, cfg, thresholds)
    df[tags["matches"]] = consistency.violation_labels(matches, list(matches.columns))
    primary = df[tags["primary"]]
    passed = pd.Series(pd.NA, index=df.index, dtype="boolean")
    profiles_report = {}
    for p in ec["profiles"]:
        pid = p["id"]
        rows = primary.eq(pid).fillna(False)
        if rows.any():
            passed[rows] = wp.evaluate(df.loc[rows], p["expected_advisory_outcome"]["test_assertion"])
        cap = p["capacity_cap"]
        caps = df.loc[rows, "capacity_caps_applied"].fillna("")
        stats = fill_stats.get(pid, {})
        profiles_report[pid] = {
            "name": p["name"],
            "target_share_pct": round(100 * stats.get("quota", 0) / max(n, 1), 3),
            "rows": int(rows.sum()),
            "share_pct": round(100 * rows.mean(), 3),
            "natural_prevalence_pct": round(100 * stats.get("natural_available", 0) / max(n, 1), 3),
            "natural_selected": stats.get("natural_selected", 0),
            "injected": stats.get("injected", 0),
            "shortfall": stats.get("shortfall", 0),
            "predicate_holds_pct": _pct(matches.loc[rows, pid]),
            "assertion": p["expected_advisory_outcome"]["test_assertion"],
            "assertion_pass_pct": _pct(passed[rows]),
            "capacity_cap": cap,
            "cap_fired_pct": _pct(caps.str.contains(cap)) if cap else None,
            "relaxed_rules": p["relaxed_rules"],
        }
    df["edge_case_assertion_pass"] = passed
    cohort = primary.notna()
    report = {
        "target_total_share_pct": round(100 * sum(s["quota"] for s in fill_stats.values()) / max(n, 1), 3),
        "achieved_total_share_pct": round(100 * cohort.mean(), 3),
        "rows": int(cohort.sum()),
        "origin_counts": df.loc[cohort, tags["origin"]].value_counts().to_dict(),
        "thresholds": thresholds,
        "profiles": profiles_report,
    }
    if acceptance:
        report["acceptance"] = _acceptance(profiles_report, acceptance)
    return df, report


def _pct(mask: pd.Series):
    return None if len(mask) == 0 else round(100 * float(mask.astype(bool).mean()), 2)


def _acceptance(profiles: dict, rules: dict) -> dict:
    """model_config.yaml evaluation_config.thresholds.edge_cases applied per profile."""
    out = {}
    for pid, r in profiles.items():
        checks = {
            "share_within_tolerance": abs(r["share_pct"] - r["target_share_pct"])
            <= rules["share_tolerance_pp"],
            "min_rows": r["rows"] >= rules["min_rows_per_profile"],
            "assertions": (r["assertion_pass_pct"] or 0) >= 100 * rules["assertion_pass_rate_min"],
        }
        out[pid] = {**checks, "pass": all(checks.values())}
    return out
