# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Stage 2 + 3 orchestrator: resolve inputs -> derive wealth features -> validate (raw)
-> enforce CR01-CR07 + MiFID II -> [stage 3] edge-case cohort EC01-EC09 with per-rule
exemptions -> re-enforce (exempt rows keep their relaxed rules) -> validate -> report.

Library use (from the generators):
    from risksynth.engine.stage2 import run_stage2
    profiles, report = run_stage2(base_rows, seed=1042)

Dry run on the processed INE base (writes only under data/synthetic/):
    python -m risksynth.engine.stage2 --n 20000 --seed 42      (or: risksynth-profiles ...)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import yaml

from risksynth.engine import consistency
from risksynth.engine import edge_cases as ec
from risksynth.engine import wealth_profile as wp
from risksynth.paths import MODEL_CONFIG, SYNTHETIC_DIR

log = logging.getLogger(__name__)
# run_stage2 receives random_seed + stage_2 offset (1000); stage 3 uses random_seed + 2000
STAGE3_SEED_DELTA = 1000

NUMERIC_SUMMARY = [
    "net_income_annual_eur",
    "household_expenses_annual_eur",
    "total_debt_service_annual_eur",
    "raw_surplus_eur",
    "investable_surplus_eur",
    "surplus_ratio",
    "debt_service_to_income",
    "liquid_savings_eur",
    "risk_tolerance_score",
    "investment_horizon_years",
    "max_acceptable_drawdown_pct",
    "liquidity_need_pct",
    "risk_capacity_points",
    "risk_capacity_score",
    "effective_risk_score",
]
CATEGORICAL_SUMMARY = [
    "household_finance_source",
    "liquid_savings_source",
    "risk_tolerance_category",
    "investment_horizon_bucket",
    "primary_investment_goal",
    "esg_preference_level",
    "employment_stability",
]


def run_stage2(
    df: pd.DataFrame,
    seed: int,
    cfg: dict | None = None,
    donor_pool: pd.DataFrame | None = None,
    enforce: bool = True,
    exempt: dict | None = None,
    edge_cases: bool | None = None,
    edge_case_share: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Derive wealth features (stage 2) and the edge-case cohort (stage 3); return (profiles, report).

    `cr_violations_raw` keeps, per row, the rules violated by the raw draws and
    `cr_repaired` the rules that were repaired, so raw draws stay distinguishable
    from suitability-aligned profiles. `profile_inconsistency_flag` is True when a row
    still breaks a rule after enforcement, which only exempt edge cases (EC08) may do.

    edge_cases: None = data_config edge_case_profiles.enabled; edge_case_share overrides
    the 8% total (per-profile targets are scaled proportionally).
    """
    cfg = cfg or wp.load_config()
    rng = np.random.default_rng(seed)
    df = wp.resolve_inputs(df, cfg, donor_pool, rng)
    df = wp.derive_features(df, cfg, rng)

    raw_masks, raw_counts = consistency.validate(df, cfg, exempt, stage="raw")
    df["cr_violations_raw"] = consistency.violation_labels(raw_masks)
    repairs = {}
    if enforce:
        df, repairs = consistency.enforce(df, cfg, rng, exempt)

    use_edge_cases = cfg["edge_case_profiles"]["enabled"] if edge_cases is None else edge_cases
    ec_report, repairs_after_ec, exempt = None, {}, dict(exempt or {})
    if use_edge_cases:
        rng3 = np.random.default_rng(seed + STAGE3_SEED_DELTA)
        df, thresholds, fill_stats = ec.build_cohort(df, cfg, rng3, edge_case_share)
        for rule, mask in ec.exemptions(df, cfg).items():
            exempt[rule] = exempt[rule] | mask if rule in exempt else mask
        if enforce:  # anything a non-relaxed rule rejects is still repaired, cohort rows included
            df, repairs_after_ec = consistency.enforce(df, cfg, rng3, exempt)
        else:
            df = wp.compute_capacity(df, cfg)

    final_masks, final_counts = consistency.validate(df, cfg, exempt, stage="enforced" if enforce else "raw")
    df["profile_inconsistency_flag"] = rule_violated = consistency.rule_masks(df, cfg)[
        consistency.CR_IDS
    ].any(axis=1)
    if use_edge_cases:
        df, ec_report = ec.label_and_validate(df, cfg, thresholds, fill_stats, _acceptance_rules())
        log.info(
            "Edge cases: %.2f%% of rows (target %.2f%%), origins %s",
            ec_report["achieved_total_share_pct"],
            ec_report["target_total_share_pct"],
            ec_report["origin_counts"],
        )
    report = {
        "rows": len(df),
        "seed": seed,
        "enforced": enforce,
        "income_calibration_factor": wp.schema(cfg)["features"]["investable_surplus_eur"][
            "income_calibration_factor"
        ],
        "violations_raw": raw_counts,
        "repairs_applied": repairs,
        "repairs_after_edge_cases": repairs_after_ec,
        "violations_enforced": final_counts,
        "violations_exempted": consistency.exempted_counts(df, cfg, exempt),
        "rows_with_raw_violation": int(df["cr_violations_raw"].notna().sum()),
        "rows_inconsistent_after": int(rule_violated.sum()),
        "edge_cases": ec_report,
        "summary": summarize(df),
    }
    return df, report


def _acceptance_rules() -> dict | None:
    """model_config.yaml evaluation_config.thresholds.edge_cases (None if unavailable)."""
    try:
        cfg = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
        rules: dict = cfg["evaluation_config"]["thresholds"]["edge_cases"]
        return rules
    except (OSError, KeyError, TypeError):
        return None


def summarize(df: pd.DataFrame) -> dict:
    out = {"numeric": {}, "categorical": {}}
    for col in (c for c in NUMERIC_SUMMARY if c in df):
        s = pd.to_numeric(df[col], errors="coerce")
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        out["numeric"][col] = {
            "mean": _r(s.mean()),
            "p05": _r(q[0.05]),
            "p25": _r(q[0.25]),
            "p50": _r(q[0.5]),
            "p75": _r(q[0.75]),
            "p95": _r(q[0.95]),
            "null_pct": _r(100 * s.isna().mean()),
        }
    for col in (c for c in CATEGORICAL_SUMMARY if c in df):
        out["categorical"][col] = {
            str(k): _r(v) for k, v in (100 * df[col].value_counts(normalize=True)).items()
        }
    out["share_raw_surplus_negative_pct"] = _r(100 * (df["raw_surplus_eur"] < 0).mean())
    out["share_investable_zero_pct"] = _r(100 * (df["investable_surplus_eur"] == 0).mean())
    out["capacity_caps_pct"] = {
        cap: _r(100 * df["capacity_caps_applied"].fillna("").str.contains(cap).mean())
        for cap in [
            "CAP_NO_SURPLUS",
            "CAP_HIGH_DSTI",
            "CAP_NO_BUFFER",
            "CAP_SHORT_HORIZON",
            "CAP_HIGH_LIQUIDITY",
            "CAP_NEAR_RETIREMENT",
        ]
    }
    out["effective_below_tolerance_pct"] = _r(
        100 * (df["effective_risk_score"] < df["risk_tolerance_score"]).mean()
    )
    return out


def _r(x):
    return None if pd.isna(x) else round(float(x), 3)


def write_outputs(df: pd.DataFrame, report: dict, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"profiles": out_dir / f"{stem}.parquet", "report": out_dir / f"{stem}_report.json"}
    df.to_parquet(paths["profiles"], index=False)
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 wealth feature dry run")
    parser.add_argument("--input", type=Path, default=wp.PROCESSED_DIR / "combined_base.parquet")
    parser.add_argument("--n", type=int, default=None, help="sample size (default: all rows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-enforce", action="store_true")
    parser.add_argument("--no-edge-cases", action="store_true")
    parser.add_argument(
        "--edge-case-share",
        type=float,
        default=None,
        help="override the total edge-case share (default 0.08 from config)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    base = pd.read_parquet(args.input)
    if args.n:
        base = base.sample(n=min(args.n, len(base)), random_state=args.seed).reset_index(drop=True)
    profiles, report = run_stage2(
        base,
        seed=args.seed,
        enforce=not args.no_enforce,
        edge_cases=False if args.no_edge_cases else None,
        edge_case_share=args.edge_case_share,
    )
    report["input"] = str(args.input)
    out = args.out or SYNTHETIC_DIR / f"wealth_dryrun_{datetime.now():%Y%m%d_%H%M%S}"
    paths = write_outputs(profiles, report, out, "wealth_profiles")
    log.info("Wrote %s", paths)


if __name__ == "__main__":
    main()
