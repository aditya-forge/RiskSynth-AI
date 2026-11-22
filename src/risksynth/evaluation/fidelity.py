# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Statistical fidelity and MiFID II / consistency compliance for RiskSynth AI datasets.

Population scope (model_config.yaml evaluation_config.population_scope):
  fidelity tests     base population only: rows with edge_case_origin == "injected" are excluded
  constraint tests   all rows (exempt edge-case traits reported separately)

Reference data: the INE inputs the generator learned from.
  EPF observed households (stage-1 table) for household finances, and for demographics when
  the dataset was generated from household heads (Sampler / SDV); the INE combined base
  (ECV+EPF+EPA persons) for demographics when the rule-based generator drew from it.

Thresholds come from model_config.yaml (evaluation_config.thresholds, statistical_constraints,
dependency_constraints). Nothing here needs SDV, loguru or seaborn.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.spatial.distance import jensenshannon

from risksynth.engine import consistency, edge_cases
from risksynth.engine import wealth_profile as wp
from risksynth.paths import MODEL_CONFIG, PROCESSED_DIR, REPORTS_DIR

FINANCE_COLUMNS = ["net_income_annual_eur", "household_expenses_annual_eur", "debt_service_annual_eur",
                   "household_surplus_eur", "main_residence_value_eur"]
DEMOGRAPHIC_NUMERIC = ["edad", "n_miembros", "dependent_children"]
CATEGORICAL = ["nivel_estudios", "situacion_laboral", "sexo"]
WEALTH_FEATURES = ["edad", "net_income_annual_eur", "raw_surplus_eur", "liquid_savings_eur",
                   "risk_tolerance_score", "investment_horizon_years", "max_acceptable_drawdown_pct",
                   "liquidity_need_pct", "risk_capacity_score", "effective_risk_score"]


def load_model_config() -> dict:
    cfg: dict = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    return cfg


def _with_surplus(df: pd.DataFrame) -> pd.DataFrame:
    """Household surplus on the same basis in reference and synthetic: income - spending - EPF debt."""
    df = df.copy()
    if "household_expenses_annual_eur" not in df and "gasto_monetario" in df:
        df["household_expenses_annual_eur"] = df["gasto_monetario"]
    if {"net_income_annual_eur", "household_expenses_annual_eur"} <= set(df):
        df["household_surplus_eur"] = (df["net_income_annual_eur"] - df["household_expenses_annual_eur"]
                                       - df.get("debt_service_annual_eur", 0).fillna(0))
    return df


def reference_tables(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(finance_reference, demographic_reference) for 'households' or 'persons'."""
    households = _with_surplus(wp.build_stage1_table())
    if kind == "persons":
        persons = pd.read_parquet(PROCESSED_DIR / "combined_base.parquet")
        return households, persons
    return households, households


def reference_kind(df: pd.DataFrame) -> str:
    """Rows drawn from the combined persons base carry fuente_base (rule-based generator) or
    fuente (stage-2 dry runs on combined_base); Sampler/SDV outputs are household heads."""
    return "persons" if {"fuente_base", "fuente"} & set(df.columns) else "households"


def base_population(df: pd.DataFrame) -> pd.DataFrame:
    if "edge_case_origin" in df:
        return df[df["edge_case_origin"] != "injected"]
    return df


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _grade(norm: float, grades: dict) -> str:
    if norm < grades["excellent"]:
        return "Excellent"
    if norm < grades["good"]:
        return "Good"
    if norm < grades["acceptable"]:
        return "Acceptable"
    return "Poor"


def numeric_tests(ref: pd.DataFrame, syn: pd.DataFrame, columns: list, thr: dict) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in ref or col not in syn:
            continue
        r, s = pd.to_numeric(ref[col], errors="coerce").dropna(), pd.to_numeric(syn[col], errors="coerce").dropna()
        if len(r) < 20 or len(s) < 20:
            continue
        ks = stats.ks_2samp(r, s)
        span = r.quantile(0.99) - r.quantile(0.01)   # robust range: one outlier must not hide gaps
        wd = stats.wasserstein_distance(r, s)
        norm = wd / span if span > 0 else 0.0
        grade = _grade(norm, thr["numerical"]["wasserstein_normalized"])
        passed = ks.statistic <= thr["numerical"]["ks_statistic_max"] and grade != "Poor"
        rows.append({"column": col, "n_reference": len(r), "n_synthetic": len(s),
                     "ks_d": round(ks.statistic, 4), "ks_p_value": float(ks.pvalue),
                     "wasserstein": round(wd, 2), "wasserstein_norm": round(norm, 4),
                     "grade": grade, "status": "PASS" if passed else "FAIL"})
    return pd.DataFrame(rows)


def categorical_tests(ref: pd.DataFrame, syn: pd.DataFrame, columns: list, thr: dict) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in ref or col not in syn:
            continue
        p = ref[col].astype("string").value_counts(normalize=True)
        q = syn[col].astype("string").value_counts(normalize=True)
        cats = p.index.union(q.index)
        p, q = p.reindex(cats, fill_value=0), q.reindex(cats, fill_value=0)
        jsd = float(jensenshannon(p, q, base=2) ** 2)
        max_pp = float((100 * (p - q).abs()).max())
        passed = (jsd <= thr["categorical"]["jensen_shannon_max"]
                  and max_pp <= thr["categorical"]["category_share_tolerance_pp"])
        rows.append({"column": col, "categories": len(cats), "jensen_shannon": round(jsd, 4),
                     "max_share_diff_pp": round(max_pp, 2), "status": "PASS" if passed else "FAIL"})
    return pd.DataFrame(rows)


def target_mix_tests(syn: pd.DataFrame, model_cfg: dict) -> pd.DataFrame:
    """Categorical targets from statistical_constraints (risk tier mix, ESG mix)."""
    rows = []
    for c in model_cfg["constraints"]["statistical_constraints"]:
        if c.get("distribution") != "categorical" or c["column"] not in syn or "tolerance_pp" not in c:
            continue
        actual = 100 * syn[c["column"]].astype("string").value_counts(normalize=True)
        for cat, share in c["params"].items():
            diff = actual.get(str(cat), 0.0) - 100 * share
            rows.append({"column": c["column"], "category": str(cat), "target_pct": round(100 * share, 2),
                         "actual_pct": round(float(actual.get(str(cat), 0.0)), 2), "diff_pp": round(float(diff), 2),
                         "status": "PASS" if abs(diff) <= c["tolerance_pp"] else "FAIL"})
    return pd.DataFrame(rows)


def correlation_tests(ref: pd.DataFrame, syn: pd.DataFrame, columns: list, thr: dict) -> dict:
    cols = [c for c in columns if c in ref and c in syn
            and pd.to_numeric(ref[c], errors="coerce").notna().sum() > 20]
    r = ref[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    s = syn[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    diff = (s - r)
    off = ~np.eye(len(cols), dtype=bool)
    mean_abs = float(diff.abs().to_numpy()[off].mean()) if len(cols) > 1 else 0.0
    max_abs = float(diff.abs().to_numpy()[off].max()) if len(cols) > 1 else 0.0
    t = thr["correlation"]
    return {"columns": cols, "reference": r, "synthetic": s, "difference": diff,
            "mean_abs_diff": round(mean_abs, 4), "max_abs_diff": round(max_abs, 4),
            "status": "PASS" if mean_abs <= t["mean_abs_spearman_diff_max"]
            and max_abs <= t["max_abs_spearman_diff_max"] else "FAIL"}


def dependency_tests(syn: pd.DataFrame, model_cfg: dict) -> pd.DataFrame:
    rows = []
    for c in model_cfg["constraints"].get("dependency_constraints", []):
        a, b = c["pair"]
        if a not in syn or b not in syn:
            continue
        rho = float(syn[a].corr(syn[b], method="spearman"))
        ok = rho >= c.get("min_rho", -1) and rho <= c.get("max_rho", 1)
        bounds = " ".join(f"{k}={v}" for k, v in c.items() if "rho" in k)
        rows.append({"pair": f"{a} ~ {b}", "expected": c["expected"], "bounds": bounds,
                     "spearman_rho": round(rho, 3), "status": "PASS" if ok else "FAIL"})
    return pd.DataFrame(rows)


def compliance_tests(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    """CR01-CR07 and MiFID II checks on all rows; intentional edge-case breaks counted apart."""
    masks = consistency.rule_masks(df, data_cfg)
    exempt = edge_cases.exemptions(df, data_cfg) if "edge_case_id" in df else {}
    rules = {r["id"]: r["rule"] for r in data_cfg["wealth_profile_schema"]["consistency_rules"]}
    rules.update({"MIFID_EFFECTIVE_RISK": "effective_risk_score == min(tolerance, capacity)",
                  "MIFID_CAPACITY_CURRENT": "risk_capacity_score matches a fresh recomputation",
                  "MIFID_ESG_COLLECTED": "sustainability preference collected"})
    rows = []
    for rule in masks.columns:
        ex = exempt.get(rule, pd.Series(False, index=df.index))
        violated = int(masks[rule].sum())
        exempted = int((masks[rule] & ex).sum())
        unexpected = violated - exempted
        rows.append({"rule": rule, "definition": rules.get(rule, ""), "violations": violated,
                     "exempted": exempted, "unexpected": unexpected,
                     "status": "PASS" if unexpected == 0 else "FAIL"})
    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame, data_cfg: dict | None = None, model_cfg: dict | None = None,
            kind: str | None = None) -> dict:
    data_cfg = data_cfg or wp.load_config()
    model_cfg = model_cfg or load_model_config()
    thr = model_cfg["evaluation_config"]["thresholds"]
    kind = kind or reference_kind(df)
    fin_ref, demo_ref = reference_tables(kind)
    base = _with_surplus(base_population(df))
    numeric = pd.concat([numeric_tests(fin_ref, base, FINANCE_COLUMNS, thr),
                         numeric_tests(demo_ref, base, DEMOGRAPHIC_NUMERIC, thr)], ignore_index=True)
    corr_cols = [c for c in FINANCE_COLUMNS + ["edad", "n_miembros"] if c in base]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_kind": kind,
        "rows_total": len(df),
        "rows_base_population": len(base),
        "rows_injected_excluded": len(df) - len(base),
        "numeric": numeric,
        "categorical": categorical_tests(demo_ref, base, CATEGORICAL, thr),
        "target_mix": target_mix_tests(base, model_cfg),
        "correlation": correlation_tests(fin_ref, base, corr_cols, thr),
        "wealth_correlation": base[[c for c in WEALTH_FEATURES if c in base]].apply(
            pd.to_numeric, errors="coerce").corr(method="spearman"),
        "dependencies": dependency_tests(base, model_cfg),
        "compliance": compliance_tests(df, data_cfg),
    }


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _cell(value) -> tuple[str, str]:
    """(text, css class) for one table cell: numbers right-aligned, PASS/FAIL coloured."""
    if isinstance(value, (bool, np.bool_)):
        return ("yes" if value else "no"), ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}", "num"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        text = f"{v:.2e}" if v != 0 and abs(v) < 1e-3 else f"{v:,.4g}"
        return text, "num"
    text = str(value)
    if text in ("PASS", "FAIL"):
        return text, text.lower()
    return text, ""


def _table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>No data.</p>"
    head = []
    for c in df.columns:
        numeric = pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        head.append(f"<th class='{'num' if numeric else ''}'>{html.escape(str(c).replace('_', ' '))}</th>")
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            text, cls = _cell(row[c])
            cells.append(f"<td class='{cls}'>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='tbl'><table><thead><tr>{''.join(head)}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _passed(frame: pd.DataFrame) -> str:
    return f"{int((frame['status'] == 'PASS').sum())} / {len(frame)}" if len(frame) else "n/a"


REPORT_CSS = """
:root { --bg:#ffffff; --panel:#f6f8fa; --ink:#1f2328; --ink2:#57606a; --line:#d8dee4; --rule:#d0d7de;
        --accent:#1e88e5; --good:#1a7f37; --bad:#cf222e; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#050608; --panel:#0d1016; --ink:#d4dbe5; --ink2:#8b95a5; --line:#1a1f29; --rule:#232936;
          --accent:#58a6ff; --good:#3fb950; --bad:#f85149; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width:1120px; margin:0 auto; padding:40px 24px 56px; }
.eyebrow { font-size:11.5px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); }
h1 { font-size:26px; font-weight:600; letter-spacing:-.01em; margin:8px 0 6px; }
h2 { font-size:15px; font-weight:600; margin:36px 0 4px; }
.sub, .muted { color:var(--ink2); }
.sub { font-size:13px; margin:0 0 12px; }
.meta { color:var(--ink2); font-size:13px; margin:0 0 24px; }
.summary { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); border:1px solid var(--rule);
           border-radius:2px; overflow:hidden; gap:1px; background:var(--line); }
.summary div { background:var(--panel); padding:14px 16px; }
.summary .l { font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--ink2); }
.summary .v { font:500 22px/1.3 ui-monospace, "JetBrains Mono", Consolas, monospace; margin-top:4px; }
.tbl { border:1px solid var(--rule); border-radius:2px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-weight:600; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink2);
     background:var(--panel); padding:9px 12px; border-bottom:1px solid var(--rule); white-space:nowrap; }
td { padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
tbody tr:last-child td { border-bottom:none; }
.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.pass, td.fail { font-weight:600; font-size:12px; letter-spacing:.04em; }
td.pass { color:var(--good); } td.fail { color:var(--bad); }
.note { border-left:2px solid var(--rule); padding:2px 0 2px 12px; color:var(--ink2); font-size:13px; margin:28px 0 0; }
footer { margin-top:36px; padding-top:14px; border-top:1px solid var(--line); color:var(--ink2); font-size:12px; }
"""


def html_report(results: dict, dataset_name: str) -> str:
    corr = results["correlation"]
    corr_tbl = corr["difference"].round(3).reset_index().rename(columns={"index": "column"})
    comp = results["compliance"]
    summary = [("Numeric tests", _passed(results["numeric"])), ("Categorical tests", _passed(results["categorical"])),
               ("Target mixes", _passed(results["target_mix"])), ("Correlation drift", f"{corr['mean_abs_diff']:.3f}"),
               ("Unexpected violations", f"{int(comp['unexpected'].sum()):,}"),
               ("Intentional exemptions", f"{int(comp['exempted'].sum()):,}")]
    sections = [
        ("Numeric fidelity", "Kolmogorov-Smirnov D and Wasserstein distance per column, against INE.", results["numeric"]),
        ("Categorical fidelity", "Jensen-Shannon divergence and the largest share difference per column.",
         results["categorical"]),
        ("Target mixes", "Risk tier and ESG preference shares against the configured targets.", results["target_mix"]),
        ("Correlation difference", f"Spearman rho, synthetic minus INE. Mean |diff| {corr['mean_abs_diff']}, "
                                   f"max {corr['max_abs_diff']}: {corr['status']}.", corr_tbl),
        ("Expected dependencies", "Sign and size of the relationships the profile engine is designed to produce.",
         results["dependencies"]),
        ("Consistency rules and MiFID II checks", "CR01-CR07 and the MiFID II checks on all rows. Exempted rows are "
                                                  "intentional edge-case traits (EC08).", comp),
    ]
    reference = ("INE persons base (ECV, EPF, EPA) for demographics and EPF households for finances"
                 if results["reference_kind"] == "persons" else "INE EPF households")
    cards = "".join(f"<div><div class='l'>{html.escape(l)}</div><div class='v'>{html.escape(v)}</div></div>"
                    for l, v in summary)
    blocks = "".join(f"<h2>{html.escape(t)}</h2><p class='sub'>{html.escape(d)}</p>{_table(f)}" for t, d, f in sections)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark">
<title>RiskSynth AI evaluation report</title><style>{REPORT_CSS}</style></head><body><main>
<div class="eyebrow">RiskSynth AI &middot; Evaluation report</div>
<h1>{html.escape(dataset_name)}</h1>
<p class="meta">Generated {results['generated_at']} &middot; reference: {reference}<br>
{results['rows_total']:,} rows; fidelity tests on {results['rows_base_population']:,} base-population rows
({results['rows_injected_excluded']:,} injected edge cases excluded); rule checks on all rows.</p>
<div class="summary">{cards}</div>
{blocks}
<p class="note">Parameters marked [ASSUMPTION] in the configs (risk-tier mix, ESG mix, goals by age, tolerance model,
liquidity priors, simulator portfolios) are design choices, not measurements. Calibrate them against the
Banco de Espa&ntilde;a EFF 2022 survey before any production use.</p>
<footer>Synthetic data only. Not calibrated to any real client book; not investment advice.<br>
Generated with RiskSynth AI. &copy; 2026 aditya-forge. All rights reserved; not licensed for reuse.</footer>
</main></body></html>"""


def save_report(results: dict, dataset_name: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or REPORTS_DIR / "evaluation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"risksynth_evaluation_{datetime.now():%Y%m%d_%H%M%S}.html"
    path.write_text(html_report(results, dataset_name), encoding="utf-8")
    return path
