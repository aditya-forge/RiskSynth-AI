"""
Stage 2 of the RiskSynth AI pipeline: wealth-management feature engine.

Everything is driven by configs/data_config.yaml:
  wealth_profile_schema.features             the 7 robo-advisory features
  wealth_profile_schema.supporting_features  surplus, liquid savings, risk capacity
  wealth_profile_schema.input_resolution     where each row's household finances come from

Pipeline order (each step only uses what precedes it):
  resolve_inputs -> surplus -> liquid savings -> goal -> horizon -> risk tolerance
  -> drawdown -> liquidity need -> ESG -> risk capacity / effective risk

Consistency rules and MiFID II checks live in consistency.py; stage2.py orchestrates.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from risksynth.paths import DATA_CONFIG, PROCESSED_DIR

log = logging.getLogger(__name__)

HOUSEHOLD_COLUMNS_FROM_EPF = {        # engine name <- epf_processed column
    "edad": "edad", "sexo": "sexo", "ccaa": "ccaa", "nivel_estudios": "nivel_estudios",
    "situacion_laboral": "situacion_laboral", "contract_type": "contract_type",
    "n_miembros": "n_miembros", "dependent_children": "dependent_children",
    "net_income_annual_eur": "net_income_annual_eur",
    "household_expenses_annual_eur": "gasto_monetario",
    "debt_service_annual_eur": "debt_service_annual_eur",
    "main_residence_value_eur": "main_residence_value_eur", "peso": "peso",
}


def load_config(path: Path = DATA_CONFIG) -> dict:
    cfg: dict = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg


def schema(cfg: dict) -> dict:
    return cfg["wealth_profile_schema"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame, expr: str, constants: dict | None = None) -> pd.Series:
    """Evaluate a config condition such as "edad >= 55 and investment_horizon_years < 10"."""
    result = pd.eval(expr, resolvers=[df, constants or {}], engine="python")
    return pd.Series(result, index=df.index).fillna(False).astype(bool)


def interval_mask(values: pd.Series, spec: str) -> pd.Series:
    """Mask for config intervals: "<0", ">=0.25", "[0,0.10)", "(1,2]"."""
    spec = spec.strip()
    for op in ("<=", ">=", "<", ">"):
        if spec.startswith(op):
            bound = float(spec[len(op):])
            return {"<=": values <= bound, ">=": values >= bound,
                    "<": values < bound, ">": values > bound}[op]
    lo, hi = (float(x) for x in spec[1:-1].split(","))
    lower = values >= lo if spec[0] == "[" else values > lo
    upper = values <= hi if spec[-1] == "]" else values < hi
    return lower & upper


def points(values: pd.Series, mapping: dict) -> pd.Series:
    """Score points from an interval mapping or a categorical mapping (NaN -> 0 points)."""
    keys = list(mapping)
    if all(str(k).strip()[0] in "<>[(" for k in keys):
        out = pd.Series(0.0, index=values.index)
        for key, pts in mapping.items():
            out[interval_mask(values, key).fillna(False)] = pts
        return out
    return values.map(mapping).astype("float").fillna(0.0)


def category_from_score(score: pd.Series, categories: dict) -> pd.Series:
    out = pd.Series(pd.NA, index=score.index, dtype="string")
    for name, rng in categories.items():
        out[score.between(rng["min"], rng["max"])] = name
    return out


def bucket_from_years(years: pd.Series, buckets: dict) -> pd.Series:
    """Buckets are [min, max) with max=None meaning open-ended."""
    out = pd.Series(pd.NA, index=years.index, dtype="string")
    for name, rng in buckets.items():
        upper = np.inf if rng["max"] is None else rng["max"]
        out[(years >= rng["min"]) & (years < upper)] = name
    return out


def age_band(edad: pd.Series, labels: list[str]) -> pd.Series:
    """Map ages to band labels like "18-34" / "65+".

    Bands are contiguous: each runs from its lower bound up to the next band's lower bound,
    so fractional ages (34.5) never fall between bands; ages below the first band map to it.
    """
    lowers = sorted((float(label.replace("+", "-").partition("-")[0]), label) for label in labels)
    bounds = np.array([lo for lo, _ in lowers])
    idx = (np.searchsorted(bounds, edad.to_numpy(dtype=float), side="right") - 1).clip(min=0)
    return pd.Series(np.array([lab for _, lab in lowers], dtype=object)[idx],
                     index=edad.index, dtype="string")


def quantile_assign(latent: pd.Series, target: dict) -> tuple[pd.Series, pd.Series]:
    """Rank-map a latent score onto a target categorical distribution.

    Returns the category and the row's relative position (0-1) inside its category.
    """
    pct = (latent.rank(method="first") - 0.5) / len(latent)
    names = list(target)
    upper = np.cumsum([target[n] for n in names])
    upper[-1] = 1.0
    idx = np.searchsorted(upper, pct.to_numpy(), side="right").clip(max=len(names) - 1)
    lower = np.concatenate([[0.0], upper[:-1]])
    within = (pct.to_numpy() - lower[idx]) / (upper[idx] - lower[idx])
    return (pd.Series(np.array(names, dtype=object)[idx], index=latent.index, dtype="string"),
            pd.Series(within.clip(0, 1 - 1e-9), index=latent.index))


def employment_group(df: pd.DataFrame, cap_cfg: dict) -> pd.Series:
    """Harmonize ECV/EPF/EPA statuses; EPA 'inactivo' at retirement age counts as jubilado."""
    status = df["situacion_laboral"].astype("string")
    retired_age = cap_cfg["inactive_retired_age"]
    group = status.mask((status == "inactivo") & (df["edad"] >= retired_age), "jubilado")
    return group.fillna("otro_inactivo")


# --------------------------------------------------------------------------- #
# Stage-1 training table and donor pool
# --------------------------------------------------------------------------- #
def build_stage1_table(processed_dir: Path | None = None) -> pd.DataFrame:
    """EPF household heads with the stage-1 columns (SDV training table and donor pool)."""
    processed_dir = processed_dir or PROCESSED_DIR
    epf = pd.read_parquet(processed_dir / "epf_processed.parquet")
    table = pd.DataFrame({k: epf[v] for k, v in HOUSEHOLD_COLUMNS_FROM_EPF.items()})
    before = len(table)
    table = table[table["net_income_annual_eur"].notna()].reset_index(drop=True)
    log.info("Stage-1 table: %d EPF households (%d dropped without income)", len(table),
             before - len(table))
    return table


def resolve_inputs(df: pd.DataFrame, cfg: dict, donor_pool: pd.DataFrame | None,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Give every row a coherent household-finance block.

    Rows that already carry the block (EPF, SDV samples, earlier resolution) keep it.
    Rows without it (ECV, EPA) receive the block of a weighted random EPF household from
    the same age band and employment group (hot-deck). Observed HY020 (renta_disponible)
    is kept as income when present; it is never imputed.
    """
    s = schema(cfg)
    res = s["input_resolution"]
    cap_cfg = s["supporting_features"]["risk_capacity_score"]
    df = df.copy()
    for target, aliases in res["source_aliases"].items():
        if target not in df:
            src = next((a for a in aliases if a in df), None)
            df[target] = df[src] if src else np.nan
    if "net_income_annual_eur" not in df:
        df["net_income_annual_eur"] = np.nan
    for col in res["observed_income_priority"][1:]:
        if col in df:
            df["net_income_annual_eur"] = df["net_income_annual_eur"].fillna(df[col])

    block = res["household_finance_block"]
    for col in block:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df["contract_type"] = (df["contract_type"] if "contract_type" in df else pd.NA)
    df["contract_type"] = df["contract_type"].astype("string")
    df["employment_group"] = employment_group(df, cap_cfg)
    df["age_band_10"] = _age_band_10(df["edad"])

    prov = res["donor_imputation"]["provenance_column"]
    needs = df[block[:3]].isna().any(axis=1)          # income, expenses, debt service
    if prov not in df:
        df[prov] = pd.Series("observed", index=df.index, dtype="string")
    df.loc[needs, prov] = "donor_epf"
    if needs.any():
        if donor_pool is None:
            donor_pool = build_stage1_table()
        donors = donor_pool.copy()
        donors["employment_group"] = employment_group(donors, cap_cfg)
        donors["age_band_10"] = _age_band_10(donors["edad"])
        idx = _hot_deck(df.loc[needs], donors, res["donor_imputation"]["cells"],
                        res["donor_imputation"]["weight_column"], rng)
        for col in block + ["contract_type"]:
            values = donors.loc[idx, col].to_numpy()
            if col == "contract_type":
                values = pd.array(values, dtype="string")
            df.loc[needs, col] = df.loc[needs, col].fillna(pd.Series(values, index=df.index[needs]))
        log.info("Household finances: %d observed, %d donor-imputed from %d EPF households",
                 (~needs).sum(), needs.sum(), len(donors))
    return df


def _age_band_10(edad: pd.Series) -> pd.Series:
    """Decade bands for donor matching: 20 = 18-29, ..., 70 = 70+."""
    return (np.maximum(edad.clip(upper=79) // 10 * 10, 20)).astype(int).astype(str)


def _hot_deck(recipients: pd.DataFrame, donors: pd.DataFrame, cell_levels: list,
              weight_col: str, rng: np.random.Generator, min_donors: int = 20) -> np.ndarray:
    """Weighted donor index per recipient, falling back to coarser cells when a cell is thin."""
    chosen = pd.Series(-1, index=recipients.index)
    for cells in cell_levels + [[]]:                  # [] = whole pool, the last resort
        todo = chosen[chosen < 0].index
        if todo.empty:
            break
        groups = recipients.loc[todo].groupby(cells) if cells else [((), recipients.loc[todo])]
        pool_groups = dict(list(donors.groupby(cells))) if cells else {(): donors}
        for key, rec in groups:
            pool = pool_groups.get(key)
            if pool is None or (cells and len(pool) < min_donors):
                continue
            w = pool[weight_col].to_numpy(dtype=float)
            chosen[rec.index] = rng.choice(pool.index.to_numpy(), size=len(rec), p=w / w.sum())
    return chosen.to_numpy()


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
TERMS = {   # names referenced by the config `terms`; `p` is the generation block
    "age_centered_40": lambda d, p: d["edad"] - 40,
    "log_income_rel_median": lambda d, p: np.log(d["net_income_annual_eur"].clip(lower=1)
                                                 / p["income_reference_eur"]),
    "surplus_ratio": lambda d, p: d["surplus_ratio"].clip(-1, 1).fillna(-1),
    "university_education": lambda d, p: d["nivel_estudios"].eq("universitario"),
    "unemployed": lambda d, p: d["employment_group"].eq("desempleado"),
    "retired": lambda d, p: d["employment_group"].eq("jubilado"),
    "dependent_children": lambda d, p: d["dependent_children"].fillna(0).clip(upper=3),
    "goal_wealth_growth": lambda d, p: d["primary_investment_goal"].eq("Wealth Growth"),
    "goal_capital_preservation": lambda d, p: d["primary_investment_goal"].eq("Capital Preservation"),
    "long_horizon": lambda d, p: d["investment_horizon_bucket"].eq("Long"),
}


def latent_score(df: pd.DataFrame, gen: dict, rng: np.random.Generator) -> pd.Series:
    unknown = set(gen["terms"]) - set(TERMS)
    if unknown:
        raise KeyError(f"Config terms without implementation in TERMS: {sorted(unknown)}")
    z = pd.Series(float(gen.get("intercept", 0.0)), index=df.index)
    for name, term in gen["terms"].items():
        z += term["coef"] * TERMS[name](df, gen).astype(float).fillna(0.0)
    return z + rng.normal(0, gen["noise_sd"], len(df))


def derive_surplus(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feat = schema(cfg)["features"]["investable_surplus_eur"]
    income = df["net_income_annual_eur"] * feat["income_calibration_factor"]
    debt = df["debt_service_annual_eur"].fillna(0)
    if "consumer_debt_service_annual_eur" in df:
        debt = debt + df["consumer_debt_service_annual_eur"].fillna(0)
    lo, hi = feat["clip"]
    raw = (income - df["household_expenses_annual_eur"] - debt).round(feat["rounding"]).clip(upper=hi)
    df["total_debt_service_annual_eur"] = debt
    df["raw_surplus_eur"] = raw
    df["investable_surplus_eur"] = raw.clip(lower=lo, upper=hi)
    df["surplus_ratio"] = raw / income.where(income > 0)
    df["debt_service_to_income"] = debt / income.where(income > 0)
    return df


def derive_liquid_savings(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    spec = schema(cfg)["supporting_features"]["liquid_savings_eur"]["fallback_model"]
    if "liquid_savings_eur" not in df:
        df["liquid_savings_eur"] = np.nan
    missing = df["liquid_savings_eur"].isna()
    df["liquid_savings_source"] = pd.Series("generator", index=df.index, dtype="string").mask(
        missing, "assumption_model")
    if missing.any():
        median = points(df.loc[missing, "surplus_ratio"].fillna(-1),
                        spec["median_months_of_expenses_by_surplus_ratio"])
        months = np.exp(np.log(median) + rng.normal(0, spec["sigma"], missing.sum()))
        df.loc[missing, "liquid_savings_eur"] = (
            months * df.loc[missing, "household_expenses_annual_eur"] / 12).round(-1)
    return df


def sample_goal(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.Series:
    s = schema(cfg)
    gen = s["features"]["primary_investment_goal"]["generation"]
    goals = s["features"]["primary_investment_goal"]["values"]
    bands = age_band(df["edad"], list(gen["probabilities"]))
    table = pd.DataFrame(gen["probabilities"]).T[goals]
    probs = table.loc[bands.to_numpy()].set_axis(df.index)
    constants = {"retirement_age": s["retirement_age"]}
    for adj in gen["adjustments"]:
        hit = evaluate(df, adj["when"], constants)
        for goal, factor in adj["multiply"].items():
            probs.loc[hit, goal] *= factor
    probs = probs.div(probs.sum(axis=1), axis=0)
    u = rng.random(len(df))
    pick = (probs.cumsum(axis=1).to_numpy() < u[:, None]).sum(axis=1).clip(max=len(goals) - 1)
    return pd.Series(np.array(goals, dtype=object)[pick], index=df.index, dtype="string")


def sample_horizon(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.Series:
    s = schema(cfg)
    feat = s["features"]["investment_horizon_years"]
    by_goal = feat["generation"]["by_goal"]
    goal = df["primary_investment_goal"]
    n = len(df)
    years = pd.Series(np.nan, index=df.index)
    ret = goal == "Retirement"
    years[ret] = (np.maximum(1, s["retirement_age"] - df.loc[ret, "edad"])
                  + rng.normal(0, 1, ret.sum()))
    for name, spec in by_goal.items():
        if "distribution" not in spec:
            continue
        m = goal == name
        p = spec["params"]
        draw = (rng.uniform(p["low"], p["high"], n) if spec["distribution"] == "uniform"
                else rng.triangular(p["low"], p["mode"], p["high"], n))
        years[m] = draw[m.to_numpy()]
    lo, hi = feat["range"]
    cap = (s["life_horizon_cap_age"] - df["edad"]).clip(lower=lo)
    return years.clip(lo, hi).clip(upper=cap).round(feat["rounding"])


def derive_risk_tolerance(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    feat = schema(cfg)["features"]["risk_tolerance_score"]
    gen = feat["generation"]
    category, within = quantile_assign(latent_score(df, gen, rng), gen["target_distribution"])
    cat_min = category.map({k: v["min"] for k, v in feat["categories"].items()}).astype(int)
    cat_max = category.map({k: v["max"] for k, v in feat["categories"].items()}).astype(int)
    df["risk_tolerance_score"] = (cat_min + np.floor(within * (cat_max - cat_min + 1))).clip(
        upper=cat_max).astype(int)
    df[feat["category_column"]] = category_from_score(df["risk_tolerance_score"], feat["categories"])
    return df


def derive_drawdown(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.Series:
    f = schema(cfg)["features"]
    feat, cats = f["max_acceptable_drawdown_pct"], f["risk_tolerance_score"]["categories"]
    gen = feat["generation"]
    cat = df["risk_tolerance_category"]
    bmin = cat.map({k: v["min"] for k, v in gen["bands"].items()}).astype(float)
    bmax = cat.map({k: v["max"] for k, v in gen["bands"].items()}).astype(float)
    cmin = cat.map({k: v["min"] for k, v in cats.items()}).astype(float)
    cmax = cat.map({k: v["max"] for k, v in cats.items()}).astype(float)
    pos = (df["risk_tolerance_score"] - cmin + rng.random(len(df))) / (cmax - cmin + 1)
    value = bmin + pos * (bmax - bmin) + rng.normal(0, gen["noise_sd"], len(df))
    lo, hi = feat["range"]
    return value.round(feat["rounding"]).clip(lo, hi)


def derive_liquidity_need(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.Series:
    feat = schema(cfg)["features"]["liquidity_need_pct"]
    gen = feat["generation"]
    value = pd.Series(np.nan, index=df.index)
    for goal, ab in gen["base_beta_by_goal"].items():
        m = df["primary_investment_goal"] == goal
        value[m] = 100 * rng.beta(ab["a"], ab["b"], m.sum())
    for adj in gen["additive_adjustments"]:
        hit = evaluate(df, adj["when"])
        add = adj["add"]
        if "per_unit_of" in adj:
            add = add * df[adj["per_unit_of"]].fillna(0).clip(upper=adj["max_units"])
            add = add[hit]
        value[hit] += add
    lo, hi = gen["clip"]
    return value.clip(lo, hi).round(feat["rounding"])


def derive_esg(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.Series:
    gen = schema(cfg)["features"]["esg_preference_level"]["generation"]
    category, _ = quantile_assign(latent_score(df, gen, rng), gen["target_distribution"])
    return category


def employment_stability(df: pd.DataFrame, cap_cfg: dict) -> pd.Series:
    classes = cap_cfg["employment_stability_classes"]
    group = df["employment_group"]
    contract = df["contract_type"].astype("string")
    key = group.copy()
    working = group == "ocupado"
    key[working] = "ocupado_" + contract[working].fillna("unknown_contract")
    return key.map(classes).fillna("desempleado")


def compute_capacity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """MiFID II ability to bear losses: additive points, then hard caps, then effective risk."""
    cap_cfg = schema(cfg)["supporting_features"]["risk_capacity_score"]
    pts = cap_cfg["points"]
    df["employment_stability"] = employment_stability(df, cap_cfg)
    total = (points(df["surplus_ratio"], pts["surplus_ratio"])
             + points(df["investment_horizon_bucket"], pts["investment_horizon_bucket"])
             + points(df["debt_service_to_income"].fillna(np.inf), pts["debt_service_to_income"])
             + points(df["employment_stability"], pts["employment_stability"])
             + points(df["liquidity_need_pct"], pts["liquidity_need_pct"]))
    lo, hi = cap_cfg["clip"]
    score = total.clip(lo, hi)
    applied = pd.Series("", index=df.index, dtype="string")
    for cap in cap_cfg["caps"]:
        hit = evaluate(df, cap["when"])
        score[hit] = score[hit].clip(upper=cap["max"])
        applied[hit] = applied[hit] + "|" + cap["id"]
    df["risk_capacity_points"] = total.astype(int)
    df["risk_capacity_score"] = score.astype(int)
    df["capacity_caps_applied"] = applied.str.lstrip("|").replace("", pd.NA)
    df["effective_risk_score"] = np.minimum(df["risk_tolerance_score"], df["risk_capacity_score"])

    # Product feasibility: cannot meet the robo-advisor's minimum initial or monthly amount
    product = cfg.get("edge_case_profiles", {}).get("robo_advisor_product")
    if product:
        df["flag_minimum_investment"] = (
            (df["liquid_savings_eur"] < product["min_initial_investment_eur"])
            | (df["investable_surplus_eur"] < 12 * product["min_monthly_contribution_eur"]))
    return df


def derive_features(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Derive all wealth features in dependency order (inputs must be resolved first)."""
    s = schema(cfg)
    df = derive_surplus(df.copy(), cfg)
    df = derive_liquid_savings(df, cfg, rng)
    df["primary_investment_goal"] = sample_goal(df, cfg, rng)
    df["investment_horizon_years"] = sample_horizon(df, cfg, rng)
    horizon = s["features"]["investment_horizon_years"]
    df[horizon["bucket_column"]] = bucket_from_years(df["investment_horizon_years"], horizon["buckets"])
    df = derive_risk_tolerance(df, cfg, rng)
    df["max_acceptable_drawdown_pct"] = derive_drawdown(df, cfg, rng)
    df["liquidity_need_pct"] = derive_liquidity_need(df, cfg, rng)
    df["esg_preference_level"] = derive_esg(df, cfg, rng)
    return compute_capacity(df, cfg)
