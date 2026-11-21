#!/usr/bin/env python3
"""
INE microdata processor for RiskSynth AI.

Builds data/processed/{ecv,epf,epa}_processed.parquet and combined_base.parquet
from ECV 2024, EPF 2024 and EPA 2025T3. Column mappings, missing-value codes,
code harmonization and validation rules come from configs/data_config.yaml
(sections: ine_source_mapping, harmonization, combined_base, data_quality).

Sources (``--source``):
  raw     INE microdata under data/raw (layout in ine_source_mapping.raw_layout)
  legacy  frozen schema-1.x parquet files (ine_source_mapping.legacy_snapshot_dir),
          used when the raw microdata is not available
  auto    raw if present, otherwise legacy (default)

Per-survey outputs keep every raw INE column for auditability and add
harmonized columns (edad, sexo, ccaa, nivel_estudios, situacion_laboral, ...).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from risksynth.paths import DATA_CONFIG as DEFAULT_CONFIG
from risksynth.paths import ROOT as PROJECT_ROOT

log = logging.getLogger("process_data")

# Renames applied by the schema-1.x version of this script. They are reversed when
# rebuilding from the legacy snapshot, so both sources enter the pipeline with raw
# INE column names. ('gasto_total' was IMPEXACSP: the first column containing
# "IMPEXAC", located between PERCEPSP and INTERINPSP in the EPF household file.)
LEGACY_V1_RENAMES = {
    "ecv": {"edad": "PB140", "sexo": "PB150", "ingresos_trabajo": "PY010G",
            "ccaa": "DB040", "nivel_estudios": "PE040", "renta_disponible": "HY020",
            "situacion_laboral": "PL031"},
    "epf": {"ccaa": "CCAA", "n_miembros": "NMIEMB", "gasto_total": "IMPEXACSP",
            "gasto_monetario": "GASTMON", "regimen_vivienda": "REGTEN"},
    "epa": {"edad": "EDAD1", "sexo": "SEXO1", "ccaa": "CCAA", "situacion_laboral": "AOI",
            "ocupacion": "CNO11", "sector": "SPTS", "tramo_salario": "SALAM1"},
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_table(dir_path: Path) -> pd.DataFrame | None:
    """Read the first CSV/TAB file in a directory, sniffing the separator."""
    if not dir_path.exists():
        log.warning("Missing directory: %s", dir_path)
        return None
    files = sorted(dir_path.glob("*.csv")) + sorted(dir_path.glob("*.tab"))
    if not files:
        log.warning("No CSV/TAB file in %s", dir_path)
        return None
    for sep in ["\t", ",", ";"]:
        try:
            df = pd.read_csv(files[0], sep=sep, encoding="latin-1", low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
        if df.shape[1] > 1:
            log.info("Read %s: %d rows x %d cols", files[0].name, *df.shape)
            return df
    log.error("Could not parse %s", files[0])
    return None


def load_raw(raw_dir: Path, layout: dict) -> dict:
    """Load raw INE microdata tables, keyed by survey and table."""
    ecv_dir = raw_dir / layout["ecv"]["dir"]
    epf_dir = raw_dir / layout["epf"]["dir"]
    epa_path = raw_dir / layout["epa"]["dir"] / layout["epa"]["file"]
    epa = (pd.read_csv(epa_path, sep="\t", encoding="latin-1", low_memory=False)
           if epa_path.exists() else read_table(epa_path.parent))
    return {
        "ecv": {t: read_table(ecv_dir / layout["ecv"][t])
                for t in ("personas", "hogares", "registro_hogares")},
        "epf": {"hogar": read_table(epf_dir / layout["epf"]["hogar"])},
        "epa": {"personas": epa},
    }


def load_legacy(snapshot_dir: Path) -> dict:
    """Load the schema-1.x parquet snapshot and restore raw INE column names."""
    tables = {}
    for survey, table in [("ecv", "personas"), ("epf", "hogar"), ("epa", "personas")]:
        path = snapshot_dir / f"{survey}_processed.parquet"
        df = pd.read_parquet(path) if path.exists() else None
        if df is not None:
            renames = {k: v for k, v in LEGACY_V1_RENAMES[survey].items() if k in df.columns}
            df = df.rename(columns=renames).drop(columns=["fuente"], errors="ignore")
            log.info("Legacy %s: %d rows; restored raw names %s", survey, len(df), renames)
        else:
            log.warning("Legacy snapshot missing: %s", path)
        tables[survey] = {table: df}
    # The schema-1.x pipeline never ingested the ECV household (H) / register (D) files.
    tables["ecv"].update(hogares=None, registro_hogares=None)
    return tables


# --------------------------------------------------------------------------- #
# Cleaning and harmonization helpers
# --------------------------------------------------------------------------- #
def _canon(value) -> str | None:
    """Canonical code key: '006', 6, 6.0 and ' 6 ' all map to '6'."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def map_codes(series: pd.Series, mapping: dict, label: str) -> pd.Series:
    """Map raw INE codes to harmonized values; log codes that have no mapping."""
    lookup = {_canon(k): v for k, v in mapping.items()}
    keys = series.map(_canon)
    out = keys.map(lookup)
    unmapped = keys.notna() & out.isna()
    if unmapped.any():
        log.warning("%s: %d values without mapping, e.g. %s", label, unmapped.sum(),
                    keys[unmapped].value_counts().head(5).to_dict())
    return out


def blank_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """INE fixed-width exports leave missing values as blank strings."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].mask(df[col].astype("string").str.strip() == "")
    return df


def replace_missing_codes(df: pd.DataFrame, codes: list, label: str) -> pd.DataFrame:
    """Replace INE missing-value sentinels (e.g. -9) with NaN in numeric columns."""
    numeric = df.select_dtypes(include="number").columns
    hits = {c: int(df[c].isin(codes).sum()) for c in numeric}
    hits = {c: n for c, n in hits.items() if n}
    if hits:
        df[numeric] = df[numeric].mask(df[numeric].isin(codes))
        log.info("%s: sentinels %s replaced with NaN: %s", label, codes, hits)
    return df


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = values.argsort()
    v, w = values.to_numpy()[order], weights.to_numpy()[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


# --------------------------------------------------------------------------- #
# Survey processing
# --------------------------------------------------------------------------- #
def process_ecv(tables: dict, mapping: dict, harm: dict, reference_year: int) -> pd.DataFrame | None:
    """ECV persons (P) joined to household income (H) and region (D) on the household id."""
    people = tables.get("personas")
    if people is None:
        return None
    df = blank_to_nan(people.copy())
    cols = mapping["columns"]

    df["hogar_id"] = pd.to_numeric(df[mapping["person_id"]]) // mapping["household_id_divisor"]
    joins = [("hogares", mapping["household_file_id"], [cols["renta_disponible"]]),
             ("registro_hogares", mapping["register_file_id"], [cols["ccaa"], "DB090"])]
    for table, key, wanted in joins:
        other = tables.get(table)
        if other is None:
            log.warning("ECV %s file not available: %s will be null", table, wanted)
            continue
        other = blank_to_nan(other.copy())
        keep = [c for c in wanted if c in other.columns]
        other = other[[key] + keep].assign(**{key: pd.to_numeric(other[key])})
        df = df.merge(other, left_on="hogar_id", right_on=key, how="left", validate="m:1")
        log.info("ECV join %s on %s: %.1f%% of persons matched", table, key,
                 100 * df[key].notna().mean())

    # Age: year of birth -> age attained in the interview year (ISSUE-01)
    birth = pd.to_numeric(df[mapping["birth_year_column"]], errors="coerce")
    year_col = mapping["interview_year_column"]
    year = pd.to_numeric(df[year_col], errors="coerce") if year_col in df else reference_year
    df["edad"] = (year - birth).astype("float")
    df["edad_min"] = df["edad"]
    top_coded = birth <= mapping["birth_year_top_code"]
    df["edad_max"] = df["edad"].mask(top_coded)
    df["edad_precision"] = np.where(top_coded, "exact_topcoded", "exact")

    df["sexo"] = map_codes(df[cols["sexo"]], harm["sexo"]["ecv"], "ECV sexo").astype("Int8")
    df["nivel_estudios"] = map_codes(df[cols["nivel_estudios"]], harm["nivel_estudios"]["ecv"],
                                     "ECV nivel_estudios")
    df["situacion_laboral"] = map_codes(df[cols["situacion_laboral"]],
                                        harm["situacion_laboral"]["ecv"], "ECV situacion_laboral")
    df["ingresos_trabajo"] = pd.to_numeric(df.get(cols["ingresos_trabajo"]), errors="coerce")
    df["renta_disponible"] = pd.to_numeric(df.get(cols["renta_disponible"]), errors="coerce")
    df["ccaa"] = df[cols["ccaa"]].astype("string").str.strip() if cols["ccaa"] in df else pd.NA

    components = [c for c in mapping["personal_net_income_components"] if c in df.columns]
    df["renta_neta_personal_eur"] = (df[components].apply(pd.to_numeric, errors="coerce")
                                     .sum(axis=1, min_count=1))
    df["peso"] = pd.to_numeric(df[mapping["weight_column"]], errors="coerce")
    df["fuente"] = "ECV"
    return df


def process_epf(tables: dict, mapping: dict, harm: dict) -> pd.DataFrame | None:
    """EPF households: per-household expenditure, net income and mortgage debt service."""
    hogar = tables.get("hogar")
    if hogar is None:
        return None
    df = replace_missing_codes(hogar.copy(), mapping["missing_value_codes"], "EPF")
    cols = mapping["columns"]
    weight = df[mapping["weight_column"]]
    new = {}  # derived columns, attached in one concat (EPF has ~140 raw columns)

    for target in ("n_miembros", "regimen_vivienda", "dependent_children", "employed_members"):
        new[target] = df[cols[target]]
    new["edad"] = new["edad_min"] = new["edad_max"] = df[cols["edad"]].astype("float")
    new["edad_precision"] = "exact"
    new["sexo"] = map_codes(df[cols["sexo"]], harm["sexo"]["epf"], "EPF sexo").astype("Int8")
    new["ccaa"] = map_codes(df[cols["ccaa"]], harm["ccaa_ine_to_nuts2"], "EPF ccaa")
    new["nivel_estudios"] = map_codes(df[cols["nivel_estudios"]], harm["nivel_estudios"]["epf"],
                                      "EPF nivel_estudios")
    new["situacion_laboral"] = map_codes(df[cols["situacion_laboral"]],
                                         harm["situacion_laboral"]["epf"], "EPF situacion_laboral")
    new["contract_type"] = map_codes(df[cols["contract_type"]], harm["contract_type"]["epf"],
                                     "EPF contract_type")

    # Grossed-up expenditure -> EUR per household per year (ISSUE-02, ISSUE-03)
    for target, source in mapping["grossed_up_expenditure"].items():
        new[target] = df[source] / weight

    income = mapping["income"]
    exact = df[income["exact_monthly"]].where(df[income["exact_monthly"]] > 0)
    bracket = df[income["bracket"]].mask(df[income["bracket"]].isin(income["invalid_bracket_codes"]))
    imputed = bracket.map({int(k): v for k, v in income["bracket_impute_median"].items()})
    new["net_income_annual_eur"] = 12 * exact.fillna(imputed)
    new["net_income_source"] = (pd.Series("bracket_median", index=df.index, dtype="string")
                                .where(imputed.notna()).mask(exact.notna(), "exact"))

    debt = mapping["debt_service"]
    monthly = df[debt["principal"]].clip(lower=0).fillna(0) + df[debt["interest"]].clip(lower=0).fillna(0)
    new["debt_service_annual_eur"] = 12 * monthly
    new["main_residence_value_eur"] = df[mapping["main_residence_value"]]
    new["peso"] = weight
    new["fuente"] = "EPF"
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    lo, hi = mapping["expected_median_gasto_monetario_eur"]
    median = df["gasto_monetario"].median()
    (log.info if lo <= median <= hi else log.warning)(
        "EPF median gasto_monetario = %.0f EUR/household/year (expected %s-%s)", median, lo, hi)
    return df


def process_epa(tables: dict, mapping: dict, harm: dict, open_band_age: float) -> pd.DataFrame | None:
    """EPA persons: 5-year age bands become midpoints with explicit bounds."""
    people = tables.get("personas")
    if people is None:
        return None
    df = blank_to_nan(people.copy())
    cols = mapping["columns"]

    bands = {_canon(k): v for k, v in mapping["age_bands"].items()}
    band = df[mapping["age_band_column"]].map(_canon)
    df["edad_min"] = band.map(lambda b: bands[b][0] if b in bands else np.nan).astype("float")
    df["edad_max"] = band.map(lambda b: bands[b][1] if b in bands and bands[b][1] is not None
                              else np.nan).astype("float")
    open_band = df["edad_min"].notna() & df["edad_max"].isna()
    df["edad"] = ((df["edad_min"] + df["edad_max"]) / 2).mask(open_band, open_band_age)
    df["edad_precision"] = "band5"
    unknown = band.notna() & ~band.isin(bands)
    if unknown.any():
        log.warning("EPA: %d rows with unknown age band codes %s", unknown.sum(),
                    band[unknown].unique()[:5])

    df["sexo"] = map_codes(df[cols["sexo"]], harm["sexo"]["epa"], "EPA sexo").astype("Int8")
    df["ccaa"] = map_codes(df[cols["ccaa"]], harm["ccaa_ine_to_nuts2"], "EPA ccaa")
    df["situacion_laboral"] = map_codes(df[cols["situacion_laboral"]],
                                        harm["situacion_laboral"]["epa"], "EPA situacion_laboral")
    df["nivel_estudios"] = map_codes(df[cols["nivel_estudios"]], harm["nivel_estudios"]["epa"],
                                     "EPA nivel_estudios")
    df["contract_type"] = map_codes(df[cols["contract_type"]], harm["contract_type"]["epa"],
                                    "EPA contract_type")
    for target in ("ocupacion", "sector"):
        df[target] = df[cols[target]].map(_canon) if cols[target] in df else pd.NA

    df["peso"] = pd.to_numeric(df[mapping["weight_column"]], errors="coerce")
    df["fuente"] = "EPA"
    return df


# --------------------------------------------------------------------------- #
# Validation, diagnostics, combination
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, name: str, rules: list) -> None:
    """Apply data_quality.validation_rules to the columns present in df (log only)."""
    for rule in rules:
        for col in (c for c in rule.get("columns", []) if c in df.columns):
            s = df[col]
            kind = rule["rule"]
            if kind == "not_null":
                bad = s.isna()
            elif kind == "range":
                bad = s.notna() & ~s.between(rule["min"], rule["max"])
            elif kind == "positive":
                bad = s.notna() & (s <= 0)
            elif kind == "in_list":
                bad = s.notna() & ~s.isin(rule["values"])
            else:
                continue
            level = logging.WARNING if bad.any() else logging.INFO
            log.log(level, "[%s] rule %-22s on %-18s: %d violations (%.2f%%)",
                    name, rule["name"], col, bad.sum(), 100 * bad.mean())


def summarize(df: pd.DataFrame, name: str, columns: list) -> None:
    log.info("[%s] %d rows x %d cols", name, *df.shape)
    for col in (c for c in columns if c in df.columns):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            log.info("[%s]   %-26s null %5.1f%%  min %-10.4g p50 %-10.4g max %.4g", name, col,
                     100 * s.isna().mean(), s.min(), s.median(), s.max())
        else:
            log.info("[%s]   %-26s null %5.1f%%  %d distinct, top %s", name, col,
                     100 * s.isna().mean(), s.nunique(), s.value_counts().head(3).to_dict())


def build_combined(frames: list, columns: list, age_rule: dict) -> pd.DataFrame:
    """Schema-aligned stack of independent survey samples, filtered to the valid age range.

    The surveys sample different people, so rows cannot be joined across them.
    `fuente` is the stratum and `peso_norm` makes the weights sum to 1 within each source.
    """
    combined = pd.concat([f.reindex(columns=columns) for f in frames], ignore_index=True)
    in_range = combined["edad"].between(age_rule["min"], age_rule["max"])
    dropped = combined.loc[~in_range, "fuente"].value_counts().to_dict()
    log.info("combined: dropping rows outside edad %s-%s: %s", age_rule["min"], age_rule["max"], dropped)
    combined = combined[in_range].reset_index(drop=True)
    combined["peso_norm"] = combined["peso"] / combined.groupby("fuente")["peso"].transform("sum")
    return combined


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
SUMMARY_COLUMNS = ["edad", "edad_precision", "sexo", "ccaa", "nivel_estudios", "situacion_laboral",
                   "ingresos_trabajo", "renta_disponible", "renta_neta_personal_eur",
                   "net_income_annual_eur", "gasto_total", "gasto_monetario",
                   "debt_service_annual_eur", "main_residence_value_eur", "contract_type", "peso"]


def run(source: str, config_path: Path) -> pd.DataFrame | None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping, harm = cfg["ine_source_mapping"], cfg["harmonization"]
    rules = cfg["data_quality"]["validation_rules"]
    raw_dir = PROJECT_ROOT / cfg["data_pipeline"]["raw"]["path"]
    out_dir = PROJECT_ROOT / cfg["data_pipeline"]["processed"]["path"]
    snapshot_dir = PROJECT_ROOT / mapping["legacy_snapshot_dir"]

    if source == "auto":
        source = "raw" if raw_dir.exists() and any(raw_dir.iterdir()) else "legacy"
    log.info("Source: %s", source)
    if source == "raw":
        tables = load_raw(raw_dir, mapping["raw_layout"])
    else:
        if not snapshot_dir.exists():
            log.error("Neither %s nor %s exists; nothing to process", raw_dir, snapshot_dir)
            return None
        tables = load_legacy(snapshot_dir)

    reference_year = cfg["project"]["reference_year"]
    ecv = process_ecv(tables["ecv"], mapping["ecv"], harm, reference_year)
    if ecv is not None:
        senior = ecv[ecv["edad"] >= 65]
        open_band_age = round(weighted_median(senior["edad"], senior["peso"].fillna(0)), 1)
        log.info("EPA open band 65+ represented by ECV weighted median age %.1f", open_band_age)
    else:
        open_band_age = mapping["epa"]["open_band_age_fallback"]
    epf = process_epf(tables["epf"], mapping["epf"], harm)
    epa = process_epa(tables["epa"], mapping["epa"], harm, open_band_age)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, df in [("ecv", ecv), ("epf", epf), ("epa", epa)]:
        if df is None:
            log.warning("%s not produced", name.upper())
            continue
        summarize(df, name.upper(), SUMMARY_COLUMNS)
        validate(df, name.upper(), rules)
        df.to_parquet(out_dir / f"{name}_processed.parquet", index=False)
        frames.append(df)
    if not frames:
        log.error("No survey could be processed")
        return None

    age_rule = next(r for r in rules if r["name"] == cfg["combined_base"]["age_rule"])
    combined = build_combined(frames, cfg["combined_base"]["columns"], age_rule)
    summarize(combined, "COMBINED", SUMMARY_COLUMNS)
    validate(combined, "COMBINED", rules)
    combined.to_parquet(out_dir / "combined_base.parquet", index=False)
    log.info("Wrote %d files to %s", len(frames) + 1, out_dir)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "raw", "legacy"], default="auto")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    run(args.source, args.config)


if __name__ == "__main__":
    main()
