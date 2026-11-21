"""INE microdata ingestion (risksynth.ingest.ine)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risksynth.ingest import ine as pdp

ROOT_PROCESSED = pdp.PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def mapping(data_cfg):
    return data_cfg["ine_source_mapping"]


@pytest.fixture(scope="module")
def harm(data_cfg):
    return data_cfg["harmonization"]


# --------------------------------------------------------------------------- #
# ECV: age from year of birth, P -> H -> D household join
# --------------------------------------------------------------------------- #
@pytest.fixture
def ecv_tables():
    personas = pd.DataFrame({
        "PB030": [101, 102, 201, 301], "PB040": [500.0, 500.0, 800.0, 300.0], "PB110": [2024] * 4,
        "PB140": [1980, 1938, 2000, 1990], "PB150": [1, 2, 2, 1],
        "PE041": ["500", "200", "344", "000"], "PL032": ["1", "3", "5", "2"],
        "PY010G": [30000.0, 0.0, 5000.0, 0.0], "PY010N": [24000.0, 0.0, 4500.0, 0.0],
        "PY100N": [0.0, 15000.0, 0.0, 0.0]})
    hogares = pd.DataFrame({"HB030": [1, 2], "HY020": [41000.5, 18000.0]})
    registro = pd.DataFrame({"DB030": [1, 2, 3], "DB040": ["ES30", "ES51", "ES61"], "DB090": [700, 900, 400]})
    return {"personas": personas, "hogares": hogares, "registro_hogares": registro}


def test_ecv_age_is_interview_year_minus_birth_year(ecv_tables, mapping, harm):
    out = pdp.process_ecv(ecv_tables, mapping["ecv"], harm, 2024)
    assert out["edad"].tolist() == [44, 86, 24, 34]
    assert (out["edad"] < 120).all(), "no raw 4-digit birth years may survive as age"
    assert out["PB140"].tolist() == [1980, 1938, 2000, 1990], "raw INE column kept for audit"


def test_ecv_top_coded_birth_cohort_is_flagged(ecv_tables, mapping, harm):
    out = pdp.process_ecv(ecv_tables, mapping["ecv"], harm, 2024)
    assert out.loc[1, "edad_precision"] == "exact_topcoded"
    assert np.isnan(out.loc[1, "edad_max"]), "top-coded age has no upper bound"
    assert (out.loc[[0, 2, 3], "edad_precision"] == "exact").all()


def test_ecv_household_income_join_is_many_to_one(ecv_tables, mapping, harm):
    out = pdp.process_ecv(ecv_tables, mapping["ecv"], harm, 2024)
    assert len(out) == 4, "joins must not duplicate persons"
    assert out["hogar_id"].tolist() == [1, 1, 2, 3]
    assert out["renta_disponible"].iloc[:3].tolist() == [41000.5, 41000.5, 18000.0]
    assert pd.isna(out["renta_disponible"].iloc[3]), "no HY020 is invented for households without an H row"
    assert out["ccaa"].tolist() == ["ES30", "ES30", "ES51", "ES61"]


def test_ecv_without_household_file_leaves_hy020_null(ecv_tables, mapping, harm):
    tables = {"personas": ecv_tables["personas"], "hogares": None, "registro_hogares": None}
    out = pdp.process_ecv(tables, mapping["ecv"], harm, 2024)
    assert out["renta_disponible"].isna().all()
    assert out["renta_neta_personal_eur"].tolist() == [24000.0, 15000.0, 4500.0, 0.0]


def test_ecv_code_harmonization(ecv_tables, mapping, harm):
    out = pdp.process_ecv(ecv_tables, mapping["ecv"], harm, 2024)
    assert out["sexo"].tolist() == [1, 2, 2, 1]
    assert out["nivel_estudios"].tolist() == ["universitario", "secundaria", "secundaria", "sin_estudios"]
    assert out["situacion_laboral"].tolist() == ["ocupado", "jubilado", "estudiante", "desempleado"]


# --------------------------------------------------------------------------- #
# EPF: -9 sentinels, FACTOR weighting, income imputation, debt service
# --------------------------------------------------------------------------- #
@pytest.fixture
def epf_hogar():
    return pd.DataFrame({
        "FACTOR": [1000.0, 500.0, 250.0],
        "GASTOT": [30_000_000.0, 10_000_000.0, 5_000_000.0],
        "GASTMON": [22_000_000.0, 8_000_000.0, 4_000_000.0],
        "IMPEXAC": [2500.0, -9.0, 0.0],             # exact, missing, zero -> bracket fallback
        "INTERIN": [6, 3, 1],
        "IMPEXACSP": [1800.0, -9.0, -9.0],          # main earner's INCOME: the old 'gasto_total' bug
        "CAPITMENS": [400.0, np.nan, -0.1], "INTERESMENS": [100.0, np.nan, 0.0],
        "VALOREST": [200000.0, -9.0, 90000.0],
        "CCAA": [13, 9, 19], "NMIEMB": [3, 1, 2], "REGTEN": [2, 1, 1],
        "EDADSP": [45, 70, 30], "SEXOSP": [1, 6, 6], "ESTUDIOSSP": [7, 3, 5], "SITUACTSP": [1, 3, 1],
        "NHIJOSD": [1, 0, 0], "NUMOCUP": [2, 0, 1], "TIPOCONT": [1, np.nan, 2]})


def test_epf_sentinels_become_nan(epf_hogar, mapping, harm):
    out = pdp.process_epf({"hogar": epf_hogar}, mapping["epf"], harm)
    numeric = out.select_dtypes("number")
    assert not (numeric == -9).any().any(), "-9 must never survive as a value"
    assert out["IMPEXACSP"].isna().tolist() == [False, True, True]


def test_epf_spending_divided_by_factor(epf_hogar, mapping, harm):
    out = pdp.process_epf({"hogar": epf_hogar}, mapping["epf"], harm)
    assert out["gasto_monetario"].tolist() == [22000.0, 16000.0, 16000.0]
    assert out["gasto_total"].tolist() == [30000.0, 20000.0, 20000.0]
    assert out["GASTMON"].iloc[0] == 22_000_000.0, "raw grossed value kept for audit"
    assert (out["gasto_monetario"] <= out["gasto_total"]).all()


def test_epf_gasto_total_is_not_the_income_column(epf_hogar, mapping, harm):
    out = pdp.process_epf({"hogar": epf_hogar}, mapping["epf"], harm)
    assert not out["gasto_total"].equals(out["IMPEXACSP"])
    assert (out["gasto_total"] > 0).all()


def test_epf_net_income_exact_then_bracket_median(epf_hogar, mapping, harm):
    out = pdp.process_epf({"hogar": epf_hogar}, mapping["epf"], harm)
    medians = mapping["epf"]["income"]["bracket_impute_median"]
    assert out["net_income_annual_eur"].tolist() == [12 * 2500.0, 12 * medians[3], 12 * medians[1]]
    assert out["net_income_source"].tolist() == ["exact", "bracket_median", "bracket_median"]


def test_epf_debt_service_clips_negatives_and_treats_nan_as_no_mortgage(epf_hogar, mapping, harm):
    out = pdp.process_epf({"hogar": epf_hogar}, mapping["epf"], harm)
    assert out["debt_service_annual_eur"].tolist() == [6000.0, 0.0, 0.0]


def test_epf_code_harmonization(epf_hogar, mapping, harm):
    out = pdp.process_epf({"hogar": epf_hogar}, mapping["epf"], harm)
    assert out["sexo"].tolist() == [1, 2, 2], "EPF codes female as 6"
    assert out["ccaa"].tolist() == ["ES30", "ES51", "ES64"]
    assert out["contract_type"].tolist()[0] == "permanent" and out["contract_type"].tolist()[2] == "temporary"
    assert out["edad"].tolist() == [45.0, 70.0, 30.0]


# --------------------------------------------------------------------------- #
# EPA: 5-year bands, codes
# --------------------------------------------------------------------------- #
def test_epa_age_bands_become_midpoints_with_bounds(mapping, harm):
    epa = pd.DataFrame({"EDAD1": [20, 65, 16, 0], "SEXO1": [1, 6, 6, 1], "CCAA": [13, 51, 52, 1],
                        "AOI": [4, 9, 5, np.nan], "NFORMA": ["SU", "P2", "SP", None],
                        "OCUP1": ["2", "9", "5", None], "ACT1": ["8", "0", "6", None],
                        "DUCON1": [1, np.nan, 6, np.nan], "FACTOREL": [300.0, 200.0, 100.0, 50.0]})
    out = pdp.process_epa({"personas": epa}, mapping["epa"], harm, open_band_age=75.0)
    assert out["edad"].tolist() == [22.0, 75.0, 17.5, 2.0]
    assert out["edad_min"].tolist() == [20.0, 65.0, 16.0, 0.0]
    assert np.isnan(out.loc[1, "edad_max"]), "65+ is open-ended"
    assert (out["edad_precision"] == "band5").all()
    assert out["sexo"].tolist() == [1, 2, 2, 1]
    assert out["ccaa"].tolist() == ["ES30", "ES63", "ES64", "ES61"], "EPA Ceuta/Melilla 51/52 -> NUTS2"
    assert out["situacion_laboral"].tolist()[:3] == ["ocupado", "inactivo", "desempleado"]
    assert out["contract_type"].tolist()[0] == "permanent" and out["contract_type"].tolist()[2] == "temporary"


def test_map_codes_canonicalises_code_formats():
    s = pd.Series(["006", 6, 6.0, " 6 ", None])
    out = pdp.map_codes(s, {6: "x"}, "test")
    assert out.tolist()[:4] == ["x"] * 4 and pd.isna(out.iloc[4])


def test_combined_base_filters_age_and_normalises_weights():
    a = pd.DataFrame({"fuente": "ECV", "peso": [1.0, 3.0, 5.0], "edad": [17.0, 30.0, 50.0]})
    b = pd.DataFrame({"fuente": "EPF", "peso": [2.0, 2.0], "edad": [40.0, 101.0]})
    out = pdp.build_combined([a, b], ["fuente", "peso", "peso_norm", "edad"], {"min": 18, "max": 100})
    assert out["edad"].tolist() == [30.0, 50.0, 40.0]
    assert np.allclose(out.groupby("fuente")["peso_norm"].sum(), 1.0)


# --------------------------------------------------------------------------- #
# Committed processed parquet files
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def processed(require_processed):
    return {name: pd.read_parquet(ROOT_PROCESSED / f"{name}.parquet")
            for name in ["combined_base", "epf_processed", "ecv_processed"]}


def test_processed_combined_ages_are_valid(processed):
    edad = processed["combined_base"]["edad"]
    assert edad.notna().all() and edad.between(18, 100).all()
    assert (edad < 1900).all()


def test_processed_epf_household_spending_scale(processed):
    epf = processed["epf_processed"]
    assert 22_000 <= epf["gasto_monetario"].median() <= 23_600, "EPF 2024 median ~ EUR 22.8k"
    assert np.allclose(epf["gasto_monetario"] * epf["FACTOR"], epf["GASTMON"])
    assert not (epf.select_dtypes("number") == -9).any().any()


def test_processed_codes_are_harmonised(processed):
    c = processed["combined_base"]
    assert set(c["sexo"].dropna().unique()) <= {1, 2}
    assert c["ccaa"].dropna().str.match(r"^ES\d\d$").all()
    assert np.allclose(c.groupby("fuente")["peso_norm"].sum(), 1.0)


def test_processed_income_columns_present(processed):
    ecv, c = processed["ecv_processed"], processed["combined_base"]
    assert "renta_disponible" in c, "HY020 column exists (populated once ECV_Th is in data/raw)"
    assert ecv["renta_neta_personal_eur"].notna().mean() > 0.99
    assert processed["epf_processed"]["net_income_annual_eur"].notna().mean() > 0.99
