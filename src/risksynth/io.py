# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
I/O helpers for RiskSynth AI profile files.

Why this exists: "None" is a valid value of esg_preference_level (no sustainability
preference). pandas.read_csv treats the string "None" as missing by default, which
silently turns those clients into NaN and breaks the MiFID II "ESG preference
collected" check. Always read profile CSVs with read_profiles() (or pass
keep_default_na=False, na_values=CSV_NA_VALUES yourself). Parquet is unaffected.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# pandas' default NA tokens minus "None" (and its variants), which is a real ESG category
CSV_NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
                 "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]


def read_profiles(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a RiskSynth profile file (.parquet or .csv) without losing ESG 'None'."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    options = {"keep_default_na": False, "na_values": CSV_NA_VALUES, "low_memory": False}
    options.update(kwargs)
    return pd.read_csv(path, **options)
