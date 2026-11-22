# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Repository locations shared by the package, the CLI tools and the app.

The package is meant to be used from a checkout (pip install -e .): configs and data
live next to src/, not inside the installed package.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "configs"
DATA_CONFIG = CONFIG_DIR / "data_config.yaml"
MODEL_CONFIG = CONFIG_DIR / "model_config.yaml"

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
