# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Run contracts: the parameters, rule results and assumptions behind one generated dataset.

A contract is a plain JSON document saved next to every run. It records what a model
test suite needs to pin a dataset down:

  parameters          engine, rows, seed, edge-case share, target risk mix
  cohort_proportions  EC01-EC09 targets and what the run achieved
  rule_checks         CR01-CR07 and MiFID II checks, before and after repair
  configs             SHA-256 of data_config.yaml and model_config.yaml
  assumptions         every [ASSUMPTION] marker in those configs
  output              row count, columns and a content hash of the profiles

generate() is the single code path the app and reproduce() use, so the same contract
regenerates the same rows on the same code and configs. verify() reports any drift.

    from risksynth import contract
    c = contract.load("data/synthetic/app_runs/<run>/run_contract.json")
    df = contract.reproduce(c)
    assert contract.verify(c, df) == []
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from risksynth import __version__
from risksynth.engine import wealth_profile as wp
from risksynth.paths import DATA_CONFIG, MODEL_CONFIG, PROCESSED_DIR

CONTRACT_VERSION = "1.0"
COPYRIGHT = (
    "Generated with RiskSynth AI. Copyright (c) 2026 aditya-forge. All rights reserved; "
    "not licensed for reuse without written permission. See LICENSE."
)
CALIBRATION_NOTE = (
    "Parameters marked [ASSUMPTION] are design choices, not measurements. Calibrate them against "
    "the Banco de Espana EFF 2022 survey before any production use."
)
_ASSUMPTION = re.compile(r"\[ASSUMPTION\]\s*[:\-]?\s*(.*)$")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of a profile frame: column names plus a stable per-row hash."""
    h = hashlib.sha256("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df.astype("string"), index=False).values.tobytes())
    return h.hexdigest()


def assumptions(paths: tuple[Path, ...] = (DATA_CONFIG, MODEL_CONFIG)) -> list[dict]:
    """Every line marked [ASSUMPTION] in the configs, with the key it sits on."""
    found = []
    for path in paths:
        stack: list[tuple[int, str]] = []
        pending: list[dict] = []
        for no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            body = line.split("#", 1)[0]
            key = re.match(r"^(\s*)([A-Za-z0-9_]+)\s*:", body)
            # a comment on its own line belongs to the block it is indented into
            indent = len(key.group(1)) if key else len(line) - len(line.lstrip())
            if key:
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                stack.append((indent, key.group(2)))
                for item in pending:  # a section header comment names the next block
                    item["key"] = key.group(2)
                pending.clear()
            path_keys = [k for i, k in stack] if key else [k for i, k in stack if i < indent]
            m = _ASSUMPTION.search(line)
            if m and not m.group(1).startswith("="):  # skip the legend lines that define the tag
                found.append(
                    {
                        "file": Path(path).name,
                        "line": no,
                        "key": ".".join(path_keys),
                        "note": m.group(1).strip() or None,
                    }
                )
                if not path_keys:
                    pending.append(found[-1])
    return found


def run_config(cfg: dict, risk_mix_pct: dict) -> dict:
    """data_config with the run's target risk mix applied (percentages in, shares stored)."""
    out = copy.deepcopy(cfg)
    out["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]["target_distribution"] = {
        k: v / 100 for k, v in risk_mix_pct.items()
    }
    return out


def generate(
    method: str,
    n: int,
    seed: int,
    edge_case_share: float,
    risk_mix_pct: dict,
    sdv_model: str = "copula",
    cfg: dict | None = None,
    save_dir: Path | None = None,
    progress: Callable[[str], None] = lambda _msg: None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Generate one dataset; returns (profiles, wealth report, extra files written)."""
    run_cfg = run_config(cfg or wp.load_config(), risk_mix_pct)
    if method == "Rule-based":
        from risksynth.generators import generate_synthetic as gs

        progress("Stage 1: sampling the INE persons base and resolving household finances")
        base = pd.read_parquet(PROCESSED_DIR / "combined_base.parquet")
        progress("Building banking payloads: accounts, loans, transactions")
        before = set(gs.OUTPUT_DIR.glob(f"synthetic_{n}_*_wealth_report.json"))
        out = gs.generate_synthetic_data(
            n, base, n_jobs=1, batch_size=n, seed=seed, wealth_cfg=run_cfg, edge_case_share=edge_case_share
        )
        new = sorted(set(gs.OUTPUT_DIR.glob(f"synthetic_{n}_*_wealth_report.json")) - before) or sorted(
            gs.OUTPUT_DIR.glob(f"synthetic_{n}_*_wealth_report.json")
        )
        return out, json.loads(new[-1].read_text(encoding="utf-8")), {}
    from risksynth.generators.client_generator import ClientGenerator

    model = "bootstrap" if method == "Bootstrap sampler" else sdv_model
    stage1 = wp.build_stage1_table()
    progress(f"Stage 1: fitting {model} on {len(stage1):,} EPF households")
    gen = ClientGenerator(n_samples=n, method=model, seed=seed)  # type: ignore[arg-type]
    gen.fit(stage1)
    progress("Stages 2-3: suitability profile, CR01-CR07 and MiFID II checks, edge-case cohort")
    out = gen.generate(wealth_config=run_cfg, edge_case_share=edge_case_share)
    files = gen.save(str(save_dir), out) if save_dir else {}
    return out, gen.wealth_report or {}, files


def build(
    *,
    method: str,
    stage1_model: str | None,
    sdv_model: str,
    rows: int,
    seed: int,
    edge_case_share: float,
    risk_mix_pct: dict,
    report: dict,
    frame: pd.DataFrame,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or wp.load_config()
    s = wp.schema(cfg)
    ec = report.get("edge_cases") or {}
    profiles = {}
    for p in cfg["edge_case_profiles"]["profiles"]:
        got = (ec.get("profiles") or {}).get(p["id"], {})
        profiles[p["id"]] = {
            "name": p["name"],
            "configured_share_pct": round(100 * p["target_share"], 3),
            "target_share_pct": got.get("target_share_pct"),
            "achieved_share_pct": got.get("share_pct"),
            "rows": got.get("rows", 0),
            "natural_selected": got.get("natural_selected"),
            "injected": got.get("injected"),
            "capacity_cap": p["capacity_cap"],
            "relaxed_rules": p["relaxed_rules"],
            "assertion": p["expected_advisory_outcome"]["test_assertion"],
            "assertion_pass_pct": got.get("assertion_pass_pct"),
        }
    enforced = report.get("violations_enforced", {})
    return {
        "contract_version": CONTRACT_VERSION,
        "copyright": COPYRIGHT,
        "risksynth_version": __version__,
        "schema_version": str(cfg["project"]["schema_version"]),
        "parameters": {
            "method": method,
            "stage1_model": stage1_model,
            "sdv_model": sdv_model if method.startswith("SDV") else None,
            "rows": rows,
            "seed": seed,
            "edge_case_share": edge_case_share,
            "target_risk_mix_pct": risk_mix_pct,
        },
        "cohort_proportions": {
            "target_total_share_pct": ec.get("target_total_share_pct"),
            "achieved_total_share_pct": ec.get("achieved_total_share_pct"),
            "origin_counts": ec.get("origin_counts"),
            "profiles": profiles,
        },
        "rule_checks": {
            "consistency_rules": {r["id"]: r["rule"] for r in s["consistency_rules"]},
            "mifid_checks": [k for k in enforced if k.startswith("MIFID")],
            "violations_raw": report.get("violations_raw"),
            "violations_after_repair": enforced,
            "violations_exempted": report.get("violations_exempted"),
            "rows_inconsistent_after": report.get("rows_inconsistent_after"),
        },
        "configs": {
            "data_config.yaml": file_sha256(DATA_CONFIG),
            "model_config.yaml": file_sha256(MODEL_CONFIG),
        },
        "assumptions": assumptions(),
        "calibration_note": CALIBRATION_NOTE,
        "output": {
            "rows": len(frame),
            "columns": list(map(str, frame.columns)),
            "sha256": frame_sha256(frame),
        },
        "reproduce": (
            "from risksynth import contract; c = contract.load('run_contract.json'); "
            "df = contract.reproduce(c); assert contract.verify(c, df) == []"
        ),
    }


def load(path: str | Path) -> dict:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def reproduce(c: dict, cfg: dict | None = None, save_dir: Path | None = None) -> pd.DataFrame:
    """Regenerate the dataset a contract describes (same code path as the app)."""
    p = c["parameters"]
    out, _, _ = generate(
        p["method"],
        int(p["rows"]),
        int(p["seed"]),
        float(p["edge_case_share"]),
        p["target_risk_mix_pct"],
        sdv_model=p.get("sdv_model") or "copula",
        cfg=cfg,
        save_dir=save_dir,
    )
    return out


def verify(c: dict, df: pd.DataFrame) -> list[str]:
    """Differences between a contract and a (re)generated frame or the current configs; [] = faithful."""
    problems = []
    for name, path in (("data_config.yaml", DATA_CONFIG), ("model_config.yaml", MODEL_CONFIG)):
        if c["configs"].get(name) != file_sha256(path):
            problems.append(f"{name} has changed since the run")
    if c["risksynth_version"] != __version__:
        problems.append(f"risksynth {__version__} differs from the run's {c['risksynth_version']}")
    if len(df) != c["output"]["rows"]:
        problems.append(f"{len(df)} rows, contract says {c['output']['rows']}")
    if list(map(str, df.columns)) != c["output"]["columns"]:
        problems.append("column list differs")
    elif frame_sha256(df) != c["output"]["sha256"]:
        problems.append("profile content differs")
    return problems
