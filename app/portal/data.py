"""Dataset discovery, cached loaders and the generation runner shared by all pages."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from portal import PAGES
from portal.theme import TIER_ORDER
from risksynth import contract
from risksynth.engine import wealth_profile
from risksynth.evaluation import fidelity
from risksynth.io import read_profiles
from risksynth.paths import PROCESSED_DIR, SYNTHETIC_DIR
from risksynth.scoring import robo_advisor, scorecard

RUNS_DIR = SYNTHETIC_DIR / "app_runs"
REQUIRED = {"risk_tolerance_score", "effective_risk_score", "raw_surplus_eur", "investable_surplus_eur"}
METHODS = {"Rule-based": "Full banking payload per client: accounts, loans, 12 months of transactions. Up to 5,000 rows here.",
           "Bootstrap sampler": "Weighted resampling of observed INE (EPF) households, then fresh profiles. Fastest.",
           "SDV (CTGAN / copula / TVAE)": "Generative models trained on EPF households. Requires the optional SDV install."}


def _has_profiles(path: Path) -> bool:
    try:
        if path.suffix == ".parquet":
            return "risk_tolerance_score" in pq.read_schema(path).names
        with open(path, encoding="utf-8") as f:
            return "risk_tolerance_score" in f.readline()
    except (OSError, ValueError):
        return False


def discover_datasets() -> list[Path]:
    patterns = ["app_runs/*/profiles.parquet", "synthetic_*.parquet",
                "wealth_*/wealth_profiles.parquet", "clients/*/clients_synthetic.csv"]
    found = {p for pat in patterns for p in SYNTHETIC_DIR.glob(pat) if _has_profiles(p)}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def rel(path: Path) -> str:
    return path.relative_to(SYNTHETIC_DIR).as_posix()


@st.cache_data(show_spinner=False, max_entries=6)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    """mtime is part of the cache key, so a regenerated file is reloaded."""
    return read_profiles(path)


@st.cache_data(show_spinner=False, max_entries=6)
def advisory(path: str, mtime: float) -> pd.DataFrame:
    return robo_advisor.recommend(load_dataset(path, mtime))


@st.cache_data(show_spinner=False, max_entries=6)
def edge_scorecard(path: str, mtime: float) -> pd.DataFrame:
    return scorecard.edge_case_scorecard(load_dataset(path, mtime), advisory(path, mtime))


@st.cache_data(show_spinner=False, max_entries=6)
def fidelity_results(path: str, mtime: float) -> dict:
    return fidelity.run_all(load_dataset(path, mtime))


@st.cache_data(show_spinner=False)
def baseline_records() -> int:
    return pq.read_metadata(PROCESSED_DIR / "combined_base.parquet").num_rows


def active_dataset(key: str) -> tuple[pd.DataFrame, str, float] | None:
    """Dataset selector bar. Returns (frame, path, mtime) or renders the empty state and returns None."""
    datasets = discover_datasets()
    if not datasets:
        empty_state(key)
        return None
    options = [str(p) for p in datasets]
    default = st.session_state.get("dataset_path")
    left, right = st.columns([2.2, 3], vertical_alignment="bottom")
    chosen = left.selectbox("Active dataset", options, index=options.index(default) if default in options else 0,
                            format_func=lambda s: rel(Path(s)), help="Generated profile files, newest first",
                            key=f"dataset_{key}")
    st.session_state["dataset_path"] = chosen
    mtime = Path(chosen).stat().st_mtime
    frame = load_dataset(chosen, mtime)
    right.markdown(f'<div class="rs rs-note" style="padding-bottom:10px">{len(frame):,} clients &middot; '
                   f'{len(frame.columns)} columns &middot; created {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}</div>',
                   unsafe_allow_html=True)
    if not REQUIRED <= set(frame.columns):
        st.warning("This file has no RiskSynth profile columns. Pick another dataset.")
        return None
    return frame, chosen, mtime


def empty_state(key: str) -> None:
    with st.container(border=True):
        st.markdown('<div class="rs"><div class="rs-h2">No dataset yet</div><div class="rs-sub">'
                    'This deployment starts empty. Create a 2,000-client demo dataset with the bootstrap sampler '
                    '(about 5 seconds), or configure a run in the Generator.</div></div>', unsafe_allow_html=True)
        a, b, _ = st.columns([1.3, 1.3, 3])
        if a.button("Create demo dataset", type="primary", key=f"demo_{key}"):
            cfg = wealth_profile.load_config()
            target = cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]["target_distribution"]
            run_generation("Bootstrap sampler", "copula", 2000, cfg["edge_case_profiles"]["total_target_share"], 42,
                           {t: 100 * target[t] for t in TIER_ORDER}, cfg)
        if "generator" in PAGES:
            with b:
                st.page_link(PAGES["generator"], label="Open the Generator")


def run_generation(method: str, sdv_model: str, n: int, edge_share: float, seed: int, mix: dict, cfg: dict) -> None:
    """Run one generation, write exports, metadata and the run contract, select it as the active dataset."""
    slug = {"Rule-based": "rulebased", "Bootstrap sampler": "sampler"}.get(method, sdv_model)
    run_dir = RUNS_DIR / f"{slug}_{n}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()

    with st.status(f"Generating {n:,} profiles ({method})", expanded=True) as status:
        out, report, files = contract.generate(method, n, seed, edge_share, mix, sdv_model=sdv_model, cfg=cfg,
                                               save_dir=None if method == "Rule-based" else run_dir,
                                               progress=st.write)
        st.write("Writing exports and the run contract")
        out.to_parquet(run_dir / "profiles.parquet", index=False)
        if not (run_dir / "profiles.csv").exists():
            out.to_csv(run_dir / "profiles.csv", index=False)
        stage1 = None if method == "Rule-based" else ("bootstrap" if method == "Bootstrap sampler" else sdv_model)
        run_contract = contract.build(method=method, stage1_model=stage1, sdv_model=sdv_model, rows=n, seed=seed,
                                      edge_case_share=edge_share, risk_mix_pct=mix, report=report, frame=out, cfg=cfg)
        (run_dir / "run_contract.json").write_text(json.dumps(run_contract, indent=2, default=str, ensure_ascii=False),
                                                   encoding="utf-8")
        meta = {"generated_at": started.isoformat(timespec="seconds"), "method": method,
                "stage1_model": stage1, "rows": len(out), "seed": seed, "edge_case_share": edge_share,
                "target_risk_mix_pct": mix,
                "runtime_seconds": round((datetime.now() - started).total_seconds(), 1),
                "files": {"parquet": "profiles.parquet", "csv": "profiles.csv", "contract": "run_contract.json",
                          **{k: Path(v).name for k, v in files.items() if v}},
                "wealth_report": report}
        (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False),
                                                   encoding="utf-8")
        status.update(label=f"Done: {len(out):,} profiles in {meta['runtime_seconds']}s", state="complete",
                      expanded=False)
    st.session_state["last_run"] = str(run_dir)
    st.session_state["dataset_path"] = str(run_dir / "profiles.parquet")
    st.rerun()
