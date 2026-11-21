"""
RiskSynth AI - Synthetic Client Risk Profile Generator (Streamlit UI).

Tabs
  1. Overview        population KPIs, risk-tier mix vs target, surplus distribution,
                     edge-case cohort, capacity vs stated tolerance
  2. Generator       on-demand generation (rule-based / sampler / SDV) with risk-mix and
                     edge-case overrides; CSV, Parquet and JSON metadata exports
  3. Advisory        model-portfolio simulator: tolerance-only vs capacity vs drawdown-guarded
                     allocations per client and per stress profile
  4. Fidelity        KS / Wasserstein / Jensen-Shannon vs INE, correlation heatmaps (injected
                     edge cases excluded), CR01-CR07 + MiFID II checks, HTML report

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

try:
    import risksynth  # noqa: F401  (installed with: pip install -e .)
except ImportError:      # running from a plain checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risksynth.engine import wealth_profile  # noqa: E402
from risksynth.evaluation import fidelity  # noqa: E402
from risksynth.io import read_profiles  # noqa: E402
from risksynth.paths import PROCESSED_DIR, SYNTHETIC_DIR  # noqa: E402
from risksynth.scoring import robo_advisor  # noqa: E402

SYNTH_DIR = SYNTHETIC_DIR
RUNS_DIR = SYNTH_DIR / "app_runs"

# Palette (validated with the dataviz validator, light surface)
TIER_ORDER = ["Conservative", "Moderate", "Aggressive", "Ultra-Aggressive"]
TIER_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # ordinal blue ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                          # categorical slots 1-3
SEQ_RANGE = ["#cde2fb", "#0d366b"]
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]
INK, INK_2, GRID = "#161616", "#5c5c5c", "#ececec"

EC_SHORT = {"EC01": "High income, low liquidity", "EC02": "Young, high risk, low capital",
            "EC03": "Near-retiree, aggressive", "EC04": "Negative surplus investor",
            "EC05": "Over-leveraged risk seeker", "EC06": "Retiree, income, high liquidity",
            "EC07": "Short horizon, aggressive", "EC08": "Inconsistent questionnaire",
            "EC09": "Asset-rich, cash-poor"}

st.set_page_config(page_title="RiskSynth AI", page_icon=":material/monitoring:", layout="wide")

st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }}
  .rs-brand {{ font-size: 21px; font-weight: 650; letter-spacing: -0.015em; color: {INK}; line-height: 1.2; }}
  .rs-brand b {{ color: #2a78d6; font-weight: 650; }}
  .rs-sub {{ color: {INK_2}; font-size: 13px; margin-top: 2px; }}
  .rs-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: {INK_2};
              margin: 18px 0 6px 0; font-weight: 600; }}
  .rs-note {{ color: {INK_2}; font-size: 12.5px; }}
  .rs-kpi {{ border-top: 1px solid #d6d6d6; padding-top: 10px; margin-bottom: 14px; }}
  .rs-kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: {INK_2}; font-weight: 600; }}
  .rs-kpi-value {{ font-size: 25px; font-weight: 500; font-variant-numeric: tabular-nums; color: {INK};
                  line-height: 1.35; margin-top: 2px; }}
  .rs-kpi-note {{ font-size: 12px; color: {INK_2}; min-height: 16px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid #e4e4e4; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 14px; font-weight: 550; padding: 10px 0; }}
  dl.rs-facts {{ display: grid; grid-template-columns: max-content 1fr; gap: 5px 18px; font-size: 13.5px; margin: 0; }}
  dl.rs-facts dt {{ color: {INK_2}; }}
  dl.rs-facts dd {{ margin: 0; font-weight: 550; font-variant-numeric: tabular-nums; }}
  .rs-action {{ border-left: 3px solid #d03b3b; padding: 3px 10px; margin: 5px 0; font-size: 13.5px; }}
  .rs-action.ok {{ border-left-color: #0ca30c; }}
  .rs-expect {{ border: 1px solid #e4e4e4; padding: 10px 12px; font-size: 13.5px; margin-top: 10px; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _has_profiles(path: Path) -> bool:
    try:
        if path.suffix == ".parquet":
            return "risk_tolerance_score" in pq.read_schema(path).names
        with open(path, encoding="utf-8") as f:
            return "risk_tolerance_score" in f.readline()
    except (OSError, ValueError):   # UnicodeDecodeError is a ValueError
        return False


def discover_datasets() -> list[Path]:
    patterns = ["app_runs/*/profiles.parquet", "synthetic_*.parquet",
                "wealth_*/wealth_profiles.parquet", "clients/*/clients_synthetic.csv"]
    found = {p for pat in patterns for p in SYNTH_DIR.glob(pat) if _has_profiles(p)}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False, max_entries=6)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    """mtime is part of the cache key so a regenerated file is reloaded."""
    return read_profiles(path)   # keeps the ESG category "None" when reading CSV


@st.cache_data(show_spinner=False, max_entries=6)
def advisory(path: str, mtime: float) -> pd.DataFrame:
    return robo_advisor.recommend(load_dataset(path, mtime))


@st.cache_data(show_spinner=False, max_entries=6)
def fidelity_results(path: str, mtime: float) -> dict:
    return fidelity.run_all(load_dataset(path, mtime))


@st.cache_data(show_spinner=False)
def stage1_table() -> pd.DataFrame:
    return wealth_profile.build_stage1_table()


@st.cache_data(show_spinner=False)
def combined_base() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "combined_base.parquet")


def rel(path: Path) -> str:
    return path.relative_to(SYNTH_DIR).as_posix()


# --------------------------------------------------------------------------- #
# Chart helpers (Altair): thin marks, recessive axes, tooltips everywhere
# --------------------------------------------------------------------------- #
def styled(chart, height: int = 260):
    return (chart.properties(height=height)
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor=GRID, domainColor="#cfcfcf",
                            tickColor="#cfcfcf", labelFontSize=11, titleFontSize=11, titleFontWeight=500)
            .configure_legend(labelColor=INK, titleColor=INK_2, labelFontSize=11, titleFontSize=11,
                              orient="top", direction="horizontal", symbolType="square")
            .configure_title(fontSize=13, fontWeight=600, color=INK, anchor="start", subtitleColor=INK_2,
                             subtitleFontSize=11.5))


def show(chart, height: int = 260) -> None:
    st.altair_chart(styled(chart, height), width="stretch")


def kpi(col, label: str, value: str, note: str = "") -> None:
    """Flat KPI block: uppercase label, tabular value, plain grey note (no delta pills)."""
    col.markdown(f"<div class='rs-kpi'><div class='rs-kpi-label'>{label}</div>"
                 f"<div class='rs-kpi-value'>{value}</div><div class='rs-kpi-note'>{note}</div></div>",
                 unsafe_allow_html=True)


def eur(x: float) -> str:
    return f"EUR {x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


# --------------------------------------------------------------------------- #
# Header + dataset selection
# --------------------------------------------------------------------------- #
datasets = discover_datasets()
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
with head_l:
    st.markdown('<div class="rs-brand">Risk<b>Synth</b> AI</div>'
                '<div class="rs-sub">Synthetic client risk profiles for robo-advisory testing &middot; '
                'calibrated on INE microdata (ECV, EPF, EPA) &middot; MiFID II suitability fields</div>',
                unsafe_allow_html=True)
with head_r:
    if datasets:
        default = st.session_state.get("dataset_path")
        options = [str(p) for p in datasets]
        index = options.index(default) if default in options else 0
        chosen = st.selectbox("Dataset", options, index=index, format_func=lambda s: rel(Path(s)),
                              help="Generated profile files under data/synthetic/ (newest first)")
        st.session_state["dataset_path"] = chosen
    else:
        chosen = None
        st.info("No profile datasets yet: create one in the Generator tab.")

df = None
if chosen:
    mtime = Path(chosen).stat().st_mtime
    df = load_dataset(chosen, mtime)

tab_overview, tab_generator, tab_advisory, tab_fidelity = st.tabs(
    ["Overview", "Generator", "Advisory simulator", "Fidelity & compliance"])


# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
def render_overview(df: pd.DataFrame) -> None:
    cfg = wealth_profile.load_config()
    target = cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]["target_distribution"]
    has_ec = "edge_case_id" in df
    k = st.columns(6)
    kpi(k[0], "Clients", f"{len(df):,}")
    if has_ec:
        target_ec = 100 * cfg["edge_case_profiles"]["total_target_share"]
        share = 100 * df["is_edge_case"].astype(bool).mean()
        kpi(k[1], "Edge-case cohort", pct(share, 2), f"{share - target_ec:+.2f} pp vs {target_ec:.0f}% target")
    else:
        kpi(k[1], "Edge-case cohort", "n/a", "no stage-3 columns")
    kpi(k[2], "Median investable surplus", eur(df["investable_surplus_eur"].median()), "per year")
    kpi(k[3], "Negative raw surplus", pct(100 * (df["raw_surplus_eur"] < 0).mean()), "spending + debt > income")
    bound = 100 * (df["effective_risk_score"] < df["risk_tolerance_score"]).mean()
    kpi(k[4], "Capacity-bound", pct(bound), "effective < stated risk")
    flagged = int(df.get("profile_inconsistency_flag", pd.Series(False)).astype(bool).sum())
    kpi(k[5], "Re-profiling flags", f"{flagged:,}", "inconsistent answers")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        mix = (100 * df["risk_tolerance_category"].astype("string").value_counts(normalize=True)) \
            .reindex(TIER_ORDER).fillna(0).rename("actual").to_frame()
        mix["target"] = [100 * target[t] for t in TIER_ORDER]
        mix["diff"] = mix["actual"] - mix["target"]
        mix["label_x"] = mix[["actual", "target"]].max(axis=1)   # label clears both bar and tick
        mix = mix.reset_index(names="tier")
        base = alt.Chart(mix).encode(y=alt.Y("tier:N", sort=TIER_ORDER, title=None))
        bars = base.mark_bar(height=22, cornerRadiusEnd=3).encode(
            x=alt.X("actual:Q", title="% of clients", scale=alt.Scale(domain=[0, max(45, mix.actual.max() + 5)])),
            color=alt.Color("tier:N", scale=alt.Scale(domain=TIER_ORDER, range=TIER_COLORS), legend=None),
            tooltip=[alt.Tooltip("tier:N", title="Tier"), alt.Tooltip("actual:Q", format=".2f", title="Actual %"),
                     alt.Tooltip("target:Q", format=".1f", title="Target %"),
                     alt.Tooltip("diff:Q", format="+.2f", title="Diff pp")])
        ticks = base.mark_tick(color=INK, thickness=2, size=30).encode(x="target:Q", tooltip=["tier", "target"])
        labels = base.mark_text(align="left", dx=8, color=INK, fontSize=11).encode(
            x="label_x:Q", text=alt.Text("actual:Q", format=".1f"))
        show((bars + ticks + labels).properties(
            title=alt.Title("Risk tolerance tiers", subtitle="Bars: actual share. Black ticks: configured target mix")), height=300)
    with c2:
        s = df["raw_surplus_eur"].dropna()
        lo, hi = s.quantile([0.01, 0.99])
        counts, edges = np.histogram(s.clip(lo, hi), bins=40)
        hist = pd.DataFrame({"from": edges[:-1], "to": edges[1:], "clients": counts})
        neg = 100 * (s < 0).mean()
        bars = alt.Chart(hist).mark_bar(color=SERIES[0], binSpacing=1).encode(
            x=alt.X("from:Q", bin="binned", title="Raw surplus, EUR / year (p1-p99)", axis=alt.Axis(format="~s")),
            x2="to:Q", y=alt.Y("clients:Q", title="Clients"),
            tooltip=[alt.Tooltip("from:Q", format=",.0f", title="From EUR"), alt.Tooltip("to:Q", format=",.0f", title="To EUR"),
                     alt.Tooltip("clients:Q", format=",", title="Clients")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="x:Q")
        show((bars + zero).properties(title=alt.Title(
            "Household surplus distribution",
            subtitle=f"{neg:.1f}% below zero: their investable surplus is floored at EUR 0 (dashed line)")), height=300)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        if has_ec:
            cohort = df[df["is_edge_case"].astype(bool)]
            tab = (cohort.groupby(["edge_case_id", "edge_case_origin"]).size().rename("rows").reset_index())
            tab["share"] = 100 * tab["rows"] / len(df)
            tab["profile"] = tab["edge_case_id"] + "  " + tab["edge_case_id"].map(EC_SHORT)
            order = sorted(tab["profile"].unique())
            targets = pd.DataFrame([{"profile": f"{p['id']}  {EC_SHORT[p['id']]}", "target": 100 * p["target_share"]}
                                    for p in cfg["edge_case_profiles"]["profiles"]])
            bars = alt.Chart(tab).mark_bar(stroke="white", strokeWidth=2).encode(
                y=alt.Y("profile:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("share:Q", stack="zero", title="% of all clients"),
                color=alt.Color("edge_case_origin:N", title="Origin", scale=alt.Scale(
                    domain=["natural", "injected"], range=SERIES[:2])),
                order=alt.Order("edge_case_origin:N", sort="descending"),
                tooltip=["edge_case_id", "edge_case_origin", alt.Tooltip("rows:Q", format=","),
                         alt.Tooltip("share:Q", format=".2f", title="% of clients")])
            ticks = alt.Chart(targets).mark_tick(color=INK, thickness=2, size=26).encode(
                y=alt.Y("profile:N", sort=order), x="target:Q", tooltip=["profile", "target"])
            show((bars + ticks).properties(title=alt.Title(
                "Edge-case stress cohort", subtitle="Natural matches fill each quota first; injection tops up. Ticks: target")),
                height=470)
        else:
            st.info("This dataset has no edge-case columns (generated before stage 3).")
    with c4:
        grid = (df.groupby(["risk_tolerance_score", "effective_risk_score"]).size()
                .rename("clients").reset_index())
        heat = alt.Chart(grid).mark_rect(stroke="white", strokeWidth=1).encode(
            x=alt.X("risk_tolerance_score:O", title="Stated tolerance (1-10)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("effective_risk_score:O", title="Effective risk (1-10)", sort="descending"),
            color=alt.Color("clients:Q", title="Clients", scale=alt.Scale(range=SEQ_RANGE, type="sqrt")),
            tooltip=[alt.Tooltip("risk_tolerance_score:O", title="Stated"),
                     alt.Tooltip("effective_risk_score:O", title="Effective"),
                     alt.Tooltip("clients:Q", format=",")])
        show(heat.properties(title=alt.Title(
            "Capacity vs stated tolerance",
            subtitle="Cells below the diagonal: MiFID II capacity caps the risk a client asked for")), height=470)

    with st.expander("Data tables"):
        t1, t2 = st.columns(2)
        t1.dataframe(df["risk_tolerance_category"].value_counts().rename("clients"), width="stretch")
        if has_ec:
            t2.dataframe(df.groupby("edge_case_id").agg(rows=("edge_case_id", "size"),
                                                        injected=("edge_case_origin", lambda s: int((s == "injected").sum()))),
                         width="stretch")


# --------------------------------------------------------------------------- #
# 2. Generator
# --------------------------------------------------------------------------- #
METHODS = {"Rule-based": "Full banking payload per client (accounts, loans, transactions) + wealth profile. "
                         "Sequential in the app: up to 5,000 rows.",
           "Sampler": "Weighted bootstrap of observed INE (EPF) households, then fresh wealth features. "
                      "Fast baseline; no SDV needed.",
           "SDV": "CTGAN / Gaussian copula / TVAE trained on EPF households, then stages 2-3."}


def render_generator() -> None:
    from risksynth.generators.client_generator import sdv_available
    has_sdv = sdv_available()
    cfg = wealth_profile.load_config()
    target = cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]["target_distribution"]

    with st.form("generate"):
        a, b, c = st.columns([1.2, 1, 1], gap="large")
        with a:
            method = st.radio("Method", list(METHODS), horizontal=True,
                              captions=None, help="Stage 1 engine; stages 2 and 3 always run")
            # Form widgets only update on submit, so describe every method rather than the selected one
            st.markdown("<div class='rs-note'>" + "<br>".join(f"<b>{k}</b>: {v}" for k, v in METHODS.items())
                        + "</div>", unsafe_allow_html=True)
            sdv_model = st.selectbox("SDV model", ["copula", "ctgan", "vae"], disabled=not has_sdv,
                                     help=None if has_sdv else "SDV is not installed in this environment")
        with b:
            n = st.number_input("Sample size", min_value=500, max_value=50000, value=5000, step=500)
            edge_share = st.slider("Edge-case share", 0.0, 15.0,
                                   100 * cfg["edge_case_profiles"]["total_target_share"], 0.5, format="%.1f%%")
            seed = st.number_input("Seed", min_value=0, max_value=10**6, value=42, step=1)
        with c:
            st.markdown("<div class='rs-label'>Target risk mix (%)</div>", unsafe_allow_html=True)
            mix = {}
            m1, m2 = st.columns(2)
            for i, tier in enumerate(TIER_ORDER):
                mix[tier] = (m1 if i % 2 == 0 else m2).number_input(
                    tier, min_value=0.0, max_value=100.0, value=round(100 * target[tier], 1), step=1.0)
            st.caption("Must sum to 100%; checked when you generate.")
        submitted = st.form_submit_button("Generate profiles", type="primary")

    if submitted:
        problems = []
        if abs(sum(mix.values()) - 100) > 0.01:
            problems.append("The target risk mix must add up to 100%.")
        if method == "SDV" and not has_sdv:
            problems.append("SDV is not installed (pip install sdv); use Sampler or Rule-based.")
        if method == "Rule-based" and n > 5000:
            problems.append("Rule-based runs in-process in the app: keep it at 5,000 rows or fewer "
                            "(use generate_synthetic.py for larger runs).")
        if problems:
            for msg in problems:
                st.error(msg)
        else:
            run_generation(method, sdv_model, int(n), edge_share / 100, int(seed), mix, cfg)

    last = st.session_state.get("last_run")
    if last:
        render_run_summary(last)


def run_generation(method: str, sdv_model: str, n: int, edge_share: float, seed: int,
                   mix: dict, cfg: dict) -> None:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"][
        "target_distribution"] = {k: v / 100 for k, v in mix.items()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{method.lower().replace('-', '')}_{n}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()

    with st.status(f"Generating {n:,} profiles ({method})", expanded=True) as status:
        if method == "Rule-based":
            from risksynth.generators import generate_synthetic as gs
            st.write("Stage 1: sampling INE persons base and resolving household finances")
            base = combined_base()
            st.write("Building banking payloads (accounts, loans, transactions)")
            out = gs.generate_synthetic_data(n, base, n_jobs=1, batch_size=n, seed=seed,
                                             wealth_cfg=run_cfg, edge_case_share=edge_share)
            report_file = sorted(gs.OUTPUT_DIR.glob(f"synthetic_{n}_*_wealth_report.json"))[-1]
            report = json.loads(report_file.read_text(encoding="utf-8"))
            files = {}
        else:
            from risksynth.generators.client_generator import ClientGenerator
            model = "bootstrap" if method == "Sampler" else sdv_model
            st.write(f"Stage 1: fitting {model} on {len(stage1_table()):,} EPF households")
            gen = ClientGenerator(n_samples=n, method=model, seed=seed)
            gen.fit(stage1_table())
            st.write("Stages 2-3: wealth features, CR01-CR07 / MiFID II, edge-case cohort")
            out = gen.generate(wealth_config=run_cfg, edge_case_share=edge_share)
            files = gen.save(str(run_dir), out)
            report = gen.wealth_report
        st.write("Writing exports")
        out.to_parquet(run_dir / "profiles.parquet", index=False)
        csv_path = run_dir / "profiles.csv"
        if not csv_path.exists():
            out.to_csv(csv_path, index=False)
        meta = {"generated_at": started.isoformat(timespec="seconds"), "method": method,
                "stage1_model": None if method == "Rule-based" else ("bootstrap" if method == "Sampler" else sdv_model),
                "rows": len(out), "seed": seed, "edge_case_share": edge_share, "target_risk_mix_pct": mix,
                "runtime_seconds": round((datetime.now() - started).total_seconds(), 1),
                "files": {"parquet": "profiles.parquet", "csv": "profiles.csv", **{k: Path(v).name for k, v in files.items() if v}},
                "wealth_report": report}
        (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False),
                                                   encoding="utf-8")
        status.update(label=f"Done: {len(out):,} profiles in {meta['runtime_seconds']}s", state="complete",
                      expanded=False)
    st.session_state["last_run"] = str(run_dir)
    st.session_state["dataset_path"] = str(run_dir / "profiles.parquet")
    st.rerun()


def render_run_summary(run_dir: str) -> None:
    run = Path(run_dir)
    meta_file = run / "run_metadata.json"
    if not meta_file.exists():
        return
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    rep = meta["wealth_report"]
    st.markdown("<div class='rs-label'>Last run</div>", unsafe_allow_html=True)
    k = st.columns(5)
    kpi(k[0], "Rows", f"{meta['rows']:,}", f"{meta['method']} - {meta['runtime_seconds']}s")
    ec = rep.get("edge_cases") or {}
    if ec:
        kpi(k[1], "Edge cases", pct(ec["achieved_total_share_pct"], 2), f"target {ec['target_total_share_pct']:.2f}%")
    kpi(k[2], "Raw rule violations", f"{sum(rep['violations_raw'].values()):,}", "before enforcement")
    kpi(k[3], "After enforcement", f"{sum(rep['violations_enforced'].values()):,}", f"exempted {sum(rep.get('violations_exempted', {}).values()):,} (EC08)")
    kpi(k[4], "Negative raw surplus", pct(rep["summary"]["share_raw_surplus_negative_pct"]))
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    d1.download_button("Download CSV", (run / "profiles.csv").read_bytes(), "risksynth_profiles.csv", "text/csv")
    d2.download_button("Download Parquet", (run / "profiles.parquet").read_bytes(), "risksynth_profiles.parquet",
                       "application/octet-stream")
    d3.download_button("Download JSON metadata", meta_file.read_bytes(), "risksynth_run_metadata.json",
                       "application/json")
    st.caption(f"Saved to data/synthetic/{rel(run)}/ and selected as the active dataset.")


# --------------------------------------------------------------------------- #
# 3. Advisory simulator
# --------------------------------------------------------------------------- #
SLEEVES = ["Equities", "Fixed income", "Cash"]


def render_advisory(df: pd.DataFrame, rec: pd.DataFrame) -> None:
    has_ec = "edge_case_id" in df
    groups = ["All clients"]
    if has_ec:
        groups += ["Base population"] + [f"{e}  {EC_SHORT[e]}" for e in sorted(df["edge_case_id"].dropna().unique())]
    a, b, c = st.columns([1.3, 1.6, 0.6], vertical_alignment="bottom")
    group = a.selectbox("Stress profile", groups)
    if group == "All clients":
        pool = df.index
    elif group == "Base population":
        pool = df.index[df["edge_case_id"].isna()]
    else:
        pool = df.index[df["edge_case_id"].eq(group.split()[0]).fillna(False)]
    pool = pool[:400]
    if c.button("Random client"):
        st.session_state["sim_client"] = int(np.random.default_rng().choice(pool))
    current = st.session_state.get("sim_client")
    options = list(pool)
    idx = options.index(current) if current in options else 0
    client = b.selectbox("Client", options, index=idx, format_func=lambda i: (
        f"#{i}  age {int(df.at[i, 'edad'])}  {df.at[i, 'risk_tolerance_category']}  "
        f"{df.at[i, 'edge_case_id'] if has_ec and pd.notna(df.at[i, 'edge_case_id']) else ''}"))
    row, r = df.loc[client], rec.loc[client]

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown("<div class='rs-label'>Client profile</div>", unsafe_allow_html=True)
        facts = {
            "Age / goal": f"{int(row['edad'])} / {row['primary_investment_goal']}",
            "Horizon": f"{row['investment_horizon_years']:.1f} y ({row['investment_horizon_bucket']})",
            "Stated tolerance": f"{int(row['risk_tolerance_score'])} ({row['risk_tolerance_category']})",
            "Risk capacity": f"{int(row['risk_capacity_score'])}",
            "Effective risk": f"{int(row['effective_risk_score'])}",
            "Max drawdown accepted": pct(row["max_acceptable_drawdown_pct"], 0),
            "Liquidity need": pct(row["liquidity_need_pct"], 0),
            "Investable surplus": eur(row["investable_surplus_eur"]) + " / y",
            "Liquid savings": eur(row["liquid_savings_eur"]),
            "Debt service / income": pct(100 * (row["debt_service_to_income"] if pd.notna(row["debt_service_to_income"]) else 0)),
            "ESG preference": str(row["esg_preference_level"]),
            "Capacity caps": str(row.get("capacity_caps_applied")) if pd.notna(row.get("capacity_caps_applied")) else "none",
        }
        st.markdown("<dl class='rs-facts'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts.items())
                    + "</dl>", unsafe_allow_html=True)
        if has_ec and pd.notna(row.get("edge_case_id")):
            verdict = "assertion holds" if bool(row.get("edge_case_assertion_pass")) else "ASSERTION FAILS"
            st.markdown(f"<div class='rs-expect'><b>{row['edge_case_id']} {EC_SHORT[row['edge_case_id']]}</b> "
                        f"({row['edge_case_origin']})<br>Expected behaviour: {row['expected_robo_behavior']}<br>"
                        f"<span class='rs-note'>Test: <code>{row['edge_case_assertion']}</code> - {verdict}</span></div>",
                        unsafe_allow_html=True)
        st.markdown("<div class='rs-label'>Advisor actions</div>", unsafe_allow_html=True)
        raised = [lbl for col, lbl in robo_advisor.ACTION_LABELS.items() if bool(r[col])]
        if bool(r["drawdown_guard_applied"]):
            raised.append("Equity cut to keep the stressed loss within the accepted drawdown")
        if raised:
            st.markdown("".join(f"<div class='rs-action'>{t}</div>" for t in raised), unsafe_allow_html=True)
        else:
            st.markdown("<div class='rs-action ok'>No suitability actions: tolerance-based allocation stands</div>",
                        unsafe_allow_html=True)

    with right:
        steps = [("1  Tolerance only", "naive"), ("2  + Capacity & liquidity", "capacity"), ("3  + Drawdown guard", "rec")]
        rows = []
        for label, pre in steps:
            vals = [r[f"{pre}_equity_pct"], r[f"{pre}_fixed_income_pct"], r[f"{pre}_cash_pct"]]
            rows += [{"step": label, "sleeve": s, "pct": float(v), "order": i} for i, (s, v) in enumerate(zip(SLEEVES, vals))]
        alloc = pd.DataFrame(rows)
        bars = alt.Chart(alloc).mark_bar(height=30, stroke="white", strokeWidth=2).encode(
            y=alt.Y("step:N", title=None, sort=[s for s, _ in steps], axis=alt.Axis(labelLimit=220)),
            x=alt.X("pct:Q", stack="zero", title="% of portfolio", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("sleeve:N", title=None, scale=alt.Scale(domain=SLEEVES, range=SERIES)),
            order=alt.Order("order:Q"),
            tooltip=["step", "sleeve", alt.Tooltip("pct:Q", format=".1f", title="%")])
        text = alt.Chart(alloc[alloc["pct"] >= 7]).mark_text(color="white", fontSize=11, fontWeight=600).encode(
            y=alt.Y("step:N", sort=[s for s, _ in steps]), x=alt.X("pct:Q", stack="zero", bandPosition=0.5),
            detail="sleeve:N", order=alt.Order("order:Q"), text=alt.Text("pct:Q", format=".0f"))
        show((bars + text).properties(title=alt.Title(
            "Allocation: how each protection changes the portfolio",
            subtitle="Stated tolerance -> effective risk with liquidity sleeve -> stressed-loss guard")), height=250)
        m = st.columns(4)
        kpi(m[0], "Stressed loss", pct(r["rec_stress_loss_pct"]), f"limit {row['max_acceptable_drawdown_pct']:.0f}%")
        kpi(m[1], "Tolerance-only loss", pct(r["naive_stress_loss_pct"]), "same stress")
        kpi(m[2], "Equity cut", f"{r['equity_reduction_pp']:.1f} pp", f"capacity {r['equity_cut_by_capacity_pp']:.1f} / guard {r['equity_cut_by_drawdown_pp']:.1f}")
        kpi(m[3], "ESG-screened", pct(r["rec_esg_pct"]), f"preference {row['esg_preference_level']}")

    st.markdown("<div class='rs-label'>Protection by stress profile</div>", unsafe_allow_html=True)
    key = df["edge_case_id"].fillna("Base") if has_ec else pd.Series("All", index=df.index)
    agg = pd.concat([key.rename("group"), rec], axis=1).groupby("group").agg(
        naive=("naive_equity_pct", "mean"), capacity=("capacity_equity_pct", "mean"),
        recommended=("rec_equity_pct", "mean"), clients=("rec_equity_pct", "size"),
        reprofile=("action_reprofile", "mean"), capacity_bound=("action_capacity_bound", "mean"),
        no_contrib=("action_no_new_contributions", "mean"), guard=("drawdown_guard_applied", "mean"))
    agg = agg.reset_index()
    agg["label"] = agg["group"].map(lambda g: f"{g}  {EC_SHORT[g]}" if g in EC_SHORT else f"{g} population")
    long = agg.melt(id_vars=["label"], value_vars=["naive", "capacity", "recommended"],
                    var_name="allocation", value_name="equity")
    names = {"recommended": "Recommended (3)", "naive": "Tolerance only (1)", "capacity": "Capacity (2)"}
    long["allocation"] = long["allocation"].map(names)
    order = list(agg.sort_values("group")["label"])
    rule = alt.Chart(agg).mark_rule(color="#b9b9b9", strokeWidth=2).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)), x="recommended:Q", x2="naive:Q")
    dots = alt.Chart(long).mark_circle(size=110, opacity=1, stroke="white", strokeWidth=2).encode(
        y=alt.Y("label:N", sort=order), x=alt.X("equity:Q", title="Mean equity share, % of portfolio"),
        color=alt.Color("allocation:N", title=None, scale=alt.Scale(   # legend in step order; colour fixed per step
            domain=[names["naive"], names["capacity"], names["recommended"]],
            range=[SERIES[1], SERIES[2], SERIES[0]])),
        tooltip=["label", "allocation", alt.Tooltip("equity:Q", format=".1f", title="Equity %")])
    show((rule + dots).properties(title=alt.Title(
        "Mean equity share per stress profile",
        subtitle="Gap between dots = equity the suitability layer removes; EC08 is driven to cash/bonds by its 5-10% drawdown limit")),
        height=460)
    table = agg[["group", "clients", "naive", "capacity", "recommended", "capacity_bound", "guard", "no_contrib", "reprofile"]]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "group": "Profile", "clients": st.column_config.NumberColumn("Clients", format="%d"),
        "naive": st.column_config.NumberColumn("Equity, tolerance only %", format="%.1f"),
        "capacity": st.column_config.NumberColumn("Equity, capacity %", format="%.1f"),
        "recommended": st.column_config.NumberColumn("Equity, recommended %", format="%.1f"),
        "capacity_bound": st.column_config.NumberColumn("Capacity-bound", format="percent"),
        "guard": st.column_config.NumberColumn("Drawdown guard", format="percent"),
        "no_contrib": st.column_config.NumberColumn("No contributions", format="percent"),
        "reprofile": st.column_config.NumberColumn("Re-profile", format="percent")})
    st.caption("Model portfolios are illustrative ([ASSUMPTION] parameters in model_config.yaml -> "
               "robo_advisor_simulator); stress losses: equities -50%, fixed income -10%, cash 0%.")


# --------------------------------------------------------------------------- #
# 4. Fidelity & compliance
# --------------------------------------------------------------------------- #
def status_table(frame: pd.DataFrame, **config) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "status": st.column_config.TextColumn("Status", help="PASS / FAIL against model_config.yaml thresholds"),
        **config})


SHORT = {"net_income_annual_eur": "Net income", "household_expenses_annual_eur": "Spending",
         "debt_service_annual_eur": "Debt service", "household_surplus_eur": "Surplus",
         "main_residence_value_eur": "Home value", "edad": "Age", "n_miembros": "Household size",
         "raw_surplus_eur": "Raw surplus", "liquid_savings_eur": "Liquid savings",
         "risk_tolerance_score": "Tolerance", "investment_horizon_years": "Horizon",
         "max_acceptable_drawdown_pct": "Max drawdown", "liquidity_need_pct": "Liquidity need",
         "risk_capacity_score": "Capacity", "effective_risk_score": "Effective risk"}


def heatmap(matrix: pd.DataFrame, title: str, subtitle: str, domain: float) -> alt.Chart:
    """Diverging blue-grey-red matrix; tooltips keep the full column names."""
    m = matrix.copy()
    m.index.name = "row"
    long = m.reset_index().melt(id_vars="row", var_name="col", value_name="value")
    long["row_label"] = long["row"].map(lambda c: SHORT.get(c, c))
    long["col_label"] = long["col"].map(lambda c: SHORT.get(c, c))
    long["dark"] = long["value"].abs() >= 0.6 * domain          # white text on saturated cells
    order = [SHORT.get(c, c) for c in matrix.columns]
    base = alt.Chart(long).encode(
        x=alt.X("col_label:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelLimit=140)),
        y=alt.Y("row_label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)))
    rect = base.mark_rect(stroke="white", strokeWidth=1).encode(
        # interpolate in Lab so the midpoint stays neutral grey (default HCL drifts through pink/purple)
        color=alt.Color("value:Q", title=None, scale=alt.Scale(domain=[-domain, 0, domain], range=DIVERGING,
                                                               interpolate="lab", clamp=True)),
        tooltip=[alt.Tooltip("row:N", title="Row"), alt.Tooltip("col:N", title="Column"),
                 alt.Tooltip("value:Q", format="+.3f", title="Value")])
    text = base.mark_text(fontSize=9.5).encode(
        text=alt.Text("value:Q", format=".2f"),
        color=alt.condition("datum.dark", alt.value("white"), alt.value(INK)))
    return (rect + text).properties(title=alt.Title(title, subtitle=subtitle))


def render_fidelity(path: str, mtime: float) -> None:
    with st.spinner("Running fidelity and compliance tests"):
        res = fidelity_results(path, mtime)
    ref = ("INE persons base (ECV+EPF+EPA) for demographics, EPF households for finances"
           if res["reference_kind"] == "persons" else "INE EPF households (stage-1 table)")
    st.markdown(f"<div class='rs-note'>Reference: {ref}. Fidelity tests use {res['rows_base_population']:,} "
                f"base-population rows; {res['rows_injected_excluded']:,} injected edge cases are excluded. "
                "Rule checks use all rows.</div>", unsafe_allow_html=True)

    def passed(frame):
        return f"{int((frame['status'] == 'PASS').sum())} / {len(frame)}" if len(frame) else "n/a"
    comp = res["compliance"]
    k = st.columns(6)
    kpi(k[0], "Numeric tests", passed(res["numeric"]), "KS D and Wasserstein")
    kpi(k[1], "Categorical tests", passed(res["categorical"]), "Jensen-Shannon")
    kpi(k[2], "Target mixes", passed(res["target_mix"]), "risk tier, ESG")
    kpi(k[3], "Correlation drift", f"{res['correlation']['mean_abs_diff']:.3f}", f"mean |rho diff| - {res['correlation']['status']}")
    kpi(k[4], "Unexpected violations", f"{int(comp['unexpected'].sum()):,}", "CR01-CR07 + MiFID II")
    kpi(k[5], "Intentional (exempt)", f"{int(comp['exempted'].sum()):,}", "EC08 drawdown contradiction")

    st.markdown("<div class='rs-label'>Distribution fidelity vs INE</div>", unsafe_allow_html=True)
    status_table(res["numeric"], ks_p_value=st.column_config.NumberColumn("KS p-value", format="%.2e"))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        status_table(res["categorical"])
    with c2:
        status_table(res["target_mix"])

    st.markdown("<div class='rs-label'>Correlation structure</div>", unsafe_allow_html=True)
    h1, h2 = st.columns(2, gap="large")
    with h1:
        corr = res["correlation"]
        show(heatmap(corr["difference"], "Spearman difference: synthetic - INE",
                     f"Household finance + demographics. Mean |diff| {corr['mean_abs_diff']}, max {corr['max_abs_diff']}",
                     0.3), height=470)
    with h2:
        show(heatmap(res["wealth_correlation"], "Wealth feature correlations (synthetic)",
                     "Spearman rho, base population (injected edge cases excluded)", 1.0), height=470)
    status_table(res["dependencies"])

    st.markdown("<div class='rs-label'>Consistency rules and MiFID II suitability</div>", unsafe_allow_html=True)
    status_table(comp, definition=st.column_config.TextColumn("Rule", width="large"))

    st.markdown("<div class='rs-label'>Evaluation report</div>", unsafe_allow_html=True)
    if st.button("Build HTML report"):
        out = fidelity.save_report(res, rel(Path(path)))
        st.session_state["report_path"] = str(out)
    rp = st.session_state.get("report_path")
    if rp and Path(rp).exists():
        st.download_button("Download HTML report", Path(rp).read_bytes(), Path(rp).name, "text/html")
        st.caption(f"Saved to reports/evaluation_reports/{Path(rp).name}")


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
REQUIRED = {"risk_tolerance_score", "effective_risk_score", "raw_surplus_eur", "investable_surplus_eur"}

with tab_generator:
    render_generator()

if df is None:
    for t in (tab_overview, tab_advisory, tab_fidelity):
        with t:
            st.info("Generate a dataset first.")
elif not REQUIRED <= set(df.columns):
    for t in (tab_overview, tab_advisory, tab_fidelity):
        with t:
            st.warning("The selected file has no RiskSynth wealth columns. Pick another dataset.")
else:
    active = str(chosen)
    with tab_overview:
        render_overview(df)
    with tab_advisory:
        render_advisory(df, advisory(active, mtime))
    with tab_fidelity:
        render_fidelity(active, mtime)

st.markdown("<div class='rs-note' style='margin-top:36px;border-top:1px solid #e4e4e4;padding-top:10px'>"
            "Synthetic data only. Not calibrated to any real client book; not investment advice.</div>",
            unsafe_allow_html=True)
