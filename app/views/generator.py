"""Live profile generator: configure a run, watch it, export the result."""
import json
from pathlib import Path

import streamlit as st

from portal import PAGES
from portal.data import METHODS, rel, run_generation
from portal.theme import TIER_ORDER
from portal.icons import icon
from portal.ui import kpi_grid, pct, section
from risksynth.engine import wealth_profile
from risksynth.generators.client_generator import sdv_available

cfg = wealth_profile.load_config()
target = cfg["wealth_profile_schema"]["features"]["risk_tolerance_score"]["generation"]["target_distribution"]
has_sdv = sdv_available()

default_edge = round(100 * cfg["edge_case_profiles"]["total_target_share"], 1)
default_mix = {t: round(100 * target[t], 1) for t in TIER_ORDER}

# Test scenarios for a portfolio allocation model. Each one only fills in the form below.
PRESETS = {
    "baseline": ("layers", "Baseline mix",
                 f"Configured risk mix with {default_edge:g}% edge cases. The reference run for an allocation model.",
                 default_edge, default_mix),
    "edge": ("alert", "Edge-case stress",
             "EC01-EC09 raised to 15% of rows, so every archetype has enough clients to test on its own.",
             15.0, default_mix),
    "capacity": ("shield", "Capacity-constrained",
                 "Most clients state Aggressive or higher, so MiFID II capacity caps and re-profiling bind often.",
                 12.0, dict(zip(TIER_ORDER, [15.0, 25.0, 35.0, 25.0]))),
}


def apply_preset(name: str) -> None:
    _, _, _, edge, mix = PRESETS[name]
    st.session_state.update({"gen_method": "Bootstrap sampler", "gen_n": 5000, "gen_edge": edge, "gen_seed": 42,
                             **{f"gen_mix_{t}": v for t, v in mix.items()}})


def matches(name: str) -> bool:
    """True when the form currently holds this preset's values."""
    _, _, _, edge, mix = PRESETS[name]
    ss = st.session_state
    return (ss.get("gen_method") == "Bootstrap sampler" and ss.get("gen_n") == 5000 and ss.get("gen_edge") == edge
            and all(ss.get(f"gen_mix_{t}") == v for t, v in mix.items()))


if "gen_n" not in st.session_state:
    apply_preset("baseline")

section("Profile generator",
        "Build a synthetic client book to test a robo-advisor against. No real client data is used: every profile "
        "is generated from INE survey statistics, then given a suitability profile, rule checks and an edge-case "
        "cohort.", eyebrow="Workbench", first=True)

st.markdown('<div class="rs rs-label" style="margin-top:0">Test scenario</div>', unsafe_allow_html=True)
cols = st.columns(len(PRESETS), gap="small")
for col, (key, (ic, title, desc, _, _)) in zip(cols, PRESETS.items()):
    with col, st.container(border=True):
        on = matches(key)
        st.markdown(f'<div class="rs rs-preset{" on" if on else ""}"><div class="t">{icon(ic, 16)}{title}</div>'
                    f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)
        st.button("Selected" if on else "Use this scenario", key=f"preset_{key}", disabled=on,
                  on_click=apply_preset, args=(key,), width="stretch")

with st.form("generate", border=True):
    a, b, c = st.columns([1.3, 1, 1.1], gap="large")
    with a:
        st.markdown('<div class="rs rs-label" style="margin-top:0">Engine</div>', unsafe_allow_html=True)
        method = st.radio("Engine", list(METHODS), label_visibility="collapsed", key="gen_method")
        # Form widgets only update on submit, so every engine is described up front
        st.markdown('<div class="rs rs-note" style="line-height:1.6">' +
                    "<br>".join(f"<b>{k}</b>: {v}" for k, v in METHODS.items()) + "</div>", unsafe_allow_html=True)
        sdv_model = st.selectbox("SDV model", ["copula", "ctgan", "vae"], disabled=not has_sdv,
                                 help=None if has_sdv else "SDV is not installed in this environment")
    with b:
        st.markdown('<div class="rs rs-label" style="margin-top:0">Population</div>', unsafe_allow_html=True)
        n = st.number_input("Sample size", min_value=500, max_value=50000, step=500, key="gen_n")
        edge_share = st.slider("Edge-case share", 0.0, 15.0, step=0.5, format="%.1f%%", key="gen_edge",
                               help="Share of rows assigned to the EC01-EC09 stress archetypes")
        seed = st.number_input("Seed", min_value=0, max_value=10**6, step=1, key="gen_seed",
                               help="The same seed and settings reproduce the same data")
    with c:
        st.markdown('<div class="rs rs-label" style="margin-top:0">Target risk mix (%)</div>', unsafe_allow_html=True)
        mix = {}
        m1, m2 = st.columns(2)
        for i, tier in enumerate(TIER_ORDER):
            mix[tier] = (m1 if i % 2 == 0 else m2).number_input(
                tier, min_value=0.0, max_value=100.0, step=1.0, key=f"gen_mix_{tier}")
        st.caption("Must sum to 100%; checked when you generate.")
    submitted = st.form_submit_button("Generate profiles", type="primary")

if submitted:
    problems = []
    if abs(sum(mix.values()) - 100) > 0.01:
        problems.append("The target risk mix must add up to 100%.")
    if method.startswith("SDV") and not has_sdv:
        problems.append("SDV is not installed (pip install sdv); use the bootstrap sampler or the rule-based engine.")
    if method == "Rule-based" and n > 5000:
        problems.append("The rule-based engine runs in-process here: keep it at 5,000 rows or fewer "
                        "(use risksynth-generate for larger runs).")
    for msg in problems:
        st.error(msg)
    if not problems:
        run_generation(method, sdv_model, int(n), edge_share / 100, int(seed), mix, cfg)

last = st.session_state.get("last_run")
meta_file = Path(last) / "run_metadata.json" if last else None
if meta_file and meta_file.exists():
    run = meta_file.parent
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    rep = meta["wealth_report"]
    ec = rep.get("edge_cases") or {}
    section("Last run", f"Saved to <code>data/synthetic/{rel(run)}/</code> and selected as the active dataset.")
    kpi_grid([
        ("Rows", f"{meta['rows']:,}", f"{meta['method']}, {meta['runtime_seconds']}s"),
        ("Edge cases", pct(ec["achieved_total_share_pct"], 2) if ec else "n/a",
         f"target {ec['target_total_share_pct']:.2f}%" if ec else ""),
        ("Raw rule violations", f"{sum(rep['violations_raw'].values()):,}", "before repair"),
        ("After repair", f"{sum(rep['violations_enforced'].values()):,}",
         f"{sum(rep.get('violations_exempted', {}).values()):,} intentional (EC08)"),
        ("Negative raw surplus", pct(rep["summary"]["share_raw_surplus_negative_pct"]), "of households"),
    ])
    st.write("")
    d1, d2, d3, l1, l2 = st.columns([1, 1, 1.2, 1.2, 1.2])
    d1.download_button("Download CSV", (run / "profiles.csv").read_bytes(), "risksynth_profiles.csv", "text/csv",
                       width="stretch")
    d2.download_button("Download Parquet", (run / "profiles.parquet").read_bytes(), "risksynth_profiles.parquet",
                       "application/octet-stream", width="stretch")
    d3.download_button("Download JSON metadata", meta_file.read_bytes(), "risksynth_run_metadata.json",
                       "application/json", width="stretch")
    with l1:
        st.page_link(PAGES["overview"], label="Explore the population")
    with l2:
        st.page_link(PAGES["simulator"], label="Test it in the Advisory Simulator")
