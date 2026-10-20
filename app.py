"""
SPIDER: System for Portfolio, Index, and Data Economic Relationships
Main Streamlit Application Entry Point.
"""

import datetime
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_VERSION,
    APP_FULL_NAME,
    DEFAULT_TIME_WINDOW,
    MASTER_LINKS,
    NODES,
    QUADRANTS,
    RINGS,
    TIME_WINDOWS,
    Z_SCORE_CRITICAL,
    Z_SCORE_NORMAL,
    NODE_ENCYCLOPEDIA,
)
from quant_engine import engine
from pring_engine import (
    PringCycleEngine,
    build_kst_momentum_figure,
    build_pring_cycle_dial_figure,
    build_rrg_sector_matrix_figure,
    build_sector_horizontal_bars_figure,
    PRING_STAGES,
)
from curve_regimes_engine import (
    yield_curve_engine,
    build_yield_curve_comparison_figure,
    build_regimes_matrix_preview_figure,
    build_quadrants_comparison_grid_figure,
)
from historical_crisis_engine import (
    HISTORICAL_CRISIS_LIBRARY,
    HistoricalStressEngine,
    HistoricalCrisisScenario,
)
from net_liquidity_engine import (
    net_liquidity_engine,
    NetLiquidityState,
)
from risk_parity_engine import (
    risk_parity_engine,
    RiskParityState,
    RISK_PARITY_ASSETS,
)
from diebold_yilmaz_engine import (
    diebold_yilmaz_engine,
    DieboldYilmazState,
    SPILLOVER_NODES,
)
from super_intelligence_engine import (
    si_engine,
    SuperIntelligenceState,
    SuperIntelligenceStock,
    AI_UNIVERSE_CONFIG,
)
from spider_ui import (
    build_macro_regime_matrix_figure,
    build_spider_web_figure,
    build_spider_web_3d_figure,
    build_zscore_distribution_figure,
    build_ciss_gauge_and_decomposition_figure,
    build_ciss_correlation_heatmap,
    build_net_liquidity_overview_figure,
    build_liquidity_asset_correlation_figure,
    build_risk_parity_breakdown_figure,
    build_efficient_frontier_figure,
    build_diebold_yilmaz_heatmap,
    build_spillover_net_transmission_figure,
    build_spider_si_network_figure,
    build_spider_si_network_3d_figure,
    build_si_correlation_heatmap,
    build_si_subsector_bar_figure,
    build_si_capex_vs_revenue_figure,
    build_si_peg_valuation_figure,
    generate_zscore_mini_svg,
    generate_links_dataframe,
    generate_nodes_dataframe,
)

# Page configuration
st.set_page_config(
    page_title=f"{APP_FULL_NAME} • Macro & Market Network Monitor",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }

    .spider-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    .spider-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .spider-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .spider-badge {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .tension-badge-normal {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 20px;
    }
    .tension-badge-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FBBF24;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 20px;
    }
    .tension-badge-critical {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #F87171;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 20px;
        animation: pulse-crit 2s infinite;
    }
    @keyframes pulse-crit {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Flashing animation for active What-If stress simulator banner */
    .simulator-active-flash {
        background: rgba(245, 158, 11, 0.15);
        border: 1.5px solid #F59E0B;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
        animation: flash-border 1.5s infinite ease-in-out;
    }
    .simulator-flash-title {
        color: #F59E0B;
        font-weight: 800;
        font-size: 14px;
        letter-spacing: 0.5px;
        animation: flash-text 1.2s infinite alternate;
        display: inline-block;
    }
    @keyframes flash-text {
        0% { opacity: 1; text-shadow: 0 0 8px rgba(245, 158, 11, 0.8); }
        50% { opacity: 0.35; text-shadow: none; }
        100% { opacity: 1; text-shadow: 0 0 12px rgba(245, 158, 11, 1); }
    }
    @keyframes flash-border {
        0% { border-color: rgba(245, 158, 11, 0.4); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
        50% { border-color: rgba(245, 158, 11, 1); box-shadow: 0 0 15px rgba(245, 158, 11, 0.35); }
        100% { border-color: rgba(245, 158, 11, 0.4); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }

    .spider-kpi {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .spider-top-kpi {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, rgba(10, 15, 26, 0.96) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 9px 9px 8px 9px;
        backdrop-filter: blur(8px);
        margin-bottom: 8px;
        height: 148px;
        min-height: 148px;
        max-height: 148px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        overflow: hidden;
        transition: all 0.2s ease;
        position: relative;
    }
    .spider-top-kpi:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .kpi-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 2px;
        gap: 2px;
    }
    .kpi-title {
        font-size: 9.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-stars-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-radius: 4px;
        padding: 1px 3px;
        font-size: 8px;
        line-height: 1;
        white-space: nowrap;
        cursor: help;
    }
    .kpi-stars-gold {
        color: #F59E0B;
        letter-spacing: -0.5px;
        text-shadow: 0 0 4px rgba(245, 158, 11, 0.4);
    }
    .kpi-stars-dim {
        color: rgba(148, 163, 184, 0.22);
        letter-spacing: -0.5px;
    }
    .kpi-body {
        display: flex;
        flex-direction: column;
        justify-content: center;
        flex: 1;
        margin: 2px 0;
    }
    .kpi-main-val {
        font-size: 15px;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', -apple-system, monospace;
        line-height: 1.15;
        margin: 1px 0;
        word-break: break-word;
    }
    .kpi-sub-val {
        font-size: 9.5px;
        color: #64748B;
        line-height: 1.2;
        margin-top: 2px;
        white-space: normal;
    }
    .kpi-footer-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-top: 3px;
        padding-top: 3px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .kpi-influence-pill {
        font-size: 8px;
        font-weight: 700;
        color: #F59E0B;
        letter-spacing: 0.2px;
        display: flex;
        align-items: center;
        gap: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Uniform height & style for KPI detail buttons */
    div[data-testid="stHorizontalBlock"] > div:has(.spider-top-kpi) button,
    div[data-testid="stHorizontalBlock"] > div:has(.spider-kpi) button {
        border-radius: 8px !important;
        font-size: 10.5px !important;
        font-weight: 600 !important;
        padding: 4px 2px !important;
        min-height: 32px !important;
        height: 32px !important;
        max-height: 32px !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #94A3B8 !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"] > div:has(.spider-top-kpi) button:hover,
    div[data-testid="stHorizontalBlock"] > div:has(.spider-kpi) button:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

if "force_refresh" not in st.session_state:
    st.session_state["force_refresh"] = False
if "time_window" not in st.session_state:
    st.session_state["time_window"] = DEFAULT_TIME_WINDOW
if "use_spearman" not in st.session_state:
    st.session_state["use_spearman"] = False
if "selected_node" not in st.session_state:
    st.session_state["selected_node"] = "TUTTI (Visualizzazione Globale)"
if "filter_quadrant" not in st.session_state:
    st.session_state["filter_quadrant"] = "TUTTI"
if "filter_ring" not in st.session_state:
    st.session_state["filter_ring"] = 0
if "what_if_oil" not in st.session_state:
    st.session_state["what_if_oil"] = 0.0
if "what_if_real_yield" not in st.session_state:
    st.session_state["what_if_real_yield"] = 0.0
if "what_if_hy" not in st.session_state:
    st.session_state["what_if_hy"] = 0.0
if "active_crisis_scenario" not in st.session_state:
    st.session_state["active_crisis_scenario"] = None


def reset_node_selection():
    st.session_state["selected_node"] = "TUTTI (Visualizzazione Globale)"
    if "sel_node_sidebar" in st.session_state:
        del st.session_state["sel_node_sidebar"]
    if "spider_web_chart_selection" in st.session_state:
        del st.session_state["spider_web_chart_selection"]


@st.cache_data(ttl=900, show_spinner=False)
def get_computed_engine_data(window_key: str, use_spearman: bool, refresh_id: int):
    aligned_df = engine.load_and_align_data(force_refresh=(refresh_id > 0))
    node_states = engine.compute_node_states(window_key=window_key)
    link_states = engine.compute_link_states(node_states, window_key=window_key, use_spearman=use_spearman)
    core_state = engine.compute_spider_core(node_states, link_states, window_key=window_key)
    return node_states, link_states, core_state, aligned_df


refresh_count = st.session_state.get("refresh_count", 0)
with st.spinner("Allineamento serie storiche e calcolo correlazioni in corso..."):
    node_states, link_states, core_state, aligned_df = get_computed_engine_data(
        st.session_state["time_window"],
        st.session_state["use_spearman"],
        refresh_count
    )

# 1. Historical Crisis Replay Shock Tensor (CCAR / Aladdin Engine)
active_crisis_id = st.session_state.get("active_crisis_scenario")
crisis_sim_data = None

if active_crisis_id and active_crisis_id in HISTORICAL_CRISIS_LIBRARY:
    crisis_sim_data = HistoricalStressEngine.apply_crisis_scenario(
        scenario_id=active_crisis_id,
        current_node_states=node_states,
        engine_instance=engine,
        window_key=st.session_state["time_window"],
    )
    node_states = crisis_sim_data["stressed_nodes"]
    link_states = crisis_sim_data["stressed_links"]
    core_state = crisis_sim_data["stressed_core"]

# 2. Manual What-If Sliders Stress (if active)
elif (st.session_state["what_if_oil"] != 0.0 or 
      st.session_state["what_if_real_yield"] != 0.0 or 
      st.session_state["what_if_hy"] != 0.0):
    
    if "COM_OIL" in node_states and st.session_state["what_if_oil"] != 0:
        ns = node_states["COM_OIL"]
        ns.current_value += st.session_state["what_if_oil"]
        ns.z_score += (st.session_state["what_if_oil"] / 8.0)
    if "YIELD_REAL10" in node_states and st.session_state["what_if_real_yield"] != 0:
        ns = node_states["YIELD_REAL10"]
        ns.current_value += st.session_state["what_if_real_yield"]
        ns.z_score += (st.session_state["what_if_real_yield"] / 0.35)
    if "CRED_HY" in node_states and st.session_state["what_if_hy"] != 0:
        ns = node_states["CRED_HY"]
        ns.current_value += st.session_state["what_if_hy"]
        ns.z_score += (st.session_state["what_if_hy"] / 0.50)

    # Recalculate link states and spider core to reflect the stress scenario across the entire web
    link_states = engine.compute_link_states(node_states, window_key=st.session_state["time_window"], use_spearman=st.session_state["use_spearman"])
    core_state = engine.compute_spider_core(node_states, link_states, window_key=st.session_state["time_window"])



# ================= SIDEBAR =================
with st.sidebar:
    st.markdown(f"## 🕷️ **SPIDER CORE** <span style='font-size: 12px; color: #38BDF8; font-weight: 700; background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.35); padding: 2px 7px; border-radius: 5px; vertical-align: middle;'>{APP_VERSION}</span>", unsafe_allow_html=True)
    st.caption("Macro & Market Network Monitor • Standard Istituzionale")
    st.markdown("<hr style='margin: 8px 0 16px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    t_val = core_state.systemic_web_tension
    if t_val >= Z_SCORE_CRITICAL:
        t_badge = f'<div class="tension-badge-critical">⚠️ TENSIONE CRITICA ({t_val:.2f})</div>'
    elif t_val >= Z_SCORE_NORMAL:
        t_badge = f'<div class="tension-badge-warning">🟡 TENSIONE MODERATA ({t_val:.2f})</div>'
    else:
        t_badge = f'<div class="tension-badge-normal">🟢 EQUILIBRIO FISIOLOGICO ({t_val:.2f})</div>'

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; margin-bottom: 16px;">
        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 600; margin-bottom: 6px;">Stato Tensione Rete</div>
        {t_badge}
        <div style="font-size: 11px; color: #94A3B8; margin-top: 8px;">Regime: <b style="color: #F8FAFC;">{core_state.macro_regime.split('/')[0]}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ **Parametri Motore Quantitativo**")
    selected_win = st.selectbox(
        "Finestra Temporale Correlazioni (Rolling):",
        options=list(TIME_WINDOWS.keys()),
        format_func=lambda k: f"{k} - {TIME_WINDOWS[k]['label']}",
        index=list(TIME_WINDOWS.keys()).index(st.session_state["time_window"]),
    )
    if selected_win != st.session_state["time_window"]:
        st.session_state["time_window"] = selected_win
        st.rerun()

    corr_method = st.radio(
        "Metodo Matematico Correlazione:",
        options=["Pearson (Lineare)", "Spearman (Ranghi Non-Parametrici)"],
        index=1 if st.session_state["use_spearman"] else 0,
        horizontal=True,
    )
    is_spearman = (corr_method == "Spearman (Ranghi Non-Parametrici)")
    if is_spearman != st.session_state["use_spearman"]:
        st.session_state["use_spearman"] = is_spearman
        st.rerun()

    st.markdown("<hr style='margin: 12px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    st.markdown("##### 🔍 **Filtri di Visualizzazione Tela**")
    quad_opts = ["TUTTI"] + list(QUADRANTS.keys())
    quad_names = {"TUTTI": "🌐 Tutti i Quadranti", **{k: f"{v['name']}" for k, v in QUADRANTS.items()}}
    cur_quad = st.session_state.get("filter_quadrant", "TUTTI")
    quad_idx = quad_opts.index(cur_quad) if cur_quad in quad_opts else 0
    sel_quad = st.selectbox("Filtra per Quadrante Macro:", options=quad_opts, index=quad_idx, format_func=lambda k: quad_names[k])
    st.session_state["filter_quadrant"] = sel_quad

    ring_opts = [0, 1, 2, 3]
    ring_names = {0: "🌐 Tutti gli Anelli", **{k: f"{v['name']}: {v['label']}" for k, v in RINGS.items()}}
    cur_ring = st.session_state.get("filter_ring", 0)
    ring_idx = ring_opts.index(cur_ring) if cur_ring in ring_opts else 0
    sel_ring = st.selectbox("Filtra per Anello Concentrico:", options=ring_opts, index=ring_idx, format_func=lambda k: ring_names[k])
    st.session_state["filter_ring"] = sel_ring

    all_nodes_list = ["TUTTI (Visualizzazione Globale)", "SPIDER_CORE (Centro di Controllo)"] + list(NODES.keys())
    curr_sel = st.session_state.get("selected_node", "TUTTI (Visualizzazione Globale)")
    sel_node_idx = all_nodes_list.index(curr_sel) if curr_sel in all_nodes_list else 0
    
    sel_node = st.selectbox(
        "Isola Nodo & Collegamenti (Focus):",
        options=all_nodes_list,
        index=sel_node_idx,
    )
    if sel_node != st.session_state.get("selected_node"):
        st.session_state["selected_node"] = sel_node
        if "spider_web_chart_selection" in st.session_state:
            del st.session_state["spider_web_chart_selection"]
        st.rerun()

    st.markdown("<hr style='margin: 14px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    col_ref1, col_ref2 = st.columns([1, 1])
    with col_ref1:
        if st.button("🔄 Aggiorna Dati", use_container_width=True, type="primary"):
            st.session_state["refresh_count"] = refresh_count + 1
            st.rerun()
    with col_ref2:
        if st.button("🧹 Reset Filtri", use_container_width=True):
            reset_node_selection()
            st.session_state["filter_quadrant"] = "TUTTI"
            st.session_state["filter_ring"] = 0
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.session_state["active_crisis_scenario"] = None
            st.rerun()

    st.caption("Fonti Istituzionali: FRED, BCE, Eurostat, CME, Yahoo Finance, BLS.")


# ================= BRAND HEADER =================
active_isolated_id = None
if st.session_state["selected_node"] not in ["TUTTI (Visualizzazione Globale)", "SPIDER_CORE (Centro di Controllo)"]:
    active_isolated_id = st.session_state["selected_node"]

n_crit_alerts = len(core_state.active_critical_alerts)
alert_badge_html = f'<span class="tension-badge-critical">⚡ {n_crit_alerts} ANOMALIE CRITICHE</span>' if n_crit_alerts > 0 else '<span class="tension-badge-normal">✅ RELAZIONI STORICHE NORMALI</span>'

st.markdown(f"""
<div class="spider-header">
    <div class="spider-title-row">
        <div class="spider-title">
            <span>🕷️ <b>SPIDER</b> <span style="font-size: 13px; font-weight: 700; color: #38BDF8; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); padding: 2px 7px; border-radius: 6px; vertical-align: middle;">{APP_VERSION}</span></span>
            <span class="spider-badge">Macro & Market Network Monitor</span>
        </div>
        {alert_badge_html}
    </div>
    <div style="color: #94A3B8; font-size: 13px; margin-top: 6px;">
        Centro di Controllo Macro-Finanziario Radiale • <b>30 Nodi Istituzionali</b> • <b>10 Fili Maestri di Trasmissione</b>
        <br>
        <b>Regime Attivo:</b> <span style="color: #38BDF8;">{core_state.macro_regime}</span> | 
        <b>Tensione Tela (|Z|):</b> <span style="color: #FBBF24;">{core_state.systemic_web_tension:.2f}</span> | 
        <b>Contagio CISS (ECB):</b> <span style="color: {core_state.ciss.status_color}; font-weight: 700;">{core_state.ciss.ciss_score * 100:.1f}% ({core_state.ciss.ciss_status.split('(')[0].strip()})</span> | 
        <b>Bias Banche Centrali:</b> <span style="color: #A78BFA;">{core_state.central_bank_bias}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if core_state.active_critical_alerts:
    var_warning_html = ""
    if n_crit_alerts >= 3:
        var_warning_html = f"""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1.5px solid #EF4444; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; color: #FECACA; font-size: 12.5px;">
            🚨 <b>SOGLIA CRITICA VAR SUPERATA ({n_crit_alerts}/10 Fili in Rottura Sistemica |Z| &ge; 2.0):</b><br>
            L'instabilità statistica della matrice di correlazione compromette l'affidabilità dei modelli <i>Value-at-Risk (VaR)</i> di portafoglio, esponendo le strategie multi-asset a correlazioni di coda impreviste. Attivare protocollo di de-risking o coperture di convessità asimmetriche.
        </div>
        """
    with st.expander(f"⚠️ **ATTENZIONE SISTEMICA: Rilevate {n_crit_alerts} divergenze anomale tra nodi macro** (Clicca per espandere)", expanded=True):
        if var_warning_html:
            st.markdown(var_warning_html, unsafe_allow_html=True)
        for alt in core_state.active_critical_alerts:
            st.error(alt)


# Modal Dialog definition for Landing Scenario Analysis (Soft vs Hard vs No Landing)
@st.dialog("🛬 Analisi del Ciclo di Atterraggio: Soft vs Hard vs No Landing", width="large")
def show_landing_scenario_modal():
    ls = core_state.landing_scenario
    st.markdown(f"### 🧭 Scenario Predittivo Ciclico: <span style='color: {ls.color_hex};'>{ls.scenario_title}</span>", unsafe_allow_html=True)
    st.caption("Modello di probabilità multi-variata basato sull'interazione tra Crescita del PIL, Rigidità dell'Inflazione, Dinamica Occupazionale e Spread Creditizi.")

    # Probability Bars
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1.5px solid #F59E0B; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #F59E0B; font-weight: 700;">🔥 NO LANDING</div>
            <div style="font-size: 26px; font-weight: 800; color: #F59E0B; font-family: monospace;">{ls.prob_no_landing:.1f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 4px;">Crescita forte + Inflazione alta</div>
        </div>
        """, unsafe_allow_html=True)
    with p_c2:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1.5px solid #10B981; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #10B981; font-weight: 700;">🛬 SOFT LANDING</div>
            <div style="font-size: 26px; font-weight: 800; color: #10B981; font-family: monospace;">{ls.prob_soft_landing:.1f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 4px;">Disinflazione ordinata + Crescita moderata</div>
        </div>
        """, unsafe_allow_html=True)
    with p_c3:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1.5px solid #EF4444; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #EF4444; font-weight: 700;">⚠️ HARD LANDING</div>
            <div style="font-size: 26px; font-weight: 800; color: #EF4444; font-family: monospace;">{ls.prob_hard_landing:.1f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 4px;">Forte contrazione / Recessione</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Driver Empirici Attuali Rilevati dai Nodi SPIDER:")
    for drv in ls.key_drivers:
        st.markdown(f"• **{drv}**")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Quadro Comparativo Teorico dei Tre Scenari di Atterraggio:")
    df_landing_guide = pd.DataFrame([
        {
            "Scenario": "🔥 NO LANDING (Attuale)",
            "Crescita Economica (PIL)": "Resiliente / Superiore al potenziale (> 2.0%)",
            "Dinamica Inflazione": "Sostenuta e sopra target (PCE > 2.8%, petrolio alto)",
            "Mercato del Lavoro": "Piena occupazione solida (Disoccupazione < 4.3%, bassi licenziamenti)",
            "Politica Monetaria Fed": "Tassi elevati a lungo ('Higher for longer'), stop ai tagli",
            "Impatto Asset Allocation": "Favoriti: Materie Prime, Energia, Titoli Value. Sottopesare bond a lunga duration.",
        },
        {
            "Scenario": "🛬 SOFT LANDING (Goldilocks)",
            "Crescita Economica (PIL)": "Decelerazione ordinata verso il trend potenziale (~1.5% - 2.0%)",
            "Dinamica Inflazione": "Progressivo e costante rientro verso il target 2.0%",
            "Mercato del Lavoro": "Ribilanciamento graduale senza impennata di licenziamenti (disoccupazione 4.3% - 4.8%)",
            "Politica Monetaria Fed": "Tagli preventivi graduali verso tassi neutrali (~3.0%)",
            "Impatto Asset Allocation": "Scenario ideale per azionario diversificato (Quality Growth) e bond societari.",
        },
        {
            "Scenario": "⚠️ HARD LANDING (Recessione)",
            "Crescita Economica (PIL)": "Forte contrazione o crescita negativa (< 0.5% / recessione)",
            "Dinamica Inflazione": "Rapida caduta disinflazionistica da distruzione della domanda",
            "Mercato del Lavoro": "Ondate di licenziamenti (sussidi > 260k, disoccupazione > 5.0%)",
            "Politica Monetaria Fed": "Tagli d'emergenza aggressivi (-50 bps per meeting)",
            "Impatto Asset Allocation": "Flight-to-safety: Long Treasuries (TLT), Bund, Cash monetario, Oro difensivo. Bear market equity.",
        },
    ])
    st.dataframe(df_landing_guide, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 14px; margin-top: 14px;">
        <div style="font-size: 12px; font-weight: 700; color: #38BDF8; text-transform: uppercase;">💼 Implicazione Tattica di Portafoglio SPIDER:</div>
        <div style="font-size: 13px; color: #E2E8F0; margin-top: 4px;">{ls.market_implication}</div>
        <div style="font-size: 12px; font-weight: 700; color: #F59E0B; text-transform: uppercase; margin-top: 8px;">🏛️ Reazione Attesa delle Banche Centrali:</div>
        <div style="font-size: 13px; color: #CBD5E1; margin-top: 4px;">{ls.policy_reaction}</div>
    </div>
    """, unsafe_allow_html=True)


# Modal Dialog 1: Regime Macroeconomico (Ray Dalio / Bridgewater Framework)
@st.dialog("🌐 Dettaglio Analitico: Regime Macroeconomico & 4 Quadranti", width="large")
def show_macro_regime_modal():
    st.markdown(f"### 🧭 Regime Attivo: <span style='color: #38BDF8;'>{core_state.macro_regime}</span>", unsafe_allow_html=True)
    st.caption("Mappatura bidimensionale Crescita Reale vs Pressioni Inflattive secondo il framework istituzionale Ray Dalio / All Weather.")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #38BDF8; font-weight: 700;">Score Crescita Reale</div>
            <div style="font-size: 24px; font-weight: 800; color: #38BDF8; font-family: monospace;">{core_state.growth_score:+.2f}</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 3px;">PIL, Produzione Ind., Vendite, Lavoro</div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1.5px solid #F59E0B; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #F59E0B; font-weight: 700;">Score Pressione Inflattiva</div>
            <div style="font-size: 24px; font-weight: 800; color: #F59E0B; font-family: monospace;">{core_state.inflation_score:+.2f}</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 3px;">PCE Core, CPI, Salari, Petrolio WTI</div>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div style="background: rgba(168, 85, 247, 0.1); border: 1.5px solid #A855F7; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #A855F7; font-weight: 700;">Coordinate Spaziali</div>
            <div style="font-size: 24px; font-weight: 800; color: #A855F7; font-family: monospace;">({core_state.regime_coords[0]:.2f}, {core_state.regime_coords[1]:.2f})</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 3px;">Quadrante: {'Q1 Surriscaldamento' if core_state.growth_score >= 0 and core_state.inflation_score >= 0 else ('Q2 Goldilocks' if core_state.growth_score >= 0 else ('Q3 Stagflazione' if core_state.inflation_score >= 0 else 'Q4 Deflazione'))}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📖 Fondamento Teorico e Diagnosi SPIDER:")
    st.markdown(f"""
    {core_state.macro_regime_desc}
    
    Il modello aggrega vettorialmente **30 nodi macroeconomici e finanziari istituzionali**, normalizzati in Z-Score rolling per eliminare distorsioni di scala.
    """)

    st.markdown("#### 📋 Matrice Istituzionale dei 4 Regimi Economici:")
    df_regimes = pd.DataFrame([
        {
            "Regime": "🔥 Surriscaldamento / Late-Cycle Boom (Attuale)",
            "Crescita": "Positiva (> 0)",
            "Inflazione": "Positiva (> 0)",
            "Contesto Macro": "Piena occupazione, domanda robusta, materie prime in tensione, banche centrali restrittive.",
            "Asset Vincenti": "Commodities, Petrolio, Titoli Energetici/Ciclici, TIPS.",
            "Asset Penalizzati": "Governative lunga duration (TLT, Bund), Growth ad altissimi multipli.",
        },
        {
            "Regime": "✨ Goldilocks / Espansione Disinflazionistica",
            "Crescita": "Positiva (> 0)",
            "Inflazione": "Negativa (< 0)",
            "Contesto Macro": "Crescita solida associata a disinflazione ordinata; la Fed può abbassare i tassi.",
            "Asset Vincenti": "Azionario Growth/Tech, Indici Generali (S&P 500, Nasdaq), Corporate Credit IG.",
            "Asset Penalizzati": "Cash monetario, Materie prime energetiche.",
        },
        {
            "Regime": "⚠️ Stagflazione",
            "Crescita": "Negativa (< 0)",
            "Inflazione": "Positiva (> 0)",
            "Contesto Macro": "Stagnazione dell'attività economica con inflazione persistente da costi di fornitura.",
            "Asset Vincenti": "Oro fisico, Materie prime agricole, Liquidità difensiva breve termine.",
            "Asset Penalizzati": "Azionario complessivo, Obbligazioni societarie (rischio allargamento spread).",
        },
        {
            "Regime": "❄️ Deflazione / Recessione",
            "Crescita": "Negativa (< 0)",
            "Inflazione": "Negativa (< 0)",
            "Contesto Macro": "Forte contrazione di investimenti e consumi, crollo dei prezzi per distruzione di domanda.",
            "Asset Vincenti": "Governative sovrane a lunga scadenza (Treasuries, Bund), Cash, Dollaro difensivo.",
            "Asset Penalizzati": "Azionario High-Beta, Credito High Yield, Commodities cicliche industriali.",
        },
    ])
    st.dataframe(df_regimes, use_container_width=True, hide_index=True)


# Modal Dialog 3: Tensione Sistemica Tela (Systemic Web Tension)
@st.dialog("🕸️ Dettaglio Analitico: Tensione Sistemica della Rete SPIDER", width="large")
def show_web_tension_modal():
    t_val = core_state.systemic_web_tension
    st.markdown(f"### ⚡ Tensione Sistemica: <span style='color: {'#EF4444' if t_val >= 2.0 else ('#F59E0B' if t_val >= 1.25 else '#10B981')};'>{t_val:.2f} | {core_state.web_tension_status}</span>", unsafe_allow_html=True)
    st.caption("Media ponderata degli Z-score di tensione calcolati su tutti i 10 Fili Maestri di trasmissione intermarket.")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Tensione Attiva (|Z|)</div>
            <div style="font-size: 24px; font-weight: 800; color: {'#EF4444' if t_val >= 2.0 else ('#F59E0B' if t_val >= 1.25 else '#10B981')}; font-family: monospace;">{t_val:.2f}</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 3px;">Soglia critica: 2.00 | Moderata: 1.25</div>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        n_crit = len(core_state.active_critical_alerts)
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Fili in Divergenza Critica</div>
            <div style="font-size: 24px; font-weight: 800; color: {'#EF4444' if n_crit > 0 else '#10B981'}; font-family: monospace;">{n_crit} / 10</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 3px;">Fili con |Z| &ge; 2.0</div>
        </div>
        """, unsafe_allow_html=True)
    with tc3:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Stato Strutturale Rete</div>
            <div style="font-size: 20px; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{core_state.web_tension_status.split('(')[0].strip()}</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 3px;">Pressione di trasmissione</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Significato Matematico e Meccanica:")
    st.markdown("""
    Ogni coppia di nodi legata da un Filo Maestro possiede una **correlazione teorica istituzionale** (es. *Tassi Reali e Oro* hanno correlazione inversa $\\approx -1$, *2Y Treasury e Tasso Fed* correlazione positiva $\\approx +1$).
    
    L'indice di Tensione Sistemica misura **quanto le correlazioni empiriche correnti deviano dalla loro relazione fisiologica di lungo periodo**:
    $$\\text{Z-Tension} = \\frac{|r_{\\text{rolling}} - r_{\\text{teorico}}|}{\\sigma_{r}}$$
    * **$|Z| < 1.25$ (Verde - Equilibrio Fisiologico):** I mercati trasmettono le informazioni monetarie e reali in modo fluido e ordinato.
    * **$1.25 \\le |Z| < 2.00$ (Giallo - Tensione Moderata):** Primi disallineamenti di prezzo; alcune asset class tardano ad incorporare i fondamentali.
    * **$|Z| \\ge 2.00$ (Rosso/Viola - Rottura Sistemica):** Anomalia statistica severa (probabilità $< 5\\%$ in condizioni normali). Segnale precursore di violenti repricing a catena.
    """)

    st.markdown("#### ⚠️ Monitoraggio di Dettaglio dei Fili Maestri:")
    master_links_data = []
    for l in link_states:
        if l.is_master:
            master_links_data.append({
                "Filo Maestro": l.id,
                "Nodo A": l.node_a_name,
                "Nodo B": l.node_b_name,
                "Corr. Rolling": f"{l.active_corr:+.2f}",
                "Corr. Teorica": f"{l.theoretical_corr:+d}",
                "Z-Score Tensione": f"{l.z_score_tension:.2f}",
                "Stato": l.status,
            })
    st.dataframe(pd.DataFrame(master_links_data), use_container_width=True, hide_index=True)

    with st.expander("🔬 **Decomposizione Micro-Strutturale & Diagnosi Economica Anomalie Critiche**", expanded=True):
        st.caption("Analisi quantitativa dei driver esogeni responsabili della deviazione empirica rispetto alla teoria istituzionale:")
        for l in link_states:
            if l.is_master and l.z_score_tension >= 1.25:
                tag_color = "#D946EF" if l.z_score_tension >= 2.0 else "#F59E0B"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border-left: 4px solid {tag_color}; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #F8FAFC; font-size: 13.5px;">{l.id}: {l.node_a_name} ⟷ {l.node_b_name}</span>
                        <span style="background: {tag_color}22; border: 1px solid {tag_color}; color: {tag_color}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700;">{l.status} (|Z| = {l.z_score_tension:.2f})</span>
                    </div>
                    <div style="font-size: 12.5px; color: #CBD5E1; margin-bottom: 4px;"><b>Meccanismo Istituzionale Atteso:</b> {l.mechanism}</div>
                    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 6px;"><b>Diagnosi SPIDER:</b> {l.alert_description}</div>
                    <div style="font-size: 12px; color: #38BDF8;">
                        • <b>Correlazione Attiva:</b> <code>{l.active_corr:+.2f}</code> vs <b>Teorica:</b> <code>{l.theoretical_corr:+d}</code> | 
                        • <b>Nodo A ({l.node_a_id}):</b> {l.node_a_val:,.2f} ({l.node_a_delta:+.2f}) | 
                        • <b>Nodo B ({l.node_b_id}):</b> {l.node_b_val:,.2f} ({l.node_b_delta:+.2f})
                    </div>
                </div>
                """, unsafe_allow_html=True)



# Modal Dialog: Indice di Contagio Non-Lineare CISS (ECB / OFR Framework)
@st.dialog("⚡ Indice di Contagio Sistemico Non-Lineare (ECB CISS & OFR)", width="large")
def show_ciss_modal():
    ciss = core_state.ciss
    st.markdown(f"### 🌐 ECB CISS Stress Index: <span style='color: {ciss.status_color};'>{ciss.ciss_score * 100:.1f}% • {ciss.ciss_status}</span>", unsafe_allow_html=True)
    st.caption("Modello di rischio non-lineare basato sulla teoria di portafoglio: aggrega lo stress dei 4 quadranti macroeconomici attraverso la matrice di cross-correlazione dinamica $C_t$.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1.5px solid {ciss.status_color}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Indice CISS Globale</div>
            <div style="font-size: 24px; font-weight: 800; color: {ciss.status_color}; font-family: monospace;">{ciss.ciss_score * 100:.1f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Soglia critica: 50%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Effetto Contagio Cross-Asset</div>
            <div style="font-size: 24px; font-weight: 800; color: #38BDF8; font-family: monospace;">{ciss.contagion_contribution_pct:.1f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Quota varianza da cross-correlazione</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Rischio Idiosincratico Puro</div>
            <div style="font-size: 24px; font-weight: 800; color: #A78BFA; font-family: monospace;">{ciss.idiosyncratic_risk * 100:.2f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Stress isolato dei singoli mercati</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Stato Sistemico</div>
            <div style="font-size: 16px; font-weight: 800; color: {ciss.status_color}; margin-top: 5px;">{ciss.ciss_status.split('(')[0].strip()}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 4px;">Standard BCE / OFR</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizzazioni grafiche Plotly CISS
    fig_ciss_bars = build_ciss_gauge_and_decomposition_figure(ciss, dark_mode=True)
    st.plotly_chart(fig_ciss_bars, use_container_width=True)

    col_h1, col_h2 = st.columns([1.1, 0.9])
    with col_h1:
        fig_ciss_heat = build_ciss_correlation_heatmap(ciss, dark_mode=True)
        st.plotly_chart(fig_ciss_heat, use_container_width=True)
    with col_h2:
        st.markdown("#### 🔬 Fondamento Matematico (ECB CISS)")
        st.markdown("""
        L'indice CISS sviluppato da **Hóllo, Kremer e Lo Duca (BCE Working Paper No. 1426)** supera il limite delle medie lineari aggregando i sub-indici di stress dei 4 quadranti attraverso l'equazione di varianza di portafoglio:
        $$\\text{CISS}_t = \\mathbf{s}_t' \\cdot \\mathbf{C}_t \\cdot \\mathbf{s}_t = \\sum_{i=1}^4 w_i^2 s_{i,t}^2 + 2 \\sum_{i < j} w_i w_j \\rho_{ij,t} s_{i,t} s_{j,t}$$
        * **$s_{i,t} \\in [0, 1]$:** Livello di stress standardizzato tramite la Funzione di Distribuzione Cumulativa empirica ($CDF$).
        * **$\\mathbf{C}_t$:** Matrice di cross-correlazione rolling a 90 giorni tra i quadranti.
        * **Effetto Moltiplicatore di Contagio:** Se lo stress colpisce un solo settore (es. correzione tecnologica isolata), il termine di cross-correlazione è vicino a zero e l'indice resta contenuto. Se invece tassi, inflazione, borse e credito subiscono vendite simultanee, $\\rho_{ij}$ sale ed esplode l'effetto moltiplicativo sistemico.
        """)

    st.markdown("---")
    st.markdown("#### 📋 Tabella Sotto-Indici dei Quadranti Macro:")
    q_table_data = []
    for qk, q_stress in ciss.quadrant_stress.items():
        q_name = QUADRANTS.get(qk, {}).get("name", qk)
        w_val = ciss.weights.get(qk, 0.25)
        q_table_data.append({
            "Quadrante Macro": q_name,
            "Peso Algoritmico (w_i)": f"{w_val * 100:.0f}%",
            "Stress Sub-Index (s_i)": f"{q_stress * 100:.1f}%",
            "Contributo Ponderato (w_i * s_i)": f"{w_val * q_stress * 100:.2f}%",
            "Stato Settoriale": "Critico (>50%)" if q_stress >= 0.50 else ("Attenzione (25-50%)" if q_stress >= 0.25 else "Fisiologico (<25%)")
        })
    st.dataframe(pd.DataFrame(q_table_data), use_container_width=True, hide_index=True)


# Modal Dialog 4: Bias Banche Centrali (Central Bank Policy Bias)
@st.dialog("🏛️ Dettaglio Analitico: Bias e Tassi Reali delle Banche Centrali", width="large")
def show_cb_bias_modal():
    st.markdown(f"### ⚖️ Orientamento Strategico: <span style='color: #A78BFA;'>{core_state.central_bank_bias}</span>", unsafe_allow_html=True)
    st.caption("Valutazione comparativa dei tassi reali di policy FED e BCE rispetto all'inflazione core target.")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(f"""
        <div style="background: rgba(167, 139, 250, 0.1); border: 1.5px solid #A78BFA; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #A78BFA; font-weight: 700;">Tasso Reale Fed (Ex-Post)</div>
            <div style="font-size: 24px; font-weight: 800; color: {'#10B981' if core_state.fed_real_rate >= 0.5 else '#F59E0B'}; font-family: monospace;">{core_state.fed_real_rate:+.2f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Fed Funds Nominale - US Core PCE</div>
        </div>
        """, unsafe_allow_html=True)
    with cc2:
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #38BDF8; font-weight: 700;">Tasso Reale BCE (Ex-Post)</div>
            <div style="font-size: 24px; font-weight: 800; color: {'#10B981' if core_state.ecb_real_rate >= 0.0 else '#EF4444'}; font-family: monospace;">{core_state.ecb_real_rate:+.2f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Deposit Facility - Euro HICP Core</div>
        </div>
        """, unsafe_allow_html=True)
    with cc3:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Differenziale Tassi (US - EZ)</div>
            <div style="font-size: 24px; font-weight: 800; color: #F8FAFC; font-family: monospace;">{(core_state.fed_real_rate - core_state.ecb_real_rate):+.2f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Premio reale a supporto di USD/EUR</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Canali di Trasmissione della Politica Monetaria:")
    st.markdown("""
    1. **Tasso Neutrale $r^*$:** Negli Stati Uniti il tasso neutrale reale è stimato tra $+0.8\\%$ e $+1.2\\%$. Un tasso reale di policy superiore a $+1.50\\%$ colloca la Fed in pieno territorio restrittivo (*Hawkish*).
    2. **Ancoraggio delle Aspettative:** Quando l'inflazione Core PCE rimane al di sopra del target del $2.0\\%$, la banca centrale deve mantenere condizioni finanziarie stringenti per evitare la spirale prezzi-salari.
    3. **Pressione sui Multipli Azionari:** Tassi reali elevati aumentano il costo del capitale WACC e deprimono i multipli $P/E$, premiando le aziende con elevato Free Cash Flow e solida cassa netta.
    """)

    st.markdown("#### 💼 Implicazioni di Asset Allocation:")
    st.markdown(f"""
    - **Valute:** Differenziale di tasso reale favorevole agli USA genera supporto strutturale per il Dollaro USA (**DXY**).
    - **Obbligazionario:** Curva a breve termine attraente per rendimenti monetari senza rischio duration; cautela su scadenze ultradecennali se l'inflazione non scende sotto il 2.5%.
    - **Azionario:** Favorire settori a bassa leva finanziaria (Value istituzionale, Finanziari e Materie prime) e sottopesare growth iper-indebitato.
    """)


# Modal Dialog 5: Taylor Rule Gap
@st.dialog("📏 Dettaglio Analitico: Regola di Taylor & Adeguatezza dei Tassi", width="large")
def show_taylor_rule_modal():
    st.markdown(f"### 🎯 Taylor Rule Gap: <span style='color: {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'};'>{core_state.taylor_rate_gap_bps:+.0f} bps</span>", unsafe_allow_html=True)
    st.caption("Confronto matematico tra il tasso Fed Funds ufficiale effettivo e il tasso ottimale prescritto dall'equazione istituzionale di John B. Taylor.")

    fed_rate = node_states["RATE_FED"].current_value if "RATE_FED" in node_states else 3.88
    us_pce = node_states["INF_USPCE"].current_value if "INF_USPCE" in node_states else 3.01
    gdp_val = node_states["GRO_USGDP"].current_value if "GRO_USGDP" in node_states else 2.19
    taylor_prescribed = fed_rate - (core_state.taylor_rate_gap_bps / 100.0)

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Tasso Fed Attuale</div>
            <div style="font-size: 24px; font-weight: 800; color: #F8FAFC; font-family: monospace;">{fed_rate:.2f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Upper bound Fed Funds</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #38BDF8; font-weight: 700;">Tasso Prescritto Taylor</div>
            <div style="font-size: 24px; font-weight: 800; color: #38BDF8; font-family: monospace;">{taylor_prescribed:.2f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Equazione Benchmark di Taylor</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1.5px solid {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'}; font-weight: 700;">Posizione di Policy</div>
            <div style="font-size: 20px; font-weight: 800; color: {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'}; margin-top: 3px;">{'Restrittiva (Surplus)' if core_state.taylor_rate_gap_bps >= 0 else 'Accomodante (Deficit)'}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">{core_state.taylor_rate_gap_bps:+.0f} punti base</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Formulazione Matematica della Regola di Taylor (1993):")
    st.markdown(r"""
    $$i_t = r^* + \pi_t + 0.5 \cdot (\pi_t - \pi^*) + 0.5 \cdot (y_t - y^*)$$
    * **$r^* = 1.00\%$:** Tasso neutrale reale di equilibrio di lungo termine.
    * **$\pi_t = 3.01\%$:** Tasso di inflazione Core PCE effettivo.
    * **$\pi^* = 2.00\%$:** Obiettivo statutario di stabilità dei prezzi della Federal Reserve.
    * **$y_t = 2.19\%$:** Crescita annua del PIL reale (Real GDP).
    * **$y^* = 2.00\%$:** Trend potenziale di crescita economica non inflazionistica.
    """)

    st.markdown("#### 💡 Lettura dell'Indicatore:")
    if core_state.taylor_rate_gap_bps >= 0:
        st.markdown(f"""
        - **Taylor Gap $\\ge 0$ (Valore attuale: {core_state.taylor_rate_gap_bps:+.0f} bps):** La Federal Reserve applica un tasso effettivo superiore al benchmark teorico di Taylor (**Restrittiva / Surplus di policy**). Ciò esercita una pressione disinflattiva sull'economia reale, raffreddando la domanda aggregata.
        - **Taylor Gap $< 0$:** Condizione in cui il tasso effettivo è inferiore alla prescrizione teorica (*behind the curve* / accomodante).
        """)
    else:
        st.markdown(f"""
        - **Taylor Gap $< 0$ (Valore attuale: {core_state.taylor_rate_gap_bps:+.0f} bps):** La Federal Reserve applica un tasso effettivo inferiore al benchmark teorico prescritto di Taylor ({fed_rate:.2f}% effettivo vs {taylor_prescribed:.2f}% prescritto $\\rightarrow$ **Accomodante / Deficit di policy**). La banca centrale risulta in ritardo rispetto all'inflazione ("*behind the curve*"), mantenendo condizioni finanziarie ancora espansive.
        - **Taylor Gap $> 0$:** Condizione in cui il tasso effettivo è superiore alla prescrizione teorica (*ahead of the curve* / restrittiva).
        """)


# Modal Dialog: Global Net Liquidity Dashboard (Fed H.4.1, TGA, RRP & Asset Transmission)
@st.dialog("🌊 Global Net Liquidity Dashboard (Fed H.4.1, TGA & RRP)", width="large")
def show_net_liquidity_modal():
    liq = net_liquidity_engine.compute_net_liquidity(aligned_df)
    st.markdown(f"### 🌊 Liquidità Netta Federal Reserve: <span style='color: {liq.liquidity_color_hex};'>${liq.us_net_liquidity_b:,.0f} Miliardi • {liq.liquidity_regime_badge}</span>", unsafe_allow_html=True)
    st.caption("Framework Quantitativo dei Macro Hedge Fund (CrossBorder Capital / Goldman Sachs): misura il denaro realmente disponibile nel sistema bancario e la sua trasmissione anticipatrice a 4-8 settimane verso azioni e oro.")

    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">US Net Liquidity</div>
            <div style="font-size: 24px; font-weight: 800; color: #38BDF8; font-family: monospace;">${liq.us_net_liquidity_b:,.0f} B</div>
            <div style="font-size: 11px; color: {'#10B981' if liq.us_net_liquidity_delta_4w_b >= 0 else '#EF4444'}; margin-top: 2px;">Δ 4W: {liq.us_net_liquidity_delta_4w_b:+,.0f} B ({liq.us_net_liquidity_delta_4w_pct:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with l2:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Attivo Totale Fed (WALCL)</div>
            <div style="font-size: 24px; font-weight: 800; color: #F8FAFC; font-family: monospace;">${liq.fed_total_assets_b:,.0f} B</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Dimensione Bilancio Fed</div>
        </div>
        """, unsafe_allow_html=True)
    with l3:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Conto Tesoro TGA (WTREGEN)</div>
            <div style="font-size: 24px; font-weight: 800; color: #F59E0B; font-family: monospace;">${liq.tga_balance_b:,.0f} B</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Cassa Tesoro (Drena Liquidità)</div>
        </div>
        """, unsafe_allow_html=True)
    with l4:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Reverse Repo RRP (ON RRP)</div>
            <div style="font-size: 24px; font-weight: 800; color: #A78BFA; font-family: monospace;">${liq.rrp_balance_b:,.0f} B</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Liquidità Parcheggiata Fed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Charts
    fig_overview = build_net_liquidity_overview_figure(liq, dark_mode=True)
    st.plotly_chart(fig_overview, use_container_width=True)

    col_liq_l, col_liq_r = st.columns([1.1, 0.9])
    with col_liq_l:
        fig_corr = build_liquidity_asset_correlation_figure(liq, dark_mode=True)
        st.plotly_chart(fig_corr, use_container_width=True)
    with col_liq_r:
        st.markdown("#### 🔬 Equazione Istituzionale & Meccanica:")
        st.markdown(r"""
        $$\text{US Net Liquidity} = \text{Fed Total Assets} - \text{TGA} - \text{ON RRP}$$
        * **1. Attivo Fed ($\text{WALCL}$):** Base monetaria creata tramite titoli detenuti e prestiti bancari. Durante il QT si contrae di $\sim \$25\text{-}60\text{B}$ al mese.
        * **2. Treasury General Account ($\text{TGA}$):** Il conto corrente del governo USA presso la Fed. Quando il Tesoro raccoglie tasse o emette debito, la cassa sale e **drena liquidità** dalle riserve bancarie private.
        * **3. Reverse Repo Facility ($\text{RRP}$):** Liquidità parcheggiata dai fondi monetari presso la Fed. Quando l'RRP scende, il capitale rientra nel sistema bancario agendo come **stimolo quantitativo implicito**.
        * **Beta di Trasmissione:** Empiricamente ogni **$+\$100\text{B}$** di iniezione di Liquidità Netta genera un incremento medio dell'**$+1.45\%$ sull'S&P 500** con un ritardo di trasmissione (*lag*) compreso tra **4 e 8 settimane**.
        """)

    st.markdown("---")
    st.markdown("#### 📋 Sintesi Trasmissione Quantitativa su Asset Chiave:")
    corr_table = [
        {"Asset Target": "Nasdaq 100 (EQ_NDX)", "Correlazione Lead 4W": f"+{liq.corr_ndx_4w:.2f}", "Correlazione Lead 8W": f"+{liq.corr_ndx_8w:.2f}", "Sensibilità Prevista": "Massima (Multipli Growth ad alta duration)"},
        {"Asset Target": "S&P 500 (EQ_SPX)", "Correlazione Lead 4W": f"+{liq.corr_spx_4w:.2f}", "Correlazione Lead 8W": f"+{liq.corr_spx_8w:.2f}", "Sensibilità Prevista": "Elevata (+1.45% per $100B iniezione)"},
        {"Asset Target": "Oro Fisico (COM_GOLD)", "Correlazione Lead 4W": f"+{liq.corr_gold_4w:.2f}", "Correlazione Lead 8W": f"+{liq.corr_gold_8w:.2f}", "Sensibilità Prevista": "Moderata-Alta (Hedge svalutazione fiat)"},
    ]
    st.dataframe(pd.DataFrame(corr_table), use_container_width=True, hide_index=True)


# Modal Dialog: Diebold-Yilmaz Spillover Network (Directional Contagion & Net Transmitters)
@st.dialog("🕸️ Diebold-Yılmaz Spillover Network (Chi Contagia Chi • VAR/GFEVD)", width="large")
def show_diebold_spillover_modal():
    dy_state = diebold_yilmaz_engine.calculate_spillover_state(aligned_df)
    st.markdown(f"### 🕸️ Indice di Connessione Totale (TCI): <span style='color: {dy_state.tci_color_hex};'>{dy_state.total_connectedness_index:.1f}% • {dy_state.tci_regime.split('(')[0].strip()}</span>", unsafe_allow_html=True)
    st.caption("Framework Econometrico Diebold-Yılmaz (Wharton / Journal of Econometrics): quantifica la direzione causale degli shock (VAR/GFEVD a 10 giorni) isolando i Trasmettitori Netti di stress sistemico dagli Assorbitori.")

    dy_k1, dy_k2, dy_k3, dy_k4 = st.columns(4)
    with dy_k1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1.5px solid {dy_state.tci_color_hex}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Indice Connessione (TCI)</div>
            <div style="font-size: 24px; font-weight: 800; color: {dy_state.tci_color_hex}; font-family: monospace;">{dy_state.total_connectedness_index:.1f}%</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Varianza da Spillover Cross-Asset</div>
        </div>
        """, unsafe_allow_html=True)
    with dy_k2:
        top_tx = dy_state.top_transmitters[0] if dy_state.top_transmitters else None
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #EF4444; font-weight: 700;">Top Trasmettitore Netto</div>
            <div style="font-size: 18px; font-weight: 800; color: #EF4444; margin-top: 4px;">{top_tx.symbol if top_tx else 'N/A'}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Net Spillover: <b>{top_tx.net_spillover_pct:+.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with dy_k3:
        top_rx = dy_state.top_receivers[-1] if dy_state.top_receivers else None
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #10B981; font-weight: 700;">Top Ricevitore Netto</div>
            <div style="font-size: 18px; font-weight: 800; color: #10B981; margin-top: 4px;">{top_rx.symbol if top_rx else 'N/A'}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Net Spillover: <b>{top_rx.net_spillover_pct:+.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with dy_k4:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Orizzonte Forecast (H)</div>
            <div style="font-size: 24px; font-weight: 800; color: #F8FAFC; font-family: monospace;">{dy_state.horizon_steps} Giorni</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">GFEVD Ordine-Invariante (VAR {dy_state.var_lags})</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive Banner
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid {dy_state.tci_color_hex}; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="font-size: 12px; font-weight: 700; color: {dy_state.tci_color_hex}; text-transform: uppercase;">
            🔍 Diagnosi Causale della Rete di Contagio:
        </div>
        <div style="font-size: 13.5px; color: #F8FAFC; margin-top: 4px; line-height: 1.5;">
            {dy_state.executive_diagnosis}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plotly Charts: Heatmap Matrix + Net Transmission Bars
    col_dy1, col_dy2 = st.columns([1.1, 0.9])
    with col_dy1:
        fig_dy_heat = build_diebold_yilmaz_heatmap(dy_state, dark_mode=True)
        st.plotly_chart(fig_dy_heat, use_container_width=True)
    with col_dy2:
        fig_dy_bar = build_spillover_net_transmission_figure(dy_state, dark_mode=True)
        st.plotly_chart(fig_dy_bar, use_container_width=True)

    st.markdown("#### 📋 Tabella Direzionale Completa Diebold-Yılmaz (Ruoli di Trasmissione & Infezione):")
    df_profiles = pd.DataFrame([
        {
            "Simbolo": p.symbol,
            "Nome Pilastro Sistemico": p.name,
            "Cluster": p.cluster,
            "Varianza Propria (%)": f"{p.own_variance_pct:.1f}%",
            "Ricevuto dagli Altri (FROM %)": f"{p.from_others_pct:.1f}%",
            "Trasmesso agli Altri (TO %)": f"{p.to_others_pct:.1f}%",
            "Spillover Netto (TO - FROM)": f"{p.net_spillover_pct:+.1f}%",
            "Ruolo Sistemico": p.role_badge,
            "Bersaglio Principale": p.top_transmission_target,
            "Fonte Infezione Principale": p.top_infection_source,
        }
        for p in dy_state.nodes_profile
    ])
    st.dataframe(df_profiles, use_container_width=True, hide_index=True)

    st.markdown("""
    > **Note Metodologiche (Diebold & Yılmaz, Wharton School):**  
    > * **Order-Invariance:** Grazie alla scomposizione della varianza generalizzata di Pesaran-Shin (1998), i risultati **non dipendono dall'ordinamento delle variabili nel VAR**, superando il limite della fattorizzazione di Cholesky classica.  
    > * **Lettura Causale Direzionale:**  
    >   - Se $\text{Net Spillover} > 0$ (Barra Rossa): Il nodo è un **Trasmettitore Netto** (esporta più volatilità di quanta ne importi). Quando entra in crisi, trascina il resto del mercato.  
    >   - Se $\text{Net Spillover} < 0$ (Barra Verde): Il nodo è un **Ricevitore / Assorbitore Netto** di shock esterni (subisce passivamente le turbolenze originate altrove).
    """)


# Modal Dialog 6: Nodi Monitorati & Architettura Radiale
@st.dialog("🕸️ Dettaglio Analitico: Architettura dei 30 Nodi e Topologia della Rete", width="large")
def show_network_nodes_modal():
    st.markdown(f"### 🌐 Copertura Globale: <span style='color: #F8FAFC;'>{len(node_states)} Nodi Istituzionali</span>", unsafe_allow_html=True)
    st.caption("Topologia concentrica distribuita su 3 Anelli Gerarchici e 4 Quadranti Macroeconomici.")

    nc1, nc2, nc3 = st.columns(3)
    r1_count = sum(1 for ns in node_states.values() if ns.ring == 1)
    r2_count = sum(1 for ns in node_states.values() if ns.ring == 2)
    r3_count = sum(1 for ns in node_states.values() if ns.ring == 3)
    with nc1:
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #38BDF8; font-weight: 700;">Anello 1 (Interno)</div>
            <div style="font-size: 24px; font-weight: 800; color: #38BDF8; font-family: monospace;">{r1_count} Nodi</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Politica Monetaria & Tassi Ufficiali</div>
        </div>
        """, unsafe_allow_html=True)
    with nc2:
        st.markdown(f"""
        <div style="background: rgba(168, 85, 247, 0.1); border: 1.5px solid #A855F7; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #A855F7; font-weight: 700;">Anello 2 (Intermedio)</div>
            <div style="font-size: 24px; font-weight: 800; color: #A855F7; font-family: monospace;">{r2_count} Nodi</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Fondamentali Reali, PIL, Inflazione & Lavoro</div>
        </div>
        """, unsafe_allow_html=True)
    with nc3:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1.5px solid #10B981; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #10B981; font-weight: 700;">Anello 3 (Esterno)</div>
            <div style="font-size: 24px; font-weight: 800; color: #10B981; font-family: monospace;">{r3_count} Nodi</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Asset Intermarket (Azionario, Bond, Commodity, FX)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🗺️ Suddivisione per Quadrante Macro:")
    q_counts = {}
    for n_id, ns in node_states.items():
        q_counts[ns.quadrant] = q_counts.get(ns.quadrant, 0) + 1
    
    st.write(f"- **🟡 INFLAZIONE & ASPETTATIVE (`INFLATION`):** {q_counts.get('INFLATION', 0)} serie storiche (Core PCE, Headline CPI, HICP Eurozona, 5Y Breakeven, Petrolio WTI, Oro, TIPS Reale)")
    st.write(f"- **🟢 CRESCITA & PRODUZIONE (`GROWTH`):** {q_counts.get('GROWTH', 0)} serie storiche (Real GDP, Produzione Industriale, Vendite al Dettaglio, Rame, Pendenza Curva 10Y-2Y, Borse USA/UE)")
    st.write(f"- **🔵 LAVORO & OCCUPAZIONE (`LABOR`):** {q_counts.get('LABOR', 0)} serie storiche (Tasso Disoccupazione U3, Buste Paga NFP, Sussidi di Disoccupazione Initial Claims)")
    st.write(f"- **🟣 LIQUIDITÀ, RISCHIO & TASSI (`LIQUIDITY_RISK`):** {q_counts.get('LIQUIDITY_RISK', 0)} serie storiche (Fed Funds, Tasso BCE, Treasury 2Y e 10Y, Bund 10Y, Spread BTP/Bund, Dollaro DXY, VIX, Spread High Yield)")

    st.markdown("#### 📊 Sintesi Nodi e Z-Score:")
    nodes_summary_table = []
    for n_id, ns in sorted(node_states.items(), key=lambda x: abs(x[1].z_score), reverse=True):
        nodes_summary_table.append({
            "Ticker": n_id,
            "Nome": ns.name,
            "Anello": ns.ring,
            "Valore": f"{ns.current_value:,.2f} {ns.unit}",
            "Z-Score": f"{ns.z_score:+.2f}",
            "Stato": ns.status,
            "Fonte": ns.source,
        })
    st.dataframe(pd.DataFrame(nodes_summary_table), use_container_width=True, hide_index=True)


# Modal Dialog 7: Distribuzione e Diagnosi Z-Score dei 30 Nodi
@st.dialog("📊 Z-Score degli Indicatori Macro-Finanziari (30 Nodi)", width="large")
def show_zscore_modal():
    st.markdown(
        '<div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">'
        '<div style="font-size: 14px; font-weight: 800; color: #38BDF8; margin-bottom: 4px;">'
        '📐 STANDARDIZZAZIONE STATISTICA Z-SCORE (ROLLING WINDOW 1Y)'
        '</div>'
        '<div style="font-size: 12px; color: #CBD5E1; line-height: 1.5;">'
        'Lo <b>Z-Score</b> normalizza tutte le 30 variabili macro-finanziarie calcolando quante deviazioni standard il valore odierno dista dalla propria media storica mobile: '
        '<span style="font-family: monospace; color: #F8FAFC; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">Z = (X_t - μ) / σ</span>. '
        'Ciò consente un confronto quantitativo immediato e neutrale tra scale dimensionali profondamente disomogenee (tassi in %, borse in punti, greggio in $/barile, spread di credito in bps).'
        '</div></div>',
        unsafe_allow_html=True
    )

    n_ext = sum(1 for ns in node_states.values() if abs(ns.z_score) >= 2.0)
    n_att = sum(1 for ns in node_states.values() if 1.25 <= abs(ns.z_score) < 2.0)
    n_norm = sum(1 for ns in node_states.values() if abs(ns.z_score) < 1.25)

    # 3 Colonne di KPI di sintesi dello Z-Score
    zc1, zc2, zc3 = st.columns(3)
    with zc1:
        st.markdown(
            f'<div style="background: rgba(59, 130, 246, 0.1); border: 1.5px solid #3B82F6; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 11px; text-transform: uppercase; color: #3B82F6; font-weight: 700;">🟦 Regime Estremo (|Z| ≥ 2.0)</div>'
            f'<div style="font-size: 24px; font-weight: 800; color: #3B82F6; font-family: monospace;">{n_ext} Nodi</div>'
            f'<div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Anomalie Critiche / Shock di Mercato</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with zc2:
        st.markdown(
            f'<div style="background: rgba(239, 68, 68, 0.1); border: 1.5px solid #EF4444; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 11px; text-transform: uppercase; color: #EF4444; font-weight: 700;">🟥 In Attenzione (1.25 ≤ |Z| < 2.0)</div>'
            f'<div style="font-size: 24px; font-weight: 800; color: #EF4444; font-family: monospace;">{n_att} Nodi</div>'
            f'<div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Surriscaldamento / Tensione Ciclica</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with zc3:
        st.markdown(
            f'<div style="background: rgba(245, 158, 11, 0.1); border: 1.5px solid #F59E0B; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 11px; text-transform: uppercase; color: #F59E0B; font-weight: 700;">🟨 In Equilibrio (|Z| < 1.25)</div>'
            f'<div style="font-size: 24px; font-weight: 800; color: #F59E0B; font-family: monospace;">{n_norm} Nodi</div>'
            f'<div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Range Statistico Normale (±1.25σ)</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grafico a barre orizzontali ad alta risoluzione (identico allo screenshot dell'utente)
    fig_zscore = build_zscore_distribution_figure(node_states, dark_mode=True)
    st.plotly_chart(fig_zscore, use_container_width=True)

    st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Tabella Analitica Completa Ordinata per Deviazione Z-Score:")

    table_z_data = []
    for ns in sorted(node_states.values(), key=lambda x: abs(x.z_score), reverse=True):
        table_z_data.append({
            "Indicatore": ns.name,
            "ID Ticker": ns.official_ticker,
            "Anello": f"Anello {ns.ring}",
            "Quadrante": ns.quadrant,
            "Valore Attuale": f"{ns.current_value:,.2f} {ns.unit}",
            "Z-Score Standardizzato": f"{ns.z_score:+.2f}",
            "Stato Rete": ns.status,
            "Delta Periodo": f"{ns.delta_period:+.2f} ({ns.delta_period_pct:+.2f}%)"
        })
    st.dataframe(pd.DataFrame(table_z_data), use_container_width=True, hide_index=True)



# Modal Dialog 8: SPIDER SI (Super Intelligence - 10 Giganti dell'Intelligenza Artificiale)
@st.dialog("🤖 SPIDER SI (Super Intelligence): I 10 Giganti dell'Intelligenza Artificiale", width="large")
def show_super_intelligence_modal():
    si_state = si_engine.compute_si_state()
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(56, 189, 248, 0.15)); border: 2px solid #8B5CF6; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 11px; text-transform: uppercase; color: #8B5CF6; font-weight: 800; letter-spacing: 0.8px;">SPIDER SI • SUPER INTELLIGENCE 10 ECOSYSTEM</span>
                <div style="font-size: 22px; font-weight: 800; color: #F8FAFC; margin-top: 2px;">Monitoraggio Istituzionale dell'Ecosistema AI Globale</div>
                <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 4px;">Monitoraggio congiunto della catena del valore dell'AI: Fonderie, Chip & GPU, Memorie HBM, Datacenter Hyperscalers e Piattaforme Software.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.9); border: 1.5px solid {si_state.ai_cycle_color}; border-radius: 10px; padding: 10px 16px; text-align: center; min-width: 160px;">
                <div style="font-size: 10.5px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">AI Momentum Score</div>
                <div style="font-size: 26px; font-weight: 800; color: {si_state.ai_cycle_color}; font-family: monospace;">{si_state.ai_momentum_score:.1f} / 100</div>
                <div style="font-size: 11px; color: {si_state.ai_cycle_color}; font-weight: 700;">{si_state.ai_cycle_regime.split('(')[0].strip()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Market Cap AI Aggregato</div>
            <div style="font-size: 19px; font-weight: 800; color: #38BDF8; font-family: monospace;">${si_state.total_market_cap_t:.2f} Trilioni</div>
            <div style="font-size: 10.5px; color: #64748B;">10 Giganti Globali</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Performance Media YTD</div>
            <div style="font-size: 19px; font-weight: 800; color: {'#10B981' if si_state.composite_ytd_pct >= 0 else '#EF4444'}; font-family: monospace;">{si_state.composite_ytd_pct:+.1f}%</div>
            <div style="font-size: 10.5px; color: #64748B;">1 Mese: {si_state.composite_1m_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Capex AI Globale Stimato</div>
            <div style="font-size: 19px; font-weight: 800; color: #F59E0B; font-family: monospace;">${si_state.total_annual_capex_b:,.0f} Miliardi</div>
            <div style="font-size: 10.5px; color: #64748B;">Spesa Infrastrutture Datacenter</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Top Performer YTD</div>
            <div style="font-size: 19px; font-weight: 800; color: #10B981; font-family: monospace;">{si_state.top_performer.ticker}</div>
            <div style="font-size: 10.5px; color: #10B981;">{si_state.top_performer.change_ytd_pct:+.1f}% YTD</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dual tabs inside modal: Tela Radiale & Tabella Fondamentali
    m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
        "🕸️ Tela Radiale SPIDER SI",
        "📋 Cockpit Fondamentale dei 10 Giganti",
        "🔗 Matrice Correlazioni & Sotto-Settori",
        "💡 Analisi Capex, Margini & Valutazioni PEG",
    ])
    with m_tab1:
        c_m_dim1, c_m_dim2 = st.columns([1.5, 2.5])
        with c_m_dim1:
            modal_si_dim = st.radio(
                "Prospettiva Tela SI:",
                ["🕸️ Vista 2D Radiale", "🌐 Vista 3D Spaziale (WebGL)"],
                horizontal=True,
                key="radio_si_modal_dim"
            )
        with c_m_dim2:
            st.caption("🔍 **Zoom con il mouse:** Usa la rotella per zoomare. In 3D: trascina per ruotare a 360°, tasto destro per pan orbitale.")

        is_modal_3d = ("3D" in modal_si_dim)
        if is_modal_3d:
            fig_si_net = build_spider_si_network_3d_figure(si_state, dark_mode=True)
        else:
            fig_si_net = build_spider_si_network_figure(si_state, dark_mode=True)

        st.plotly_chart(
            fig_si_net,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"] if not is_modal_3d else [],
            },
            key=f"plotly_modal_si_chart_{'3d' if is_modal_3d else '2d'}"
        )
        st.caption("ℹ️ **Legenda Tela SI:** Al centro SI_CORE. Anello 1 = Silicio & Fonderia; Anello 2 = Memorie HBM & Cloud Specializzato; Anello 3 = Hyperscalers & Software. I fili viola evidenziano i flussi di fornitura e ordini Capex tra le aziende.")
    with m_tab2:
        df_si_table = []
        for s in sorted(si_state.stocks, key=lambda x: x.market_cap_b, reverse=True):
            df_si_table.append({
                "Ticker": s.ticker,
                "Società": s.name,
                "Sotto-settore": s.subsector,
                "Prezzo ($)": f"${s.current_price:,.2f}",
                "1D %": f"{s.change_1d_pct:+.2f}%",
                "1M %": f"{s.change_1m_pct:+.1f}%",
                "YTD %": f"{s.change_ytd_pct:+.1f}%",
                "Market Cap ($B)": f"${s.market_cap_b:,.0f} B",
                "Forward P/E": f"{s.fwd_pe:.1f}x",
                "PEG Ratio": f"{s.peg_ratio:.2f}",
                "Crescita Ricavi YoY": f"+{s.rev_growth_yoy:.1f}%",
                "Margine Lordo": f"{s.gross_margin_pct:.1f}%",
                "Ruolo Catena Valore": s.supply_chain_role,
            })
        st.dataframe(pd.DataFrame(df_si_table), use_container_width=True, hide_index=True)
    with m_tab3:
        col_c1, col_c2 = st.columns([1.2, 1])
        with col_c1:
            fig_si_corr = build_si_correlation_heatmap(si_state, dark_mode=True)
            st.plotly_chart(fig_si_corr, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        with col_c2:
            fig_si_sub = build_si_subsector_bar_figure(si_state, dark_mode=True)
            st.plotly_chart(fig_si_sub, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    with m_tab4:
        st.markdown("##### 📊 Analisi Economico-Finanziaria & Valutazioni dell'Ecosistema AI:")
        
        # Two Charts: Capex vs Revenue & PEG Valuation
        col_m_ch1, col_m_ch2 = st.columns(2)
        with col_m_ch1:
            fig_m_capex = build_si_capex_vs_revenue_figure(si_state, dark_mode=True)
            st.plotly_chart(fig_m_capex, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="modal_fig_capex_rev")
        with col_m_ch2:
            fig_m_peg = build_si_peg_valuation_figure(si_state, dark_mode=True)
            st.plotly_chart(fig_m_peg, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="modal_fig_peg")

        # 3 Structured Analysis Cards
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #10B981; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="font-size: 13.5px; font-weight: 700; color: #10B981; text-transform: uppercase;">
                1. Dicotomia tra chi finanzia la rivoluzione AI e chi la monetizza oggi
            </div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 6px; line-height: 1.6;">
                <b>I Finanziatori dell'Infrastruttura (Hyperscalers):</b> Microsoft ($55B), Amazon ($70B), Alphabet ($52B), Meta ($40B) e Oracle ($10B) investono oltre <b>$227 Miliardi di Capex annuo</b> per costruire la capacità computazionale dei data center del futuro. Si tratta di un impegno di capitale massiccio (Capex-heavy) il cui ritorno sul capitale investito (ROIC) richiede anni per maturare.<br>
                <b>I Monetizzatori Immediati (Semiconduttori & Fonderie):</b> All'estremo opposto, NVIDIA (+122% YoY con soli $4.5B di capex proprio), TSMC (+36% YoY) e Broadcom (+47% YoY) assorbono direttamente questi flussi di spesa, convertendoli istantaneamente in fatturato record e Free Cash Flow immediato.
            </div>
        </div>

        <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #38BDF8; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="font-size: 13.5px; font-weight: 700; color: #38BDF8; text-transform: uppercase;">
                2. Margini operativi d'eccellenza e potere di prezzo (Pricing Power)
            </div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 6px; line-height: 1.6;">
                La redditività dell'ecosistema SI evidenzia barriere all'entrata e potere di prezzo quasi anelastico:<br>
                • <b>NVIDIA (75.1% Margine Lordo):</b> L'architettura CUDA crea un lock-in software invalicabile, consentendo prezzi unitari di $30,000–$40,000 per chip Hopper/Blackwell con margini industriali unici.<br>
                • <b>Meta Platforms (81.5% Margine Operativo/Lordo Software):</b> Straordinaria leva operativa pubblicitaria che finanzia integralmente lo sviluppo di Llama open-source e l'acquisto di centinaia di migliaia di cluster GPU.<br>
                • <b>TSMC (54.5% Margine) & Broadcom (62.0%):</b> Il monopolio globale sui wafer a 3nm e il packaging avanzato CoWoS proteggono TSMC da qualsiasi compressione dei margini anche durante i cicli macro avversi.
            </div>
        </div>

        <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;">
            <div style="font-size: 13.5px; font-weight: 700; color: #F59E0B; text-transform: uppercase;">
                3. Valutazioni di mercato ed efficienza sul tasso di crescita (Forward P/E vs. PEG Ratio)
            </div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 6px; line-height: 1.6;">
                L'analisi del semplice Forward P/E (es. NVDA a 34.5x, MSFT a 31.2x) offre una visione distorta se non normalizzata per il tasso di crescita (PEG Ratio = P/E / Crescita):<br>
                • <b>Massima Efficienza Valutativa (PEG &le; 1.25):</b> <b>META (PEG 1.10)</b> e <b>NVDA (PEG 1.15)</b> si posizionano vicini alla <i>soglia aurea di Peter Lynch (PEG = 1.0)</i>. Una crescita dei ricavi superiore al 100% rende il multiplo di NVIDIA estremamente attraente su base adjusted.<br>
                • <b>Valutazioni a Premio (PEG &gt; 1.80):</b> Società con crescite solide ma moderate (es. MSFT a 2.10x PEG, ORCL a 1.95x) scambiano a multipli che incorporano già anni di espansione futura, richiedendo una perfetta esecuzione operativa nei trimestri a venire.
            </div>
        </div>
        """, unsafe_allow_html=True)


# Helper per rendering stelle importanza ciclo economico (1 a 5 stelle)
def render_cycle_stars_badge(stars: int, max_stars: int = 5) -> str:
    gold = "★" * stars
    dim = "☆" * (max_stars - stars)
    return (
        f'<span class="kpi-stars-badge" title="Importanza & Influenza sul Ciclo Economico: {stars}/{max_stars}">'
        f'<span class="kpi-stars-gold">{gold}</span><span class="kpi-stars-dim">{dim}</span>'
        f'</span>'
    )


# 11 Colonne di KPI Superiori (con SI - Super Intelligence, Liquidità Netta, CISS, Diebold-Yilmaz Spillover & Z-Score)
k1, k2, k3, k4, k5, k_dy, k_si, k6, k7, k8, k9 = st.columns([1.0, 1.0, 1.0, 1.02, 1.05, 1.05, 1.08, 1.0, 1.0, 0.95, 1.25])

with k1:
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid #38BDF8;">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Regime Macroeconomico Primario">REGIME MACRO</span>
            {render_cycle_stars_badge(5)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="font-size: 13.5px; color: #38BDF8;">{core_state.macro_regime.split('/')[0].split('(')[0].strip()}</div>
            <div class="kpi-sub-val">Growth: {core_state.growth_score:+.2f} | Inf: {core_state.inflation_score:+.2f}</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Influenza Massima (5/5): la combinazione Crescita/Inflazione governa l'allocazione primaria">
                <span>★ 5/5</span> <span style="color: #94A3B8; font-weight: 500;">Driver Ciclo</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ℹ️ Dett. Regime", key="btn_open_regime_modal", use_container_width=True, type="secondary"):
        show_macro_regime_modal()

with k2:
    ls_obj = core_state.landing_scenario
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid {ls_obj.color_hex};">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Scenario Atterraggio Ciclico (Landing)">SCENARIO LANDING</span>
            {render_cycle_stars_badge(4)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="font-size: 14px; color: {ls_obj.color_hex};">{ls_obj.scenario_badge}</div>
            <div class="kpi-sub-val">Conf: {ls_obj.confidence_pct:.0f}% (NoLand {ls_obj.prob_no_landing:.0f}%)</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Importanza Elevata (4/5): anticipa i punti di svolta tra soft landing, hard landing e no landing">
                <span>★ 4/5</span> <span style="color: #94A3B8; font-weight: 500;">Svolta Ciclica</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ℹ️ Dett. Landing", key="btn_open_landing_modal", use_container_width=True, type="secondary"):
        show_landing_scenario_modal()

with k3:
    t_val = core_state.systemic_web_tension
    t_col = '#EF4444' if t_val >= 2.0 else ('#F59E0B' if t_val >= 1.25 else '#10B981')
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid {t_col};">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Tensione Sistemica Tela Intermarket (|Z|)">TENSIONE TELA</span>
            {render_cycle_stars_badge(5)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: {t_col}; font-size: 18px;">|Z| = {t_val:.2f}</div>
            <div class="kpi-sub-val">{core_state.web_tension_status.split('(')[0].strip()}</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Influenza Massima (5/5): misura la stabilità d'insieme della rete e guida il de-risking">
                <span>★ 5/5</span> <span style="color: #94A3B8; font-weight: 500;">De-risking Rete</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ℹ️ Dett. Tensione", key="btn_open_tension_modal", use_container_width=True, type="secondary"):
        show_web_tension_modal()

with k4:
    ciss_obj = core_state.ciss
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid {ciss_obj.status_color};">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Composite Indicator of Systemic Stress (BCE)">CONTAGIO CISS</span>
            {render_cycle_stars_badge(4)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: {ciss_obj.status_color}; font-size: 17px;">{ciss_obj.ciss_score * 100:.1f}%</div>
            <div class="kpi-sub-val">{ciss_obj.ciss_status.split('(')[0].strip()}</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Importanza Elevata (4/5): rileva la sincronizzazione dello stress nei 5 quadranti finanziari">
                <span>★ 4/5</span> <span style="color: #94A3B8; font-weight: 500;">Stress BCE</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⚡ Dett. CISS", key="btn_open_ciss_modal", use_container_width=True, type="secondary"):
        show_ciss_modal()

with k5:
    liq_curr = net_liquidity_engine.compute_net_liquidity(aligned_df)
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid {liq_curr.liquidity_color_hex};">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Federal Reserve Net Liquidity = Assets - TGA - RRP">NET LIQUIDITY</span>
            {render_cycle_stars_badge(5)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: #38BDF8; font-size: 16px;">${liq_curr.us_net_liquidity_b:,.0f} B</div>
            <div class="kpi-sub-val">Δ 4W: {liq_curr.us_net_liquidity_delta_4w_b:+,.0f}B ({liq_curr.us_net_liquidity_delta_4w_pct:+.1f}%)</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Influenza Massima (5/5): la liquidità quantitativa anticipa le asset class di 4-8 settimane">
                <span>★ 5/5</span> <span style="color: #94A3B8; font-weight: 500;">Driver Fed 4-8W</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🌊 Dett. Liquidità", key="btn_open_liquidity_modal", use_container_width=True, type="secondary"):
        show_net_liquidity_modal()

with k_dy:
    dy_curr = diebold_yilmaz_engine.calculate_spillover_state(aligned_df)
    top_tx_label = dy_curr.top_transmitters[0].symbol if dy_curr.top_transmitters else "N/A"
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid {dy_curr.tci_color_hex};">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Total Connectedness Index (Diebold-Yilmaz VAR/GFEVD)">SPILLOVER TCI</span>
            {render_cycle_stars_badge(4)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: {dy_curr.tci_color_hex}; font-size: 17px;">{dy_curr.total_connectedness_index:.1f}%</div>
            <div class="kpi-sub-val">Leader: <b style="color: #EF4444;">{top_tx_label}</b> ({dy_curr.top_transmitters[0].net_spillover_pct:+.1f}%)</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Importanza Elevata (4/5): misura la propagazione e la causalità direzionale del contagio">
                <span>★ 4/5</span> <span style="color: #94A3B8; font-weight: 500;">Chi Contagia</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🕸️ Chi Contagia", key="btn_open_diebold_modal", use_container_width=True, type="secondary"):
        show_diebold_spillover_modal()

with k_si:
    si_curr = si_engine.compute_si_state()
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid #8B5CF6;">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Super Intelligence (SI): I 10 Giganti dell'Intelligenza Artificiale Mondiale">SI - SUPER INT.</span>
            {render_cycle_stars_badge(5)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: #8B5CF6; font-size: 16px;">{si_curr.composite_ytd_pct:+.1f}% YTD</div>
            <div class="kpi-sub-val">Cap: <b>${si_curr.total_market_cap_t:.1f}T</b> • Top: <b>{si_curr.top_performer.ticker}</b></div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Influenza Massima (5/5): Il super-ciclo AI e i Capex dei semiconduttori guidano la produttività mondiale">
                <span>★ 5/5</span> <span style="color: #94A3B8; font-weight: 500;">Megatrend AI</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🤖 Dettagli SI", key="btn_open_si_modal", use_container_width=True, type="secondary"):
        show_super_intelligence_modal()

with k6:
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid #A78BFA;">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Orientamento di Politica Monetaria Fed & BCE">BANCHE CENTRALI</span>
            {render_cycle_stars_badge(5)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="font-size: 15px; color: #A78BFA;">{core_state.central_bank_bias.split('(')[0].strip()}</div>
            <div class="kpi-sub-val">FED: {core_state.fed_real_rate:+.2f}% | BCE: {core_state.ecb_real_rate:+.2f}%</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Influenza Massima (5/5): stabilisce il costo del denaro e i tassi reali che pilotano il ciclo">
                <span>★ 5/5</span> <span style="color: #94A3B8; font-weight: 500;">Tassi Reali</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ℹ️ Dett. Banche", key="btn_open_cb_bias_modal", use_container_width=True, type="secondary"):
        show_cb_bias_modal()

with k7:
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'};">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Scostamento del Tasso Effettivo dalla Regola di Taylor">TAYLOR GAP</span>
            {render_cycle_stars_badge(3)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'}; font-size: 17px;">{core_state.taylor_rate_gap_bps:+.0f} bps</div>
            <div class="kpi-sub-val">{'Tasso Restrittivo' if core_state.taylor_rate_gap_bps >= 0 else 'Accomodante'}</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Importanza Media (3/5): misura la deviazione dal tasso neutrale di lungo termine">
                <span>★ 3/5</span> <span style="color: #94A3B8; font-weight: 500;">Tasso Neutrale</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ℹ️ Dett. Taylor", key="btn_open_taylor_modal", use_container_width=True, type="secondary"):
        show_taylor_rule_modal()

with k8:
    st.markdown(f"""
    <div class="spider-top-kpi" style="border-left: 3px solid rgba(255,255,255,0.25);">
        <div class="kpi-top-row">
            <span class="kpi-title" title="Completezza dell'Universo dei Nodi SPIDER">NODI RETE</span>
            {render_cycle_stars_badge(3)}
        </div>
        <div class="kpi-body">
            <div class="kpi-main-val" style="color: #F8FAFC; font-size: 18px;">{len(node_states)} / 30</div>
            <div class="kpi-sub-val">10 Fili • 16 Spoke • 4 Sat</div>
        </div>
        <div class="kpi-footer-row">
            <span class="kpi-influence-pill" title="Importanza Media (3/5): granularità informativa e copertura multi-mercato">
                <span>★ 3/5</span> <span style="color: #94A3B8; font-weight: 500;">Copertura Rete</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("ℹ️ Dett. Nodi", key="btn_open_nodes_modal", use_container_width=True, type="secondary"):
        show_network_nodes_modal()

with k9:
    n_ext_tot = sum(1 for ns in node_states.values() if abs(ns.z_score) >= 2.0)
    n_att_tot = sum(1 for ns in node_states.values() if 1.25 <= abs(ns.z_score) < 2.0)
    n_norm_tot = sum(1 for ns in node_states.values() if abs(ns.z_score) < 1.25)
    z_mini_svg = generate_zscore_mini_svg(node_states)
    st.markdown(
        f'<div class="spider-top-kpi" style="border-left: 3px solid #38BDF8; padding: 7px 7px 6px 7px;">'
        f'<div class="kpi-top-row">'
        f'<span class="kpi-title" title="Distribuzione Statistica Z-Score dei 30 Nodi">Z-SCORE 30N</span>'
        f'{render_cycle_stars_badge(4)}'
        f'</div>'
        f'<div class="kpi-body" style="margin: 0;">'
        f'{z_mini_svg}'
        f'<div style="font-size: 8.5px; color: #CBD5E1; text-align: center; margin-top: 1px; display: flex; justify-content: space-between;">'
        f'<span><b style="color: #3B82F6;">{n_ext_tot}</b> Estr</span>'
        f'<span><b style="color: #EF4444;">{n_att_tot}</b> Att</span>'
        f'<span><b style="color: #F59E0B;">{n_norm_tot}</b> Norm</span>'
        f'</div>'
        f'</div>'
        f'<div class="kpi-footer-row">'
        f'<span class="kpi-influence-pill" title="Importanza Elevata (4/5): rileva le dislocazioni e anomalie estreme dei prezzi">'
        f'<span>★ 4/5</span> <span style="color: #94A3B8; font-weight: 500;">Dislocazioni</span>'
        f'</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("ℹ️ Dett. Z-Score", key="btn_open_zscore_modal", use_container_width=True, type="secondary"):
        show_zscore_modal()


# ================= GENERAZIONE SNAPSHOT GLOBALE (FILE A & FILE B) =================
now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
yc_state_curr = yield_curve_engine.get_curve_regime_state()
pring_engine_curr = PringCycleEngine(aligned_df)
pring_res_curr = pring_engine_curr.analyze()
rp_state_curr = risk_parity_engine.optimize_risk_parity_portfolio(aligned_df, systemic_tension_z=core_state.systemic_web_tension)

# Analisi e Formattazione Z-Score dei 30 Nodi
sorted_nodes_z = sorted(node_states.values(), key=lambda x: x.z_score)
n_ext_tot_snap = sum(1 for ns in node_states.values() if abs(ns.z_score) >= 2.0)
n_att_tot_snap = sum(1 for ns in node_states.values() if 1.25 <= abs(ns.z_score) < 2.0)
n_norm_tot_snap = sum(1 for ns in node_states.values() if abs(ns.z_score) < 1.25)
min_node_z = sorted_nodes_z[0] if sorted_nodes_z else None
max_node_z = sorted_nodes_z[-1] if sorted_nodes_z else None

z_score_rows_text = []
for rank_i, ns in enumerate(sorted_nodes_z, 1):
    z_class = "ESTREMO (Blu)" if abs(ns.z_score) >= 2.0 else ("ATTENZIONE (Rosso)" if abs(ns.z_score) >= 1.25 else "NORMALE (Giallo)")
    z_score_rows_text.append(
        f"{rank_i:2d}. [{ns.id:13s}] {ns.name:32s} | Z: {ns.z_score:+6.2f} | Val: {ns.current_value:10.2f} {ns.unit:6s} | Classe: {z_class:18s} | Q: {ns.quadrant}"
    )
z_score_table_formatted = "\n".join(z_score_rows_text)

file_a_text = f"""================================================================================
SPIDER {APP_VERSION} • SNAPSHOT DI STATO GLOBALE DEL SISTEMA MACRO-FINANZIARIO
Data e Ora Salvataggio: {now_iso}
Ambiente: SPIDER {APP_VERSION} Institutional Network Engine (30 Nodi, 10 Fili Maestri)
Versione Software: {APP_VERSION} (Institutional Quantitative Architecture)
================================================================================

1. PARAMETRI DI CONFIGURAZIONE & FILTRI ATTIVI (CONFIGURAZIONE DI PARTENZA)
--------------------------------------------------------------------------------
- Finestra Temporale Correlazioni (Rolling Window): {st.session_state.get('time_window', '1Y')}
- Metodo Matematico Correlazione: {'Spearman (Ranghi Non-Parametrici)' if st.session_state.get('use_spearman', False) else 'Pearson (Lineare)'}
- Filtro Quadrante Macro Attivo: {st.session_state.get('filter_quadrant', 'TUTTI')}
- Filtro Anello Concentrico Attivo: {st.session_state.get('filter_ring', 0)} ({'Tutti gli Anelli' if st.session_state.get('filter_ring', 0) == 0 else f'Anello {st.session_state.get("filter_ring")}'})
- Focus Nodo Selezionato: {st.session_state.get('selected_node', 'TUTTI (Visualizzazione Globale)')}
- Replay Storico Crisi Attivo (CCAR/Aladdin): {HISTORICAL_CRISIS_LIBRARY[active_crisis_id].name if active_crisis_id and active_crisis_id in HISTORICAL_CRISIS_LIBRARY else 'Nessuno (Dati Correnti o What-If Manuale)'}
{f'''  * Periodo Storico: {HISTORICAL_CRISIS_LIBRARY[active_crisis_id].period}
  * Catalizzatore: {HISTORICAL_CRISIS_LIBRARY[active_crisis_id].catalyst}
  * Max Drawdown Stimato 60/40: {HISTORICAL_CRISIS_LIBRARY[active_crisis_id].historical_max_drawdown_60_40:.1f}%
  * Max Drawdown Stimato All Weather: {HISTORICAL_CRISIS_LIBRARY[active_crisis_id].historical_max_drawdown_all_weather:.1f}%
  * Max Drawdown Stimato SPIDER Dynamic: {HISTORICAL_CRISIS_LIBRARY[active_crisis_id].historical_max_drawdown_spider:.1f}%''' if active_crisis_id and active_crisis_id in HISTORICAL_CRISIS_LIBRARY else ''}
- Parametri What-If Stress Manuali:
  * Shock Petrolio (WTI Delta %): {st.session_state.get('what_if_oil', 0.0):+.1f}%
  * Tassi Reali TIPS 10Y (Delta bps): {st.session_state.get('what_if_real_yield', 0.0):+.0f} bps
  * Spread High Yield (Delta bps): {st.session_state.get('what_if_hy', 0.0):+.0f} bps

2. SPIDER_CORE: CENTRO DI CONTROLLO & DIAGNOSI SISTEMICA
--------------------------------------------------------------------------------
- Regime Macroeconomico Attivo: {core_state.macro_regime}
- Growth Score: {core_state.growth_score:+.2f}
- Inflation Score: {core_state.inflation_score:+.2f}
- Coordinate Spaziali Regimi: ({core_state.regime_coords[0]:.2f}, {core_state.regime_coords[1]:.2f})
- Tensione Sistemica della Tela: |Z| = {core_state.systemic_web_tension:.2f} ({core_state.web_tension_status})
- Indice di Contagio Non-Lineare CISS (ECB / OFR): {core_state.ciss.ciss_score * 100:.1f}% ({core_state.ciss.ciss_status})
  * Quota Varianza da Contagio Cross-Asset: {core_state.ciss.contagion_contribution_pct:.1f}%
  * Rischio Idiosincratico Puro: {core_state.ciss.idiosyncratic_risk * 100:.2f}%
  * Stress Quadrante Inflazione & Commodities: {core_state.ciss.quadrant_stress.get('INFLATION', 0.0) * 100:.1f}%
  * Stress Quadrante Crescita & Produzione: {core_state.ciss.quadrant_stress.get('GROWTH', 0.0) * 100:.1f}%
  * Stress Quadrante Lavoro & Occupazione: {core_state.ciss.quadrant_stress.get('LABOR', 0.0) * 100:.1f}%
  * Stress Quadrante Liquidità, Rischio & Tassi: {core_state.ciss.quadrant_stress.get('LIQUIDITY_RISK', 0.0) * 100:.1f}%
- Global Net Liquidity Dashboard (Fed H.4.1 / CrossBorder Capital):
  * Formula Istituzionale: US Net Liquidity = Fed Total Assets - TGA - Reverse Repo (RRP)
  * US Net Liquidity Attuale: ${liq_curr.us_net_liquidity_b:,.0f} B (${liq_curr.us_net_liquidity_t:.2f} Trillion)
  * Fed Total Assets (WALCL): ${liq_curr.fed_total_assets_b:,.0f} B
  * Treasury General Account TGA (WTREGEN): ${liq_curr.tga_balance_b:,.0f} B
  * Overnight Reverse Repo RRP (RRPONTSYD): ${liq_curr.rrp_balance_b:,.0f} B
  * Variazione 4 Settimane (Δ 4W): {liq_curr.us_net_liquidity_delta_4w_b:+,.0f} B ({liq_curr.us_net_liquidity_delta_4w_pct:+.1f}%)
  * Variazione 8 Settimane (Δ 8W): {liq_curr.us_net_liquidity_delta_8w_b:+,.0f} B ({liq_curr.us_net_liquidity_delta_8w_pct:+.1f}%)
  * Impulso di Liquidità: {liq_curr.liquidity_regime_badge} ({liq_curr.macro_interpretation})
  * Correlazioni Forward Empirical Lead-Lag (Trasmissione Variazione Liquidità Netta):
    - Nasdaq 100 (EQ_NDX): 4-Settimane r = {liq_curr.corr_ndx_4w:+.2f} | 8-Settimane r = {liq_curr.corr_ndx_8w:+.2f}
    - S&P 500 (EQ_SPX):    4-Settimane r = {liq_curr.corr_spx_4w:+.2f} | 8-Settimane r = {liq_curr.corr_spx_8w:+.2f}
    - Oro Fisico (COM_GOLD): 4-Settimane r = {liq_curr.corr_gold_4w:+.2f} | 8-Settimane r = {liq_curr.corr_gold_8w:+.2f}
- Bias Banche Centrali: {core_state.central_bank_bias}
- Tasso Reale FED: {core_state.fed_real_rate:+.2f}% | Tasso Reale BCE: {core_state.ecb_real_rate:+.2f}%
- Taylor Rule Gap: {core_state.taylor_rate_gap_bps:+.0f} bps (Posizione: {'Restrittiva (Surplus)' if core_state.taylor_rate_gap_bps >= 0 else 'Accomodante (Deficit / Behind the curve)'})
- Numero Anomalie Critiche (|Z| >= 2.0): {len(core_state.active_critical_alerts)}
- Allarmi Critici:
  * {chr(10).join('  * ' + alt for alt in core_state.active_critical_alerts) if core_state.active_critical_alerts else 'Nessuna rottura sistemica critica rilevata.'}

3. SCENARIO DI ATTERRAGGIO (LANDING SCENARIO)
--------------------------------------------------------------------------------
- Scenario Rilevato: {core_state.landing_scenario.scenario_title} ({core_state.landing_scenario.scenario_badge})
- Confidenza Modello: {core_state.landing_scenario.confidence_pct:.1f}%
- Probabilità No Landing: {core_state.landing_scenario.prob_no_landing:.1f}%
- Probabilità Soft Landing: {core_state.landing_scenario.prob_soft_landing:.1f}%
- Probabilità Hard Landing: {core_state.landing_scenario.prob_hard_landing:.1f}%
- Implicazione di Mercato: {core_state.landing_scenario.market_implication}
- Reazione Banche Centrali: {core_state.landing_scenario.policy_reaction}

4. TEORIA DEI CICLI INTERMARKET (MARTIN PRING)
--------------------------------------------------------------------------------
- Fase Ciclica Attuale: {pring_res_curr.stage_title}
- Descrizione Fase: {pring_res_curr.stage_desc}
- Confidenza: {pring_res_curr.confidence_pct:.1f}%
- Regime Pring: {pring_res_curr.stage_regime}
- Stato Obbligazioni: {'BULL' if pring_res_curr.asset_statuses['BONDS'].is_bullish else 'BEAR'} (Trend: {pring_res_curr.asset_statuses['BONDS'].trend_200d}, Mom: {pring_res_curr.asset_statuses['BONDS'].momentum_60d_pct:+.2f}%)
- Stato Azioni: {'BULL' if pring_res_curr.asset_statuses['STOCKS'].is_bullish else 'BEAR'} (Trend: {pring_res_curr.asset_statuses['STOCKS'].trend_200d}, Mom: {pring_res_curr.asset_statuses['STOCKS'].momentum_60d_pct:+.2f}%)
- Stato Materie Prime: {'BULL' if pring_res_curr.asset_statuses['COMMODITIES'].is_bullish else 'BEAR'} (Trend: {pring_res_curr.asset_statuses['COMMODITIES'].trend_200d}, Mom: {pring_res_curr.asset_statuses['COMMODITIES'].momentum_60d_pct:+.2f}%)
- Asset Favoriti da Pring: {', '.join(pring_res_curr.favored_assets)}
- Settori Azionari Favoriti da Pring: {', '.join(pring_res_curr.favored_sectors)}

5. CURVA DEI RENDIMENTI & REGIMI (STENO RESEARCH FRAMEWORK)
--------------------------------------------------------------------------------
- Regime Curva Rilevato: {yc_state_curr.regime_name} (Rank Ciclo: #{yc_state_curr.rank})
- Delta Tasso Breve (2Y): {yc_state_curr.short_rate_delta_bps:+.1f} bps
- Delta Tasso Lungo (10Y): {yc_state_curr.long_rate_delta_bps:+.1f} bps
- Variazione Slope (Pendenza): {yc_state_curr.slope_delta_bps:+.1f} bps
- Impatto Motore del Credito: {yc_state_curr.impact_credit_engine}
- Diagnosi SLOOS & Fido: {yc_state_curr.credit_analysis.bank_lending_standards}
- Asset Curva Favoriti: {', '.join(yc_state_curr.favored_assets)}
- Asset Curva Sfavoriti: {', '.join(yc_state_curr.unfavored_assets)}

6. ASSET ALLOCATION STRATEGICA & SCENARIO DINAMICO SPIDER
--------------------------------------------------------------------------------
- Modello Selezionato: SPIDER Dynamic (Macro-Adaptive)
- Allocazione Ottimizzata Multi-Fattore:
  * 35% Azioni (Quality FCF + Energy Tilt)
  * 15% Obbligazioni (Short-Duration 1-3Y / T-Bills)
  * 20% Oro Fisico (Hedge Debito & De-dollarizzazione LINK_03)
  * 15% Commodity (Focus Petrolio / Offerta rigida LINK_04)
  * 10% Cash / Liquidità Monetaria (Rendimento privo di rischio al 5%)
  * 5% Crypto (Bitcoin - Debasement hedge asimmetrico e liquidità globale)

7. RISK PARITY & VOLATILITY TARGETING (AQR / BRIDGEWATER / MAN AHL)
--------------------------------------------------------------------------------
- Framework: Equal Risk Contribution (w_i ~ 1/sigma_i) vincolato alla Tensione Sistemica (|Z|)
- Regime Attivo Tensione: {rp_state_curr.tension_badge}
- Tensione Sistemica Rete: |Z| = {rp_state_curr.systemic_tension_z:.2f}
- Fattore Scala Volatility Targeting: {rp_state_curr.vol_scale_factor * 100:.0f}%
- Volatilità Complessiva Portafoglio: {rp_state_curr.portfolio_volatility_pct:.2f}% (Rendimento Atteso: {rp_state_curr.expected_return_pct:.2f}%, Sharpe: {rp_state_curr.sharpe_ratio:.2f})
- Esposizione Beta Ciclici (Azioni, HY, Commodity): {rp_state_curr.cyclical_exposure_pct:.1f}%
- Copertura Difensiva & Cash (T-Bills, 10Y Bond, Oro): {rp_state_curr.defensive_exposure_pct:.1f}%
- Verdetto Esecutivo: {rp_state_curr.executive_summary}
- Pesi Attivi e Contributi al Rischio (Equal Risk Contribution):
{chr(10).join(f"  * {it.asset_name:32s}: Peso {it.tension_adjusted_weight_pct:5.1f}% | Contributo Rischio (RC): {it.risk_contribution_pct:5.1f}% | Vol: {it.volatility_annual_pct:4.1f}% [{it.allocation_status}]" for it in rp_state_curr.items)}

8. DIEBOLD-YILMAZ SPILLOVER NETWORK (CHI CONTAGIA CHI • VAR/GFEVD)
--------------------------------------------------------------------------------
- Framework: Generalized Forecast Error Variance Decomposition (Pesaran-Shin order-invariant)
- Indice di Connessione Totale (TCI): {dy_curr.total_connectedness_index:.1f}% ({dy_curr.tci_regime})
- Quota Varianza da Shock Esterni (Cross-Spillover): {dy_curr.total_connectedness_index:.1f}% | Varianza Idiosincratica: {100.0 - dy_curr.total_connectedness_index:.1f}%
- Orizzonte di Previsione (H): {dy_curr.horizon_steps} Giorni | Ritardi VAR (p): {dy_curr.var_lags}
- Diagnosi Direzionale: {dy_curr.executive_diagnosis}
- Ruoli Sistemici dei Nodi (Trasmettitori Netti vs Ricevitori):
{chr(10).join(f"  * [{p.symbol:8s}] {p.name:32s}: Net Spillover {p.net_spillover_pct:+5.1f}% (TO: {p.to_others_pct:4.1f}% | FROM: {p.from_others_pct:4.1f}%) -> {p.role_badge}" for p in dy_curr.nodes_profile)}

9. SPIDER SI: SUPER INTELLIGENCE (I 10 GIGANTI DELL'INTELLIGENZA ARTIFICIALE)
--------------------------------------------------------------------------------
- Indice Composito SI (Base 1000): {si_curr.composite_index:.1f} (YTD: {si_curr.composite_ytd_pct:+.1f}% | 1 Mese: {si_curr.composite_1m_pct:+.1f}%)
- Capitalizzazione Totale 10 Giganti AI: ${si_curr.total_market_cap_t:.2f} Trilioni
- Spesa in Conto Capitale Totale (Annual AI Capex): ${si_curr.total_annual_capex_b:.1f} Miliardi
- AI Momentum Score Sistemico: {si_curr.ai_momentum_score:.1f} / 100
- Regime Ciclo AI: {si_curr.ai_cycle_regime}
- Top Performer: {si_curr.top_performer.name} ({si_curr.top_performer.ticker}) con {si_curr.top_performer.change_ytd_pct:+.1f}% YTD
- Azioni Monitorate nel Cockpit Fondamentale (10 Titoli):
{chr(10).join(f"  * [{s.ticker:5s}] {s.name:22s}: Cap ${s.market_cap_b:5.0f}B | P/E {s.pe_ratio:4.1f} | Rev YoY {s.rev_growth_yoy_pct:+5.1f}% | Capex ${s.annual_capex_b:4.1f}B | YTD {s.change_ytd_pct:+5.1f}% [{s.subsector}]" for s in sorted(si_curr.stocks, key=lambda x: x.market_cap_b, reverse=True))}

10. DISTRIBUZIONE & DIAGNOSI Z-SCORE DEGLI INDICATORI (30 NODI)
--------------------------------------------------------------------------------
- Metodologia: Standardizzazione statistica rolling 1Y Z = (X_t - μ) / σ
- Soglie Quantitative e Codifica Cromatica Grafica:
  * Regime Estremo (|Z| >= 2.0): Deviazione critica / Shock sistemico [Blu #3B82F6]
  * In Attenzione (1.25 <= |Z| < 2.0): Tensione ciclica / Surriscaldamento [Rosso #EF4444]
  * In Equilibrio (|Z| < 1.25): Range statistico fisiologico [Giallo #F59E0B]
- Sintesi Conteggi Nodi:
  * Nodi Estremi: {n_ext_tot_snap} / 30 ({n_ext_tot_snap / 30 * 100:.1f}%)
  * Nodi in Attenzione: {n_att_tot_snap} / 30 ({n_att_tot_snap / 30 * 100:.1f}%)
  * Nodi Normali: {n_norm_tot_snap} / 30 ({n_norm_tot_snap / 30 * 100:.1f}%)
- Massimo Z-Score Rilevato: {max_node_z.name if max_node_z else 'N/A'} (Z = {max_node_z.z_score:+.2f} {max_node_z.unit if max_node_z else ''})
- Minimo Z-Score Rilevato: {min_node_z.name if min_node_z else 'N/A'} (Z = {min_node_z.z_score:+.2f} {min_node_z.unit if min_node_z else ''})

- Classifica Completa Ordinata per Z-Score (Asse Grafico Standardizzato):
{z_score_table_formatted}

================================================================================
Fine Report Snapshot SPIDER {APP_VERSION}.
================================================================================
"""

nodes_num_data = []
for n_id, ns in node_states.items():
    nd = NODES.get(n_id)
    nodes_num_data.append({
        "Versione_SPIDER": APP_VERSION,
        "Sezione": "NODO_TELA",
        "ID": n_id,
        "Nome": ns.name,
        "Categoria": nd.category if nd else "",
        "Anello": ns.ring,
        "Quadrante": ns.quadrant,
        "Valore_Attuale": ns.current_value,
        "Valore_Precedente": ns.prev_value,
        "Unita": ns.unit,
        "Delta_Periodo": ns.delta_period,
        "Delta_Periodo_Pct": ns.delta_period_pct,
        "Delta_YoY_1A": ns.delta_yoy,
        "Z_Score": round(ns.z_score, 4),
        "Stato": ns.status,
        "Ticker_Ufficiale": ns.official_ticker,
        "Fonte": ns.source,
        "Ultimo_Aggiornamento": ns.last_timestamp,
    })
df_file_b_nodes = pd.DataFrame(nodes_num_data)

links_num_data = []
for l in link_states:
    links_num_data.append({
        "Versione_SPIDER": APP_VERSION,
        "ID_Filo": l.id,
        "Nodo_A": l.node_a_id,
        "Nome_A": l.node_a_name,
        "Valore_A": l.node_a_val,
        "Nodo_B": l.node_b_id,
        "Nome_B": l.node_b_name,
        "Valore_B": l.node_b_val,
        "Tipo": "Filo Maestro" if l.is_master else "Spoke",
        "Correlazione_Rolling": round(l.active_corr, 4),
        "Correlazione_Teorica": l.theoretical_corr,
        "Tensione_ZScore": round(l.z_score_tension, 4),
        "Stato_Tensione": l.status,
        "Meccanismo": l.mechanism,
        "Diagnosi_Anomalia": l.alert_description,
    })
df_file_b_links = pd.DataFrame(links_num_data)

# Tabella dedicata Distribuzione Z-Score 30 Nodi ordinata identica al grafico
zscore_table_data = []
for rank_i, ns in enumerate(sorted_nodes_z, 1):
    abs_z = abs(ns.z_score)
    if abs_z >= 2.0:
        cls_name = "Estremo"
        col_hex = "#3B82F6"
    elif abs_z >= 1.25:
        cls_name = "Attenzione"
        col_hex = "#EF4444"
    else:
        cls_name = "Normale"
        col_hex = "#F59E0B"
    nd = NODES.get(ns.id)
    zscore_table_data.append({
        "Versione_SPIDER": APP_VERSION,
        "Rank_Grafico": rank_i,
        "ID_Nodo": ns.id,
        "Nome_Indicatore": ns.name,
        "Z_Score_Standardizzato": round(ns.z_score, 4),
        "Classe_ZScore": cls_name,
        "Colore_Grafico_Hex": col_hex,
        "Valore_Attuale": ns.current_value,
        "Valore_Precedente": ns.prev_value,
        "Unita": ns.unit,
        "Delta_Periodo": ns.delta_period,
        "Delta_Periodo_Pct": ns.delta_period_pct,
        "Delta_YoY_1A": ns.delta_yoy,
        "Quadrante": ns.quadrant,
        "Anello": ns.ring,
        "Categoria": nd.category if nd else "",
        "Ticker_Ufficiale": ns.official_ticker,
        "Fonte": ns.source,
        "Ultimo_Aggiornamento": ns.last_timestamp,
    })
df_file_b_zscore = pd.DataFrame(zscore_table_data)

# Tabella dedicata CISS Quadranti & Correlazioni
ciss_quad_data = []
for qk, q_stress in core_state.ciss.quadrant_stress.items():
    q_name = QUADRANTS.get(qk, {}).get("name", qk)
    w_val = core_state.ciss.weights.get(qk, 0.25)
    ciss_quad_data.append({
        "Versione_SPIDER": APP_VERSION,
        "ID_Quadrante": qk,
        "Nome_Quadrante": q_name,
        "Peso_Algoritmico": w_val,
        "Stress_CDF_SubIndex": q_stress,
        "Contributo_Ponderato": round(w_val * q_stress, 4),
        "Stato_Settoriale": "Critico (>50%)" if q_stress >= 0.50 else ("Attenzione (25-50%)" if q_stress >= 0.25 else "Fisiologico (<25%)")
    })
df_file_b_ciss_quad = pd.DataFrame(ciss_quad_data)

# Tabella matrice di correlazione tra quadranti C_t
ciss_corr_data = []
for q1, row_corr in core_state.ciss.corr_matrix.items():
    row_dict = {"Versione_SPIDER": APP_VERSION, "Quadrante_Asse_Y": q1}
    for q2, val in row_corr.items():
        row_dict[f"Corr_{q2}"] = val
    ciss_corr_data.append(row_dict)
df_file_b_ciss_corr = pd.DataFrame(ciss_corr_data)

# Tabella Scenari Storici CCAR / Aladdin
crisis_summary_data = []
for cid, c_obj in HISTORICAL_CRISIS_LIBRARY.items():
    crisis_summary_data.append({
        "Versione_SPIDER": APP_VERSION,
        "ID_Scenario": cid,
        "Nome_Crisi": c_obj.name,
        "Periodo_Storico": c_obj.period,
        "Durata_Giorni": c_obj.duration_days,
        "Catalizzatore": c_obj.catalyst,
        "Max_Drawdown_60_40": c_obj.historical_max_drawdown_60_40,
        "Max_Drawdown_All_Weather": c_obj.historical_max_drawdown_all_weather,
        "Max_Drawdown_SPIDER": c_obj.historical_max_drawdown_spider,
        "Picco_VIX": c_obj.vix_peak,
        "Picco_Spread_HY": c_obj.hy_spread_peak,
        "Fili_Chiave_Rotti": ", ".join(c_obj.key_ruptured_links),
    })
df_file_b_crises = pd.DataFrame(crisis_summary_data)

excel_b_buffer = io.BytesIO()
with pd.ExcelWriter(excel_b_buffer, engine="openpyxl") as writer:
    df_file_b_nodes.to_excel(writer, index=False, sheet_name="Dati_Numerici_30_Nodi")
    df_file_b_links.to_excel(writer, index=False, sheet_name="Correlazioni_10_Fili")
    df_file_b_zscore.to_excel(writer, index=False, sheet_name="Distribuzione_ZScore_30_Nodi")
    df_file_b_ciss_quad.to_excel(writer, index=False, sheet_name="Stress_Quadranti_CISS")
    df_file_b_ciss_corr.to_excel(writer, index=False, sheet_name="Matrice_CrossCorr_CISS")
    df_file_b_crises.to_excel(writer, index=False, sheet_name="Scenari_Crisi_CCAR")
    if not liq_curr.historical_timeseries.empty:
        df_liq_export = liq_curr.historical_timeseries.copy().reset_index()
        df_liq_export.insert(0, "Versione_SPIDER", APP_VERSION)
        df_liq_export.to_excel(writer, index=False, sheet_name="Liquidita_Netta_Fed")
    # Tabella Risk Parity Execution
    rp_export_data = []
    for it in rp_state_curr.items:
        rp_export_data.append({
            "Versione_SPIDER": APP_VERSION,
            "Asset_ID": it.asset_id,
            "Asset_Name": it.asset_name,
            "Asset_Class": it.asset_class,
            "Volatilta_Annua_Pct": it.volatility_annual_pct,
            "Peso_Benchmark_60_40": it.unconstrained_weight_pct,
            "Peso_Risk_Parity_Base": it.raw_risk_parity_weight_pct,
            "Peso_Attivo_Tension_Adjusted": it.tension_adjusted_weight_pct,
            "Contributo_Rischio_RC_Pct": it.risk_contribution_pct,
            "Fattore_Leva_Scala": it.leverage_multiplier,
            "Stato_Allocativo": it.allocation_status,
        })
    pd.DataFrame(rp_export_data).to_excel(writer, index=False, sheet_name="Risk_Parity_Allocation")
    # Tabella Profili Spillover Diebold-Yilmaz
    dy_export_data = []
    for p in dy_curr.nodes_profile:
        dy_export_data.append({
            "Versione_SPIDER": APP_VERSION,
            "Symbol": p.symbol,
            "Nome_Nodo": p.name,
            "Varianza_Propria_Pct": p.own_variance_pct,
            "Trasmesso_Ad_Altri_TO_Pct": p.to_others_pct,
            "Ricevuto_Da_Altri_FROM_Pct": p.from_others_pct,
            "Spillover_Netto_Pct": p.net_spillover_pct,
            "Ruolo_Sistemico": p.role_badge,
            "Is_Transmitter": p.is_net_transmitter,
        })
    pd.DataFrame(dy_export_data).to_excel(writer, index=False, sheet_name="Profili_Spillover_Diebold")
    if not dy_curr.spillover_matrix.empty:
        df_dy_mat = dy_curr.spillover_matrix.copy().reset_index()
        df_dy_mat.rename(columns={"index": "Source_Node (FROM)"}, inplace=True)
        df_dy_mat.to_excel(writer, index=False, sheet_name="Matrice_Spillover_Diebold")
    # Tabella SPIDER SI: 10 Giganti dell'Intelligenza Artificiale
    si_export_data = []
    for s in si_curr.stocks:
        si_export_data.append({
            "Versione_SPIDER": APP_VERSION,
            "Ticker": s.ticker,
            "Nome_Azienda": s.name,
            "Sotto_Settore": s.subsector,
            "Ruolo_Supply_Chain": s.supply_chain_role,
            "Market_Cap_USD_B": s.market_cap_b,
            "Prezzo_Attuale": s.price,
            "Performance_YTD_Pct": s.change_ytd_pct,
            "Performance_1M_Pct": s.change_1m_pct,
            "Forward_PE": s.pe_ratio,
            "PEG_Ratio": s.peg_ratio,
            "Fatturato_Growth_YoY_Pct": s.rev_growth_yoy_pct,
            "Margine_Operativo_Pct": s.operating_margin_pct,
            "Annual_Capex_USD_B": s.annual_capex_b,
            "Volatilita_Annua_Pct": s.volatility_annual_pct,
            "ZScore_Fondamentale": s.z_score,
            "Descrizione_Business": s.description,
        })
    pd.DataFrame(si_export_data).to_excel(writer, index=False, sheet_name="SPIDER_SI_10_Azioni_AI")

    pd.DataFrame([{
        "Versione_SPIDER": APP_VERSION,
        "Data_Snapshot": now_iso,
        "Macro_Regime": core_state.macro_regime,
        "Tensione_Sistemica_Z": core_state.systemic_web_tension,
        "SI_Composite_Index": si_curr.composite_index,
        "SI_Composite_YTD_Pct": si_curr.composite_ytd_pct,
        "SI_Composite_1M_Pct": si_curr.composite_1m_pct,
        "SI_Total_Market_Cap_T": si_curr.total_market_cap_t,
        "SI_Total_Annual_Capex_B": si_curr.total_annual_capex_b,
        "SI_AI_Momentum_Score": si_curr.ai_momentum_score,
        "SI_Cycle_Regime": si_curr.ai_cycle_regime,
        "SI_Top_Performer_Ticker": si_curr.top_performer.ticker,
        "SI_Top_Performer_YTD": si_curr.top_performer.change_ytd_pct,
        "Risk_Parity_Regime": rp_state_curr.tension_badge,
        "Risk_Parity_Vol_Target_Pct": rp_state_curr.portfolio_volatility_pct,
        "Risk_Parity_Vol_Scale_Factor": rp_state_curr.vol_scale_factor,
        "Risk_Parity_Cyclical_Exposure": rp_state_curr.cyclical_exposure_pct,
        "Risk_Parity_Defensive_Exposure": rp_state_curr.defensive_exposure_pct,
        "Risk_Parity_Sharpe": rp_state_curr.sharpe_ratio,
        "Diebold_TCI_Pct": dy_curr.total_connectedness_index,
        "Diebold_TCI_Regime": dy_curr.tci_regime,
        "Diebold_Top_Transmitter": dy_curr.top_transmitters[0].symbol if dy_curr.top_transmitters else "",
        "Diebold_Top_Transmitter_Net_Pct": dy_curr.top_transmitters[0].net_spillover_pct if dy_curr.top_transmitters else 0.0,
        "Diebold_Top_Receiver": dy_curr.top_receivers[0].symbol if dy_curr.top_receivers else "",
        "Diebold_Top_Receiver_Net_Pct": dy_curr.top_receivers[0].net_spillover_pct if dy_curr.top_receivers else 0.0,
        "CISS_Score_BCE": core_state.ciss.ciss_score,
        "CISS_Stato": core_state.ciss.ciss_status,
        "CISS_Contagion_Contribution_Pct": core_state.ciss.contagion_contribution_pct,
        "CISS_Idiosyncratic_Risk": core_state.ciss.idiosyncratic_risk,
        "US_Net_Liquidity_B": liq_curr.us_net_liquidity_b,
        "Fed_Total_Assets_B": liq_curr.fed_total_assets_b,
        "TGA_Balance_B": liq_curr.tga_balance_b,
        "RRP_Balance_B": liq_curr.rrp_balance_b,
        "Net_Liquidity_Delta_4W_B": liq_curr.us_net_liquidity_delta_4w_b,
        "Net_Liquidity_Delta_4W_Pct": liq_curr.us_net_liquidity_delta_4w_pct,
        "Net_Liquidity_Delta_8W_B": liq_curr.us_net_liquidity_delta_8w_b,
        "Net_Liquidity_Delta_8W_Pct": liq_curr.us_net_liquidity_delta_8w_pct,
        "Liquidity_Regime": liq_curr.liquidity_regime_badge,
        "Corr_NDX_4W": liq_curr.corr_ndx_4w,
        "Corr_NDX_8W": liq_curr.corr_ndx_8w,
        "Corr_SPX_4W": liq_curr.corr_spx_4w,
        "Corr_SPX_8W": liq_curr.corr_spx_8w,
        "Corr_Gold_4W": liq_curr.corr_gold_4w,
        "Corr_Gold_8W": liq_curr.corr_gold_8w,
        "Landing_Scenario": core_state.landing_scenario.scenario_title,
        "Prob_No_Landing": core_state.landing_scenario.prob_no_landing,
        "Pring_Stage": pring_res_curr.stage_title,
        "Yield_Curve_Regime": yc_state_curr.regime_name,
        "Taylor_Gap_Bps": core_state.taylor_rate_gap_bps,
        "Fed_Real_Rate": core_state.fed_real_rate,
        "ECB_Real_Rate": core_state.ecb_real_rate,
        "ZScore_Nodi_Estremi_Count": n_ext_tot_snap,
        "ZScore_Nodi_Attenzione_Count": n_att_tot_snap,
        "ZScore_Nodi_Normali_Count": n_norm_tot_snap,
        "ZScore_Max_Nodo": max_node_z.name if max_node_z else "",
        "ZScore_Max_Val": round(max_node_z.z_score, 4) if max_node_z else 0.0,
        "ZScore_Min_Nodo": min_node_z.name if min_node_z else "",
        "ZScore_Min_Val": round(min_node_z.z_score, 4) if min_node_z else 0.0,
    }]).to_excel(writer, index=False, sheet_name="Sintesi_SPIDER_CORE")
file_b_excel_bytes = excel_b_buffer.getvalue()


# ================= DIALOG: SPIDER CORE FOCUS =================
@st.dialog(f"🕷️ SPIDER_CORE {APP_VERSION}: Centro di Controllo Macro-Finanziario & Tensione Sistemica", width="large")
def show_spider_core_modal():
    t_val = core_state.systemic_web_tension
    t_color = "#EF4444" if t_val >= 2.0 else ("#F59E0B" if t_val >= 1.25 else "#10B981")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(168, 85, 247, 0.12)); border: 2px solid #38BDF8; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 11px; text-transform: uppercase; color: #38BDF8; font-weight: 800; letter-spacing: 0.8px;">SINTESI SISTEMICA CENTRALE (RAGGIO 0) • {APP_VERSION}</span>
                <div style="font-size: 24px; font-weight: 800; color: #F8FAFC; margin-top: 2px;">SPIDER_CORE {APP_VERSION} • Centro di Controllo Istituzionale</div>
                <div style="font-size: 13px; color: #CBD5E1; margin-top: 4px;">Topologia causale della rete: riceve i vettori di impulso monetario, fondamentale e intermarket sintetizzandoli in un unico tensore di stato.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.9); border: 1.5px solid {t_color}; border-radius: 10px; padding: 12px 18px; text-align: center; min-width: 170px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Tensione Globale Tela</div>
                <div style="font-size: 28px; font-weight: 800; color: {t_color}; font-family: monospace;">|Z| = {t_val:.2f}</div>
                <div style="font-size: 11px; color: {t_color}; font-weight: 700;">{core_state.web_tension_status.split('(')[0].strip()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # BARRA SALVATAGGIO SITUAZIONE GLOBALE DENTRO IL MODALE SPIDER CORE
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.9); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;">
        <div style="font-size: 13px; font-weight: 800; color: #38BDF8; margin-bottom: 4px;">
            💾 SALVATAGGIO SITUAZIONE GLOBALE DA SPIDER_CORE
        </div>
        <div style="font-size: 11.5px; color: #94A3B8; margin-bottom: 10px;">
            Scarica l'intera fotografia operativa di SPIDER sincronizzata con tutti i parametri di calcolo e i dati certificati:
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_save_core1, col_save_core2 = st.columns(2)
    with col_save_core1:
        st.download_button(
            "📄 Scarica File A (.txt) • Parametri Completi",
            data=file_a_text,
            file_name=f"SPIDER_{APP_VERSION}_Snapshot_Parametri_{datetime.date.today()}.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
            key="dl_btn_core_file_a",
            help="Scarica il file di testo con tutti i parametri salvati di tutte le sezioni."
        )
    with col_save_core2:
        st.download_button(
            "📊 Scarica File B (.xlsx) • Tabella Dati Numerici",
            data=file_b_excel_bytes,
            file_name=f"SPIDER_{APP_VERSION}_Dati_Numerici_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_btn_core_file_b",
            help="Scarica la cartella Excel completa di tutti i 30 nodi e 10 fili maestri."
        )

    st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
    with c_sc1:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Regime Macro Attivo</div>
            <div style="font-size: 16px; font-weight: 800; color: #38BDF8; margin-top: 4px;">{core_state.macro_regime.split('/')[0].strip()}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Growth: {core_state.growth_score:+.2f} | Inf: {core_state.inflation_score:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_sc2:
        ls_s = core_state.landing_scenario
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Scenario Ciclico Landing</div>
            <div style="font-size: 16px; font-weight: 800; color: {ls_s.color_hex}; margin-top: 4px;">{ls_s.scenario_badge}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Confidenza: {ls_s.confidence_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c_sc3:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Banche Centrali & Policy</div>
            <div style="font-size: 16px; font-weight: 800; color: #A78BFA; margin-top: 4px;">{core_state.central_bank_bias.split('(')[0].strip()}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Taylor Gap: {core_state.taylor_rate_gap_bps:+.0f} bps</div>
        </div>
        """, unsafe_allow_html=True)
    with c_sc4:
        yc_s = yield_curve_engine.get_curve_regime_state()
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Curva dei Rendimenti</div>
            <div style="font-size: 16px; font-weight: 800; color: {yc_s.color_hex}; margin-top: 4px;">{yc_s.regime_name}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">Rank #{yc_s.rank} Ciclo Economico</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Termometro di Rischio Sistemico & Contagio Multi-Mercato (ECB CISS):")
    st.plotly_chart(build_ciss_gauge_and_decomposition_figure(core_state.ciss, dark_mode=True), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Matrice Anomalie Critiche & Fili Maestri in Tensione:")
    crit_links = [l for l in link_states if l.is_master]
    df_core_links = pd.DataFrame([
        {
            "Filo": l.id,
            "Relazione": f"{l.node_a_name} ↔ {l.node_b_name}",
            "Corr. Rolling": f"{l.active_corr:+.2f}",
            "Corr. Teorica": f"{l.theoretical_corr:+d}",
            "Tensione (|Z|)": f"{l.z_score_tension:.2f}",
            "Stato": l.status,
            "Diagnosi Macro": l.alert_description,
        }
        for l in crit_links
    ])
    st.dataframe(df_core_links, use_container_width=True, hide_index=True)

c_glob_box1, c_glob_box2 = st.columns([1.8, 1.2])
with c_glob_box1:
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid #38BDF8; border-radius: 10px; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; gap: 14px;">
        <div>
            <div style="font-size: 13.5px; font-weight: 800; color: #F8FAFC;">
                💾 <b>SALVATAGGIO SITUAZIONE GLOBALE</b>
            </div>
            <div style="font-size: 11.5px; color: #94A3B8; margin-top: 2px;">
                Estrai istantaneamente l'intero stato operativo della piattaforma in due formati sincronizzati:
                <br>• <b>File A:</b> Report di testo (.txt) con tutti i parametri salvati di ogni pagina e configurazione iniziale.
                <br>• <b>File B:</b> Tabella completa di tutti i dati numerici (.csv / .xlsx).
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_glob_box2:
    st.markdown("<div style='display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px;'>", unsafe_allow_html=True)
    c_btn_a, c_btn_b, c_btn_core = st.columns([1.1, 1.2, 1.1])
    with c_btn_a:
        st.download_button(
            "📄 File A (.txt)\nParametri Completi",
            data=file_a_text,
            file_name=f"SPIDER_{APP_VERSION}_Snapshot_Parametri_{datetime.date.today()}.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
            help="Scarica il file di testo con tutti i parametri delle 8 pagine e la configurazione di partenza."
        )
    with c_btn_b:
        st.download_button(
            "📊 File B (.xlsx)\nTabella Dati Numerici",
            data=file_b_excel_bytes,
            file_name=f"SPIDER_{APP_VERSION}_Dati_Numerici_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Scarica la cartella Excel completa con i dati numerici certificati di tutti i 30 Nodi e 10 Fili."
        )
    with c_btn_core:
        if st.button("🕷️ SPIDER CORE\nVista Centro Controllo", use_container_width=True, type="secondary", help="Apri il pannello di dettaglio visivo dello SPIDER_CORE"):
            show_spider_core_modal()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


tabs = st.tabs([
    "🕸️ Tela di Ragno (Spider Web Graph)",
    "🎯 Centro di Controllo (Macro Matrix)",
    "🔄 Teoria dei Cicli di Martin Pring",
    "📈 Curva dei Rendimenti & Regimi (Steno Framework)",
    "📋 Supervisione Tabellare & Export",
    "🔬 Focus Nodo & Relazioni",
    "⚡ Simulatore Stress What-If",
    "💼 Struttura Portafoglio (Asset Allocation)",
    "🤖 SPIDER SI (Super Intelligence 10)",
])


# TAB 1
with tabs[0]:
    st.markdown("#### 🕸️ Visualizzazione Radiale Concentrica a Tela di Ragno")
    st.caption("I cerchi concentrici separano Politica Monetaria (Anello 1), Fondamentali Macro (Anello 2) e Asset Intermarket (Anello 3). Lo spessore e il colore dei fili indicano la forza e la tensione della relazione.")

    col_ctl1, col_ctl2, col_ctl3, col_ctl4 = st.columns([1.5, 1.5, 1.0, 1.0])
    with col_ctl1:
        view_dimension = st.radio(
            "📐 Prospettiva della Tela:",
            ["🕸️ Vista 2D Standard", "🌐 Vista 3D Spaziale (Elevazione Dinamica)"],
            index=0,
            horizontal=True,
            key="spider_view_dimension"
        )
        if active_isolated_id:
            st.info(f"🔍 **Focus Attivo:** Nodo `{active_isolated_id}` ({node_states[active_isolated_id].name if active_isolated_id in node_states else ''}) elevato in 3D con fasci di connessione isolati.")
        else:
            st.caption("💡 Seleziona un nodo per vederlo **sollevarsi in 3D** sopra la tela radiale.")
    with col_ctl2:
        c_mode1, c_mode2 = st.columns(2)
        with c_mode1:
            toggle_labels = st.checkbox("Mostra Ticker", value=True)
            enable_scroll_zoom = st.checkbox("🔍 Zoom Rotella", value=True, help="Scorri la rotella del mouse per zoomare fluidamente dentro e fuori dalla tela")
        with c_mode2:
            if "3D" not in view_dimension:
                sel_tool = st.selectbox("Strumento Mouse:", ["Zoom & Pan Libero", "Lazo Libero (Lasso)", "Rettangolo (Box)"], index=0, help="Scegli la funzione del mouse sulla tela")
                if "Zoom" in sel_tool:
                    active_dragmode = "pan"
                elif "Lazo" in sel_tool:
                    active_dragmode = "lasso"
                else:
                    active_dragmode = "select"
            else:
                st.caption("🎮 **Controlli 3D:** Trascina per ruotare, tasto destro/rotella per zoom e pan orbitale.")
    with col_ctl3:
        if active_isolated_id and st.button("Mostra Tutta la Tela", use_container_width=True):
            reset_node_selection()
            st.rerun()

    # Modal Dialog definition for Spider Web User Manual & Guide
    @st.dialog(f"📖 Manuale Istituzionale SPIDER {APP_VERSION}: Guida alla Struttura & Lettura della Tela", width="large")
    def show_spider_info_modal():
        st.markdown(f"### 🕸️ Architettura & Manuale di Utilizzo della Tela di Ragno SPIDER {APP_VERSION}")
        st.caption("Guida operativa istituzionale alla comprensione dei 3 Anelli Concentrici, dei 4 Quadranti Macro, della Tensione dei Fili e della Sintesi Sistemica.")

        st.markdown("""
        ---
        #### 1. 🎯 I Tre Cerchi Concentrici (Topologia Causa $\\rightarrow$ Effetto)
        La rete radiale organizza le 30 variabili monitorate lungo una catena di trasmissione logica:
        * **Centro (`SPIDER_CORE` - Raggio 0):** Il nucleo di controllo. Misura la tensione aggregata dell'intera rete ($|Z|$ sistemico), il regime macroeconomico e l'orientamento delle banche centrali (*Hawkish / Dovish / Neutral*).
        * **Anello 1 (Interno - Raggio 1.0): Politica Monetaria, Tassi & Rendimenti**  
          *Nodi:* `RATE_FED`, `RATE_ECB`, `YIELD_US02`, `YIELD_US10`, `YIELD_REAL10`, `CURVE_10_2`, `YIELD_DE10`, `SPREAD_BTP`.  
          *Ruolo:* Punto di origine del costo del capitale privo di rischio e delle aspettative sui tassi di interesse guida a 1-2 anni.
        * **Anello 2 (Intermedio - Raggio 2.0): Macroeconomia Reale (Fondamentali)**  
          *Nodi:* `INF_USPCE`, `INF_USCPI`, `INF_EXP5Y`, `INF_EZHICP`, `GRO_USGDP`, `GRO_INDPRO`, `GRO_RETAIL`, `LAB_UNRATE`, `LAB_NFP`, `LAB_CLAIMS`.  
          *Ruolo:* Misura la risposta dell'economia reale (inflazione, consumi, PIL e dinamiche occupazionali) agli impulsi monetari, con il fisiologico ritardo temporale (*transmission lag* di 6-18 mesi).
        * **Anello 3 (Esterno - Raggio 3.0): Indici Azionari, Credito & Asset Intermarket**  
          *Nodi:* Borse USA ed Europee (`EQ_SPX`, `EQ_NDX`, `EQ_RUT`, `EQ_DAX`, `EQ_MIB`), Commodities (`COM_OIL`, `COM_GOLD`, `COM_COPPER`), Valute (`FX_DXY`, `FX_EURUSD`), Credito (`CRED_HY`) e Volatilità (`RISK_VIX`).  
          *Ruolo:* Mercati di contrattazione liquida: dove si formano i prezzi finali, si riflettono le aspettative degli investitori e si prezzano i premi per il rischio.

        ---
        #### 2. 🧭 I 4 Quadranti Macroeconomici Angolari
        La mappa è orientata a 360° per isolare i settori macro-finanziari:
        * **🟡 INFLAZIONE & COMMODITIES (Angolo 30° - 90°):** Prezzi al consumo (`CPI`, `PCE`, `HICP`), aspettative break-even a 5 anni e materie prime energetiche/preziose (`Petrolio WTI`, `Oro`).
        * **🟢 CRESCITA & PRODUZIONE (Angolo 90° - 160°):** PIL reale, produzione industriale, vendite al dettaglio, rame (*Dr. Copper*) e curva di espansione 10Y-2Y.
        * **🔵 LAVORO & OCCUPAZIONE (Angolo 190° - 260°):** Tasso di disoccupazione U3, buste paga Non-Farm Payrolls e richieste settimanali di sussidi di disoccupazione.
        * **🟣 LIQUIDITÀ & RISCHIO (Angolo 280° - 350°):** Tassi ufficiali, curva dei titoli di stato USA e UE, differenziale BTP/Bund, Dollaro USA (`DXY`) e spread High Yield.

        ---
        #### 3. 🎨 Codifica Cromatica dei Fili (Links) & Segnali di Tensione
        Ogni filo tra due nodi esprime il grado di coerenza empirica rispetto alla teoria macroeconomica classica:
        """)

        st.markdown("""
        | Colore del Filo | Relazione Empirica | Significato & Diagnosi Quantitativa |
        | :--- | :--- | :--- |
        | **🟢 Verde Smeraldo** | **Positiva Diretta Normale** | I due nodi si muovono concordi come previsto dalla teoria (es. `COM_OIL` $\\leftrightarrow$ `INF_EXP5Y`: il rialzo del greggio trasmette fisiologicamente inflazione forward). |
        | **🔴 Rosso Corallo** | **Inversa Normale** | I due nodi divergono fisiologicamente (es. `YIELD_US02` $\\leftrightarrow$ `FX_EURUSD`: tassi USA a breve più alti attraggono capitali e deprimono EUR/USD). |
        | **🟡 Giallo Ambra (Tratteggio)** | **Tensione Moderata ($1,25 \\le \|Z\| < 2,0$)** | La correlazione storica si sta indebolendo o c'è attrito ciclico tra i fondamentali e i prezzi di mercato. |
        | **🟣 Viola Neon Pulsante** | **Rottura / Divergenza Critica ($\|Z\| \\ge 2,0$)** | **Segnale di Allarme:** Rottura anomala della correlazione storica (es. `DXY` forte e `S&P 500` ai massimi simultanei, o azionario euforico con tassi reali restrittivi). |
        """)

        st.markdown("""
        * **Spessore della Linea:** Proporzionale all'intensità del legame rolling ($|r|$). Un filo spesso indica una catena di trasmissione dominante e sincronizzata.
        * **Badge Numerico sul Filo:** Riporta il coefficiente di correlazione rolling attivo calcolato sulla finestra temporale impostata (30D, 90D, 1Y, 5Y).

        ---
        #### 4. 🔘 Dimensione & Colore dei Nodi (Deviazione Z-Score)
        * **Dimensione del cerchio:** Più il pallino è grande, maggiore è la deviazione della serie storica dalla sua media ($|Z\text{-score}|$).
        * **Bordo Bianco Acceso / Viola Neon:** Nodo in territorio statistico estremo ($|Z| \ge 2,0$).
        * **Colore Centrale `SPIDER_CORE`:**
          * 🟢 **Verde:** Tensione Sistemica Bassa ($< 1,25$) $\rightarrow$ Mercato in perfetto equilibrio armonico.
          * 🟡 **Arancione:** Tensione Sistemica Moderata ($1,25 - 2,0$) $\rightarrow$ Frizioni tra tassi e valutazioni.
          * 🔴 **Rosso:** Tensione Sistemica Critica ($> 2,0$) $\rightarrow$ Rischio elevato di violento repricing o anomalia macroeconomica.

        ---
        #### 5. 🕹️ Modalità di Interazione Focus & Scheda Esterna
        * **Click su qualsiasi Nodo:** Isola istantaneamente il nodo selezionato, **abbassando l'intensità di tutti gli altri nodi** e mettendo in piena luce solo i suoi fili attivi.
        * **Scheda Esterna al Grafico:** Sotto la tela compare la scheda istituzionale completa con Valore Attuale, $\Delta$ Periodo, YoY, Z-Score dinamico, Fonte ufficiale, Ruolo macroeconomico e l'elenco delle connessioni dirette, **senza mai coprire il grafico o i nodi**.
        """)

        st.markdown("---")
        st.markdown("#### 6. 📚 Enciclopedia Analitica dei 30 Nodi Istituzionali SPIDER")
        st.caption("Scheda tecnica strutturata per ciascuno dei 30 nodi: **Definizione**, **Cosa misura esattamente**, **Interpretazione**, **Perché è rilevante**, **Fonti e ticker**.")

        ring_tabs = st.tabs([
            "🏛️ Anello 1: Tassi & Rendimenti (8)",
            "🏭 Anello 2: Macroeconomia Reale (10)",
            "📊 Anello 3: Asset Intermarket (12)",
            "🌐 Tutti i 30 Nodi"
        ])

        with ring_tabs[0]:
            st.caption("Origine dell'impulso monetario e costo del capitale privo di rischio.")
            for nid, entry in NODE_ENCYCLOPEDIA.items():
                if entry.ring == 1:
                    with st.expander(f"🔹 {entry.name} (`{entry.id}`) • {QUADRANTS[entry.quadrant]['name']}", expanded=False):
                        st.markdown(
                            f"- **Definizione:** {entry.definizione}\n\n"
                            f"- **Cosa misura esattamente:** {entry.cosa_misura}\n\n"
                            f"- **Interpretazione:** {entry.interpretazione}\n\n"
                            f"- **Perché è rilevante:** {entry.perche_rilevante}\n\n"
                            f"- **Fonti e ticker:** `{entry.fonti_ticker}`"
                        )

        with ring_tabs[1]:
            st.caption("Risposta differita dei fondamentali macroeconomici (inflazione, lavoro, consumi, PIL).")
            for nid, entry in NODE_ENCYCLOPEDIA.items():
                if entry.ring == 2:
                    with st.expander(f"🔹 {entry.name} (`{entry.id}`) • {QUADRANTS[entry.quadrant]['name']}", expanded=False):
                        st.markdown(
                            f"- **Definizione:** {entry.definizione}\n\n"
                            f"- **Cosa misura esattamente:** {entry.cosa_misura}\n\n"
                            f"- **Interpretazione:** {entry.interpretazione}\n\n"
                            f"- **Perché è rilevante:** {entry.perche_rilevante}\n\n"
                            f"- **Fonti e ticker:** `{entry.fonti_ticker}`"
                        )

        with ring_tabs[2]:
            st.caption("Mercati finanziari a pronti, premi al rischio azionari/creditizi e materie prime.")
            for nid, entry in NODE_ENCYCLOPEDIA.items():
                if entry.ring == 3:
                    with st.expander(f"🔹 {entry.name} (`{entry.id}`) • {QUADRANTS[entry.quadrant]['name']}", expanded=False):
                        st.markdown(
                            f"- **Definizione:** {entry.definizione}\n\n"
                            f"- **Cosa misura esattamente:** {entry.cosa_misura}\n\n"
                            f"- **Interpretazione:** {entry.interpretazione}\n\n"
                            f"- **Perché è rilevante:** {entry.perche_rilevante}\n\n"
                            f"- **Fonti e ticker:** `{entry.fonti_ticker}`"
                        )

        with ring_tabs[3]:
            for nid, entry in NODE_ENCYCLOPEDIA.items():
                with st.expander(f"🔹 [{entry.id}] {entry.name} • {RINGS[entry.ring]['name']} • {QUADRANTS[entry.quadrant]['name']}", expanded=False):
                    st.markdown(
                        f"- **Definizione:** {entry.definizione}\n\n"
                        f"- **Cosa misura esattamente:** {entry.cosa_misura}\n\n"
                        f"- **Interpretazione:** {entry.interpretazione}\n\n"
                        f"- **Perché è rilevante:** {entry.perche_rilevante}\n\n"
                        f"- **Fonti e ticker:** `{entry.fonti_ticker}`"
                    )

    with col_ctl4:
        if st.button("ℹ️ Information (Manuale)", use_container_width=True, type="secondary"):
            show_spider_info_modal()

    is_3d = ("3D" in view_dimension)

    if is_3d:
        fig_web = build_spider_web_3d_figure(
            node_states=node_states,
            link_states=link_states,
            core_state=core_state,
            selected_node_id=active_isolated_id,
            filter_quadrant=st.session_state["filter_quadrant"],
            filter_ring=st.session_state["filter_ring"],
            show_labels=toggle_labels,
            dark_mode=True,
            elevation_boost=1.4,
        )
    else:
        fig_web = build_spider_web_figure(
            node_states=node_states,
            link_states=link_states,
            core_state=core_state,
            selected_node_id=active_isolated_id,
            filter_quadrant=st.session_state["filter_quadrant"],
            filter_ring=st.session_state["filter_ring"],
            show_labels=toggle_labels,
            dark_mode=True,
        )
        if "active_dragmode" in locals():
            fig_web.update_layout(dragmode=active_dragmode)
    
    chart_event = st.plotly_chart(
        fig_web,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["points", "box", "lasso"] if not is_3d else ["points"],
        config={
            "scrollZoom": enable_scroll_zoom,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"] if not is_3d else [],
        },
        key=f"spider_web_chart_selection_{'3d' if is_3d else '2d'}"
    )

    # If the user selects a node or an area directly on the web, isolate it instantly!
    if chart_event and "selection" in chart_event and chart_event["selection"].get("points"):
        pts = chart_event["selection"]["points"]
        if pts:
            # If multiple nodes are captured by the lasso or box, select the primary one or last clicked
            first_pt = pts[0]
            if "customdata" in first_pt and first_pt["customdata"]:
                clicked_node = first_pt["customdata"]
                if isinstance(clicked_node, list):
                    clicked_node = clicked_node[0]
                if clicked_node in NODES and st.session_state.get("selected_node") != clicked_node:
                    st.session_state["selected_node"] = clicked_node
                    st.rerun()

    # External dedicated analytical card (displayed cleanly outside the graph canvas so it never blocks nodes)
    displayed_node_id = active_isolated_id if (active_isolated_id and active_isolated_id in node_states) else None
    
    if displayed_node_id:
        c_card_top1, c_card_top2 = st.columns([3, 1])
        with c_card_top2:
            if st.button("❌ Chiudi Focus / Reset Asset", key="btn_close_card_focus", type="primary", use_container_width=True):
                reset_node_selection()
                st.rerun()

        ns = node_states[displayed_node_id]
        nd = NODES[displayed_node_id]
        connected_links = [l for l in link_states if l.node_a_id == displayed_node_id or l.node_b_id == displayed_node_id]
        
        status_color = "#D946EF" if abs(ns.z_score) >= 2.0 else ("#F59E0B" if abs(ns.z_score) >= 1.25 else "#10B981")
        quad_info = QUADRANTS.get(nd.quadrant, {"name": nd.quadrant, "color": "#38BDF8"})

        links_badges = "".join([
            f'<span style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 3px 8px; margin-right: 6px; margin-bottom: 6px; display: inline-block; font-size: 11px; font-family: monospace; color: #E2E8F0;">'
            f'<b>{l.node_b_id if l.node_a_id == displayed_node_id else l.node_a_id}</b> ({l.active_corr:+.2f} | |Z|={l.z_score_tension:.1f})'
            f'</span>'
            for l in connected_links
        ])

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1.5px solid #38BDF8; border-radius: 12px; padding: 20px 24px; margin-top: 14px; box-shadow: 0 8px 30px rgba(0,0,0,0.5);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #38BDF8; font-weight: 700;">🔍 SCHEDA NODO ISTITUZIONALE (ESTERNA AL GRAFICO)</span>
                        <span style="background: {status_color}22; border: 1px solid {status_color}; color: {status_color}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700;">{ns.status.upper()}</span>
                        <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); color: #94A3B8; border-radius: 4px; padding: 2px 8px; font-size: 11px;">Anello {nd.ring} • {nd.category}</span>
                    </div>
                    <div style="font-size: 24px; font-weight: 800; color: #F8FAFC;">{nd.name} <span style="font-size: 16px; color: #94A3B8; font-family: monospace;">[{displayed_node_id}]</span></div>
                    <div style="font-size: 13px; color: #CBD5E1; margin-top: 6px; max-width: 850px;">{nd.description}</div>
                    <div style="font-size: 12.5px; color: #94A3B8; margin-top: 4px;"><b>Ruolo Intermarket & Dinamica:</b> {nd.role}</div>
                </div>
                <div style="display: flex; gap: 20px; align-items: center;">
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; text-align: right;">
                        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Valore Attuale</div>
                        <div style="font-size: 22px; font-weight: 800; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">{ns.current_value:,.2f} <span style="font-size: 13px; color: #94A3B8;">{ns.unit}</span></div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; text-align: right;">
                        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Z-Score Dinamico</div>
                        <div style="font-size: 22px; font-weight: 800; color: {status_color}; font-family: 'JetBrains Mono', monospace;">{ns.z_score:+.2f}</div>
                    </div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 12.5px; color: #CBD5E1;">
                <div><b>Variazione Periodo:</b> <span style="color: {'#10B981' if ns.delta_period >= 0 else '#EF4444'}; font-weight: 600;">{ns.delta_period:+,.2f} ({ns.delta_period_pct:+.2f}%)</span></div>
                <div><b>Variazione YoY (1 Anno):</b> <span style="color: {'#10B981' if ns.delta_yoy >= 0 else '#EF4444'}; font-weight: 600;">{ns.delta_yoy:+,.2f}</span></div>
                <div><b>Fonte Ufficiale Certificata:</b> <span style="color: #F8FAFC; font-weight: 600;">{nd.source}</span> (<span style="color: #38BDF8; font-family: monospace;">{nd.official_ticker}</span>)</div>
                <div><b>Data Ultimo Rilascio:</b> <span style="color: #F8FAFC; font-weight: 600;">{ns.last_timestamp}</span></div>
                <div><b>Quadrante Macro:</b> <span style="color: {quad_info['color']}; font-weight: 600;">{quad_info['name']}</span></div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed rgba(255,255,255,0.08);">
                <div style="font-size: 11.5px; color: #94A3B8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                    🔗 <b>Collegamenti Diretti nella Tela ({len(connected_links)} Connessioni Attive):</b>
                </div>
                <div>{links_badges if links_badges else '<span style=\"color:#64748B;\">Nessun collegamento diretto filtrato</span>'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px dashed rgba(255,255,255,0.15); border-radius: 12px; padding: 14px 20px; margin-top: 14px; display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 13px; color: #94A3B8;">
                💡 <i>Suggerimento Istituzionale: Clicca su qualsiasi nodo della tela di ragno o selezionalo dal menu laterale per bloccare l'evidenziazione ed esaminare la scheda analitica dettagliata con tutti i parametri di trasmissione macro.</i>
            </div>
            <span style="font-size: 11.5px; color: #38BDF8; font-family: monospace; white-space: nowrap;">30 Nodi Disponibili</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 16px; margin-top: 12px; font-size: 12px; color: #94A3B8; display: flex; flex-wrap: wrap; gap: 20px;">
        <div>🟢 <b>Verde Smeraldo:</b> Correlazione Positiva Diretta Normale</div>
        <div>🔴 <b>Rosso Corallo:</b> Correlazione Inversa Normale</div>
        <div>🟡 <b>Giallo Ambra:</b> Tensione Moderata (|Z| ≥ 1.25)</div>
        <div>🟣 <b>Viola Neon Pulsante:</b> Rottura / Divergenza Critica (|Z| ≥ 2.0)</div>
        <div><b>Spessore Filo:</b> Proporzionale all'intensità del legame rolling (|r|)</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Manuale Operativo Rapido: Come Leggere & Interpretare la Tela di Ragno (Clicca qui)"):
        st.markdown("""
        ### 🕸️ Manuale d'Uso della Tela di Ragno SPIDER

        #### 1. I Tre Livelli Concentrici (Dalla Causa all'Effetto)
        - **Anello 1 (Interno): Politica Monetaria & Tassi Sovrani**  
          Contiene i tassi guida della FED e della BCE (`RATE_FED`, `RATE_ECB`), i rendimenti a 2 e 10 anni (`YIELD_US02`, `YIELD_US10`), i tassi reali TIPS (`YIELD_REAL10`) e la pendenza della curva (`CURVE_10_2`). Qui si originano gli impulsi del costo del denaro.
        - **Anello 2 (Intermedio): Macroeconomia Reale & Fondamentali**  
          Contiene i dati macro con ritardo di trasmissione: Inflazione (`INF_USPCE`, `INF_USCPI`), Occupazione e Lavoro (`LAB_UNRATE`, `LAB_CLAIMS`) e Produzione/Consumi (`GRO_USGDP`, `GRO_INDPRO`, `GRO_RETAIL`).
        - **Anello 3 (Esterno): Asset Finanziari, Borse & Materie Prime**  
          Contiene le quotazioni dei mercati a pronti: Indici Azionari (`S&P 500`, `Nasdaq 100`, `Russell 2000`, `DAX`, `FTSE MIB`), Petrolio (`COM_OIL`), Oro (`COM_GOLD`), Rame (`COM_COPPER`), Dollaro (`FX_DXY`) e Credito High Yield (`CRED_HY`).

        #### 2. Significato dei Colori dei Fili di Connessione
        - **🟢 Verde Smeraldo:** Relazione teorica positiva rispettata (due variabili che devono salire/scendere insieme lo stanno facendo).
        - **🔴 Rosso Corallo:** Relazione teorica inversa rispettata (due variabili che divergono lo stanno facendo normalmente).
        - **🟡 Giallo Ambra (Tratteggiato):** Tensione moderata ($1,25 \le |Z| < 2,0$). Il filo sta accumulando attrito statistico rispetto alla media storica.
        - **🟣 Viola Neon Pulsante:** Anomalia o divergenza critica ($|Z| \ge 2,0$). La relazione storica è temporaneamente rotta (es. Dollaro e Borsa che salgono insieme come polo di attrazione globale della liquidità).
        - **Spessore del Filo:** Indica la forza assoluta del coefficiente di correlazione rolling ($|r|$).

        #### 3. Centro di Controllo `SPIDER_CORE`
        Il nucleo al centro calcola la **Tensione Sistemica della Tela**:
        - 🟢 **Verde ($< 1,25$):** Tela in equilibrio armonico.
        - 🟡 **Giallo/Arancio ($1,25 - 2,0$):** Attrito macroeconomico (fase di tassi restrittivi o shock energetici).
        - 🔴 **Rosso ($> 2,0$):** Rottura sistemica con elevato rischio di violento repricing intermarket.

        #### 4. Come Usare l'Interazione sulla Rete
        - **Hover:** Passando con il mouse su un nodo o su un badge numerico del filo compaiono i valori istantanei.
        - **Click sul Nodo:** Cliccando su un nodo nella tela, tutti i nodi non collegati vengono attenuati (sfocati) e appare la **Scheda Esterna Dettagliata** sotto al grafico con tutte le informazioni quantitative e la fonte certificata.
        """)

        st.markdown("---")
        st.markdown(f"#### 5. 📚 Enciclopedia Analitica dei 30 Nodi Istituzionali SPIDER {APP_VERSION}")
        st.caption("Scheda tecnica strutturata per ciascuno dei 30 nodi: **Definizione**, **Cosa misura esattamente**, **Interpretazione**, **Perché è rilevante**, **Fonti e ticker**.")
        for nid, entry in NODE_ENCYCLOPEDIA.items():
            with st.expander(f"🔹 [{entry.id}] {entry.name} • {RINGS[entry.ring]['name']} • {QUADRANTS[entry.quadrant]['name']}", expanded=False):
                st.markdown(
                    f"- **Definizione:** {entry.definizione}\n\n"
                    f"- **Cosa misura esattamente:** {entry.cosa_misura}\n\n"
                    f"- **Interpretazione:** {entry.interpretazione}\n\n"
                    f"- **Perché è rilevante:** {entry.perche_rilevante}\n\n"
                    f"- **Fonti e ticker:** `{entry.fonti_ticker}`"
                )

# TAB 2
with tabs[1]:
    st.markdown("#### 🎯 Centro di Controllo (SPIDER_CORE): Matrice dei Regimi Economici")
    st.caption("Il Centro di Controllo sintetizza le traiettorie dell'economia e dei mercati mappando Crescita e Inflazione nei 4 quadranti istituzionali di Ray Dalio / Bridgewater.")

    col_mat_l, col_mat_r = st.columns([1, 1])

    with col_mat_l:
        st.markdown("##### 🧭 Posizionamento 2D nel Ciclo Economico")
        fig_matrix = build_macro_regime_matrix_figure(core_state, dark_mode=True)
        st.plotly_chart(fig_matrix, use_container_width=True)

    with col_mat_r:
        st.markdown("##### 📊 Diagnosi Istituzionale del Regime & Scenario di Atterraggio")
        ls_tab = core_state.landing_scenario
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 18px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Regime Corrente Certificato</div>
                <span style="background: {ls_tab.color_hex}22; border: 1px solid {ls_tab.color_hex}; color: {ls_tab.color_hex}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700;">{ls_tab.scenario_badge}</span>
            </div>
            <div style="font-size: 22px; font-weight: 800; color: #38BDF8; margin-top: 4px;">{core_state.macro_regime}</div>
            <p style="font-size: 13px; color: #CBD5E1; margin-top: 8px; line-height: 1.5;">{core_state.macro_regime_desc}</p>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.1); font-size: 12px; color: #94A3B8;">
                <b>Scenario di Atterraggio (Landing):</b> <span style="color: {ls_tab.color_hex}; font-weight: 700;">{ls_tab.scenario_title}</span> ({ls_tab.confidence_pct:.1f}% confidenza).
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔎 Esplora Modello & Matrice di Atterraggio (Soft vs Hard vs No Landing)", key="btn_open_landing_modal_tab2", use_container_width=True, type="secondary"):
            show_landing_scenario_modal()

        st.markdown("##### 🏛️ Termometro Politica Monetaria (FED & BCE)")
        col_fed1, col_fed2 = st.columns(2)
        with col_fed1:
            st.metric("Tasso Reale FED (Fed Funds - PCE)", f"{core_state.fed_real_rate:+.2f}%", delta="Restrittivo" if core_state.fed_real_rate > 1.5 else "Neutrale")
        with col_fed2:
            st.metric("Tasso Reale BCE (DFR - HICP)", f"{core_state.ecb_real_rate:+.2f}%", delta="Restrittivo" if core_state.ecb_real_rate > 0.5 else "Accomodante")

        st.markdown("##### ⚖️ Regola di Taylor Istituzionale")
        st.write(f"Differenziale tra tasso effettivo FED e obiettivo stimato dalla Taylor Rule: **{core_state.taylor_rate_gap_bps:+.0f} bps**.")
        if core_state.taylor_rate_gap_bps > 50:
            st.success("La politica monetaria è in territorio ampiamente restrittivo (adatta a frenare l'inflazione).")
        elif core_state.taylor_rate_gap_bps < -50:
            st.warning("La politica monetaria è dietro la curva e richiede rialzi o cautela sui tagli.")
        else:
            st.info("I tassi attuali sono in sostanziale equilibrio con i fondamentali macroeconomici.")

    with st.expander("📚 Distinzione Teorica Fondamentale: Reflazione vs Surriscaldamento Ciclico (Overheating)"):
        st.markdown("""
        ### 🔍 Teoria Macroeconomica: Perché la Reflazione non è un Surriscaldamento

        * **1. Che cos'è la vera Reflazione (Irving Fisher / Keynes / Cole):**
          - La **Reflazione** è un programma deliberato di **politica monetaria e fiscale ultra-espansiva** (taglio aggressivo dei tassi, espansione della base monetaria/QE, deficit spending) finalizzato a contrastare una **spirale deflazionistica** o una **depressione economica**.
          - Interviene quando l'economia opera **al di sotto della piena capacità produttiva** (output gap negativo, disoccupazione ciclica elevata) e l'inflazione è caduta **sotto** il trend di lungo periodo (es. post-crisi 2008 o post-lockdown 2020).
          - **Obiettivo:** Riportare la domanda aggregata e i prezzi *dal basso* verso il target neutrale ($2\%$).

        * **2. Il Contesto Attuale: Surriscaldamento da Tardo Ciclo (Late-Cycle Overheating):**
          - L'economia attuale opera in regime di **piena occupazione** (tasso di disoccupazione al $4,2\%$, richieste di sussidi a $199.000$).
          - L'inflazione è **già al di sopra del target** (Core PCE al $3,01\%$, CPI Headline al $3,72\%$, Petrolio WTI a $\$101/\text{bbl}$).
          - La politica monetaria è **restrittiva** (Fed Funds al $3,88\%$, tasso reale positivo a $+0,87\%$, bias **Hawkish / Vigile**).
          - **Conclusione Istituzionale:** Questo scenario NON è una manovra di reflazione anti-deflazione, bensì un **Surriscaldamento Ciclico da domanda/costi delle materie prime** (perfettamente descritto dalla *Fase 4 del ciclo di Martin Pring*), in cui le banche centrali contrastano l'eccesso di domanda anziché stimolarla.
        """)

# TAB 3: MARTIN PRING BUSINESS CYCLE
with tabs[2]:
    st.markdown("#### 🔄 Teoria Macro dei Cicli Economici di Martin Pring")
    st.caption("Modello di rotazione intermarket basato sulla sequenza cronologica di picchi e minimi delle 3 macro-classi di asset: **Obbligazioni (Bonds) → Azioni (Stocks) → Materie Prime (Commodities)**.")

    pring_engine = PringCycleEngine(aligned_df)
    pring_res = pring_engine.analyze()

    p_col1, p_col2 = st.columns([1.3, 1.7])
    with p_col1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 18px; margin-bottom: 14px;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 700;">FASE CORRENTE RILEVATA DA SPIDER</div>
            <div style="font-size: 22px; font-weight: 800; color: {PRING_STAGES[pring_res.current_stage]['color']}; margin: 6px 0;">{pring_res.stage_title}</div>
            <div style="font-size: 13px; color: #E2E8F0; line-height: 1.5; margin-bottom: 12px;">{pring_res.stage_desc}</div>
            <div style="font-size: 12px; color: #38BDF8;"><b>Confidenza del Modello:</b> {pring_res.confidence_pct:.1f}%</div>
            <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;"><b>Regime Economico:</b> {pring_res.stage_regime}</div>
            <div style="font-size: 12px; color: #F59E0B; margin-top: 4px;"><b>{pring_res.sequence_state}</b></div>
        </div>
        """, unsafe_allow_html=True)

        fig_dial = build_pring_cycle_dial_figure(pring_res, dark_mode=True)
        st.plotly_chart(fig_dial, use_container_width=True)

    with p_col2:
        st.markdown("##### 🧭 Sequenza dei Punti di Svolta & Stato delle Classi di Asset")
        st.markdown("""
        > **Regola Aurea di Martin Pring:** I mercati non invertono mai contemporaneamente. La sequenza fisiologica di svolta è:  
        > **1. Massimi (Tops):** Obbligazioni ➔ Azioni ➔ Materie Prime  
        > **2. Minimi (Bottoms):** Obbligazioni ➔ Azioni ➔ Materie Prime
        """)

        b_st = pring_res.asset_statuses["BONDS"]
        s_st = pring_res.asset_statuses["STOCKS"]
        c_st = pring_res.asset_statuses["COMMODITIES"]

        c_a1, c_a2, c_a3 = st.columns(3)
        with c_a1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid {'#10B981' if b_st.is_bullish else '#EF4444'}; border-radius: 10px; padding: 12px;">
                <div style="font-size: 11px; color: #94A3B8; font-weight: 600;">1. OBBLIGAZIONI</div>
                <div style="font-size: 16px; font-weight: 800; color: {'#10B981' if b_st.is_bullish else '#EF4444'}; margin: 4px 0;">{'BULL' if b_st.is_bullish else 'BEAR'}</div>
                <div style="font-size: 11px; color: #CBD5E1;">Trend: <b>{b_st.trend_200d}</b></div>
                <div style="font-size: 11px; color: #CBD5E1;">Mom 60G: <b>{b_st.momentum_60d_pct:+.2f}%</b></div>
                <div style="font-size: 11px; color: #94A3B8;">KST: <b>{b_st.kst_val:+.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with c_a2:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid {'#10B981' if s_st.is_bullish else '#EF4444'}; border-radius: 10px; padding: 12px;">
                <div style="font-size: 11px; color: #94A3B8; font-weight: 600;">2. AZIONI</div>
                <div style="font-size: 16px; font-weight: 800; color: {'#10B981' if s_st.is_bullish else '#EF4444'}; margin: 4px 0;">{'BULL' if s_st.is_bullish else 'BEAR'}</div>
                <div style="font-size: 11px; color: #CBD5E1;">Trend: <b>{s_st.trend_200d}</b></div>
                <div style="font-size: 11px; color: #CBD5E1;">Mom 60G: <b>{s_st.momentum_60d_pct:+.2f}%</b></div>
                <div style="font-size: 11px; color: #94A3B8;">KST: <b>{s_st.kst_val:+.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with c_a3:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid {'#10B981' if c_st.is_bullish else '#EF4444'}; border-radius: 10px; padding: 12px;">
                <div style="font-size: 11px; color: #94A3B8; font-weight: 600;">3. MATERIE PRIME</div>
                <div style="font-size: 16px; font-weight: 800; color: {'#10B981' if c_st.is_bullish else '#EF4444'}; margin: 4px 0;">{'BULL' if c_st.is_bullish else 'BEAR'}</div>
                <div style="font-size: 11px; color: #CBD5E1;">Trend: <b>{c_st.trend_200d}</b></div>
                <div style="font-size: 11px; color: #CBD5E1;">Mom 60G: <b>{c_st.momentum_60d_pct:+.2f}%</b></div>
                <div style="font-size: 11px; color: #94A3B8;">KST: <b>{c_st.kst_val:+.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 💼 Asset Allocation Tattica Suggerita da Martin Pring")
        alloc_col1, alloc_col2 = st.columns(2)
        with alloc_col1:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px;">
                <div style="font-size: 12px; font-weight: 700; color: #10B981;">✅ ASSET DA PRIVILEGIARE (OVERWEIGHT)</div>
                <ul style="font-size: 12px; color: #E2E8F0; margin-top: 6px; padding-left: 18px;">
                    {''.join(f'<li>{a}</li>' for a in pring_res.favored_assets)}
                </ul>
                <div style="font-size: 12px; font-weight: 700; color: #38BDF8; margin-top: 8px;">🏷️ SETTORI AZIONARI FAVORITI:</div>
                <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;">{', '.join(pring_res.favored_sectors)}</div>
            </div>
            """, unsafe_allow_html=True)
        with alloc_col2:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 12px;">
                <div style="font-size: 12px; font-weight: 700; color: #EF4444;">⚠️ ASSET DA SOTTOPESARE (UNDERWEIGHT)</div>
                <ul style="font-size: 12px; color: #E2E8F0; margin-top: 6px; padding-left: 18px;">
                    {''.join(f'<li>{a}</li>' for a in pring_res.unfavored_assets)}
                </ul>
                <div style="font-size: 12px; font-weight: 700; color: #F59E0B; margin-top: 8px;">🛡️ ROTAZIONE SETTORIALE ATTIVA:</div>
                <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;">Spostare capitale verso i settori leading del ciclo corrente prima del picco dei tassi.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    st.markdown("#### 📈 Strumenti Quantitativi di Pring: Oscillatore KST & Rotazione Settoriale RRG")
    c_tool1, c_tool2 = st.columns(2)
    with c_tool1:
        st.markdown("##### 📐 Indicatore KST (Know Sure Thing)")
        st.caption("Oscillatore multi-temporale di momentum sviluppato da Martin Pring per individuare le svolte primarie su S&P 500.")
        if aligned_df is not None and "EQ_SPX" in aligned_df:
            fig_kst = build_kst_momentum_figure(aligned_df["EQ_SPX"], title="S&P 500: Pring KST Multi-Timeframe")
            st.plotly_chart(fig_kst, use_container_width=True)

    with c_tool2:
        st.markdown("##### 🎯 Relative Rotation Graph (RRG) Settori S&P 500")
        st.caption("Forza Relativa (RS-Ratio) vs Momentum (RS-Momentum) rispetto all'indice benchmark SPY.")
        fig_rrg = build_rrg_sector_matrix_figure(dark_mode=True)
        st.plotly_chart(fig_rrg, use_container_width=True)

    # Horizontal Bar Chart ranking sectors
    st.markdown("##### 📊 Livelli di Forza Relativa dei Settori S&P 500 (Classifica Linee Orizzontali)")
    st.caption("Visualizzazione orizzontale dei livelli di sovraperformance/sottoperformance rispetto alla parità (benchmark SPY = 0%). I colori indicano il quadrante di appartenenza nel ciclo di rotazione.")
    fig_bars = build_sector_horizontal_bars_figure(dark_mode=True)
    st.plotly_chart(fig_bars, use_container_width=True)

    with st.expander("📖 Guida Enciclopedica: Le 6 Fasi del Ciclo, KST e Relative Rotation Graph (RRG)"):
        st.markdown("### 🔄 Le 6 Fasi del Ciclo Economico di Martin Pring")
        for stg_num, stg_data in PRING_STAGES.items():
            st.markdown(f"""
            **Fase {stg_num}: {stg_data['title']}**  
            - **Regime:** {stg_data['regime']}  
            - **Obbligazioni (Prezzo):** {stg_data['bonds']} | **Azioni:** {stg_data['stocks']} | **Materie Prime:** {stg_data['commodities']}  
            - **Dinamica:** {stg_data['description']}  
            - **Asset Favoriti (Overweight):** {', '.join(stg_data['favored_assets'])}  
            - **Settori:** {', '.join(stg_data['favored_sectors'])}  
            ---
            """)

        st.markdown("### 📐 Indicatore KST (Know Sure Thing): Matematica e Interpretazione")
        st.markdown("""
        L'indicatore **KST (Know Sure Thing)** è un oscillatore di momentum formulato da Martin Pring per catturare la tendenza primaria del mercato eliminando i falsi segnali tipici degli oscillatori a singolo periodo (come l'RSI a 14 giorni).
        
        * **Formula e Costruzione Multi-Temporale:**
          Il KST combina 4 diversi tassi di variazione (Rate of Change - ROC), ciascuno smussato con una media mobile semplice (SMA) e ponderato con pesi crescenti all'aumentare dell'orizzonte temporale:
          $$\\text{RCMA}_1 = \\text{SMA}(\\text{ROC}(10), 10) \\times 1$$
          $$\\text{RCMA}_2 = \\text{SMA}(\\text{ROC}(15), 10) \\times 2$$
          $$\\text{RCMA}_3 = \\text{SMA}(\\text{ROC}(20), 10) \\times 3$$
          $$\\text{RCMA}_4 = \\text{SMA}(\\text{ROC}(30), 15) \\times 4$$
          $$\\mathbf{\\text{KST}} = \\text{RCMA}_1 + \\text{RCMA}_2 + \\text{RCMA}_3 + \\text{RCMA}_4$$
          $$\\mathbf{\\text{Signal Line}} = \\text{SMA}(\\text{KST}, 9)$$
        
        * **Segnali Operativi:**
          1. **Inversione Ciclica:** Quando la linea KST (blu) attraversa al rialzo la Signal Line (tratteggiata arancione), conferma l'inizio di una nuova fase espansiva dell'asset.
          2. **Conferma dello Zero:** Il superamento della linea dello zero verso l'alto sancisce l'ingresso in un trend rialzista consolidato a medio-lungo termine.
          3. **Divergenze:** Se il prezzo segna nuovi massimi ma il KST non li conferma, preannuncia l'esaurimento del ciclo e il passaggio alla fase successiva di Pring.
        """)

        st.markdown("---")
        st.markdown("### 🎯 Relative Rotation Graph (RRG): Funzionamento e Rotazione Settoriale")
        st.markdown("""
        Il **Relative Rotation Graph (RRG)**, ideato da Julius de Kempenaer e ampiamente integrato nella scuola di Martin Pring, è uno strumento visuale che mappa la forza relativa di molteplici asset o settori contemporaneamente rispetto a un benchmark comune (l'S&P 500 / ETF `SPY`).
        
        * **I Due Assi Fondamentali (Normalizzati a 100):**
          - **Asse X: RS-Ratio (Forza Relativa):** Misura se il settore sta sovraperformando ($>100$) o sottoperformando ($<100$) il benchmark di mercato.
          - **Asse Y: RS-Momentum (Velocità della Forza Relativa):** Misura se il trend di sovraperformance sta accelerando ($>100$) o decelerando ($<100$).
        
        * **La Rotazione Oraria nei 4 Quadranti:**
          Nel corso del ciclo economico, i settori tendono a ruotare in **senso orario** attraversando 4 stadi:
          1. **🟢 LEADING (In Alto a Destra):** Forza Relativa $>100$ e Momentum $>100$. Il settore sta sovraperformando il mercato con velocità crescente (es. *Energia e Materiali in Fase 4*).
          2. **🟡 WEAKENING (In Basso a Destra):** Forza Relativa $>100$ ma Momentum $<100$. Il settore è ancora più forte del mercato, ma sta perdendo slancio (inizio rallentamento).
          3. **🔴 LAGGING (In Basso a Sinistra):** Forza Relativa $<100$ e Momentum $<100$. Il settore sottoperforma l'indice ed è in trend ribassista relativo (es. *Tecnologici compressi dai tassi alti in Fase 4/5*).
          4. **🔵 IMPROVING (In Alto a Sinistra):** Forza Relativa $<100$ ma Momentum $>100$. Il settore è ancora debole rispetto al mercato, ma il momentum sta invertendo al rialzo (fase di accumulazione/ripresa anticipata, es. *Utilities difensive*).
        """)

# TAB 4: CURVA DEI RENDIMENTI & REGIMI (STENO FRAMEWORK)
with tabs[3]:
    st.markdown("#### 📈 Dinamica della Curva dei Rendimenti & Regimi Macro (Steno Research Framework)")
    st.markdown("##### *«Steepeners feed the credit engine.. flatteners starve it»*")
    st.caption("Analisi dei 4 regimi della curva dei tassi (Bull/Bear Steepener & Bull/Bear Flattener) classificati in ordine di impatto sul ciclo economico e sulla creazione del credito bancario.")

    yc_state = yield_curve_engine.get_curve_regime_state()

    # Top KPI summary cards
    yc_c1, yc_c2, yc_c3, yc_c4 = st.columns(4)
    with yc_c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {yc_state.color_hex};">
            <div class="kpi-title">Regime Rilevato</div>
            <div class="kpi-main-val" style="color: {yc_state.color_hex}; font-size: 18px;">{yc_state.regime_name}</div>
            <div class="kpi-sub-val">Rango Ciclo: {yc_state.rank} di 4</div>
        </div>
        """, unsafe_allow_html=True)

    with yc_c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Variazione Tasso Breve (2Y)</div>
            <div class="kpi-main-val" style="color: {'#EF4444' if yc_state.short_rate_delta_bps > 0 else '#10B981'}; font-size: 20px;">
                {yc_state.short_rate_delta_bps:+.1f} bps
            </div>
            <div class="kpi-sub-val">Rispetto a 30gg fa</div>
        </div>
        """, unsafe_allow_html=True)

    with yc_c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Variazione Tasso Lungo (10Y)</div>
            <div class="kpi-main-val" style="color: {'#EF4444' if yc_state.long_rate_delta_bps > 0 else '#10B981'}; font-size: 20px;">
                {yc_state.long_rate_delta_bps:+.1f} bps
            </div>
            <div class="kpi-sub-val">Rispetto a 30gg fa</div>
        </div>
        """, unsafe_allow_html=True)

    with yc_c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Inclinazione Pendenza (Slope Δ)</div>
            <div class="kpi-main-val" style="color: {'#10B981' if yc_state.slope_delta_bps > 0 else '#F59E0B'}; font-size: 20px;">
                {yc_state.slope_delta_bps:+.1f} bps
            </div>
            <div class="kpi-sub-val">Spread 2Y-10Y: {yc_state.current_slope_2s10s_bps:+.1f} bps</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Inversion Status Banner & Dual-Spread Inversion Cards
    inv = yc_state.inversion
    inv_bg = "rgba(239, 68, 68, 0.15)" if inv.inversion_status == "INVERTED" else (
        "rgba(245, 158, 11, 0.15)" if "UNINVERTING" in inv.inversion_status else "rgba(16, 185, 129, 0.12)"
    )
    inv_border = "#EF4444" if inv.inversion_status == "INVERTED" else (
        "#F59E0B" if "UNINVERTING" in inv.inversion_status else "#10B981"
    )

    st.markdown(f"""
    <div style="background: {inv_bg}; border: 1.5px solid {inv_border}; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 14px; font-weight: 800; color: #FFFFFF;">STATO GEOMETRICO CURVA:</span> 
                <span style="font-size: 14px; font-weight: 700; color: {inv_border}; margin-left: 8px;">{inv.inversion_status_badge}</span>
            </div>
            <div>
                <span style="font-size: 12px; color: #94A3B8;">Rischio Recessione da Curva:</span> 
                <b style="font-size: 13px; color: #FFFFFF; background: rgba(0,0,0,0.3); padding: 3px 8px; border-radius: 4px;">{inv.recession_warning_level}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Inversion Spreads Row (2s10s and 3m10s)
    col_inv1, col_inv2, col_inv3 = st.columns(3)
    with col_inv1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {'#EF4444' if inv.is_2s10s_inverted else '#10B981'};">
            <div class="kpi-title">Spread Benchmark 2Y-10Y</div>
            <div class="kpi-main-val" style="color: {'#EF4444' if inv.is_2s10s_inverted else '#10B981'}; font-size: 20px;">
                {inv.spread_2s10s_bps:+.1f} bps
            </div>
            <div class="kpi-sub-val">{"⚠️ INVERTITO" if inv.is_2s10s_inverted else "🟢 POSITIVO (Normale)"} (Var 30gg: {inv.delta_2s10s_bps:+.1f} bps)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_inv2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {'#EF4444' if inv.is_3m10y_inverted else '#10B981'};">
            <div class="kpi-title">Spread Fed Monitor 3M-10Y</div>
            <div class="kpi-main-val" style="color: {'#EF4444' if inv.is_3m10y_inverted else '#10B981'}; font-size: 20px;">
                {inv.spread_3m10y_bps:+.1f} bps
            </div>
            <div class="kpi-sub-val">{"⚠️ INVERTITO" if inv.is_3m10y_inverted else "🟢 POSITIVO"} (Var 30gg: {inv.delta_3m10y_bps:+.1f} bps)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_inv3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Spread Estremo 3M-30Y</div>
            <div class="kpi-main-val" style="color: #38BDF8; font-size: 20px;">
                {yc_state.current_slope_3m30y_bps:+.1f} bps
            </div>
            <div class="kpi-sub-val">Inclinazione complessiva del ciclo</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main comparison chart and 2x2 regime matrix
    yc_col_left, yc_col_right = st.columns([3, 2])
    with yc_col_left:
        fig_yc = build_yield_curve_comparison_figure(yc_state, dark_mode=True)
        st.plotly_chart(fig_yc, use_container_width=True)

    with yc_col_right:
        fig_matrix = build_regimes_matrix_preview_figure(yc_state.regime_id, dark_mode=True)
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Popup Modal Trigger & Definition
        @st.dialog("🔍 Dettaglio Grafico & Caratteristiche dei 4 Regimi della Curva", width="large")
        def show_quadrants_modal():
            st.markdown("### 📊 I 4 Scenari della Curva dei Rendimenti (Framework Steno Research)")
            st.caption("Visualizzazione grafica comparativa ('Before' tratteggiato vs 'After' continuo) delle quattro dinamiche di traslazione della curva, con frecce direzionali sulle scadenze guida 3M e 30Y.")

            # 2x2 Subplots Figure
            fig_grid = build_quadrants_comparison_grid_figure(dark_mode=True)
            st.plotly_chart(fig_grid, use_container_width=True)

            st.markdown("#### 📋 Tabella Sinottica delle Caratteristiche dei Tassi & del Credito")
            df_regimes_table = pd.DataFrame([
                {
                    "Regime": "#1 BULL STEEPENER",
                    "Dinamica Tasso Breve (3M / 2Y)": "⬇️ Crollo marcato (Tagli Fed)",
                    "Dinamica Tasso Lungo (10Y / 30Y)": "➡️ Stabile o discesa lieve",
                    "Pendenza Curva (Slope)": "↗️ Forte irripidimento (Steeper)",
                    "Motore del Credito": "🟢 Massimo stimolo (NIM banche alle stelle)",
                    "Driver Macro": "Allentamento monetario per rilanciare la crescita (Trump/Bessent dream)",
                    "Asset Favoriti": "Small Cap (RUT), Banche, Real Estate",
                },
                {
                    "Regime": "#2 BEAR STEEPENER",
                    "Dinamica Tasso Breve (3M / 2Y)": "➡️ Stabile o lieve rialzo",
                    "Dinamica Tasso Lungo (10Y / 30Y)": "⬆️ Forte rialzo (Premio a termine)",
                    "Pendenza Curva (Slope)": "↗️ Irripidimento guidato dai lunghi",
                    "Motore del Credito": "🟡 Il motore bancario gira ancora (Yield curve steep)",
                    "Driver Macro": "Aspettative inflattive e massicce emissioni di debito pubblico",
                    "Asset Favoriti": "Materie Prime, Energia, Titoli Value (Short TLT)",
                },
                {
                    "Regime": "#3 BULL FLATTENER",
                    "Dinamica Tasso Breve (3M / 2Y)": "➡️ Stabile o lieve discesa",
                    "Dinamica Tasso Lungo (10Y / 30Y)": "⬇️ Forte calo (Compressione term premia)",
                    "Pendenza Curva (Slope)": "↘️ Appiattimento guidato dai lunghi",
                    "Motore del Credito": "🟠 Margini di intermediazione compressi",
                    "Driver Macro": "Forte disinflazione o timori di rallentamento economico",
                    "Asset Favoriti": "TLT / Long Treasuries, Big Tech ad alta duration, Oro",
                },
                {
                    "Regime": "#4 BEAR FLATTENER",
                    "Dinamica Tasso Breve (3M / 2Y)": "⬆️ Forte rialzo (Fed restrittiva)",
                    "Dinamica Tasso Lungo (10Y / 30Y)": "➡️ Rialzo contenuto o freno",
                    "Pendenza Curva (Slope)": "↘️ Appiattimento / Rischio inversione",
                    "Motore del Credito": "🔴 Pressione sui margini bancari (Inasprimento selettivo)",
                    "Driver Macro": "Politica monetaria restrittiva ('higher for longer') per frenare l'inflazione",
                    "Asset Favoriti": "Cash monetario 1-3M (BIL/SGOV), USD (DXY), Quality",
                },
            ])
            st.dataframe(df_regimes_table, use_container_width=True, hide_index=True)

            st.markdown("""
            > **Regola aurea di Sintesi:**  
            > * I regimi **STEEPENER** (#1 e #2) aumentano la pendenza della curva favorendo la redditività del credito e il moltiplicatore monetario.  
            > * I regimi **FLATTENER** (#3 e #4) comprimono il differenziale temporale, riducendo l'incentivo bancario alla trasformazione delle scadenze.
            """)

        # Button placed directly beneath the matrix
        if st.button("🔎 Esplora Grafici Specifici & Tabella dei 4 Quadranti (Pop-up)", use_container_width=True, type="secondary"):
            show_quadrants_modal()

        st.markdown(f"""
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 14px; margin-top: 10px;">
            <div style="font-size: 13px; font-weight: 700; color: {yc_state.color_hex}; margin-bottom: 6px;">
                Diagnosi del Credito:
            </div>
            <div style="font-size: 13px; color: #F8FAFC; margin-bottom: 8px;">
                {yc_state.impact_credit_engine}
            </div>
            <div style="font-size: 12px; color: #94A3B8; line-height: 1.5;">
                {yc_state.macro_description}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DEEP CREDIT TRANSMISSION PANEL
    st.markdown("##### 🏦 Matrice di Trasmissione del Credito Bancario & Standard di Fido")
    cred = yc_state.credit_analysis
    c_card1, c_card2, c_card3 = st.columns(3)
    with c_card1:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px; height: 100%;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 6px;">Margine di Interesse (NIM)</div>
            <div style="font-size: 13px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">{cred.net_interest_margin_impact}</div>
            <div style="font-size: 11px; color: #94A3B8; line-height: 1.4;">{cred.sloos_interpretation}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_card2:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px; height: 100%;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 6px;">Standard di Prestito (SLOOS)</div>
            <div style="font-size: 13px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">{cred.bank_lending_standards}</div>
            <div style="font-size: 11px; color: #94A3B8; line-height: 1.4;">Direzione Impulso Credito: <b style="color: #FFFFFF;">{cred.credit_impulse_direction}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c_card3:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px; height: 100%;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 6px;">Rischio Rollover & Rifinanziamento</div>
            <div style="font-size: 13px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">{cred.corporate_refinancing_risk}</div>
            <div style="font-size: 11px; color: #94A3B8; line-height: 1.4;"><b>Verdetto SPIDER:</b> {cred.summary_verdict}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Points table and asset allocation guidance
    col_pts, col_alloc = st.columns([1, 1])
    with col_pts:
        st.markdown("##### 📊 Tabella Punti di Scadenza Curva USA (3M → 30Y)")
        df_yc_points = pd.DataFrame([
            {
                "Scadenza": p.tenor,
                "Anni": p.maturity_years,
                "Rendimento Attuale": f"{p.current_yield:.2f}%",
                "Rendimento Precedente": f"{p.prev_yield:.2f}%",
                "Variazione (bps)": f"{p.delta_bps:+.1f} bps",
                "Fonte Dati": p.source,
            }
            for p in yc_state.curve_points
        ])
        st.dataframe(df_yc_points, use_container_width=True, hide_index=True)

    with col_alloc:
        st.markdown("##### 💼 Posizionamento Portafoglio nel Regime Corrente")
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px;">
            <div style="margin-bottom: 12px;">
                <span style="color: #10B981; font-weight: 700;">🟢 ASSET FAVORITI:</span><br>
                <span style="font-size: 13px; color: #F8FAFC;">{" • ".join(yc_state.favored_assets)}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #EF4444; font-weight: 700;">🔴 ASSET SFAVORITI / DA SOTTOPESARE:</span><br>
                <span style="font-size: 13px; color: #F8FAFC;">{" • ".join(yc_state.unfavored_assets)}</span>
            </div>
            <div style="font-size: 12px; color: #94A3B8;">
                <b>Spiegazione TLT / Reddito Fisso:</b> In un regime di <i>{yc_state.regime_name}</i>, {
                    'il TLT è vulnerabile alle perdite in conto capitale dovute al rialzo dei tassi e alla compressione dei premi al termine.'
                    if 'BEAR' in yc_state.regime_id else
                    'il TLT beneficia del calo dei tassi a lungo termine e del declino dell\'inflazione attesa.'
                }
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed pedagogical expander with Steno Research framework
    with st.expander("📖 Guida Enciclopedica: I 4 Regimi della Curva dei Rendimenti (Steno Research)", expanded=False):
        st.markdown("""
        ### I 4 Regimi della Curva dei Rendimenti (Ranked for the Business Cycle)
        La forma e il movimento della curva dei rendimenti sono il motore più potente del credito macroeconomico:
        > **«Steepeners feed the credit engine.. flatteners starve it»**  
        *(Le curve che si irripidiscono alimentano il motore del credito bancario.. le curve che si appiattiscono lo affamano)*

        ---

        #### 1. #1 BULL STEEPENER (Espansione / "Trump/Bessent Dream")
        * **Dinamica dei Tassi:** La parte a breve scende sensibilmente mentre la parte a lunga scende meno o tiene.
        * **Causa:** La Federal Reserve taglia aggressivamente i tassi per sostenere la crescita o per normalizzazione monetaria.
        * **Impatto sul Credito:** **Massimo stimolo.** Le banche si rifinanziano a tassi bassi e prestano a tassi a lungo più alti, massimizzando il margine di interesse (NIM) e facendo correre il credito.
        * **Asset Chiave:** Azioni Cicliche, Small Cap (Russell 2000), Titoli Bancari, Immobiliari/REITs, Oro.

        #### 2. #2 BEAR STEEPENER (Surriscaldamento / "The Bank Engine Still Runs")
        * **Dinamica dei Tassi:** La parte a lungo sale più velocemente della parte a breve.
        * **Causa:** Aspettative di inflazione in aumento, forte crescita economica nominale o abbondante emissione di debito pubblico da parte del Tesoro (aumento del *term premium*).
        * **Impatto sul Credito:** **Positivo per le banche.** La curva ripida preserva la redditività bancaria, ma i tassi più alti a lungo aumentano il costo dei mutui e dei prestiti societari.
        * **Asset Chiave:** Titoli Value, Materie Prime, Petrolio, Titoli Energetici; **forte sottopeso su TLT / Long Treasuries**.

        #### 3. #3 BULL FLATTENER (Disinflazione / "Scarce-Asset Heaven")
        * **Dinamica dei Tassi:** I premi al termine si comprimono, la parte a lungo scende molto più di quella a breve.
        * **Causa:** Rallentamento della crescita o crollo delle aspettative di inflazione; gli investitori comprano bond a lungo termine per bloccare i rendimenti.
        * **Impatto sul Credito:** Il margine di interesse si assottiglia. Il credito rallenta progressivamente.
        * **Asset Chiave:** **TLT e Treasury a lungo termine (rally dei prezzi)**, Big Tech / Growth a lungo orizzonte, Asset scarsi (Oro, Bitcoin), Difensivi.

        #### 4. #4 BEAR FLATTENER (Stretta Monetaria / Late-Cycle Test)
        * **Dinamica dei Tassi:** La parte a breve sale più velocemente della parte a lunga.
        * **Causa:** Le banche centrali mantengono o rialzano i tassi guida a fronte di un'inflazione ancora resiliente (*higher for longer*).
        * **Impatto sul Credito:** 
          - *Se la curva è invertita (< 0 bps):* Diventa il classico **'Cycle Killer'** che affama il credito e costringe le banche a chiudere i rubinetti del prestito.
          - *Se la curva è ancora positiva (> +15/30 bps, contesto attuale):* Il credito non subisce un credit crunch (i dati ufficiali **Fed SLOOS** mostrano standard fermi e l'indice **Chicago Fed NFCI a -0.50** segnala condizioni distese); si assiste invece a un test selettivo sui margini di interesse e a un rallentamento controllato.
        * **Asset Chiave:** Cash / Liquidità monetaria a 1-3 mesi (BIL/SGOV), Dollaro USA (DXY), Titoli difensivi Quality; **sottopesare TLT / Long Treasuries per via dei tassi terminali alti**.
        """)


# TAB 5: SUPERVISIONE TABELLARE
with tabs[4]:
    st.markdown("#### 📋 Output Tabellare Interattivo & Supervisione Dati")
    st.caption("Tutte le metriche, i valori correnti, le variazioni, gli Z-score e i metadati di certificazione istituzionale.")

    tab_choice = st.radio("Seleziona Tabella da Supervisionare:", ["Nodi della Tela (30 Asset & Indicatori)", "Fili della Tela (10 Fili Maestri & Connessioni)"], horizontal=True)

    df_nodes = generate_nodes_dataframe(node_states)
    df_links = generate_links_dataframe(link_states)

    if tab_choice == "Nodi della Tela (30 Asset & Indicatori)":
        c_search, c_filter_stat = st.columns([2, 1])
        with c_search:
            search_txt = st.text_input("🔍 Cerca indicatore o ticker:", placeholder="es. CPI, US10, S&P 500, Oro...")
        with c_filter_stat:
            stat_filter = st.selectbox("Filtra per Stato Z-Score:", ["TUTTI", "Estremo (|Z| ≥ 2.0)", "Attenzione (|Z| ≥ 1.25)", "Normale"])

        filtered_nodes_df = df_nodes.copy()
        if search_txt:
            filtered_nodes_df = filtered_nodes_df[
                filtered_nodes_df["ID Nodo"].str.contains(search_txt, case=False) |
                filtered_nodes_df["Nome Indicatore / Asset"].str.contains(search_txt, case=False) |
                filtered_nodes_df["Ticker Ufficiale"].str.contains(search_txt, case=False)
            ]
        if stat_filter != "TUTTI":
            stat_name = stat_filter.split(" ")[0]
            filtered_nodes_df = filtered_nodes_df[filtered_nodes_df["Stato"] == stat_name]

        st.dataframe(filtered_nodes_df, use_container_width=True, height=450)

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            csv_data = filtered_nodes_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Scarica Nodi in CSV", data=csv_data, file_name=f"SPIDER_Nodi_{datetime.date.today()}.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                filtered_nodes_df.to_excel(writer, index=False, sheet_name="Nodi_SPIDER")
                df_links.to_excel(writer, index=False, sheet_name="Fili_SPIDER")
            st.download_button("📊 Scarica Cartella Excel (.xlsx)", data=excel_buffer.getvalue(), file_name=f"SPIDER_Report_{datetime.date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col_dl3:
            txt_report = f"SPIDER REPORT • {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            txt_report += f"Macro Regime: {core_state.macro_regime}\n"
            txt_report += f"Systemic Web Tension: {core_state.systemic_web_tension:.2f}\n"
            txt_report += "=" * 70 + "\n\n"
            txt_report += filtered_nodes_df.to_string(index=False)
            st.download_button("📑 Scarica Report ASCII (.txt)", data=txt_report, file_name=f"SPIDER_Report_{datetime.date.today()}.txt", mime="text/plain", use_container_width=True)

    else:
        st.dataframe(df_links, use_container_width=True, height=450)
        csv_links = df_links.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Scarica Fili e Connessioni in CSV", data=csv_links, file_name=f"SPIDER_Fili_Connessioni_{datetime.date.today()}.csv", mime="text/csv")

# TAB 6: FOCUS NODO & RELAZIONI
with tabs[5]:
    st.markdown("#### 🔬 Focus Singolo Nodo & Correlazioni di Dettaglio")
    inspect_node_id = st.selectbox(
        "Seleziona Indicatore o Asset da Analizzare in Profondità:",
        options=list(NODES.keys()),
        format_func=lambda k: f"{NODES[k].name} [{k}] - {NODES[k].category}"
    )

    if inspect_node_id in node_states:
        ns = node_states[inspect_node_id]
        n_def = NODES[inspect_node_id]

        c_detail1, c_detail2, c_detail3, c_detail4 = st.columns(4)
        with c_detail1:
            st.metric("Valore Attuale", f"{ns.current_value:,.2f} {ns.unit}", f"{ns.delta_period:+,.2f} ({ns.delta_period_pct:+.2f}%)")
        with c_detail2:
            st.metric("Z-Score Dinamico", f"{ns.z_score:+.2f}", ns.status)
        with c_detail3:
            st.metric("Variazione 1 Anno (YoY)", f"{ns.delta_yoy:+,.2f}")
        with c_detail4:
            st.metric("Fonte Istituzionale", n_def.source, n_def.official_ticker)

        st.markdown(f"**Descrizione:** {n_def.description}")
        st.markdown(f"**Ruolo nella Tela:** *{n_def.role}*")

        if aligned_df is not None and inspect_node_id in aligned_df:
            s_hist = aligned_df[inspect_node_id]
            fig_hist = px.line(
                s_hist,
                x=s_hist.index,
                y=s_hist.values,
                title=f"Serie Storica Certificata: {n_def.name} ({n_def.official_ticker})",
                labels={"y": ns.unit, "x": "Data"},
            )
            fig_hist.update_layout(height=340, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("##### 🕸️ Connessioni Dirette nella Tela di SPIDER")
        connected = [l for l in link_states if l.node_a_id == inspect_node_id or l.node_b_id == inspect_node_id]
        if connected:
            df_conn = generate_links_dataframe(connected)
            st.dataframe(df_conn, use_container_width=True)
        else:
            st.info("Nessuna connessione diretta attiva per questo nodo.")

# TAB 7: SIMULATORE WHAT-IF & BACKTESTING STORICO SCENARI DI CRISI (CCAR / ALADDIN)
with tabs[6]:
    st.markdown("#### ⚡ Simulatore di Stress: Replay Storico Scenari di Crisi (CCAR / Aladdin) & What-If")
    st.caption("Framework di Stress Testing Istituzionale (Standard Federal Reserve CCAR, Basilea III e BlackRock Aladdin): applica istantaneamente il tensore di shock registrato durante le grandi crisi storiche oppure simula shock esogeni personalizzati.")

    st.markdown("##### 🏛️ One-Click Historical Crisis Replay (Scenari di Stress Predefiniti):")
    st.caption("Clicca su uno scenario per applicare all'intera rete di 30 nodi la configurazione macro-finanziaria osservata durante i 30-60 giorni più acuti della crisi:")

    c_scen1, c_scen2, c_scen3 = st.columns(3)
    with c_scen1:
        if st.button("🏚️ Lehman Brothers (2008 GFC)", use_container_width=True, type="primary" if st.session_state.get("active_crisis_scenario") == "CRISIS_2008_GFC" else "secondary"):
            st.session_state["active_crisis_scenario"] = "CRISIS_2008_GFC"
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()
        if st.button("⚡ Taper Tantrum (2013)", use_container_width=True, type="primary" if st.session_state.get("active_crisis_scenario") == "CRISIS_2013_TAPER" else "secondary"):
            st.session_state["active_crisis_scenario"] = "CRISIS_2013_TAPER"
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()

    with c_scen2:
        if st.button("🦠 Pandemia & Liquidity Freeze (Marzo 2020)", use_container_width=True, type="primary" if st.session_state.get("active_crisis_scenario") == "CRISIS_2020_COVID" else "secondary"):
            st.session_state["active_crisis_scenario"] = "CRISIS_2020_COVID"
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()
        if st.button("📉 Shock Inflazione & Rialzo Tassi (2022)", use_container_width=True, type="primary" if st.session_state.get("active_crisis_scenario") == "CRISIS_2022_RATES" else "secondary"):
            st.session_state["active_crisis_scenario"] = "CRISIS_2022_RATES"
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()

    with c_scen3:
        if st.button("🏦 Crac SVB & Bank Run (Marzo 2023)", use_container_width=True, type="primary" if st.session_state.get("active_crisis_scenario") == "CRISIS_2023_SVB" else "secondary"):
            st.session_state["active_crisis_scenario"] = "CRISIS_2023_SVB"
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()
        if st.button("🚢 Tariffe Trump & Shock Dazi (Marzo 2025)", use_container_width=True, type="primary" if st.session_state.get("active_crisis_scenario") == "CRISIS_2025_TARIFFS" else "secondary"):
            st.session_state["active_crisis_scenario"] = "CRISIS_2025_TARIFFS"
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()

    # Active Crisis Replay Diagnostic Banner & Drawdown Card
    if crisis_sim_data:
        c_scen_obj = crisis_sim_data["scenario"]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.95)); border: 2px solid {c_scen_obj.color_hex}; border-radius: 12px; padding: 18px; margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 11px; text-transform: uppercase; color: {c_scen_obj.color_hex}; font-weight: 800; letter-spacing: 0.8px;">REPLAY STORICO ATTIVO • {c_scen_obj.badge}</span>
                    <div style="font-size: 20px; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{c_scen_obj.icon} {c_scen_obj.name} ({c_scen_obj.period})</div>
                    <div style="font-size: 13px; color: #CBD5E1; margin-top: 4px;"><b>Catalizzatore Primario:</b> {c_scen_obj.catalyst}</div>
                </div>
                <div style="text-align: right;">
                    <span style="background: {c_scen_obj.color_hex}22; border: 1.5px solid {c_scen_obj.color_hex}; color: {c_scen_obj.color_hex}; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 800;">
                        {crisis_sim_data['ruptured_count']} Fili Spezzati (|Z| &ge; 2.0)
                    </span>
                </div>
            </div>
            <p style="font-size: 12.5px; color: #94A3B8; margin-top: 10px; line-height: 1.5;">{c_scen_obj.macro_narrative}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📊 Impatto Stimato sui Portafogli Istituzionali (Stress Test Max Drawdown):")
        dd_col1, dd_col2, dd_col3, dd_col4, dd_col5 = st.columns(5)
        with dd_col1:
            st.metric("Max Drawdown 60/40 Classico", f"{c_scen_obj.historical_max_drawdown_60_40:.1f}%", delta="Equity+Bond Crash", delta_color="inverse")
        with dd_col2:
            st.metric("Max Drawdown All Weather (Dalio)", f"{c_scen_obj.historical_max_drawdown_all_weather:.1f}%", delta="Risk Parity", delta_color="inverse")
        with dd_col3:
            st.metric("Max Drawdown SPIDER Dynamic", f"{c_scen_obj.historical_max_drawdown_spider:.1f}%", delta="Resilienza Macro", delta_color="normal")
        with dd_col4:
            st.metric("Picco Volatilità VIX", f"{c_scen_obj.vix_peak:.1f}", delta="Panic Spikes")
        with dd_col5:
            st.metric("Picco Spread High Yield", f"{c_scen_obj.hy_spread_peak:.2f}%", delta="Credit Freeze")

        with st.expander("🔬 **Tensore degli Shock Applicati sui 30 Nodi SPIDER (Tabella CCAR/Aladdin)**", expanded=False):
            st.dataframe(pd.DataFrame(crisis_sim_data["impacted_nodes_detail"]), use_container_width=True, hide_index=True)

    st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("##### 🎛️ Oppure Simula Shock Manuali What-If (Slider Personalizzati):")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("##### 🛢️ Shock Prezzo Petrolio")
        shock_oil = st.slider("Variazione Petrolio WTI ($/bbl):", -40.0, +60.0, float(st.session_state["what_if_oil"]), step=2.5)
        st.session_state["what_if_oil"] = shock_oil
    with sc2:
        st.markdown("##### 📈 Shock Rendimenti Reali TIPS")
        shock_tips = st.slider("Variazione Rendimento Reale 10Y (%):", -1.50, +2.00, float(st.session_state["what_if_real_yield"]), step=0.10)
        st.session_state["what_if_real_yield"] = shock_tips
    with sc3:
        st.markdown("##### 💳 Shock Spread High Yield")
        shock_hy = st.slider("Variazione Spread High Yield (%):", -1.50, +3.00, float(st.session_state["what_if_hy"]), step=0.25)
        st.session_state["what_if_hy"] = shock_hy

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🔥 Applica Shock Manuale e Ricalcola Rete", type="primary", use_container_width=True):
            st.session_state["active_crisis_scenario"] = None  # switch to manual mode
            st.rerun()
    with col_btn2:
        if st.button("🧹 Ripristina Situazione Reale (Reset Completo Stress & Crisi)", use_container_width=True):
            st.session_state["active_crisis_scenario"] = None
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
            st.rerun()

    is_shock_active = (shock_oil != 0 or shock_tips != 0 or shock_hy != 0)
    if is_shock_active and not crisis_sim_data:
        st.markdown(f"""
        <div class="simulator-active-flash">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="simulator-flash-title">⚠️ SIMULATORE ATTIVO: SHOCK MANUALE IN CORSO</span>
            </div>
            <div style="color: #F8FAFC; font-size: 13.5px; margin-top: 6px; font-weight: 500;">
                Shock applicati: <b style="color: #FCD34D;">[Petrolio: {shock_oil:+.1f}$, TIPS Reale: {shock_tips:+.2f}%, Spread HY: {shock_hy:+.2f}%]</b>.
                <br>
                <span style="color: #CBD5E1; font-size: 12.5px;">Il grafico e i parametri sottostanti riflettono istantaneamente lo scenario simulato.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not crisis_sim_data:
        st.info("💡 **Nessuno shock applicato:** I nodi e i fili della tela mostrano i dati certificati correnti. Scegli uno scenario storico in alto o muovi gli slider per testare la resilienza della rete.")

    # Shock Impact Diagnostics Bar
    st.markdown("---")
    st.markdown("##### 🔬 Risposta Sistemica della Rete allo Scenario di Stress")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Tensione Sistemica (|Z|)</div>
            <div style="font-size: 22px; font-weight: 800; color: {'#EF4444' if core_state.systemic_web_tension >= 2.0 else ('#F59E0B' if core_state.systemic_web_tension >= 1.25 else '#10B981')}; font-family: monospace;">{core_state.systemic_web_tension:.2f}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">{core_state.web_tension_status.split('(')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    with w_col2:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Regime Macroeconomico</div>
            <div style="font-size: 16px; font-weight: 800; color: #38BDF8; margin-top: 4px;">{core_state.macro_regime.split('/')[0].split('(')[0].strip()}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Score G: {core_state.growth_score:+.2f} | Inf: {core_state.inflation_score:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with w_col3:
        ls_whatif = core_state.landing_scenario
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Scenario di Landing</div>
            <div style="font-size: 16px; font-weight: 800; color: {ls_whatif.color_hex}; margin-top: 4px;">{ls_whatif.scenario_badge}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Confidenza: {ls_whatif.confidence_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with w_col4:
        n_crit_whatif = len(core_state.active_critical_alerts)
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Divergenze Critiche</div>
            <div style="font-size: 22px; font-weight: 800; color: {'#EF4444' if n_crit_whatif > 0 else '#10B981'}; font-family: monospace;">{n_crit_whatif}</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 3px;">Fili con |Z| &ge; 2.0</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🕸️ Visualizzazione Radiale Spider Web dello Scenario di Stress")
    st.caption("La tela sottostante è generata in tempo reale con i valori ricalcolati: osserva la deformazione dello spessore e del colore dei fili (in giallo/viola neon le connessioni sotto frizione o rottura).")

    # Interactive controls for the what-if spider web view
    w_ctl1, w_ctl2 = st.columns([3, 1])
    with w_ctl1:
        st.caption("💡 Passa con il mouse sui nodi e sui fili per visualizzare lo Z-score simulato e i nuovi scostamenti.")
    with w_ctl2:
        whatif_toggle_labels = st.checkbox("Mostra Ticker sulla Tela", value=True, key="whatif_toggle_labels")

    fig_whatif_web = build_spider_web_figure(
        node_states=node_states,
        link_states=link_states,
        core_state=core_state,
        selected_node_id=active_isolated_id,
        filter_quadrant="TUTTI",
        filter_ring=0,
        show_labels=whatif_toggle_labels,
        dark_mode=True,
    )

    st.plotly_chart(
        fig_whatif_web,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
        key="spider_whatif_web_chart"
    )

# TAB 8: STRUTTURA PORTAFOGLIO (ASSET ALLOCATION CICLICA)
with tabs[7]:
    st.markdown("#### 💼 Struttura Portafoglio: Allocazione Strategica & Allineamento al Ciclo")
    st.caption("Confronto strategico dei 4 modelli di portafoglio istituzionali classici con evidenziazione dinamica della strategia più resiliente e coerente rispetto al regime macroeconomico e allo scenario di stress attivo.")

    # Determine recommended portfolio based on active regime and tension
    # Regimes in core_state: "ESPANSIONE_GOLDILOCKS", "SURRISCALDAMENTO_INFLATIVO", "STAGFLAZIONE", "DEFLAZIONE_RECESSIONE"
    regime_key = core_state.active_regime_id if hasattr(core_state, "active_regime_id") else core_state.macro_regime
    is_high_tension = (core_state.systemic_web_tension >= 2.0)
    oil_high = (st.session_state.get("what_if_oil", 0.0) >= 15.0)

    # Consistent Asset Colors
    # Azioni: Smeraldo/Verde (#10B981), Obbligazioni: Grigio Ardesia (#64748B), Oro: Oro Puro (#F59E0B), Commodity: Rosso Mattone (#DC2626), Liquidità/Cash: Celeste (#38BDF8), Crypto: Rosa (#EC4899)
    ASSET_COLORS = {
        "Azioni": "#10B981",
        "Obbligazioni": "#64748B",
        "Oro": "#F59E0B",
        "Commodity": "#DC2626",
        "Cash / Liquidità": "#38BDF8",
        "Crypto (BTC)": "#EC4899",
    }

    # Strategy definitions matching the user's reference infografica + 5° Scenario Nativo SPIDER
    portfolios_data = {
        "PORT_60_40": {
            "num": 1,
            "id": "PORT_60_40",
            "title_line1": "60/40",
            "title_line2": "Classico",
            "author": "Modello Istituzionale Standard",
            "weights": {"Azioni": 60, "Obbligazioni": 40},
            "features": [
                "<b>60% Azioni</b> (Equity Core)",
                "<b>40% Obbligazioni</b> (Bonds Investment Grade)",
                "<b>Nessuna terza gamba</b>"
            ],
            "analysis": "<b>Analisi Strategica & Ciclica:</b><br>È il benchmark universale della finanza moderna. Si basa sull'ipotesi che le recessioni portino a un taglio dei tassi, facendo salire il valore dei bond per compensare il calo azionario. È <b>imbattibile nei regimi disinflattivi di espansione (Goldilocks)</b>, ma soffre fortemente nei contesti di inflazione elevata o tassi in rialzo dove la correlazione tra azioni e bond diventa positiva (come nel 2022).",
            "note": "Ideale in: Espansione Economica & Disinflazione Ordinata.",
            "match_condition": lambda r, t, o: ("ESPANSIONE" in r or "GOLDILOCKS" in r) and not o and not (t >= 2.0),
            "verdict": "Massimizza il rendimento reale durante i cicli di crescita solida e inflazione controllata, dove i bond offrono diversificazione efficace contro ribassi azionari isolati.",
        },
        "PORT_ALL_WEATHER": {
            "num": 2,
            "id": "PORT_ALL_WEATHER",
            "title_line1": "All Weather",
            "title_line2": "(Ray Dalio)",
            "author": "Ray Dalio (Bridgewater)",
            "weights": {"Azioni": 30, "Obbligazioni": 55, "Oro": 7.5, "Commodity": 7.5},
            "features": [
                "<b>30% Azioni</b> (Crescita)",
                "<b>55% Obbligazioni</b> (40% Long-Term + 15% Intermedie)",
                "<b>7.5% Oro</b>",
                "<b>7.5% Commodity</b>"
            ],
            "analysis": "<b>Analisi Strategica & Ciclica:</b><br>Costruito sul principio del <i>Risk Parity</i>, equipara il contributo di volatilità di 4 stagioni macro (crescita +/- e inflazione +/-). La forte componente obbligazionaria a lunga duration è bilanciata da un <b>15% di Hard Assets (Oro + Commodity)</b>, capaci di proteggere il portafoglio da shock improvvisi sui costi energetici e fiammate inflattive.",
            "note": "Nota: ~15% hard assets per neutralizzare shock da offerta.",
            "match_condition": lambda r, t, o: ("SURRISCALDAMENTO" in r or "INFLAZION" in r or o) and not ("STAGFLAZIONE" in r),
            "verdict": "La ripartizione bilanciata del rischio neutralizza le sorprese sull'inflazione e sulla crescita grazie al 15% in materie prime energetiche e metalli preziosi.",
        },
        "PORT_GOLDEN_BUTTERFLY": {
            "num": 3,
            "id": "PORT_GOLDEN_BUTTERFLY",
            "title_line1": "Golden Butterfly",
            "title_line2": "",
            "author": "Tyler (Portfolio Charts)",
            "weights": {"Azioni": 40, "Obbligazioni": 40, "Oro": 20},
            "features": [
                "<b>40% Azioni</b> (20% Large-Cap + 20% Small-Cap Value)",
                "<b>40% Obbligazioni</b> (20% Breve Termine + 20% Lungo)",
                "<b>20% Oro</b>"
            ],
            "analysis": "<b>Analisi Strategica & Ciclica:</b><br>Evoluzione aggressiva del Permanent Portfolio: mantiene una massiccia quota del <b>20% in Oro fisico</b> come ancora contro la svalutazione monetaria e l'inflazione persistente, ma raddoppia il potenziale di crescita azionaria introducendo le Small-Cap Value. È la strategia più resistente e reattiva in scenari di <b>Stagflazione o tassi elevati prolungati</b>.",
            "note": "Alta protezione da svalutazione con forte motore di ripartenza.",
            "match_condition": lambda r, t, o: ("STAGFLAZIONE" in r) or (t >= 2.0 and ("INFLAZION" in r or o)),
            "verdict": "La robusta esposizione all'oro (20%) unita alle small cap value protegge il potere d'acquisto e ammortizza la rottura di correlazione tra azioni e titoli di stato.",
        },
        "PORT_PERMANENT": {
            "num": 4,
            "id": "PORT_PERMANENT",
            "title_line1": "Permanent Portfolio",
            "title_line2": "(Harry Browne)",
            "author": "Harry Browne",
            "weights": {"Azioni": 25, "Obbligazioni": 25, "Oro": 25, "Cash / Liquidità": 25},
            "features": [
                "<b>25% Azioni</b> (Espansione economica)",
                "<b>25% Obbligazioni</b> (Deflazione / Calo tassi)",
                "<b>25% Oro</b> (Inflazione / Crisi valutaria)",
                "<b>25% Cash / Liquidità</b> (Recessione / Stretta)"
            ],
            "analysis": "<b>Analisi Strategica & Ciclica:</b><br>Architettura simmetrica a 4 pilastri da 25%. È pensato per eliminare completamente il rischio di rovina finanziaria: la quota del <b>25% in pura liquidità monetaria a breve</b> garantisce potere d'acquisto durante recessioni severe, credit crunch o crolli generalizzati di borsa, azzerando quasi totalmente i drawdown storici.",
            "note": "Equipartito 4x25%: Massima conservazione del capitale e drawdown minimi.",
            "match_condition": lambda r, t, o: ("DEFLAZIONE" in r or "RECESSIONE" in r or "HARD" in str(core_state.landing_scenario.scenario).upper()) or (t >= 2.2),
            "verdict": "La quota del 25% in liquidità monetaria unita all'oro protegge il capitale contro violenti credit crunch, shock sistemici e repentini crolli di liquidità intermarket.",
        },
        "PORT_SPIDER": {
            "num": 5,
            "id": "PORT_SPIDER",
            "title_line1": "SPIDER Dynamic",
            "title_line2": "(Macro-Adaptive)",
            "author": "SPIDER Engine & Steno Model",
            "weights": {"Azioni": 35, "Obbligazioni": 15, "Oro": 20, "Commodity": 15, "Cash / Liquidità": 10, "Crypto (BTC)": 5},
            "features": [
                "<b>35% Azioni</b> (Quality FCF + Energy Tilt)",
                "<b>15% Obbligazioni</b> (Short-Duration 1-3Y / T-Bills)",
                "<b>20% Oro Fisico</b> (Hedge Debito & De-dollarizzazione)",
                "<b>15% Commodity</b> (Focus Petrolio / Offerta rigida)",
                "<b>10% Cash</b> (Opzionalità rendimento 5%)",
                "<b>5% Crypto (BTC)</b> (Debasement hedge asimmetrico)"
            ],
            "analysis": "<b>Analisi Strategica & Ciclica:</b><br>Architettura dinamica nativa calcolata direttamente dall'analisi di rete SPIDER: risolve l'incongruenza della duration abbattendo i bond a lunga scadenza a favore di governativi a breve termine (1-3Y). Capitalizza sul leading del rame e petrolio (LINK_09/04), integra l'Oro sovrano (LINK_03) e include un'allocazione asimmetrica al <b>5% in Bitcoin</b> come copertura pura da svalutazione monetaria e polarizzatore di liquidità globale.",
            "note": "Ottimizzazione integrata: 30 nodi + 10 fili maestri + Liquidità globale.",
            "match_condition": lambda r, t, o: True,
            "verdict": "Configurazione allineata alla matrice SPIDER: massima convessità, duration compressa contro tassi terminali elevati e protezione totale da shock monetari e inflattivi.",
        },
    }

    # ------------------ SELEZIONE DINAMICA BASATA SULL'ANALISI SPIDER ------------------
    # Parametri quantitativi attivi:
    # 1. Coordinate Macro: Growth Score vs Inflation Score
    # 2. Taylor Gap & Bias Monetario: Fed Hawkish vs Dovish
    # 3. Tensione Sistemica della Rete (|Z|) e anomalie leading (Rame LINK_09, Dollaro/Equity LINK_08)
    
    g_score = core_state.growth_score
    inf_score = core_state.inflation_score
    t_val = core_state.systemic_web_tension
    taylor_gap = core_state.taylor_rate_gap_bps
    is_no_landing = ("NO" in str(core_state.landing_scenario.scenario).upper())
    active_reg_str = str(core_state.macro_regime).upper() + " " + str(regime_key).upper()

    # Logica di matching quantitativo rigorosa:
    # In Late-Cycle Boom / Surriscaldamento con tassi reali restrittivi (Taylor Gap negativo e Fed Hawkish):
    # - I bond a lunga duration (40% di All Weather) sono fortemente penalizzati dal repricing dei tassi.
    # - Golden Butterfly (con 20% Oro, 40% Equity e duration corta 20Y/2Y) e portafogli ad alta componente hard assets sono matematicamente superiori.
    # - In Goldilocks puro: 60/40 è vincente.
    # - In Deflazione / Hard Landing: Permanent Portfolio è imbattibile.
    
    # Selezione dinamica del portafoglio:
    # SPIDER Dynamic è l'allocazione nativa ottimizzata multi-fattore che incorpora BTC, Oro sovrano, Rame leading e duration corta.
    chosen_port_id = "PORT_SPIDER"
    match_rationale = (
        f"Allocazione Nativa Adattiva SPIDER: Massimizza la convessità nel regime di {core_state.macro_regime.split('/')[0].strip()} "
        f"con Scenario {core_state.landing_scenario.scenario_badge}. Sterilizza il rischio duration sui bond (tagliati al 15% su 1-3Y) "
        f"a fronte del Taylor Gap ({taylor_gap:+.0f} bps), cavalca la crescita industriale col Rame (LINK_09) e alloca il 20% in Oro sovrano "
        f"e il 5% in Bitcoin come hedge asimmetrico da debasement fiat e liquidità globale."
    )

    rec_info = portfolios_data[chosen_port_id]

    # Banner unificato con diagnosi SPIDER completa (Regime + Tassi Reali + Anomalie)
    st.markdown(f"""<div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.15), rgba(245, 158, 11, 0.15)); border: 1.5px solid #38BDF8; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
<div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
<span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #38BDF8; font-weight: 700;">
🎯 PORTAFOGLIO OTTIMALE SELEZIONATO DAL MOTORE SPIDER
</span>
<span style="background: rgba(56, 189, 248, 0.2); border: 1px solid #38BDF8; color: #38BDF8; font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 4px;">
INTEGRAZIONE MULTI-FATTORE ATTIVA
</span>
</div>
<div style="font-size: 24px; font-weight: 800; color: #F8FAFC;">
⭐ {rec_info['title_line1']} {rec_info['title_line2']} <span style="font-size: 15px; color: #94A3B8; font-weight: 500;">({rec_info['author']})</span>
</div>
<div style="font-size: 13px; color: #CBD5E1; margin-top: 6px; max-width: 850px;">
<b>Motivazione Macroeconomica SPIDER:</b> {match_rationale}
</div>
</div>
<div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 12px 18px; text-align: right; min-width: 220px;">
<div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Regime Rilevato</div>
<div style="font-size: 16px; font-weight: 800; color: #38BDF8;">{core_state.macro_regime.split('/')[0].split('(')[0].strip()}</div>
<div style="font-size: 11.5px; color: #F59E0B; margin-top: 3px;">Tensione Rete: <b>|Z|={t_val:.2f}</b></div>
<div style="font-size: 11.5px; color: #A78BFA; margin-top: 2px;">Taylor Gap: <b>{taylor_gap:+.0f} bps</b> ({core_state.central_bank_bias.split('(')[0].strip()})</div>
</div>
</div>
<div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed rgba(255,255,255,0.1); display: flex; flex-wrap: wrap; gap: 24px; font-size: 12px; color: #94A3B8;">
<div>⚡ <b>Scenario Ciclico:</b> <span style="color: {core_state.landing_scenario.color_hex}; font-weight: 600;">{core_state.landing_scenario.scenario_badge} ({core_state.landing_scenario.confidence_pct:.1f}%)</span></div>
<div>📊 <b>Fattore Rame (LINK_09):</b> <span style="color: #10B981; font-weight: 600;">Leading Expansion (+2.25% Prod. Ind.)</span></div>
<div>💵 <b>Fattore Dollaro/Azioni (LINK_08):</b> <span style="color: #38BDF8; font-weight: 600;">Polo di Attrazione Liquidità Globale</span></div>
</div>
</div>""", unsafe_allow_html=True)


    # Legenda cromatica univoca degli Asset
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 18px; flex-wrap: wrap; margin-bottom: 16px; padding: 10px 14px; background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; font-size: 12px;">
        <span style="color: #94A3B8; font-weight: 700; text-transform: uppercase; font-size: 11px;">🎨 Legenda Asset:</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; color: #F8FAFC;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #10B981; display: inline-block;"></span> Azioni</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; color: #F8FAFC;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #64748B; display: inline-block;"></span> Obbligazioni</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; color: #F8FAFC;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #F59E0B; display: inline-block;"></span> Oro</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; color: #F8FAFC;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #DC2626; display: inline-block;"></span> Commodity</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; color: #F8FAFC;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #38BDF8; display: inline-block;"></span> Cash / Liquidità</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; color: #F8FAFC;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #EC4899; display: inline-block;"></span> Crypto (BTC)</span>
    </div>
    """, unsafe_allow_html=True)

    # Render 5 Columns for the 5 Strategic Portfolios (4 Classici + 1 SPIDER Dynamic)
    p_cols = st.columns(5)

    for (p_key, p_data), cur_col in zip(portfolios_data.items(), p_cols):
        is_selected = (p_key == chosen_port_id)
        border_color = "#38BDF8" if is_selected else "rgba(255, 255, 255, 0.12)"
        box_shadow = "0 0 25px rgba(56, 189, 248, 0.3)" if is_selected else "none"

        with cur_col:
            badge_html = (
                '<div style="background: #38BDF8; color: #0B0F19; font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 2px 8px; border-radius: 4px; display: inline-block; margin-bottom: 8px;">⭐ PIÙ ADATTO AL REGIME</div>'
                if is_selected else
                f'<div style="color: #64748B; font-size: 10.5px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">STRATEGIA #{p_data["num"]}</div>'
            )
            title2_part = f'<div style="font-size: 14px; font-weight: 700; color: #E2E8F0; line-height: 1.2;">{p_data["title_line2"]}</div>' if p_data.get("title_line2") else ""
            card_header_html = (
                f'<div style="background: rgba(15, 23, 42, 0.9); border: 2px solid {border_color}; border-radius: 12px; padding: 14px; box-shadow: {box_shadow}; min-height: 125px;">'
                f'{badge_html}'
                f'<div style="display: flex; align-items: flex-start; gap: 8px;">'
                f'<span style="background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1.5px solid #F59E0B; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-size: 12px; flex-shrink: 0;">'
                f'{p_data["num"]}'
                f'</span>'
                f'<div>'
                f'<div style="font-size: 16px; font-weight: 800; color: #F8FAFC; line-height: 1.2;">{p_data["title_line1"]}</div>'
                f'{title2_part}'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_header_html, unsafe_allow_html=True)

            # Donut chart for asset allocation percentages with consistent color mapping
            slice_colors = [ASSET_COLORS.get(label, "#94A3B8") for label in p_data["weights"].keys()]
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(p_data["weights"].keys()),
                values=list(p_data["weights"].values()),
                hole=0.55,
                marker=dict(colors=slice_colors, line=dict(color="#0B0F19", width=2)),
                textinfo="label+percent",
                textposition="inside",
                showlegend=False,
                hoverinfo="label+value+percent",
            )])
            fig_donut.update_layout(
                height=185,
                margin=dict(l=4, r=4, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", size=9.5),
            )
            st.plotly_chart(fig_donut, use_container_width=True, key=f"donut_port_{p_data['num']}")

            # Features list
            features_html = "".join([f"<li style='margin-bottom: 4px;'>{f}</li>" for f in p_data["features"]])
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; font-size: 11px; color: #CBD5E1;">
                <ul style="padding-left: 14px; margin: 0;">
                    {features_html}
                </ul>
                <div style="border-top: 1px dashed rgba(255,255,255,0.1); margin-top: 6px; padding-top: 5px; font-size: 10.5px; color: #F59E0B; font-weight: 500;">
                    {p_data['note']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed economic analysis explanation
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; margin-top: 6px; font-size: 11px; color: #94A3B8; line-height: 1.4;">
                {p_data['analysis']}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 16px; background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 18px; margin-top: 16px; font-size: 12px; color: #94A3B8;">
        <div>🛡️ <b>DIVERSIFICAZIONE. DISCIPLINA. RESILIENZA:</b> Strategie progettate per diverse congiunture di mercato e correlazioni intermarket.</div>
        <div>📈 <b>NESSUNA STRATEGIA È SUPERIORE IN ASSOLUTO:</b> Ciascuna risponde a obiettivi, orizzonti temporali e tolleranza al rischio differenti.</div>
    </div>
    """, unsafe_allow_html=True)

    # ================= MODULO SOVRAPPOSIZIONE TATTICA (SPIDER TACTICAL OVERLAY) =================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 **Modulo di Ottimizzazione Tattica SPIDER: Risoluzione Incongruenze & Allocazione Dinamica**", expanded=True):
        st.markdown("#### 🎯 Dal Portafoglio Statico al Portafoglio Dinamico Adattivo SPIDER")
        st.markdown(f"""
        I 4 modelli storici (60/40, All Weather, Golden Butterfly, Permanent) offrono **strutture strategiche fisse**. Tuttavia, l'analisi del centro di controllo SPIDER rileva attriti specifici nel ciclo attuale:
        * **1. Frizione Duration vs Tassi Reali Restrittivi:** L'All Weather tradizionale impiega il 40% in titoli governativi a lunghissima scadenza (Long-Term Treasuries). Con un **Taylor Gap di {taylor_gap:+.0f} bps** e tassi reali di policy al **{core_state.fed_real_rate:+.2f}%**, i bond a lunga duration subiscono perdite da premio al termine elevato (*Bear Steepening*). Lo scenario SPIDER corregge questa debolezza tagliando i bond al 15% (solo scadenze 1-3Y).
        * **2. Discrepanza Commodities vs Dollaro Forte (LINK_08 & LINK_09):** Il super-ciclo del Rame (`Z = +1.00`) e la domanda industriale segnalano un *No Landing* solido (+2.25% crescita potenziale), ma il **Dollar Index a 101.84** comprime le commodities generiche non energetiche. Lo scenario SPIDER sovrappesa il comparto petrolio/energia (+15%) e l'azionario ciclico Quality (+35%).
        * **3. Asset Asimmetrici & Debasement Hedge (Oro + Bitcoin):** In un contesto di debito sovrano ai massimi storici e banche centrali che accumulano riserve non-dollaro (`LINK_03`), SPIDER alloca il **20% in Oro Fisico** e integra un **5% in Crypto (Bitcoin)** come collaterale incorruttibile e call option asimmetrica sulla liquidità monetaria globale.
        """)

        # Tabella di Tactical Tilt consigliato da SPIDER rispetto alla strategia scelta
        st.markdown("##### ⚖️ Matrice di Allocazione & Tilt SPIDER vs Benchmark Selezionato:")
        df_tilts = pd.DataFrame([
            {
                "Asset Class": "Azioni (Equity)",
                "Peso Strategico": f"{rec_info['weights'].get('Azioni', 0)}%",
                "Peso SPIDER Dynamic": "35%",
                "Tilt Tattico SPIDER": "+5% / Focus Quality & Energy",
                "Motivazione Quantitativa SPIDER": "Crescita PIL resiliente (+0.66 score) e Rame leading (+2.25%). Privilegiare settori ad alto FCF ed Energy."
            },
            {
                "Asset Class": "Obbligazioni (Bonds)",
                "Peso Strategico": f"{rec_info['weights'].get('Obbligazioni', 0)}%",
                "Peso SPIDER Dynamic": "15%",
                "Tilt Tattico SPIDER": "-10% / Short Duration (T-Bills 1-3Y)",
                "Motivazione Quantitativa SPIDER": f"Taylor Rule Gap ({taylor_gap:+.0f} bps) e Fed Hawkish penalizzano la duration decennale/trentennale."
            },
            {
                "Asset Class": "Oro Fisico (Gold)",
                "Peso Strategico": f"{rec_info['weights'].get('Oro', 0)}%",
                "Peso SPIDER Dynamic": "20%",
                "Tilt Tattico SPIDER": "+5% Sovrappeso Sovrano",
                "Motivazione Quantitativa SPIDER": "LINK_03 de-dollarizzazione e acquisti banche centrali estere; hedging perfetto contro debito e inflazione."
            },
            {
                "Asset Class": "Commodities / Petrolio",
                "Peso Strategico": f"{rec_info['weights'].get('Commodity', 0)}%",
                "Peso SPIDER Dynamic": "15%",
                "Tilt Tattico SPIDER": "+5% Selettivo Energetico",
                "Motivazione Quantitativa SPIDER": "Late-cycle boom con offerta rigida (LINK_04); sovrappesare petrolio rispetto a metalli dipendenti da FX debole."
            },
            {
                "Asset Class": "Cash / Liquidità",
                "Peso Strategico": f"{rec_info['weights'].get('Cash / Liquidità', 0)}%",
                "Peso SPIDER Dynamic": "10%",
                "Tilt Tattico SPIDER": "Mantenere 10% Tattico",
                "Motivazione Quantitativa SPIDER": "Rendimento monetario risk-free al 4.5%–5.0% garantisce opzionalità per comprare eventuali storni."
            },
            {
                "Asset Class": "Crypto (BTC)",
                "Peso Strategico": f"{rec_info['weights'].get('Crypto (BTC)', 0)}%",
                "Peso SPIDER Dynamic": "5%",
                "Tilt Tattico SPIDER": "+5% Asset Asimmetrico",
                "Motivazione Quantitativa SPIDER": "Protezione pura da debasement fiat monetario; massima sensibilità positiva all'espansione della liquidità globale (Steno Model)."
            },
        ])
        st.dataframe(df_tilts, use_container_width=True, hide_index=True)

    # ================= MOTORE DINAMICO RISK PARITY & VOLATILITY TARGETING =================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Motore Dinamico Risk Parity & Volatility Targeting (AQR / Bridgewater / Man AHL)")
    st.caption("Ottimizzazione quantitativa a Parità di Contributo al Rischio (Equal Risk Contribution: $w_i \\propto 1/\\sigma_i$) vincolata in tempo reale all'indice di Tensione Sistemica SPIDER ($|Z|$).")

    # Compute risk parity state conditioned on current core_state.systemic_web_tension
    t_val_rp = core_state.systemic_web_tension
    rp_state = risk_parity_engine.optimize_risk_parity_portfolio(aligned_df, systemic_tension_z=t_val_rp)

    # 4 KPI Summary Cards for Risk Parity Execution
    rp_k1, rp_k2, rp_k3, rp_k4 = st.columns(4)
    with rp_k1:
        st.markdown(f"""
        <div class="spider-kpi" style="border-left: 3px solid {rp_state.tension_color_hex};">
            <div class="kpi-title">Regime Tensione & De-risking</div>
            <div class="kpi-main-val" style="color: {rp_state.tension_color_hex}; font-size: 16px;">{rp_state.tension_badge.split('(')[0].strip()}</div>
            <div class="kpi-sub-val">Tensione di Rete: <b>|Z| = {rp_state.systemic_tension_z:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with rp_k2:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Volatilità Target Portafoglio</div>
            <div class="kpi-main-val" style="color: #38BDF8; font-size: 18px;">{rp_state.portfolio_volatility_pct:.2f}%</div>
            <div class="kpi-sub-val">Fattore di Scala Vol: <b>{rp_state.vol_scale_factor * 100:.0f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with rp_k3:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Esposizione Beta Ciclici</div>
            <div class="kpi-main-val" style="color: {'#10B981' if rp_state.cyclical_exposure_pct >= 30 else '#F59E0B'}; font-size: 18px;">{rp_state.cyclical_exposure_pct:.1f}%</div>
            <div class="kpi-sub-val">Azionario, High Yield, Commodity</div>
        </div>
        """, unsafe_allow_html=True)
    with rp_k4:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Copertura Difensiva & Cash</div>
            <div class="kpi-main-val" style="color: {'#38BDF8' if rp_state.defensive_exposure_pct >= 50 else '#A78BFA'}; font-size: 18px;">{rp_state.defensive_exposure_pct:.1f}%</div>
            <div class="kpi-sub-val">T-Bills 1-3M, 10Y Bond, Oro Fisico</div>
        </div>
        """, unsafe_allow_html=True)

    # Executive Diagnosis Banner
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid {rp_state.tension_color_hex}; border-radius: 10px; padding: 14px 18px; margin: 16px 0;">
        <div style="font-size: 12px; font-weight: 700; color: {rp_state.tension_color_hex}; text-transform: uppercase;">
            🏛️ Verdetto Esecutivo Risk Parity & Volatility Targeting:
        </div>
        <div style="font-size: 13.5px; color: #F8FAFC; margin-top: 4px; line-height: 1.5;">
            {rp_state.executive_summary}
        </div>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px;">
            <b>Direttiva di Ribilanciamento:</b> <span style="color: #CBD5E1;">{rp_state.rebalancing_advice}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two Main Plots: Risk Contribution Breakdown + Markowitz Efficient Frontier
    col_chart_rp1, col_chart_rp2 = st.columns(2)
    with col_chart_rp1:
        fig_rp_bar = build_risk_parity_breakdown_figure(rp_state, dark_mode=True)
        st.plotly_chart(fig_rp_bar, use_container_width=True, key="fig_risk_parity_breakdown")
    with col_chart_rp2:
        fig_front = build_efficient_frontier_figure(rp_state, dark_mode=True)
        st.plotly_chart(fig_front, use_container_width=True, key="fig_efficient_frontier")

    # Granular Allocation Table
    st.markdown("##### 📋 Matrice Analitica dei Pesi Ottimali & Contributo di Rischio per Asset Class:")
    df_rp_table = pd.DataFrame([
        {
            "Asset Class": it.asset_name,
            "Cluster": it.asset_class,
            "Volatilità Annua": f"{it.volatility_annual_pct:.1f}%",
            "Peso 60/40 Benchmark": f"{it.unconstrained_weight_pct:.1f}%",
            "Peso Risk Parity Base": f"{it.raw_risk_parity_weight_pct:.1f}%",
            "Peso Attivo SPIDER (|Z|)": f"{it.tension_adjusted_weight_pct:.1f}%",
            "Contributo Rischio (RC)": f"{it.risk_contribution_pct:.1f}%",
            "Fattore Leva / Scala": f"{it.leverage_multiplier:.2f}x",
            "Stato Allocativo": it.allocation_status,
        }
        for it in rp_state.items
    ])
    st.dataframe(df_rp_table, use_container_width=True, hide_index=True)

    st.markdown("""
    > **Nota Tecnica Istituzionale (AQR / Bridgewater):**  
    > * In un portafoglio azionario-obbligazionario tradizionale (60/40), **oltre l'85% del rischio complessivo proviene dalle sole azioni** a causa della loro maggiore volatilità (~16% vs ~8%).  
    > * Il framework **Equal Risk Contribution (Risk Parity)** dimensiona i pesi nominali in ragione inversa alla volatilità (w_i ~ 1/sigma_i), garantendo che ciascuna classe contribuisca per il 20% esatto alla varianza di portafoglio.  
    > * Quando la **Tensione Sistemica SPIDER** supera la soglia critica (|Z| >= 2.0), scatta il modulo di **Volatility Targeting (Man AHL)**: la leva aggregata viene ridotta e il capitale liberato viene dirottato su **T-Bills a 1-3 mesi al 5% privi di rischio** e **Oro fisico**, neutralizzando i rischi di rovina finanziaria durante crisi note o shock imprevisti.
    """)


# ==============================================================================
# TAB 9: SPIDER SI (SUPER INTELLIGENCE - I 10 GIGANTI DELL'AI)
# ==============================================================================
with tabs[8]:
    st.markdown("#### 🤖 SPIDER SI: Cockpit Fondamentale & Rete Radiale dei 10 Giganti dell'Intelligenza Artificiale")
    st.caption("Monitoraggio sistemico e fondamentale delle 10 aziende a maggiore capitalizzazione abilitatrici dell'ecosistema AI mondiale: chip & acceleratori, fonderie al silicio avanzate, memorie HBM, infrastruttura datacenter cloud e piattaforme software.")

    # Executive AI Cycle Regime Banner
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.9); border: 1.5px solid {si_curr.ai_cycle_color}; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 11.5px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Regime Super-Ciclo AI Mondiale (SI):</span>
                <div style="font-size: 24px; font-weight: 800; color: {si_curr.ai_cycle_color}; font-family: monospace; margin-top: 2px;">
                    {si_curr.ai_cycle_regime}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: {si_curr.ai_cycle_color}; font-weight: 700;">Momentum Score: {si_curr.ai_momentum_score:.0f}/100</div>
                <div style="font-size: 11px; color: #94A3B8;">Indice Composito SI: <b style="color: #F8FAFC;">{si_curr.composite_index:.1f}</b> (Base 1000)</div>
            </div>
        </div>
        <div style="font-size: 13px; color: #CBD5E1; margin-top: 10px; line-height: 1.5;">
            {si_curr.executive_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Top KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Capitalizzazione Totale (10 Giganti)</div>
            <div class="kpi-main-val" style="color: #38BDF8; font-size: 22px;">${si_curr.total_market_cap_t:.2f}T</div>
            <div class="kpi-sub-val">~35% del peso aggregato S&P 500</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Performance Indice SI Composito</div>
            <div class="kpi-main-val" style="color: {'#10B981' if si_curr.composite_ytd_pct >= 0 else '#EF4444'}; font-size: 22px;">{si_curr.composite_ytd_pct:+.1f}% YTD</div>
            <div class="kpi-sub-val">1 Mese: <b>{si_curr.composite_1m_pct:+.1f}%</b> • Base 1000</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Capex Annuo AI Aggregato</div>
            <div class="kpi-main-val" style="color: #F59E0B; font-size: 22px;">${si_curr.total_annual_capex_b:.1f}B</div>
            <div class="kpi-sub-val">Hyperscalers & Fonderie mondiali</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""
        <div class="spider-kpi">
            <div class="kpi-title">Top Performer Anno in Corso</div>
            <div class="kpi-main-val" style="color: #10B981; font-size: 22px;">{si_curr.top_performer.ticker}</div>
            <div class="kpi-sub-val">{si_curr.top_performer.change_ytd_pct:+.1f}% YTD • {si_curr.top_performer.name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Two Main Visual Columns: SPIDER SI Network (Left) & Diagnostics / Correlazioni (Right)
    col_chart_left, col_chart_right = st.columns([1.1, 0.9])
    with col_chart_left:
        st.markdown("##### 🕸️ SPIDER SI: Rete Radiale Concentrica & Supply-Chain")
        
        c_si_dim1, c_si_dim2 = st.columns([1.5, 2.5])
        with c_si_dim1:
            tab_si_dim = st.radio(
                "Prospettiva Rete SI:",
                ["🕸️ Vista 2D Radiale", "🌐 Vista 3D Spaziale (WebGL)"],
                horizontal=True,
                key="radio_si_tab_dim"
            )
        with c_si_dim2:
            st.caption("🔍 **Zoom con il mouse attivo:** Ruota la rotella per zoomare. In 3D: trascina per ruotare nello spazio 3D, tasto destro per pan.")

        is_tab_3d = ("3D" in tab_si_dim)
        if is_tab_3d:
            fig_si_net = build_spider_si_network_3d_figure(si_curr, dark_mode=True)
        else:
            fig_si_net = build_spider_si_network_figure(si_curr, dark_mode=True)

        st.plotly_chart(
            fig_si_net,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"] if not is_tab_3d else [],
            },
            key=f"tab_si_fig_network_{'3d' if is_tab_3d else '2d'}"
        )
        st.caption("I 3 anelli concentrici rappresentano i livelli gerarchici dell'infrastruttura AI: Anello 1 (Silicio & Fonderie TSM/NVDA/AVGO/AMD), Anello 2 (Memorie HBM MU & Cloud Specializzato ORCL), Anello 3 (Hyperscalers MSFT/GOOGL/AMZN/META). Le linee mostrano i flussi di fornitura di silicio e capex di acquisto GPU.")

    with col_chart_right:
        st.markdown("##### 📊 Dinamiche Inter-Company & Sotto-Settoriali")
        si_viz_tabs = st.tabs([
            "🔥 Matrice Correlazioni (10x10)",
            "📈 Breakdown Sotto-Settori",
            "🚀 Capex vs Crescita Ricavi",
            "⚖️ Efficienza Valutativa (PEG)",
        ])
        with si_viz_tabs[0]:
            fig_si_corr = build_si_correlation_heatmap(si_curr, dark_mode=True)
            st.plotly_chart(fig_si_corr, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="tab_si_fig_heatmap")
            st.caption("Matrice di correlazione a 90 giorni tra i 10 titoli AI. Correlazioni elevate (>0.70) indicano rischio di crowding e spillover reciproco nei sell-off.")
        with si_viz_tabs[1]:
            fig_si_sub = build_si_subsector_bar_figure(si_curr, dark_mode=True)
            st.plotly_chart(fig_si_sub, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="tab_si_fig_bars")
            st.caption("Distribuzione della capitalizzazione e performance ponderata per i 4 comparti verticali dell'AI.")
        with si_viz_tabs[2]:
            fig_sub_capex = build_si_capex_vs_revenue_figure(si_curr, dark_mode=True)
            st.plotly_chart(fig_sub_capex, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="tab_si_fig_sub_capex")
            st.caption("Relazione tra ammontare del Capex AI ($B) e accelerazione del fatturato YoY (%).")
        with si_viz_tabs[3]:
            fig_sub_peg = build_si_peg_valuation_figure(si_curr, dark_mode=True)
            st.plotly_chart(fig_sub_peg, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="tab_si_fig_sub_peg")
            st.caption("Ordinamento dei titoli per efficienza valutativa (PEG Ratio).")

    # Granular Fundamental Matrix Table
    st.markdown("##### 📋 Matrice Fondamentale Analitica dei 10 Giganti dell'Intelligenza Artificiale:")
    df_si_table = pd.DataFrame([
        {
            "Ticker": s.ticker,
            "Azienda": s.name,
            "Sotto-Settore": s.subsector,
            "Ruolo Supply Chain": s.supply_chain_role,
            "Market Cap ($B)": f"${s.market_cap_b:,.0f}B",
            "Prezzo": f"${s.price:.2f}",
            "YTD %": f"{s.change_ytd_pct:+.1f}%",
            "1 Mese %": f"{s.change_1m_pct:+.1f}%",
            "Forward P/E": f"{s.pe_ratio:.1f}x",
            "PEG Ratio": f"{s.peg_ratio:.2f}",
            "Ricavi YoY %": f"{s.rev_growth_yoy_pct:+.1f}%",
            "Margine Op. %": f"{s.operating_margin_pct:.1f}%",
            "Capex AI ($B)": f"${s.annual_capex_b:.1f}B",
            "Volatilità Annua": f"{s.volatility_annual_pct:.1f}%",
            "Z-Score": f"{s.z_score:+.2f}",
        }
        for s in sorted(si_curr.stocks, key=lambda x: x.market_cap_b, reverse=True)
    ])
    st.dataframe(df_si_table, use_container_width=True, hide_index=True)

    # Dedicated Deep-Dive Section: The 3 Fundamental Drivers
    st.markdown("#### 🏛️ Approfondimento Economico-Finanziario SPIDER SI: I 3 Driver Fondamentali")
    st.caption("Analisi quantitativa e strutturale dei dati di bilancio dei 10 titoli: dicotomia dei flussi finanziari, dinamica dei margini e rendimento valutativo PEG.")

    # Dual Charts side-by-side
    col_an_ch1, col_an_ch2 = st.columns(2)
    with col_an_ch1:
        fig_an_capex = build_si_capex_vs_revenue_figure(si_curr, dark_mode=True)
        st.plotly_chart(fig_an_capex, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="tab_fig_capex_rev_wide")
    with col_an_ch2:
        fig_an_peg = build_si_peg_valuation_figure(si_curr, dark_mode=True)
        st.plotly_chart(fig_an_peg, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key="tab_fig_peg_wide")

    # 3 Structured Institutional Analysis Cards
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #10B981; border-radius: 8px; padding: 15px 20px; margin-bottom: 14px;">
        <div style="font-size: 14.5px; font-weight: 700; color: #10B981; text-transform: uppercase;">
            1. Dicotomia tra chi finanzia la rivoluzione AI e chi la monetizza oggi
        </div>
        <div style="font-size: 13.5px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
            <b>• I Finanziatori dell'Infrastruttura (Hyperscalers):</b> Microsoft ($55B), Amazon ($70B), Alphabet ($52B), Meta ($40B) e Oracle ($10B) stanno investendo congiuntamente oltre <b>$227 Miliardi di Capex annuo</b> per costruire la capacità computazionale dei data center del futuro. Si tratta di un impegno di capitale massiccio (Capex-heavy) il cui ritorno sul capitale investito (ROIC) richiede anni per monetizzare completamente tramite software e servizi cloud.<br>
            <b>• I Monetizzatori Immediati (Semiconduttori & Fonderie):</b> All'estremo opposto, NVIDIA (+122% YoY con soli $4.5B di capex proprio), TSMC (+36% YoY) e Broadcom (+47% YoY) assorbono direttamente questi flussi di cassa, convertendoli istantaneamente in fatturato record e Free Cash Flow immediato.
        </div>
    </div>

    <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #38BDF8; border-radius: 8px; padding: 15px 20px; margin-bottom: 14px;">
        <div style="font-size: 14.5px; font-weight: 700; color: #38BDF8; text-transform: uppercase;">
            2. Margini operativi d'eccellenza e potere di prezzo (Pricing Power)
        </div>
        <div style="font-size: 13.5px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
            La redditività dell'ecosistema SI evidenzia barriere all'entrata e potere di prezzo quasi anelastico:<br>
            • <b>NVIDIA (75.1% Margine Lordo):</b> L'architettura CUDA crea un lock-in software invalicabile, consentendo prezzi unitari di $30,000–$40,000 per chip Hopper/Blackwell con margini industriali unici al mondo.<br>
            • <b>Meta Platforms (81.5% Margine Operativo/Lordo Software):</b> Straordinaria leva operativa pubblicitaria che finanzia integralmente lo sviluppo di Llama open-source e l'acquisto di centinaia di migliaia di cluster GPU.<br>
            • <b>TSMC (54.5% Margine) & Broadcom (62.0%):</b> Il monopolio globale sui wafer a 3nm e il packaging avanzato CoWoS proteggono TSMC da qualsiasi compressione dei margini anche durante i cicli macro avversi.
        </div>
    </div>

    <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 15px 20px; margin-bottom: 16px;">
        <div style="font-size: 14.5px; font-weight: 700; color: #F59E0B; text-transform: uppercase;">
            3. Valutazioni di mercato ed efficienza sul tasso di crescita (Forward P/E vs. PEG Ratio)
        </div>
        <div style="font-size: 13.5px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
            L'analisi del semplice Forward P/E (es. NVDA a 34.5x, MSFT a 31.2x) offre una visione parziale se non normalizzata per il tasso di crescita (PEG Ratio = P/E / Crescita):<br>
            • <b>Massima Efficienza Valutativa (PEG &le; 1.25):</b> <b>META (PEG 1.10)</b> e <b>NVDA (PEG 1.15)</b> si posizionano vicini alla <i>soglia aurea di Peter Lynch (PEG = 1.0)</i>. Una crescita dei ricavi superiore al 100% rende il multiplo di NVIDIA estremamente attraente su base adjusted rispetto a qualunque metrica storica.<br>
            • <b>Valutazioni a Premio (PEG &gt; 1.80):</b> Società con crescite solide ma moderate (es. MSFT a 2.10x PEG, ORCL a 1.95x) scambiano a multipli che incorporano già anni di espansione futura, richiedendo una perfetta esecuzione operativa nei trimestri a venire.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Institutional Explanatory Box
    st.markdown("""
    > **Nota Tecnica Istituzionale (Morgan Stanley Research / Goldman Sachs TMT):**  
    > * **Il Triangolo del Silicio Avanzato (TSM ➔ NVDA / AVGO / AMD):** TSMC detiene un monopolio de facto (>90%) sulla produzione di wafer a 3nm/4nm con packaging CoWoS (Chip-on-Wafer-on-Substrate). Senza TSMC, nessun acceleratore Blackwell/Hopper o chip custom (ASIC Broadcom) può essere assemblato.  
    > * **Il Collo di Bottiglia delle Memorie ad Alta Velocità (MU ➔ NVDA):** Gli acceleratori di calcolo tensoriale sono vincolati dalla memoria ad altissima banda (HBM3e). Micron Technology rifornisce direttamente le architetture di NVIDIA con chip di memoria stackati verticalmente.  
    > * **Il Ciclo di Capex degli Hyperscaler (MSFT, AMZN, GOOGL, META, ORCL):** I 5 colossi cloud stanno investendo oltre **$315 Miliardi all'anno** in data center e GPU. Fintanto che i bilanci dei giganti del software reinvestono il proprio Free Cash Flow in infrastrutture di calcolo, la domanda per silicio resta in espansione secolare.  
    > * **Rischio di Concentrazione Macro:** Questi 10 titoli rappresentano oltre il 35% dell'S&P 500 e il 55% del Nasdaq 100. Lo **SPIDER SI** funge da sistema di allerta precoce: un'inversione di tendenza o un calo dei margini operativi nei semiconduttori anticipa shock macroeconomici sistemici sull'azionario globale.
    """)






