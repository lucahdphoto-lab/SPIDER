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
    DEFAULT_TIME_WINDOW,
    MASTER_LINKS,
    NODES,
    QUADRANTS,
    RINGS,
    TIME_WINDOWS,
    Z_SCORE_CRITICAL,
    Z_SCORE_NORMAL,
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
from spider_ui import (
    build_macro_regime_matrix_figure,
    build_spider_web_figure,
    generate_links_dataframe,
    generate_nodes_dataframe,
)

# Page configuration
st.set_page_config(
    page_title="SPIDER • Macro & Market Network Monitor",
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

    .spider-kpi {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .kpi-main-val {
        font-size: 20px;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub-val {
        font-size: 11.5px;
        color: #64748B;
        margin-top: 3px;
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
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


@st.cache_data(ttl=900, show_spinner=False)
def get_computed_engine_data(window_key: str, use_spearman: bool, refresh_id: int):
    aligned_df = engine.load_and_align_data(force_refresh=(refresh_id > 0))
    node_states = engine.compute_node_states(window_key=window_key)
    link_states = engine.compute_link_states(node_states, window_key=window_key, use_spearman=use_spearman)
    core_state = engine.compute_spider_core(node_states, link_states)
    return node_states, link_states, core_state, aligned_df


refresh_count = st.session_state.get("refresh_count", 0)
with st.spinner("Allineamento serie storiche e calcolo correlazioni in corso..."):
    node_states, link_states, core_state, aligned_df = get_computed_engine_data(
        st.session_state["time_window"],
        st.session_state["use_spearman"],
        refresh_count
    )

if (st.session_state["what_if_oil"] != 0.0 or 
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


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 🕷️ **SPIDER CORE**")
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
    sel_quad = st.selectbox("Filtra per Quadrante Macro:", options=quad_opts, format_func=lambda k: quad_names[k])
    st.session_state["filter_quadrant"] = sel_quad

    ring_opts = [0, 1, 2, 3]
    ring_names = {0: "🌐 Tutti gli Anelli", **{k: f"{v['name']}: {v['label']}" for k, v in RINGS.items()}}
    sel_ring = st.selectbox("Filtra per Anello Concentrico:", options=ring_opts, format_func=lambda k: ring_names[k])
    st.session_state["filter_ring"] = sel_ring

    all_nodes_list = ["TUTTI (Visualizzazione Globale)", "SPIDER_CORE (Centro di Controllo)"] + list(NODES.keys())
    sel_node = st.selectbox(
        "Isola Nodo & Collegamenti (Focus):",
        options=all_nodes_list,
        index=all_nodes_list.index(st.session_state["selected_node"]) if st.session_state["selected_node"] in all_nodes_list else 0
    )
    st.session_state["selected_node"] = sel_node

    st.markdown("<hr style='margin: 14px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    col_ref1, col_ref2 = st.columns([1, 1])
    with col_ref1:
        if st.button("🔄 Aggiorna Dati", use_container_width=True, type="primary"):
            st.session_state["refresh_count"] = refresh_count + 1
            st.rerun()
    with col_ref2:
        if st.button("🧹 Reset Filtri", use_container_width=True):
            st.session_state["selected_node"] = "TUTTI (Visualizzazione Globale)"
            st.session_state["filter_quadrant"] = "TUTTI"
            st.session_state["filter_ring"] = 0
            st.session_state["what_if_oil"] = 0.0
            st.session_state["what_if_real_yield"] = 0.0
            st.session_state["what_if_hy"] = 0.0
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
            <span>🕷️ <b>SPIDER</b></span>
            <span class="spider-badge">Macro & Market Network Monitor</span>
        </div>
        {alert_badge_html}
    </div>
    <div style="color: #94A3B8; font-size: 13px; margin-top: 6px;">
        Centro di Controllo Macro-Finanziario Radiale • <b>30 Nodi Istituzionali</b> • <b>10 Fili Maestri di Trasmissione</b>
        <br>
        <b>Regime Attivo:</b> <span style="color: #38BDF8;">{core_state.macro_regime}</span> | 
        <b>Tensione Tela (|Z|):</b> <span style="color: #FBBF24;">{core_state.systemic_web_tension:.2f}</span> | 
        <b>Bias Banche Centrali:</b> <span style="color: #A78BFA;">{core_state.central_bank_bias}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if core_state.active_critical_alerts:
    with st.expander(f"⚠️ **ATTENZIONE: Rilevate {n_crit_alerts} divergenze anomale tra nodi macro** (Clicca per espandere)", expanded=True):
        for alt in core_state.active_critical_alerts:
            st.error(alt)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="spider-kpi">
        <div class="kpi-title">Regime Macroeconomico</div>
        <div class="kpi-main-val" style="font-size: 16px; color: #38BDF8;">{core_state.macro_regime.split('/')[0]}</div>
        <div class="kpi-sub-val">Growth: {core_state.growth_score:+.2f} | Inf: {core_state.inflation_score:+.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="spider-kpi">
        <div class="kpi-title">Tensione Sistemica Tela</div>
        <div class="kpi-main-val" style="color: {'#EF4444' if core_state.systemic_web_tension >= 2.0 else ('#F59E0B' if core_state.systemic_web_tension >= 1.25 else '#10B981')};">{core_state.systemic_web_tension:.2f}</div>
        <div class="kpi-sub-val">{core_state.web_tension_status.split('(')[0]}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="spider-kpi">
        <div class="kpi-title">Bias Banche Centrali</div>
        <div class="kpi-main-val" style="font-size: 16px; color: #A78BFA;">{core_state.central_bank_bias.split('(')[0]}</div>
        <div class="kpi-sub-val">FED Reale: {core_state.fed_real_rate:+.2f}% | BCE: {core_state.ecb_real_rate:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="spider-kpi">
        <div class="kpi-title">Taylor Rule Gap</div>
        <div class="kpi-main-val" style="color: {'#10B981' if core_state.taylor_rate_gap_bps >= 0 else '#EF4444'};">{core_state.taylor_rate_gap_bps:+.0f} bps</div>
        <div class="kpi-sub-val">Tasso Reale Restrittivo</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="spider-kpi">
        <div class="kpi-title">Nodi Attivi / Monitorati</div>
        <div class="kpi-main-val" style="color: #F8FAFC;">{len(node_states)} / 30</div>
        <div class="kpi-sub-val">10 Fili Maestri + 16 Spokes</div>
    </div>
    """, unsafe_allow_html=True)


tabs = st.tabs([
    "🕸️ Tela di Ragno (Spider Web Graph)",
    "🎯 Centro di Controllo (Macro Matrix)",
    "🔄 Teoria dei Cicli di Martin Pring",
    "📋 Supervisione Tabellare & Export",
    "🔬 Focus Nodo & Relazioni",
    "⚡ Simulatore Stress What-If"
])

# TAB 1
with tabs[0]:
    st.markdown("#### 🕸️ Visualizzazione Radiale Concentrica a Tela di Ragno")
    st.caption("I cerchi concentrici separano Politica Monetaria (Anello 1), Fondamentali Macro (Anello 2) e Asset Intermarket (Anello 3). Lo spessore e il colore dei fili indicano la forza e la tensione della relazione.")

    col_ctl1, col_ctl2, col_ctl3 = st.columns([2, 1, 1])
    with col_ctl1:
        if active_isolated_id:
            st.info(f"🔍 **Modalità Focus Attiva:** Isolamento del nodo `{active_isolated_id}` ({node_states[active_isolated_id].name if active_isolated_id in node_states else ''}) e di tutte le sue connessioni dirette.")
        else:
            st.write("Passa il mouse su qualsiasi nodo o filo per visualizzare la scheda analitica istituzionale.")
    with col_ctl2:
        toggle_labels = st.checkbox("Mostra Etichette Ticker", value=True)
    with col_ctl3:
        if active_isolated_id and st.button("Mostra Tutta la Tela"):
            st.session_state["selected_node"] = "TUTTI (Visualizzazione Globale)"
            st.rerun()

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
    st.plotly_chart(fig_web, use_container_width=True)

    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 16px; font-size: 12px; color: #94A3B8; display: flex; flex-wrap: wrap; gap: 20px;">
        <div>🟢 <b>Verde Smeraldo:</b> Correlazione Positiva Diretta Normale</div>
        <div>🔴 <b>Rosso Corallo:</b> Correlazione Inversa Normale</div>
        <div>🟡 <b>Giallo Ambra:</b> Tensione Moderata (|Z| ≥ 1.25)</div>
        <div>🟣 <b>Viola Neon Pulsante:</b> Rottura / Divergenza Critica (|Z| ≥ 2.0)</div>
        <div><b>Spessore Filo:</b> Proporzionale all'intensità del legame rolling (|r|)</div>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("##### 📊 Diagnosi Istituzionale del Regime")
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 18px; margin-bottom: 16px;">
            <div style="font-size: 13px; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Regime Corrente Certificato</div>
            <div style="font-size: 22px; font-weight: 800; color: #38BDF8; margin-top: 4px;">{core_state.macro_regime}</div>
            <p style="font-size: 13px; color: #CBD5E1; margin-top: 8px; line-height: 1.5;">{core_state.macro_regime_desc}</p>
        </div>
        """, unsafe_allow_html=True)

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

# TAB 4: SUPERVISIONE TABELLARE
with tabs[3]:
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

# TAB 5
with tabs[4]:
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

# TAB 6
with tabs[5]:
    st.markdown("#### ⚡ Simulatore What-If: Test di Stress e Rottura Relazioni")
    st.caption("Applica shock esogeni alle variabili chiave per testare istantaneamente la reazione della tensione della tela e degli Z-score (senza alterare i dati ufficiali salvati nel database).")

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

    if st.button("🔥 Applica Shock e Ricalcola Tensione Tela", type="primary"):
        st.rerun()

    if (shock_oil != 0 or shock_tips != 0 or shock_hy != 0):
        st.warning(f"⚠️ **Simulatore Attivo:** Shock in corso [Petrolio: {shock_oil:+.1f}$, TIPS Reale: {shock_tips:+.2f}%, Spread HY: {shock_hy:+.2f}%]. Clicca su 'Reset Filtri' nella sidebar per tornare ai dati certificati.")
