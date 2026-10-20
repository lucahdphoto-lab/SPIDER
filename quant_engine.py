"""
SPIDER Quantitative Engine:
Mathematical modeling of macro-financial relationships, mixed-frequency time series alignment,
rolling Pearson/Spearman correlations, link tension Z-scores, Macro Regime Matrix,
Systemic Web Tension, and Central Bank Policy Bias.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DEFAULT_TIME_WINDOW,
    MASTER_LINKS,
    NODES,
    RINGS,
    SECONDARY_SPOKES,
    TIME_WINDOWS,
    Z_SCORE_CRITICAL,
    Z_SCORE_NORMAL,
    MasterLinkDef,
    NodeDef,
)
from data_fetcher import db, fetcher


@dataclass
class NodeState:
    id: str
    name: str
    ring: int
    quadrant: str
    category: str
    official_ticker: str
    source: str
    frequency: str
    unit: str
    role: str
    current_value: float
    prev_value: float
    delta_period: float
    delta_period_pct: float
    delta_yoy: float
    z_score: float
    mean_val: float
    std_val: float
    status: str
    last_timestamp: str
    sparkline: List[float]


@dataclass
class LinkState:
    id: str
    node_a_id: str
    node_b_id: str
    node_a_name: str
    node_b_name: str
    node_a_val: float
    node_b_val: float
    node_a_delta: float
    node_b_delta: float
    theoretical_corr: int
    rolling_pearson: float
    rolling_spearman: float
    active_corr: float
    z_score_tension: float
    status: str
    color_hex: str
    line_width: float
    line_dash: str
    mechanism: str
    alert_description: str
    is_master: bool
    weight: float


@dataclass
class LandingScenarioState:
    scenario: str              # "NO_LANDING", "SOFT_LANDING", "HARD_LANDING"
    scenario_title: str        # e.g., "No Landing (Riacidificazione Inflattiva & Crescita Forte)"
    scenario_badge: str        # e.g., "🔥 NO LANDING"
    color_hex: str             # "#F59E0B" (orange/amber) for No Landing, "#10B981" for Soft, "#EF4444" for Hard
    confidence_pct: float      # e.g., 78.5%
    prob_no_landing: float     # 0-100
    prob_soft_landing: float   # 0-100
    prob_hard_landing: float   # 0-100
    key_drivers: List[str]     # Key empirical observations justifying the diagnosis
    market_implication: str    # Summary of investment implications
    policy_reaction: str       # Anticipated Fed/central bank move


@dataclass
class CISSState:
    """
    ECB CISS (Composite Indicator of Systemic Stress) & Non-Linear Contagion State.
    Aggregates sub-indices of systemic stress across 4 macro quadrants using portfolio theory:
    CISS = s' * C * s = sum_i sum_j w_i * w_j * rho_ij * s_i * s_j
    """
    ciss_score: float                   # Overall index in [0, 1]
    ciss_status: str                    # "Calma Fisiologica", "Allerta Settoriale", "Contagio Sistemico Severo"
    status_color: str                   # "#10B981", "#F59E0B", "#EF4444"
    quadrant_stress: Dict[str, float]   # Stress sub-index s_i in [0, 1] for each quadrant
    weights: Dict[str, float]           # Weights w_i summing to 1.0
    corr_matrix: Dict[str, Dict[str, float]] # Dynamic rolling correlation matrix C_t
    idiosyncratic_risk: float           # Risk assuming zero cross-correlation: sum(w_i^2 * s_i^2)
    contagion_effect: float             # Cross-market contagion term: sum_{i != j} w_i * w_j * rho_ij * s_i * s_j
    contagion_contribution_pct: float   # Percentage of total CISS variance explained by systemic contagion


@dataclass
class SpiderCoreState:
    macro_regime: str
    macro_regime_desc: str
    growth_score: float
    inflation_score: float
    regime_coords: Tuple[float, float]
    trajectory_coords: List[Tuple[str, float, float]]
    systemic_web_tension: float
    web_tension_status: str
    central_bank_bias: str
    fed_real_rate: float
    ecb_real_rate: float
    taylor_rate_gap_bps: float
    active_critical_alerts: List[str]
    landing_scenario: LandingScenarioState
    ciss: CISSState


class SpiderQuantEngine:
    """Quantitative engine performing all statistical and econometric computations."""

    def __init__(self):
        self.raw_series: Dict[str, pd.Series] = {}
        self.aligned_df: pd.DataFrame = pd.DataFrame()
        self.transformed_df: pd.DataFrame = pd.DataFrame()

    def load_and_align_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Align all 30 peripheral series on a unified business calendar without look-ahead bias."""
        self.raw_series = fetcher.fetch_all_nodes(force_refresh=force_refresh)

        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=5)
        master_calendar = pd.date_range(start=start_date, end=end_date, freq="B")

        df_aligned = pd.DataFrame(index=master_calendar)
        df_trans = pd.DataFrame(index=master_calendar)

        for n_id, n_def in NODES.items():
            s = self.raw_series.get(n_id)
            if s is None or s.empty:
                continue

            s = s[~s.index.duplicated(keep="last")].sort_index()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)

            # Align on master calendar using forward-fill
            s_reindexed = s.reindex(master_calendar.union(s.index)).sort_index()
            s_ffilled = s_reindexed.ffill().reindex(master_calendar)
            s_clean = s_ffilled.bfill()

            df_aligned[n_id] = s_clean

            # Economic stationary transformations
            if n_def.transform == "yoy":
                if n_id in ["INF_USCPI", "INF_USPCE", "INF_EZHICP"]:
                    s_yoy = s_clean.pct_change(periods=252) * 100.0
                    s_yoy = s_yoy.fillna(s_clean if s_clean.mean() < 20 else 2.5)
                    df_trans[n_id] = s_yoy
                else:
                    df_trans[n_id] = s_clean.pct_change(periods=252) * 100.0
            elif n_def.transform == "mom":
                df_trans[n_id] = s_clean.diff(periods=21).fillna(0.0)
            else:
                df_trans[n_id] = s_clean

        self.aligned_df = df_aligned.ffill().bfill()
        self.transformed_df = df_trans.ffill().bfill()
        return self.aligned_df

    def compute_node_states(self, window_key: str = DEFAULT_TIME_WINDOW) -> Dict[str, NodeState]:
        """Compute rolling Z-scores, period deltas, YoY changes, and status for each node."""
        if self.aligned_df.empty:
            self.load_and_align_data()

        days = TIME_WINDOWS.get(window_key, TIME_WINDOWS[DEFAULT_TIME_WINDOW])["days"]
        lookback_pts = max(15, int(days * (252 / 365)))
        recent_df = self.transformed_df.iloc[-lookback_pts:]
        full_df = self.transformed_df

        node_states: Dict[str, NodeState] = {}
        cached_meta = db.get_all_metadata()

        for n_id, n_def in NODES.items():
            s_full = full_df[n_id] if n_id in full_df else pd.Series(dtype=float)
            s_recent = recent_df[n_id] if n_id in recent_df else s_full

            if s_full.empty:
                continue

            curr_val = float(s_full.iloc[-1])
            prev_val = float(s_full.iloc[-2]) if len(s_full) > 1 else curr_val
            delta_period = curr_val - prev_val
            delta_period_pct = (delta_period / abs(prev_val) * 100.0) if prev_val != 0 else 0.0

            yoy_pts = min(252, len(s_full) - 1)
            val_1y_ago = float(s_full.iloc[-yoy_pts]) if yoy_pts > 0 else curr_val
            delta_yoy = curr_val - val_1y_ago

            # Econometrically calibrated lookback for Z-scores:
            # - Ring 1 & 2 (Monetary policy & Macro): 3Y structural lookback (756 business days)
            # - Ring 3 (Equities, FX, Commodities, Spreads): 1Y intermediate lookback (252 business days)
            # This avoids short-term window compression where high-flying assets get negative Z-scores.
            if n_def.ring in [1, 2]:
                hist_lookback = min(len(s_full), 756)
                s_baseline = s_full.iloc[-hist_lookback:]
                mean_val = float(s_baseline.mean())
                std_val = float(s_baseline.std())
            else:
                lookback_len = min(len(s_full), 252)
                s_baseline = s_full.iloc[-lookback_len:]
                mean_val = float(s_baseline.mean())
                std_val = float(s_baseline.std())

            if std_val > 1e-4:
                z_score = (curr_val - mean_val) / std_val
            else:
                z_score = 0.0

            abs_z = abs(z_score)
            if abs_z >= Z_SCORE_CRITICAL:
                status = "Estremo"
            elif abs_z >= Z_SCORE_NORMAL:
                status = "Attenzione"
            else:
                status = "Normale"

            meta = cached_meta.get(n_id, {})
            last_ts = meta.get("last_data_point_date", str(s_full.index[-1].strftime("%Y-%m-%d")))
            spark_series = s_full.iloc[-30:].tolist()

            node_states[n_id] = NodeState(
                id=n_id,
                name=n_def.name,
                ring=n_def.ring,
                quadrant=n_def.quadrant,
                category=n_def.category,
                official_ticker=n_def.official_ticker,
                source=n_def.source,
                frequency=n_def.frequency,
                unit=n_def.unit,
                role=n_def.role,
                current_value=curr_val,
                prev_value=prev_val,
                delta_period=delta_period,
                delta_period_pct=delta_period_pct,
                delta_yoy=delta_yoy,
                z_score=float(np.clip(z_score, -5.0, 5.0)),
                mean_val=mean_val,
                std_val=std_val,
                status=status,
                last_timestamp=last_ts,
                sparkline=spark_series,
            )

        return node_states

    def compute_link_states(
        self,
        node_states: Dict[str, NodeState],
        window_key: str = DEFAULT_TIME_WINDOW,
        use_spearman: bool = False
    ) -> List[LinkState]:
        """Compute rolling correlation and tension Z-score for all master links and spokes."""
        if self.transformed_df.empty:
            self.load_and_align_data()

        days = TIME_WINDOWS.get(window_key, TIME_WINDOWS[DEFAULT_TIME_WINDOW])["days"]
        lookback_pts = max(20, int(days * (252 / 365)))
        recent_df = self.transformed_df.iloc[-lookback_pts:]

        link_states: List[LinkState] = []

        # 1. Master Links
        for link_id, m_def in MASTER_LINKS.items():
            nA_id, nB_id = m_def.node_a, m_def.node_b
            if nA_id not in recent_df or nB_id not in recent_df:
                continue

            sA = recent_df[nA_id]
            sB = recent_df[nB_id]

            r_pearson, _ = stats.pearsonr(sA, sB) if len(sA) > 3 and sA.std() > 0 and sB.std() > 0 else (0.0, 1.0)
            r_spearman, _ = stats.spearmanr(sA, sB) if len(sA) > 3 and sA.std() > 0 and sB.std() > 0 else (0.0, 1.0)
            active_r = r_spearman if use_spearman else r_pearson

            zA = node_states[nA_id].z_score if nA_id in node_states else 0.0
            zB = node_states[nB_id].z_score if nB_id in node_states else 0.0
            theo = m_def.theoretical_corr

            if theo == +1:
                dir_divergence = abs(zA - zB) / np.sqrt(2.0 * max(0.1, 1.0 - active_r))
            else:
                dir_divergence = abs(zA + zB) / np.sqrt(2.0 * max(0.1, 1.0 + active_r))

            corr_gap = max(0.0, (theo - active_r) if theo == +1 else (active_r - theo))
            z_tension = float(np.clip(0.6 * dir_divergence + 0.8 * corr_gap, 0.0, 6.0))

            if z_tension >= Z_SCORE_CRITICAL:
                status = "Divergenza Critica"
                color_hex = "#D946EF"
                line_width = 3.8
                line_dash = "solid"
            elif z_tension >= Z_SCORE_NORMAL:
                status = "Tensione Moderata"
                color_hex = "#F59E0B"
                line_width = 2.8
                line_dash = "dash"
            else:
                status = "Normale"
                color_hex = "#10B981" if active_r >= 0 else "#EF4444"
                line_width = max(1.2, float(abs(active_r) * 3.5))
                line_dash = "solid" if abs(active_r) > 0.25 else "dot"

            # Contextual Macro Diagnostic Generator:
            # Replaces generic hypothetical text with exact empirical status of the link
            dynamic_alert = m_def.alert_description
            if link_id == "LINK_01":
                if zA > 0.5 and zB < -0.5:
                    dynamic_alert = f"Divergenza Recessiva: Inflazione elevata ({node_states[nA_id].current_value:.2f}%) con crollo rendimenti 2Y ({node_states[nB_id].current_value:.2f}%) segnala aspettative di pivot forzato."
                elif zA > 0.0 and zB > 0.0:
                    dynamic_alert = f"Coerenza Restrittiva: Inflazione sopra target ({node_states[nA_id].current_value:.2f}%) mantiene elevati i rendimenti a 2Y ({node_states[nB_id].current_value:.2f}%)."
                else:
                    dynamic_alert = f"Divergenza empirica su Inflazione Core ({node_states[nA_id].current_value:.2f}%) vs Treasury 2Y ({node_states[nB_id].current_value:.2f}%)."
            elif link_id == "LINK_02":
                if zA > 0.5 and zB > 0.5:
                    dynamic_alert = f"Divergenza Euforica: Rendimenti reali alti ({node_states[nA_id].current_value:.2f}%) e Nasdaq ai massimi, multipli a rischio di violento repricing."
                elif zA > 0.5 and zB <= 0.5:
                    dynamic_alert = f"Pressione Valutazioni: Rendimenti reali restrittivi ({node_states[nA_id].current_value:.2f}%) frenano l'espansione dei multipli Big Tech."
                else:
                    dynamic_alert = f"Dinamica Tassi Reali ({node_states[nA_id].current_value:.2f}%) vs Nasdaq ({node_states[nB_id].current_value:,.0f})."
            elif link_id == "LINK_03":
                if zA > 0.5 and zB > 0.5:
                    dynamic_alert = f"Accumulazione Istituzionale: Oro in rally (${node_states[nB_id].current_value:,.0f}/oz) nonostante tassi reali positivi ({node_states[nA_id].current_value:.2f}%), domanda di protezione sovrana/de-dollarizzazione."
                elif zA > 0.5 and zB <= 0.5:
                    dynamic_alert = f"Concorrenza Cedolare: Tassi reali positivi ({node_states[nA_id].current_value:.2f}%) aumentano il costo opportunità di detenere oro."
                else:
                    dynamic_alert = f"Dinamica TIPS Reali ({node_states[nA_id].current_value:.2f}%) vs Oro (${node_states[nB_id].current_value:,.0f}/oz)."
            elif link_id == "LINK_04":
                if zA < -0.5 and zB > 0.5:
                    dynamic_alert = f"Inflazione Strutturale: Petrolio debole (${node_states[nA_id].current_value:.2f}) ma aspettative d'inflazione elevate ({node_states[nB_id].current_value:.2f}%), pressioni salariali radicate."
                elif zA > 0.5 and zB > 0.5:
                    dynamic_alert = f"Spinta Energetica: Petrolio elevato (${node_states[nA_id].current_value:.2f}/bbl) sostiene le aspettative d'inflazione forward ({node_states[nB_id].current_value:.2f}%)."
                else:
                    dynamic_alert = f"Trasmissione Petrolio (${node_states[nA_id].current_value:.2f}) su Aspettative 5Y5Y ({node_states[nB_id].current_value:.2f}%)."
            elif link_id == "LINK_05":
                if zA > 0.5 and zB < 0.0:
                    dynamic_alert = f"Curva Dietro la Curva: Sussidi disoccupazione in rialzo ({node_states[nA_id].current_value:,.0f}) con curva ancora compressa/invertita."
                elif zB > 0.0:
                    dynamic_alert = f"Normalizzazione Pendenza: Curva positiva (+{node_states[nB_id].current_value:.0f} bps) coerente con sussidi disoccupazione stabili ({node_states[nA_id].current_value:,.0f})."
                else:
                    dynamic_alert = f"Dinamica Sussidi ({node_states[nA_id].current_value:,.0f}) vs Spread 10Y-2Y ({node_states[nB_id].current_value:+.0f} bps)."
            elif link_id == "LINK_06":
                if zA > 0.5 and zB > 0.5:
                    dynamic_alert = f"Fragilità Creditizia: Small Cap in rally mentre gli spread High Yield si allargano a {node_states[nA_id].current_value:.2f}%."
                elif zA <= 0.0:
                    dynamic_alert = f"Condizioni Credito Distese: Spread High Yield compressi ({node_states[nA_id].current_value:.2f}%), assenza di stress creditizio sistemico per le small cap."
                else:
                    dynamic_alert = f"Divergenza Spread HY ({node_states[nA_id].current_value:.2f}%) vs Russell 2000 ({node_states[nB_id].current_value:,.0f})."
            elif link_id == "LINK_07":
                dynamic_alert = f"Differenziale Tassi: US 2Y al {node_states[nA_id].current_value:.2f}% guida i flussi sul dollaro con EUR/USD a {node_states[nB_id].current_value:.4f}."
            elif link_id == "LINK_08":
                if zA > 0.5 and zB > 0.5:
                    dynamic_alert = f"Divergenza Dollaro/Azionario: Dollar Index DXY tonico ({node_states[nA_id].current_value:.2f}) e S&P 500 ai massimi ({node_states[nB_id].current_value:,.0f}), gli asset USA agiscono da polo di attrazione globale della liquidità."
                elif zA > 0.5 and zB < -0.5:
                    dynamic_alert = f"Restrizione Finanziaria: Dollaro forte ({node_states[nA_id].current_value:.2f}) deprime i multipli e gli utili esteri dell'azionario."
                else:
                    dynamic_alert = f"Condizioni Finanziarie: Dollar Index DXY a {node_states[nA_id].current_value:.2f} vs S&P 500 a {node_states[nB_id].current_value:,.0f}."
            elif link_id == "LINK_09":
                if zA < -0.8 and zB > 0.5:
                    dynamic_alert = f"Allarme Rallentamento: Rame in forte ribasso (${node_states[nA_id].current_value:.2f}) anticipa un ciclo manifatturiero in frenata."
                elif zA > 0.5 and zB <= 0.0:
                    dynamic_alert = f"Divergenza Anticipatoria: Rame tonico (${node_states[nA_id].current_value:.2f}/lb, Z={zA:+.2f}) anticipa una riaccelerazione manifatturiera non ancora visibile nei dati industriali storici (+{node_states[nB_id].current_value:.2f}% YoY)."
                elif zA >= 0.0 and zB >= 0.0:
                    dynamic_alert = f"Ciclo Manifatturiero Espansivo: Rame tonico (${node_states[nA_id].current_value:.2f}/lb) coerente con produzione industriale in espansione (+{node_states[nB_id].current_value:.2f}% YoY)."
                else:
                    dynamic_alert = f"Dinamica Rame (${node_states[nA_id].current_value:.2f}/lb) vs Produzione Industriale (+{node_states[nB_id].current_value:.2f}% YoY)."
            elif link_id == "LINK_10":
                if zA > 0.5 and zB > 0.5:
                    dynamic_alert = f"Rischio Asimmetrico: FTSE MIB sale nonostante allargamento anomalo dello spread BTP/Bund ({node_states[nA_id].current_value:.0f} bps)."
                elif zA <= 0.0:
                    dynamic_alert = f"Stabilità Sovrana: Spread BTP/Bund contenuto a {node_states[nA_id].current_value:.0f} bps mantiene solido il settore bancario e il FTSE MIB."
                else:
                    dynamic_alert = f"Dinamica Spread BTP/Bund ({node_states[nA_id].current_value:.0f} bps) vs FTSE MIB ({node_states[nB_id].current_value:,.0f})."

            link_states.append(LinkState(
                id=link_id,
                node_a_id=nA_id,
                node_b_id=nB_id,
                node_a_name=node_states[nA_id].name if nA_id in node_states else nA_id,
                node_b_name=node_states[nB_id].name if nB_id in node_states else nB_id,
                node_a_val=node_states[nA_id].current_value if nA_id in node_states else 0.0,
                node_b_val=node_states[nB_id].current_value if nB_id in node_states else 0.0,
                node_a_delta=node_states[nA_id].delta_period if nA_id in node_states else 0.0,
                node_b_delta=node_states[nB_id].delta_period if nB_id in node_states else 0.0,
                theoretical_corr=theo,
                rolling_pearson=float(r_pearson),
                rolling_spearman=float(r_spearman),
                active_corr=float(active_r),
                z_score_tension=z_tension,
                status=status,
                color_hex=color_hex,
                line_width=line_width,
                line_dash=line_dash,
                mechanism=m_def.mechanism,
                alert_description=dynamic_alert,
                is_master=True,
                weight=m_def.weight,
            ))

        # 2. Secondary Radial Spokes
        for idx, (src, tgt, theo, desc) in enumerate(SECONDARY_SPOKES):
            spoke_id = f"SPOKE_{idx+1:02d}"
            if src == "SPIDER_CORE":
                if tgt in node_states:
                    n_state = node_states[tgt]
                    z_tens = float(abs(n_state.z_score))
                    status = "Divergenza Critica" if z_tens >= Z_SCORE_CRITICAL else ("Tensione Moderata" if z_tens >= Z_SCORE_NORMAL else "Normale")
                    color_hex = "#38BDF8" if status == "Normale" else ("#F59E0B" if status == "Tensione Moderata" else "#D946EF")
                    link_states.append(LinkState(
                        id=spoke_id,
                        node_a_id=src,
                        node_b_id=tgt,
                        node_a_name="SPIDER_CORE",
                        node_b_name=n_state.name,
                        node_a_val=0.0,
                        node_b_val=n_state.current_value,
                        node_a_delta=0.0,
                        node_b_delta=n_state.delta_period,
                        theoretical_corr=theo,
                        rolling_pearson=0.0,
                        rolling_spearman=0.0,
                        active_corr=0.0,
                        z_score_tension=z_tens,
                        status=status,
                        color_hex=color_hex,
                        line_width=2.0 if status == "Normale" else 3.2,
                        line_dash="dot",
                        mechanism=desc,
                        alert_description=f"Stress sul nodo guida {n_state.name} (|Z| = {z_tens:.2f})",
                        is_master=False,
                        weight=0.8,
                    ))
            elif src in recent_df and tgt in recent_df:
                sA = recent_df[src]
                sB = recent_df[tgt]
                r_p, _ = stats.pearsonr(sA, sB) if len(sA) > 3 and sA.std() > 0 and sB.std() > 0 else (0.0, 1.0)
                zA = node_states[src].z_score if src in node_states else 0.0
                zB = node_states[tgt].z_score if tgt in node_states else 0.0
                diff = abs(zA - zB) if theo == +1 else abs(zA + zB)
                z_tens = float(np.clip(diff / 2.0, 0.0, 5.0))
                status = "Divergenza Critica" if z_tens >= Z_SCORE_CRITICAL else ("Tensione Moderata" if z_tens >= Z_SCORE_NORMAL else "Normale")
                color_hex = "#94A3B8" if status == "Normale" else ("#F59E0B" if status == "Tensione Moderata" else "#D946EF")
                link_states.append(LinkState(
                    id=spoke_id,
                    node_a_id=src,
                    node_b_id=tgt,
                    node_a_name=node_states[src].name if src in node_states else src,
                    node_b_name=node_states[tgt].name if tgt in node_states else tgt,
                    node_a_val=node_states[src].current_value if src in node_states else 0.0,
                    node_b_val=node_states[tgt].current_value if tgt in node_states else 0.0,
                    node_a_delta=node_states[src].delta_period if src in node_states else 0.0,
                    node_b_delta=node_states[tgt].delta_period if tgt in node_states else 0.0,
                    theoretical_corr=theo,
                    rolling_pearson=float(r_p),
                    rolling_spearman=float(r_p),
                    active_corr=float(r_p),
                    z_score_tension=z_tens,
                    status=status,
                    color_hex=color_hex,
                    line_width=1.5,
                    line_dash="dot",
                    mechanism=desc,
                    alert_description=f"Divergenza sulla catena di trasmissione {src} -> {tgt}",
                    is_master=False,
                    weight=0.6,
                ))

        return link_states

    def compute_spider_core(
        self,
        node_states: Dict[str, NodeState],
        link_states: List[LinkState],
        window_key: str = "90D",
    ) -> SpiderCoreState:
        """Synthesize the state of the central control hub."""
        # 1. Macro Regime Composites (Standard Institutional Benchmarking)
        # Growth indicator components:
        # - Real GDP growth vs historical trend (target 2.0%)
        # - Industrial Production growth vs historical trend (target 1.5%)
        # - Retail Sales growth vs trend (target 4.0%)
        # - Doctor Copper (vs historical median ~$4.50/lb)
        # - Yield curve slope (10Y-2Y positive spread indicates expansion, non-inversion)
        # - Labor resilience: Unemployment rate (below 4.5% = expansion) and Claims (below 230k = expansion)
        growth_components = []
        if "GRO_USGDP" in node_states:
            gdp_val = node_states["GRO_USGDP"].current_value
            growth_components.append((gdp_val - 2.0) / 0.8)
        if "GRO_INDPRO" in node_states:
            ind_val = node_states["GRO_INDPRO"].current_value
            growth_components.append((ind_val - 1.5) / 1.5)
        if "GRO_RETAIL" in node_states:
            ret_val = node_states["GRO_RETAIL"].current_value
            growth_components.append((ret_val - 4.0) / 2.5)
        if "COM_COPPER" in node_states:
            cop_val = node_states["COM_COPPER"].current_value
            growth_components.append((cop_val - 4.5) / 1.5)
        if "CURVE_10_2" in node_states:
            curv_val = node_states["CURVE_10_2"].current_value
            # +35 bps is a normal positive slope (expansionary), whereas < 0 is inverted
            growth_components.append(curv_val / 50.0)
        if "LAB_UNRATE" in node_states:
            unrate = node_states["LAB_UNRATE"].current_value
            growth_components.append((4.5 - unrate) / 0.5)
        if "LAB_CLAIMS" in node_states:
            claims_val = node_states["LAB_CLAIMS"].current_value
            growth_components.append((230000.0 - claims_val) / 30000.0)

        growth_score = float(np.mean(growth_components)) if growth_components else 0.5

        # Inflation indicator components:
        # - Core PCE vs Fed 2.0% target
        # - Headline CPI vs target
        # - 5Y5Y Forward Inflation Expectations vs 2.2% target
        # - Eurozone HICP vs 2.0% target
        # - Oil price pressure (WTI benchmark ~$75/bbl)
        inflation_components = []
        if "INF_USPCE" in node_states:
            pce_val = node_states["INF_USPCE"].current_value
            inflation_components.append((pce_val - 2.0) / 0.8)
        if "INF_USCPI" in node_states:
            cpi_val = node_states["INF_USCPI"].current_value
            inflation_components.append((cpi_val - 2.0) / 1.0)
        if "INF_EXP5Y" in node_states:
            exp_val = node_states["INF_EXP5Y"].current_value
            inflation_components.append((exp_val - 2.2) / 0.3)
        if "INF_EZHICP" in node_states:
            hicp_val = node_states["INF_EZHICP"].current_value
            inflation_components.append((hicp_val - 2.0) / 1.0)
        if "COM_OIL" in node_states:
            oil_val = node_states["COM_OIL"].current_value
            inflation_components.append((oil_val - 75.0) / 15.0)

        inflation_score = float(np.mean(inflation_components)) if inflation_components else 0.5

        if growth_score >= 0 and inflation_score < 0:
            regime = "Goldilocks / Espansione Disinflazionistica"
            desc = "Crescita solida e inflazione in calo: contesto ideale per multipli azionari (Risk-On)."
        elif growth_score >= 0 and inflation_score >= 0:
            regime = "Surriscaldamento Ciclico (Late-Cycle Inflationary Boom)"
            desc = "Crescita resiliente in piena occupazione con inflazione e materie prime sopra target: banche centrali su tassi restrittivi (Hawkish), commodities favorite e bond penalizzati."
        elif growth_score < 0 and inflation_score >= 0:
            regime = "Stagflazione"
            desc = "Rallentamento economico accompagnato da inflazione elevata: regime avverso ai mercati, difendere con Oro e Liquidità."
        else:
            regime = "Deflazione / Recessione"
            desc = "Crescita e inflazione entrambe in contrazione: flight-to-quality verso governativi difensivi (Risk-Off)."

        trajectory: List[Tuple[str, float, float]] = []
        if not self.transformed_df.empty:
            for months_back in [12, 9, 6, 3, 0]:
                pts = min(len(self.transformed_df) - 1, int(months_back * 21))
                row = self.transformed_df.iloc[-pts - 1]
                t_date = row.name.strftime("%b %Y")
                g_val = float((row.get("GRO_USGDP", 2.0) - 2.0) / 1.0)
                i_val = float((row.get("INF_USCPI", 2.5) - 2.5) / 1.0)
                trajectory.append((t_date, g_val, i_val))

        # 2. Systemic Web Tension Index
        master_links = [l for l in link_states if l.is_master]
        if master_links:
            weighted_z = sum(l.z_score_tension * l.weight for l in master_links)
            total_w = sum(l.weight for l in master_links)
            web_tension = float(weighted_z / total_w)
        else:
            web_tension = 1.0

        if web_tension >= Z_SCORE_CRITICAL:
            web_status = "CRITICA (Rottura Correlazioni Sistemiche)"
        elif web_tension >= Z_SCORE_NORMAL:
            web_status = "Tensione Moderata (Attrito Macro)"
        else:
            web_status = "Bassa Tensione (Equilibrio Fisiologico)"

        # 3. Central Bank Policy Bias & Real Rates
        fed_rate = node_states["RATE_FED"].current_value if "RATE_FED" in node_states else 3.88
        ecb_rate = node_states["RATE_ECB"].current_value if "RATE_ECB" in node_states else 2.50
        us_pce = node_states["INF_USPCE"].current_value if "INF_USPCE" in node_states else 3.01
        ez_hicp = node_states["INF_EZHICP"].current_value if "INF_EZHICP" in node_states else 3.75
        us_02y = node_states["YIELD_US02"].current_value if "YIELD_US02" in node_states else 4.75

        # Real policy rates: Nominal Policy Rate - Core Inflation
        fed_real_rate = fed_rate - us_pce
        ecb_real_rate = ecb_rate - ez_hicp

        # Standard Institutional Taylor Rule:
        # Target = r* + pi + 0.5*(pi - pi*) + 0.5*(y - y*)
        # with r* = 1.0%, pi* = 2.0%, y* = 2.0%
        gdp_val = node_states["GRO_USGDP"].current_value if "GRO_USGDP" in node_states else 2.19
        taylor_target = 1.0 + us_pce + 0.5 * (us_pce - 2.0) + 0.5 * (gdp_val - 2.0)
        taylor_gap_bps = (fed_rate - taylor_target) * 100.0

        # Institutional Policy Bias Classification
        if fed_real_rate >= 1.5 or (us_02y > fed_rate + 0.5):
            cb_bias = "Hawkish (Stretta Monetaria Rigida)"
        elif fed_real_rate >= 0.5 or (fed_rate >= 3.5 and us_pce > 2.5):
            cb_bias = "Restrittivo / Vigile (Ancoraggio Aspettative)"
        elif fed_real_rate < 0.0 or us_02y < (fed_rate - 0.75):
            cb_bias = "Dovish (Orientamento Accomodante)"
        else:
            cb_bias = "Neutral / Data-Dependent (In Pausa Strategica)"

        critical_alerts = [
            f"⚡ {l.id} [{l.node_a_name} ↔ {l.node_b_name}]: {l.alert_description}"
            for l in master_links if l.status == "Divergenza Critica"
        ]

        # 4. Quantitative Landing Scenario Engine (Soft vs Hard vs No Landing)
        # Factors:
        # - Growth resilience (GDP > 2.0%, IndPro > 1.5%, Retail > 4.0%, Claims < 220k)
        # - Inflation stickiness (Core PCE > 2.7%, Headline CPI > 3.0%, Oil > $85)
        # - Yield curve geometry (Inverted vs Normalizing vs Steep)
        # - Credit spreads (High yield spread compression < 3.8% vs stress > 5.0%)
        unrate_val = node_states["LAB_UNRATE"].current_value if "LAB_UNRATE" in node_states else 4.2
        claims_val = node_states["LAB_CLAIMS"].current_value if "LAB_CLAIMS" in node_states else 199000.0
        hy_val = node_states["CRED_HY"].current_value if "CRED_HY" in node_states else 3.23
        curve_val = node_states["CURVE_10_2"].current_value if "CURVE_10_2" in node_states else 0.35
        oil_val = node_states["COM_OIL"].current_value if "COM_OIL" in node_states else 99.0

        # Raw scenario score accumulator
        score_no = 0.0
        score_soft = 0.0
        score_hard = 0.0

        # Growth contribution:
        if gdp_val >= 2.0:
            score_no += 30.0
            score_soft += 20.0
        elif gdp_val >= 1.0:
            score_soft += 35.0
            score_no += 10.0
        else:
            score_hard += 40.0

        # Labor market contribution:
        if unrate_val <= 4.3 and claims_val <= 215000:
            score_no += 30.0
            score_soft += 20.0
        elif unrate_val <= 4.8:
            score_soft += 30.0
            score_hard += 10.0
        else:
            score_hard += 35.0

        # Inflation & Commodity contribution:
        if us_pce >= 2.8 or oil_val >= 85.0:
            score_no += 35.0
            score_soft += 5.0
        elif us_pce <= 2.3 and oil_val < 75.0:
            score_soft += 35.0
        else:
            score_soft += 25.0
            score_no += 15.0

        # Credit & Curve contribution:
        if hy_val < 3.6 and curve_val >= 0:
            score_no += 20.0
            score_soft += 20.0
        elif hy_val >= 4.8 or curve_val < -20:
            score_hard += 35.0
        else:
            score_soft += 15.0

        tot_score = max(1.0, score_no + score_soft + score_hard)
        prob_no = round((score_no / tot_score) * 100.0, 1)
        prob_soft = round((score_soft / tot_score) * 100.0, 1)
        prob_hard = round((score_hard / tot_score) * 100.0, 1)

        # Select dominant scenario
        if prob_no >= prob_soft and prob_no >= prob_hard:
            l_scen = "NO_LANDING"
            l_title = "No Landing (Crescita Forte & Riacidificazione Inflattiva)"
            l_badge = "🔥 NO LANDING"
            l_color = "#F59E0B"
            l_conf = prob_no
            l_drivers = [
                f"PIL Reale al {gdp_val:.2f}% (superiore al potenziale del 2.0%).",
                f"Piena occupazione solida (Disoccupazione al {unrate_val:.1f}%, Sussidi a {claims_val:,.0f}).",
                f"Core PCE al {us_pce:.2f}% e Petrolio a ${oil_val:.2f}/bbl: inflazione 'appiccicosa' sopra il target Fed.",
                f"Spread High Yield compressi a {hy_val:.2f}%: nessun sintomo di deterioramento creditizio.",
            ]
            l_market = "Tassi d'interesse a lungo termine elevati ('higher for longer'), dollaro solido, sovraperformance di materie prime, energia e ciclici; pressione sui titoli growth ad alto multiplo."
            l_policy = "La Federal Reserve è costretta a posticipare o cancellare i tagli dei tassi, mantenendo un orientamento restrittivo (Hawkish Pause o rialzi tattici)."
        elif prob_soft >= prob_no and prob_soft >= prob_hard:
            l_scen = "SOFT_LANDING"
            l_title = "Soft Landing (Atterraggio Morbido & Disinflazione Ordinata)"
            l_badge = "🛬 SOFT LANDING"
            l_color = "#10B981"
            l_conf = prob_soft
            l_drivers = [
                f"Crescita moderata sostenibile senza contrazione dell'output.",
                f"Progressivo rientro dell'inflazione verso il target del 2.0%.",
                f"Mercato del lavoro in graduale ribilanciamento senza ondate di licenziamenti.",
            ]
            l_market = "Scenario ideale 'Goldilocks' per l'azionario diversificato, compressione dei premi al rischio e rally obbligazionario progressivo."
            l_policy = "La Federal Reserve avvia tagli calibrati e preventivi (tassi verso il livello neutrale ~3.0%)."
        else:
            l_scen = "HARD_LANDING"
            l_title = "Hard Landing (Atterraggio Brusco / Rischio Recessione)"
            l_badge = "⚠️ HARD LANDING"
            l_color = "#EF4444"
            l_conf = prob_hard
            l_drivers = [
                f"Forte rallentamento del PIL o contrazione dell'attività economica.",
                f"Deterioramento accelerato dei sussidi di disoccupazione e allargamento spread creditizi.",
                f"Impatti ritardati della stretta monetaria sui bilanci di famiglie e imprese.",
            ]
            l_market = "Forte avversione al rischio (Risk-Off), sell-off su equity e credito corporate; flight-to-safety verso Treasury a lunga scadenza (TLT) e cash.",
            l_policy = "La Federal Reserve interviene con tagli d'emergenza da 50 bps per arginare il ciclo recessivo."

        landing_state = LandingScenarioState(
            scenario=l_scen,
            scenario_title=l_title,
            scenario_badge=l_badge,
            color_hex=l_color,
            confidence_pct=l_conf,
            prob_no_landing=prob_no,
            prob_soft_landing=prob_soft,
            prob_hard_landing=prob_hard,
            key_drivers=l_drivers,
            market_implication=l_market,
            policy_reaction=l_policy,
        )

        # 4. ECB CISS (Composite Indicator of Systemic Stress) Non-Linear Model
        ciss_state = self.compute_ecb_ciss(node_states, window_key=window_key)

        return SpiderCoreState(
            macro_regime=regime,
            macro_regime_desc=desc,
            growth_score=growth_score,
            inflation_score=inflation_score,
            regime_coords=(growth_score, inflation_score),
            trajectory_coords=trajectory,
            systemic_web_tension=web_tension,
            web_tension_status=web_status,
            central_bank_bias=cb_bias,
            fed_real_rate=fed_real_rate,
            ecb_real_rate=ecb_real_rate,
            taylor_rate_gap_bps=taylor_gap_bps,
            active_critical_alerts=critical_alerts,
            landing_scenario=landing_state,
            ciss=ciss_state,
        )

    def compute_ecb_ciss(self, node_states: Dict[str, NodeState], window_key: str = "90D") -> CISSState:
        """
        Computes the ECB CISS (Composite Indicator of Systemic Stress) using Portfolio Theory:
        CISS_t = s' * C_t * s = sum_i sum_j w_i * w_j * rho_ij * s_i * s_j
        where:
          s_i in [0, 1] is the CDF-transformed systemic stress for quadrant i
          C_t is the dynamic rolling cross-correlation matrix between quadrants
        """
        quad_keys = ["INFLATION", "GROWTH", "LABOR", "LIQUIDITY_RISK"]
        quad_weights = {
            "INFLATION": 0.25,
            "GROWTH": 0.25,
            "LABOR": 0.20,
            "LIQUIDITY_RISK": 0.30,  # Money market, rates, credit & systemic risk carry higher weight
        }

        # Calculate quadrant stress sub-index s_i in [0, 1] using empirical CDF of |Z-Score|
        quad_stresses: Dict[str, float] = {}
        for qk in quad_keys:
            q_nodes = [ns for ns in node_states.values() if ns.quadrant == qk]
            if not q_nodes:
                quad_stresses[qk] = 0.25
                continue

            # In financial stress models (e.g. CISS / OFR), individual stress indicators are mapped to [0, 1]
            # through standard normal or empirical CDF: Phi(|Z|), where |Z| > 1.96 yields ~0.95 stress.
            z_scores = [abs(ns.z_score) for ns in q_nodes]
            # Average Z of quadrant
            avg_z = float(np.mean(z_scores)) if z_scores else 0.0
            # Transform to [0, 1] using logistic/error function CDF mapping:
            # 2 * Phi(|Z|) - 1 or erf(|Z| / sqrt(2)), mapping 0 -> 0.0, 1.25 -> ~0.78, 2.0 -> ~0.95
            stress_cdf = float(math.erf(avg_z / (math.sqrt(2.0) * 1.2)))
            stress_cdf = max(0.01, min(0.99, stress_cdf))
            quad_stresses[qk] = stress_cdf

        # Compute dynamic cross-correlation matrix C_t across the 4 quadrants
        days = TIME_WINDOWS.get(window_key, {}).get("days", 90)
        corr_matrix: Dict[str, Dict[str, float]] = {q1: {q2: 1.0 if q1 == q2 else 0.0 for q2 in quad_keys} for q1 in quad_keys}

        if not self.aligned_df.empty:
            sub_df = self.aligned_df.iloc[-days:].copy() if len(self.aligned_df) >= days else self.aligned_df.copy()
            # Build quadrant-level daily returns / changes
            quad_series: Dict[str, pd.Series] = {}
            for qk in quad_keys:
                q_cols = [nid for nid, nd in NODES.items() if nd.quadrant == qk and nid in sub_df.columns]
                if q_cols:
                    # Normalized z-score rolling series for each node in quadrant, then mean
                    normed_cols = []
                    for c in q_cols:
                        series = sub_df[c].dropna()
                        if len(series) >= 10:
                            s_std = series.std()
                            if s_std > 1e-6:
                                normed = (series - series.mean()) / s_std
                                normed_cols.append(normed)
                    if normed_cols:
                        quad_series[qk] = pd.concat(normed_cols, axis=1).mean(axis=1)

            # Compute empirical Pearson correlation matrix
            if len(quad_series) == len(quad_keys):
                comb_df = pd.DataFrame(quad_series).dropna()
                if len(comb_df) >= 15:
                    r_mat = comb_df.corr(method="pearson")
                    for q1 in quad_keys:
                        for q2 in quad_keys:
                            val = float(r_mat.loc[q1, q2])
                            if not np.isnan(val):
                                corr_matrix[q1][q2] = round(val, 4)

        # Portfolio-theoretic CISS Aggregation:
        # Idiosyncratic risk = sum_i (w_i^2 * s_i^2)
        # Contagion effect = sum_{i != j} (w_i * w_j * rho_ij * s_i * s_j)
        # Total CISS = sqrt(Idiosyncratic + max(0, Contagion))
        s_vec = np.array([quad_stresses[qk] for qk in quad_keys])
        w_vec = np.array([quad_weights[qk] for qk in quad_keys])
        c_mat = np.array([[corr_matrix[q1][q2] for q2 in quad_keys] for q1 in quad_keys])

        idiosyncratic_term = float(np.sum((w_vec * s_vec) ** 2))

        # Total portfolio variance term: s' * diag(w) * C * diag(w) * s
        w_s = w_vec * s_vec
        total_variance = float(w_s.T @ c_mat @ w_s)
        total_variance = max(0.0001, total_variance)

        contagion_term = float(total_variance - idiosyncratic_term)
        ciss_score = float(math.sqrt(total_variance))
        ciss_score = max(0.0, min(1.0, ciss_score))

        # Contagion contribution percentage
        if total_variance > 0:
            contagion_pct = max(0.0, min(100.0, (contagion_term / total_variance) * 100.0))
        else:
            contagion_pct = 0.0

        # Status & classification according to ECB CISS empirical percentiles
        if ciss_score >= 0.50:
            ciss_status = "Contagio Sistemico Severo (Rischio Multi-Mercato)"
            status_color = "#EF4444"
        elif ciss_score >= 0.25:
            ciss_status = "Allerta Settoriale Moderata (Primi Focolai Cross-Asset)"
            status_color = "#F59E0B"
        else:
            ciss_status = "Calma Fisiologica (Rischio Isolato o Minimo)"
            status_color = "#10B981"

        return CISSState(
            ciss_score=round(ciss_score, 4),
            ciss_status=ciss_status,
            status_color=status_color,
            quadrant_stress={qk: round(quad_stresses[qk], 4) for qk in quad_keys},
            weights=quad_weights,
            corr_matrix=corr_matrix,
            idiosyncratic_risk=round(idiosyncratic_term, 5),
            contagion_effect=round(contagion_term, 5),
            contagion_contribution_pct=round(contagion_pct, 1),
        )


engine = SpiderQuantEngine()

