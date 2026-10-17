"""
SPIDER Quantitative Engine:
Mathematical modeling of macro-financial relationships, mixed-frequency time series alignment,
rolling Pearson/Spearman correlations, link tension Z-scores, Macro Regime Matrix,
Systemic Web Tension, and Central Bank Policy Bias.
"""

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

            # For high-frequency market asset nodes, compute Z-score relative to selected rolling window
            # For macro/rate series with low frequency (Ring 1 & 2), standardize relative to 3Y history (756 business days)
            # to avoid artificially inflating Z-scores when 30D variance is close to zero.
            if n_def.ring in [1, 2]:
                hist_lookback = min(len(s_full), 756)
                s_baseline = s_full.iloc[-hist_lookback:]
                mean_val = float(s_baseline.mean())
                std_val = float(s_baseline.std())
            else:
                mean_val = float(s_recent.mean())
                std_val = float(s_recent.std())

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
                alert_description=m_def.alert_description,
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
        link_states: List[LinkState]
    ) -> SpiderCoreState:
        """Synthesize the state of the central control hub."""
        # 1. Macro Regime Composites
        # Growth indicator components:
        # - Real GDP growth vs historical trend (target 2.0%)
        # - Industrial Production growth
        # - Retail Sales growth
        # - Copper (Doctor Copper economic barometer)
        # - Yield curve slope (10Y-2Y positive spread indicates economic expansion)
        # - Labor resilience (Unemployment rate and Initial Jobless claims inverted)
        growth_components = []
        if "GRO_USGDP" in node_states:
            # Baseline benchmark around 2.0% real growth
            gdp_state = node_states["GRO_USGDP"]
            g_dev = (gdp_state.current_value - 2.0) / 0.8
            growth_components.append(g_dev * 1.5)
        if "GRO_INDPRO" in node_states:
            growth_components.append(node_states["GRO_INDPRO"].z_score * 1.2)
        if "GRO_RETAIL" in node_states:
            growth_components.append(node_states["GRO_RETAIL"].z_score * 1.0)
        if "COM_COPPER" in node_states:
            growth_components.append(node_states["COM_COPPER"].z_score * 0.8)
        if "CURVE_10_2" in node_states:
            growth_components.append(node_states["CURVE_10_2"].z_score * 0.8)
        if "LAB_UNRATE" in node_states:
            # Low unemployment (< 4.5%) supports resilient growth
            unrate = node_states["LAB_UNRATE"].current_value
            growth_components.append((4.5 - unrate) * 1.2)
        if "LAB_CLAIMS" in node_states:
            growth_components.append(-node_states["LAB_CLAIMS"].z_score * 0.8)

        growth_score = float(np.mean(growth_components)) if growth_components else 0.2

        # Inflation indicator components:
        # - Core PCE vs Fed 2.0% target
        # - Headline CPI vs target
        # - 5Y5Y Forward Inflation Expectations
        # - Eurozone HICP
        # - Oil price pressure (energy shock)
        inflation_components = []
        if "INF_USPCE" in node_states:
            pce_val = node_states["INF_USPCE"].current_value
            inflation_components.append((pce_val - 2.0) * 1.5)
        if "INF_USCPI" in node_states:
            cpi_val = node_states["INF_USCPI"].current_value
            inflation_components.append((cpi_val - 2.5) * 1.2)
        if "INF_EXP5Y" in node_states:
            inflation_components.append(node_states["INF_EXP5Y"].z_score * 0.8)
        if "INF_EZHICP" in node_states:
            hicp_val = node_states["INF_EZHICP"].current_value
            inflation_components.append((hicp_val - 2.0) * 1.0)
        if "COM_OIL" in node_states:
            inflation_components.append(node_states["COM_OIL"].z_score * 1.0)

        inflation_score = float(np.mean(inflation_components)) if inflation_components else 0.1

        if growth_score >= 0 and inflation_score < 0:
            regime = "Goldilocks / Espansione Disinflazionistica"
            desc = "Crescita solida e inflazione in calo: contesto ideale per multipli azionari (Risk-On)."
        elif growth_score >= 0 and inflation_score >= 0:
            regime = "Reflazione / Surriscaldamento Controllato"
            desc = "Crescita resiliente con inflazione sostenuta da shock energetici: banche centrali vigili sui tassi, ciclici e materie prime favoriti."
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
        )


engine = SpiderQuantEngine()
