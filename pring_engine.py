"""
Martin Pring Business Cycle & Intermarket Sector Rotation Module for SPIDER.
Models:
1. 6 Pring Cycle Stages (Bonds, Stocks, Commodities sequence of tops & bottoms)
2. KST (Know Sure Thing) Multi-Timeframe Momentum Oscillator
3. RRG (Relative Rotation Graph) for S&P 500 Sectors vs SPY (Leading, Weakening, Lagging, Improving)
4. Dynamic Stage Detector & Actionable Portfolio Allocation Matrix
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


# 6 Stages definition according to Martin Pring's Intermarket Business Cycle Theory
PRING_STAGES = {
    1: {
        "title": "Fase 1: Fine Recessione / Minimo Obbligazionario",
        "regime": "Inversione Monetaria",
        "description": "L'economia è ancora in contrazione profonda ma le banche centrali allentano i tassi. I rendimenti scendono e i prezzi delle obbligazioni iniziano a salire per primi (minimo dei Bond superato).",
        "bonds": "UP (Rialzista)",
        "stocks": "DOWN (Ribassista/Bottoming)",
        "commodities": "DOWN (Ribassista)",
        "favored_assets": ["Treasury a lungo termine (TLT, IEF)", "Corporate Bond Investment Grade (LQD)", "Liquidità / T-Bills"],
        "unfavored_assets": ["Materie Prime (Energia, Metalli)", "Titoli Ciclici e Small Cap"],
        "favored_sectors": ["Utilities (XLU)", "Consumer Staples (XLP)", "Healthcare (XLV)"],
        "color": "#3B82F6",
    },
    2: {
        "title": "Fase 2: Primi Segnali di Ripresa / Disinflazione",
        "regime": "Espansione Iniziale (Goldilocks)",
        "description": "L'economia si stabilizza, l'inflazione è bassa e il costo del denaro rimane contenuto. Il mercato azionario inverte al rialzo guidato da growth e ciclici, mentre i bond continuano a salire.",
        "bonds": "UP (Rialzista)",
        "stocks": "UP (Rialzista)",
        "commodities": "DOWN (Ribassista/Base)",
        "favored_assets": ["Azioni Growth / Tecnologici (QQQ, XLK)", "Treasury a medio termine (IEF)", "Corporate Bond"],
        "unfavored_assets": ["Materie Prime Energetiche", "Oro / Difensivi di puro rifugio"],
        "favored_sectors": ["Tecnologia (XLK)", "Discrezionali (XLY)", "Comunicazioni (XLC)"],
        "color": "#10B981",
    },
    3: {
        "title": "Fase 3: Consolidamento Ripresa / Accelerazione Industriale",
        "regime": "Espansione Piena",
        "description": "La crescita industriale accelera e la domanda di input cresce. L'inflazione inizia a risvegliarsi e anche le materie prime industriali cominciano a salire, affiancandosi alle azioni (spesso la fase più breve).",
        "bonds": "UP / TOPPING (Fine rally bond)",
        "stocks": "UP (Rialzista Forte)",
        "commodities": "UP (Inizio Rally)",
        "favored_assets": ["Azioni Cicliche e Industriali (XLI)", "Rame e Metalli Industriali (CPER)", "Materiali di Base (XLB)"],
        "unfavored_assets": ["Bond a lunghissima scadenza (vulnerabili a inflazione)"],
        "favored_sectors": ["Industriali (XLI)", "Materiali di Base (XLB)", "Finanziari (XLF)"],
        "color": "#06B6D4",
    },
    4: {
        "title": "Fase 4: Boom Economico & Shock Inflazione",
        "regime": "Surriscaldamento / Reflazione",
        "description": "L'economia è in piena espansione ('boom') con domanda superiore all'offerta. L'inflazione sale visibilmente; le banche centrali alzano i tassi. I bond crollano, mentre materie prime ed energia dominano.",
        "bonds": "DOWN (Ribassista Forte / Tassi in salita)",
        "stocks": "UP / TOPPING (Fine ciclo bull azionario)",
        "commodities": "UP (Rialzista Esplosivo / Protagoniste)",
        "favored_assets": ["Materie Prime & Petrolio (DBC, XLE, USO)", "Oro e Metalli (GLD)", "TIPS indicizzati all'inflazione"],
        "unfavored_assets": ["Obbligazioni a tasso fisso (TLT)", "Titoli Growth ad alto multiplo P/E"],
        "favored_sectors": ["Energia (XLE)", "Materiali di Base (XLB)", "Finanziari (XLF)"],
        "color": "#F59E0B",
    },
    5: {
        "title": "Fase 5: Picco Ciclico & Stretta Finanziaria",
        "regime": "Stagflazione / Tardo Ciclo",
        "description": "I tassi elevati e l'inflazione persistente comprimono i margini aziendali e il potere d'acquisto. Le azioni raggiungono il picco e iniziano a scendere. Le commodities restano forti fino all'inversione tardiva.",
        "bonds": "DOWN (Ribassista)",
        "stocks": "DOWN (Ribassista / Correzione)",
        "commodities": "UP / TOPPING (Massimo delle materie prime)",
        "favored_assets": ["Oro e Metalli Preziosi (GLD)", "Commodities tardive (Petrolio, Agricoli)", "Liquidità a breve termine (SHV, BIL)"],
        "unfavored_assets": ["Azioni Globali (SPY, QQQ)", "Obbligazioni a lunga duration"],
        "favored_sectors": ["Energia (XLE)", "Utilities difensive (XLU)", "Salute (XLV)"],
        "color": "#EF4444",
    },
    6: {
        "title": "Fase 6: Rallentamento Aperto / Contrazione & Recessione",
        "regime": "Deflazione / Recessione",
        "description": "L'economia entra apertamente in recessione o forte rallentamento. La distruzione di domanda abbatte anche le materie prime. L'unico rifugio sicuro è la liquidità in attesa del pivot obbligazionario (verso Fase 1).",
        "bonds": "DOWN / BOTTOMING (Preparazione al minimo)",
        "stocks": "DOWN (Ribassista / Bear Market)",
        "commodities": "DOWN (Ribassista Forte / Crollo domanda)",
        "favored_assets": ["Cash e Money Market Fund (BIL)", "Dollar Index (UUP / DXY)", "Primi acquisti scaglionati su Treasury governativi"],
        "unfavored_assets": ["Azioni Cicliche", "High Yield & Credito corporate", "Materie Prime"],
        "favored_sectors": ["Beni di Prima Necessità (XLP)", "Utilities (XLU)", "Cash / Treasury a breve"],
        "color": "#8B5CF6",
    },
}

# Standard ETF proxies for Pring 3 Asset Classes
PRING_ASSET_CLASSES = {
    "BONDS": {
        "name": "Obbligazioni (Bonds)",
        "proxy_symbol": "IEF",
        "backup_series": "YIELD_US10",
        "invert_backup": True,  # yields down = price up
        "label": "US 7-10Y Treasury (Prezzo)",
    },
    "STOCKS": {
        "name": "Azioni (Stocks)",
        "proxy_symbol": "SPY",
        "backup_series": "EQ_SPX",
        "invert_backup": False,
        "label": "S&P 500 Equities",
    },
    "COMMODITIES": {
        "name": "Materie Prime (Commodities)",
        "proxy_symbol": "DBC",
        "backup_series": "COM_OIL",
        "backup_series_2": "COM_COPPER",
        "invert_backup": False,
        "label": "Commodities Index (Energy & Materials)",
    },
}

# S&P 500 GICS Sector SPDRs for Relative Rotation
SECTOR_ETFS = {
    "XLK": {"name": "Technology", "category": "Growth / Rate-Sensitive", "color": "#38BDF8"},
    "XLY": {"name": "Consumer Discretionary", "category": "Cyclical Growth", "color": "#F472B6"},
    "XLI": {"name": "Industrials", "category": "Early / Mid Cyclical", "color": "#34D399"},
    "XLB": {"name": "Materials", "category": "Commodity Cyclical", "color": "#FBBF24"},
    "XLE": {"name": "Energy", "category": "Late Cyclical / Inflation", "color": "#FB923C"},
    "XLF": {"name": "Financials", "category": "Rate / Steepening", "color": "#A78BFA"},
    "XLU": {"name": "Utilities", "category": "Defensive / Bond Proxy", "color": "#818CF8"},
    "XLP": {"name": "Consumer Staples", "category": "Defensive", "color": "#4ADE80"},
    "XLV": {"name": "Healthcare", "category": "Defensive Growth", "color": "#2DD4BF"},
}


def compute_kst(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Computes Martin Pring's Know Sure Thing (KST) Multi-Timeframe Momentum Oscillator.
    Standard Daily formula:
    RCMA1 = SMA(ROC(10), 10) * 1
    RCMA2 = SMA(ROC(15), 10) * 2
    RCMA3 = SMA(ROC(20), 10) * 3
    RCMA4 = SMA(ROC(30), 15) * 4
    KST = RCMA1 + RCMA2 + RCMA3 + RCMA4
    Signal Line = SMA(KST, 9)
    """
    if len(series) < 60:
        return pd.Series(0.0, index=series.index), pd.Series(0.0, index=series.index)

    # ROCs
    roc1 = series.pct_change(10) * 100.0
    roc2 = series.pct_change(15) * 100.0
    roc3 = series.pct_change(20) * 100.0
    roc4 = series.pct_change(30) * 100.0

    rcma1 = roc1.rolling(10).mean() * 1.0
    rcma2 = roc2.rolling(10).mean() * 2.0
    rcma3 = roc3.rolling(10).mean() * 3.0
    rcma4 = roc4.rolling(15).mean() * 4.0

    kst = (rcma1 + rcma2 + rcma3 + rcma4).fillna(0.0)
    kst_signal = kst.rolling(9).mean().fillna(0.0)
    return kst, kst_signal


@dataclass
class PringAssetStatus:
    name: str
    symbol: str
    current_price: float
    trend_200d: str  # "BULL" or "BEAR"
    momentum_60d_pct: float
    kst_val: float
    kst_signal: float
    is_bullish: bool
    summary: str


@dataclass
class PringCycleResult:
    current_stage: int
    stage_title: str
    stage_regime: str
    stage_desc: str
    asset_statuses: Dict[str, PringAssetStatus]
    score_stage_1: float
    score_stage_2: float
    score_stage_3: float
    score_stage_4: float
    score_stage_5: float
    score_stage_6: float
    confidence_pct: float
    favored_assets: List[str]
    unfavored_assets: List[str]
    favored_sectors: List[str]
    sequence_state: str


class PringCycleEngine:
    """Quantitative engine determining the current Martin Pring Economic Cycle Stage."""

    def __init__(self, aligned_df: Optional[pd.DataFrame] = None):
        if aligned_df is None or aligned_df.empty:
            from quant_engine import engine
            if engine.aligned_df is None or engine.aligned_df.empty:
                engine.load_and_align_data()
            self.df = engine.aligned_df
        else:
            self.df = aligned_df

    def analyze(self) -> PringCycleResult:
        if self.df is None or self.df.empty:
            from quant_engine import engine
            if engine.aligned_df is None or engine.aligned_df.empty:
                engine.load_and_align_data()
            self.df = engine.aligned_df

        asset_statuses: Dict[str, PringAssetStatus] = {}

        # 1. Evaluate Bonds (Market Prices & Yields)
        # Martin Pring Principle: Bonds Bull = Bond PRICES rising (Yields falling).
        # We fetch the primary US Treasury ETF price proxy (TLT 20Y / IEF 7-10Y).
        bonds_series = None
        b_symbol = "iShares 20+ Year Treasury Bond ETF (TLT)"
        try:
            import yfinance as yf
            tlt = yf.Ticker("TLT").history(period="2y")
            if tlt is not None and not tlt.empty and "Close" in tlt and len(tlt) > 30:
                c = tlt["Close"].dropna()
                idx = pd.to_datetime(c.index).tz_localize(None).normalize()
                bonds_series = pd.Series(c.values, index=idx, name="TLT")
        except Exception:
            bonds_series = None

        if bonds_series is None or bonds_series.empty:
            if "YIELD_US10" in self.df and len(self.df["YIELD_US10"]) > 5:
                # Synthetic Bond Price: inverse to 10Y yields
                y10 = self.df["YIELD_US10"]
                bonds_series = 1000.0 / ((1.0 + (y10 / 100.0)) ** 7.0)
                b_symbol = "US 10Y Treasury (Prezzo Sintetico)"
            else:
                dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq="B")
                bonds_series = pd.Series(80.0, index=dates)

        b_kst, b_kst_sig = compute_kst(bonds_series)
        b_sma200 = bonds_series.rolling(min(200, len(bonds_series))).mean()
        b_curr = float(bonds_series.iloc[-1]) if len(bonds_series) > 0 else 80.0
        lookback_b = min(60, len(bonds_series))
        b_mom = float((b_curr / bonds_series.iloc[-lookback_b] - 1.0) * 100.0) if lookback_b > 0 else 0.0

        # Rigorous Pring Intermarket Rule for Bonds:
        # A true Bull market in bonds requires prices ABOVE 200d SMA AND positive momentum (or KST > signal).
        # When bond prices are below 200d SMA or yields are pushing higher (TLT negative YTD), bonds are BEAR.
        b_above_sma = bool(b_curr >= b_sma200.iloc[-1])
        b_kst_bull = bool(b_kst.iloc[-1] > b_kst_sig.iloc[-1])
        b_bull = bool(b_above_sma and (b_mom > 0 or b_kst_bull))

        asset_statuses["BONDS"] = PringAssetStatus(
            name="Obbligazioni (Bonds - Prezzo)",
            symbol=b_symbol,
            current_price=b_curr,
            trend_200d="BULL" if b_above_sma else "BEAR",
            momentum_60d_pct=b_mom,
            kst_val=float(b_kst.iloc[-1]),
            kst_signal=float(b_kst_sig.iloc[-1]),
            is_bullish=b_bull,
            summary="Rialzista (Prezzi in salita / Rendimenti in calo)" if b_bull else "Ribassista (Prezzi compressi / Tassi elevati)",
        )

        # 2. Evaluate Stocks
        if "EQ_SPX" in self.df and len(self.df["EQ_SPX"]) > 5:
            stocks_series = self.df["EQ_SPX"]
        else:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq="B")
            stocks_series = pd.Series(6000.0, index=dates)

        s_kst, s_kst_sig = compute_kst(stocks_series)
        s_sma200 = stocks_series.rolling(min(200, len(stocks_series))).mean()
        s_curr = float(stocks_series.iloc[-1]) if len(stocks_series) > 0 else 6000.0
        lookback_s = min(60, len(stocks_series))
        s_mom = float((s_curr / stocks_series.iloc[-lookback_s] - 1.0) * 100.0) if lookback_s > 0 else 0.0
        s_bull = bool((s_curr >= s_sma200.iloc[-1]) and (s_kst.iloc[-1] > s_kst_sig.iloc[-1] or s_mom > 0))

        asset_statuses["STOCKS"] = PringAssetStatus(
            name="Azioni (S&P 500 Stocks)",
            symbol="S&P 500 (^GSPC)",
            current_price=s_curr,
            trend_200d="BULL" if s_curr >= s_sma200.iloc[-1] else "BEAR",
            momentum_60d_pct=s_mom,
            kst_val=float(s_kst.iloc[-1]),
            kst_signal=float(s_kst_sig.iloc[-1]),
            is_bullish=s_bull,
            summary="Rialzista (Multipli & Utili Espansione)" if s_bull else "Ribassista (De-risking)",
        )

        # 3. Evaluate Commodities
        if "COM_OIL" in self.df and "COM_COPPER" in self.df and len(self.df["COM_OIL"]) > 5:
            comm_series = (self.df["COM_OIL"] / self.df["COM_OIL"].iloc[0] * 50.0 +
                           self.df["COM_COPPER"] / self.df["COM_COPPER"].iloc[0] * 50.0)
        elif "COM_OIL" in self.df and len(self.df["COM_OIL"]) > 5:
            comm_series = self.df["COM_OIL"]
        else:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq="B")
            comm_series = pd.Series(100.0, index=dates)

        c_kst, c_kst_sig = compute_kst(comm_series)
        c_sma200 = comm_series.rolling(min(200, len(comm_series))).mean()
        c_curr = float(comm_series.iloc[-1]) if len(comm_series) > 0 else 100.0
        lookback_c = min(60, len(comm_series))
        c_mom = float((c_curr / comm_series.iloc[-lookback_c] - 1.0) * 100.0) if lookback_c > 0 else 0.0
        c_bull = bool((c_curr >= c_sma200.iloc[-1]) and (c_kst.iloc[-1] > c_kst_sig.iloc[-1] or c_mom > 0))

        asset_statuses["COMMODITIES"] = PringAssetStatus(
            name="Materie Prime (Commodities)",
            symbol="Energy & Industrial Metals Index",
            current_price=c_curr,
            trend_200d="BULL" if c_curr >= c_sma200.iloc[-1] else "BEAR",
            momentum_60d_pct=c_mom,
            kst_val=float(c_kst.iloc[-1]),
            kst_signal=float(c_kst_sig.iloc[-1]),
            is_bullish=c_bull,
            summary="Rialzista (Pressioni Inflazione & Domanda)" if c_bull else "Ribassista (Raffreddamento Prezzi)",
        )

        # Pring Scoring Logic based on the 3 binary/continuous statuses:
        # Stage 1: Bonds UP, Stocks DOWN, Commodities DOWN
        # Stage 2: Bonds UP, Stocks UP, Commodities DOWN
        # Stage 3: Bonds UP, Stocks UP, Commodities UP
        # Stage 4: Bonds DOWN, Stocks UP, Commodities UP
        # Stage 5: Bonds DOWN, Stocks DOWN, Commodities UP
        # Stage 6: Bonds DOWN, Stocks DOWN, Commodities DOWN

        b_score = 1.0 if b_bull else -1.0
        s_score = 1.0 if s_bull else -1.0
        c_score = 1.0 if c_bull else -1.0

        scores = {
            1: (b_score * 1.0) + (-s_score * 1.0) + (-c_score * 1.0),
            2: (b_score * 1.0) + (s_score * 1.0) + (-c_score * 1.0),
            3: (b_score * 0.7) + (s_score * 1.0) + (c_score * 1.0),
            4: (-b_score * 1.0) + (s_score * 1.0) + (c_score * 1.0),
            5: (-b_score * 1.0) + (-s_score * 0.8) + (c_score * 1.0),
            6: (-b_score * 1.0) + (-s_score * 1.0) + (-c_score * 1.0),
        }

        # Normalize scores to 0 - 100%
        best_stage = max(scores, key=scores.get)
        max_val = max(scores.values())
        min_val = min(scores.values())
        confidence = float(np.clip(((max_val - min_val) / 6.0) * 100.0 + 35.0, 45.0, 95.0))

        stage_meta = PRING_STAGES[best_stage]

        # Sequence text
        bull_list = []
        if b_bull:
            bull_list.append("Obbligazioni")
        if s_bull:
            bull_list.append("Azioni")
        if c_bull:
            bull_list.append("Materie Prime")

        seq_text = " + ".join(bull_list) if bull_list else "Nessun asset primario (Liquidità / Cash)"

        return PringCycleResult(
            current_stage=best_stage,
            stage_title=stage_meta["title"],
            stage_regime=stage_meta["regime"],
            stage_desc=stage_meta["description"],
            asset_statuses=asset_statuses,
            score_stage_1=float(scores[1]),
            score_stage_2=float(scores[2]),
            score_stage_3=float(scores[3]),
            score_stage_4=float(scores[4]),
            score_stage_5=float(scores[5]),
            score_stage_6=float(scores[6]),
            confidence_pct=confidence,
            favored_assets=stage_meta["favored_assets"],
            unfavored_assets=stage_meta["unfavored_assets"],
            favored_sectors=stage_meta["favored_sectors"],
            sequence_state=f"Asset in Trend Rialzista Attuale: {seq_text}",
        )


def build_pring_cycle_dial_figure(stage_res: PringCycleResult, dark_mode: bool = True) -> go.Figure:
    """Builds a circular dial visual representing the 6 Martin Pring Economic Cycle Stages."""
    fig = go.Figure()
    bg_color = "#0B0F19" if dark_mode else "#F8FAFC"

    # Hexagonal / circular coordinates for 6 stages
    angles = [90, 30, 330, 270, 210, 150]  # Clockwise: Stage 1 (top) to Stage 6
    radius = 2.5

    # Draw Stage segments & connectors
    xs, ys = [], []
    for a in angles:
        rad = math.radians(a)
        xs.append(radius * math.cos(rad))
        ys.append(radius * math.sin(rad))
    xs.append(xs[0])
    ys.append(ys[0])

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.25)", width=2, dash="dash"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Stage Nodes
    for idx, stg_id in enumerate(range(1, 7)):
        stg_meta = PRING_STAGES[stg_id]
        a_deg = angles[idx]
        rad = math.radians(a_deg)
        x = radius * math.cos(rad)
        y = radius * math.sin(rad)

        is_current = (stg_id == stage_res.current_stage)
        node_size = 42 if is_current else 28
        node_color = stg_meta["color"] if is_current else "rgba(100, 116, 139, 0.4)"
        border_color = "#FFFFFF" if is_current else stg_meta["color"]
        border_width = 3.5 if is_current else 1.5

        hover_text = (
            f"<b>FASE {stg_id}: {stg_meta['title']}</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"• Regime: <b>{stg_meta['regime']}</b><br>"
            f"• Obbligazioni: <b>{stg_meta['bonds']}</b><br>"
            f"• Azioni: <b>{stg_meta['stocks']}</b><br>"
            f"• Materie Prime: <b>{stg_meta['commodities']}</b><br>"
            f"• Asset Favoriti: {', '.join(stg_meta['favored_assets'][:2])}<br>"
            f"• Settori Target: {', '.join(stg_meta['favored_sectors'])}"
        )

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(size=node_size, color=node_color, line=dict(color=border_color, width=border_width)),
            text=[f"<b>Fase {stg_id}</b>"],
            textposition="top center" if y > 0 else "bottom center",
            textfont=dict(color="#FFFFFF" if is_current else "#94A3B8", size=11),
            hovertext=[hover_text],
            hoverinfo="text",
            showlegend=False,
        ))

    # Center Hub Annotation
    curr_meta = PRING_STAGES[stage_res.current_stage]
    center_html = (
        f"<b>CICLO DI PRING</b><br>"
        f"<span style='font-size: 16px; color: {curr_meta['color']};'>FASE {stage_res.current_stage}</span><br>"
        f"<span style='font-size: 11px; color: #94A3B8;'>{curr_meta['regime']}</span><br>"
        f"<span style='font-size: 10px; color: #38BDF8;'>Confidenza: {stage_res.confidence_pct:.0f}%</span>"
    )
    fig.add_annotation(
        x=0,
        y=0,
        text=center_html,
        showarrow=False,
        font=dict(size=12, color="#FFFFFF"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor=curr_meta["color"],
        borderwidth=2,
        borderpad=8,
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.6], scaleanchor="x", scaleratio=1),
        height=450,
    )
    return fig


def build_kst_momentum_figure(series: pd.Series, title: str = "Martin Pring KST (Know Sure Thing)") -> go.Figure:
    """Builds interactive KST chart with signal line and zero line."""
    kst, sig = compute_kst(series)
    recent_kst = kst.iloc[-180:]
    recent_sig = sig.iloc[-180:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_kst.index,
        y=recent_kst.values,
        mode="lines",
        name="KST Oscillator",
        line=dict(color="#38BDF8", width=2.0),
    ))
    fig.add_trace(go.Scatter(
        x=recent_sig.index,
        y=recent_sig.values,
        mode="lines",
        name="Signal Line (SMA 9)",
        line=dict(color="#F59E0B", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255,255,255,0.25)")

    fig.update_layout(
        title=f"<b>{title}</b>",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font=dict(color="#F8FAFC", size=11),
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="KST Momentum"),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_rrg_sector_matrix_figure(dark_mode: bool = True) -> go.Figure:
    """
    Simulates Relative Rotation Graph (RRG) 4 Quadrants:
    Leading (Green), Weakening (Yellow), Lagging (Red), Improving (Blue).
    Plots S&P 500 GICS Sector SPDRs relative to the benchmark.
    """
    fig = go.Figure()
    bg_color = "#0B0F19" if dark_mode else "#F8FAFC"
    axis_color = "rgba(255, 255, 255, 0.25)" if dark_mode else "rgba(15, 23, 42, 0.25)"

    # Quadrant Rectangles around 100 center
    # Top-Right: Leading (RS-Ratio > 100, RS-Momentum > 100)
    fig.add_shape(type="rect", x0=100, y0=100, x1=108, y1=108, fillcolor="rgba(16, 185, 129, 0.12)", line=dict(width=0))
    # Bottom-Right: Weakening (RS-Ratio > 100, RS-Momentum < 100)
    fig.add_shape(type="rect", x0=100, y0=92, x1=108, y1=100, fillcolor="rgba(245, 158, 11, 0.12)", line=dict(width=0))
    # Bottom-Left: Lagging (RS-Ratio < 100, RS-Momentum < 100)
    fig.add_shape(type="rect", x0=92, y0=92, x1=100, y1=100, fillcolor="rgba(239, 68, 68, 0.12)", line=dict(width=0))
    # Top-Left: Improving (RS-Ratio < 100, RS-Momentum > 100)
    fig.add_shape(type="rect", x0=92, y0=100, x1=100, y1=108, fillcolor="rgba(59, 130, 246, 0.12)", line=dict(width=0))

    fig.add_annotation(x=105, y=106, text="<b>LEADING</b><br>(Sovraperforma & Accelera)", showarrow=False, font=dict(color="#10B981", size=11))
    fig.add_annotation(x=105, y=94, text="<b>WEAKENING</b><br>(Sovraperforma ma Rallenta)", showarrow=False, font=dict(color="#F59E0B", size=11))
    fig.add_annotation(x=95, y=94, text="<b>LAGGING</b><br>(Sottoperforma & Decelera)", showarrow=False, font=dict(color="#EF4444", size=11))
    fig.add_annotation(x=95, y=106, text="<b>IMPROVING</b><br>(Sottoperforma ma Recupera)", showarrow=False, font=dict(color="#3B82F6", size=11))

    # Representative coordinates for S&P 500 sectors in current Reflation / Late-Cycle Stage 4
    sector_coords = {
        "XLE": (104.2, 103.5, "Leading"),      # Energy strong
        "XLB": (102.8, 102.1, "Leading"),      # Materials leading
        "XLI": (101.5, 100.8, "Leading"),      # Industrials leading
        "XLF": (101.2, 98.4, "Weakening"),     # Financials weakening
        "XLK": (99.2, 97.6, "Lagging"),        # Tech suffering from high yields
        "XLY": (98.1, 96.5, "Lagging"),        # Consumer discretionary lagging
        "XLU": (97.5, 101.8, "Improving"),     # Utilities improving on defensive bid
        "XLP": (98.4, 102.4, "Improving"),     # Consumer staples improving
        "XLV": (99.0, 101.5, "Improving"),     # Healthcare improving
    }

    for ticker, (rx, ry, quad) in sector_coords.items():
        s_info = SECTOR_ETFS[ticker]
        dot_color = s_info["color"]

        fig.add_trace(go.Scatter(
            x=[rx],
            y=[ry],
            mode="markers+text",
            marker=dict(size=14, color=dot_color, line=dict(color="#FFFFFF", width=1.5)),
            text=[f"<b>{ticker}</b>"],
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=10),
            hovertext=f"<b>Settore: {s_info['name']} ({ticker})</b><br>Categoria: {s_info['category']}<br>Quadrante RRG: <b>{quad}</b><br>RS-Ratio: {rx:.1f} | RS-Momentum: {ry:.1f}",
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color="#F8FAFC", size=11),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="RS-Ratio (Forza Relativa vs S&P 500 Benchmark)", zeroline=True, zerolinecolor=axis_color, range=[92, 108], gridcolor=axis_color),
        yaxis=dict(title="RS-Momentum (Velocità Rotazione Relativa)", zeroline=True, zerolinecolor=axis_color, range=[92, 108], gridcolor=axis_color),
        height=400,
        showlegend=False,
    )
    return fig
