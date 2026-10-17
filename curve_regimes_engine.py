"""
Yield Curve Regimes Engine (Steno Research / Macro Framework):
Analyzes the four yield curve regimes ranked for the business cycle:
#1 BULL STEEPENER  : Short end drops, curve steepens (Fed easing, cycle expansion/Trump-Bessent dream)
#2 BEAR STEEPENER  : Long end up more than short (Inflation expectations / term premium, bank engine runs)
#3 BULL FLATTENER  : Term premia compress, long end drops (Disinflation, scarce-asset heaven)
#4 BEAR FLATTENER  : Short end up more than long (Monetary tightening, cycle killer)

"Steepeners feed the credit engine.. flatteners starve it."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from data_fetcher import db, fetcher


@dataclass
class YieldCurvePoint:
    tenor: str
    maturity_years: float
    current_yield: float
    prev_yield: float
    delta_bps: float
    source: str


@dataclass
class YieldCurveRegimeState:
    regime_id: str          # "BULL_STEEPENER", "BEAR_STEEPENER", "BULL_FLATTENER", "BEAR_FLATTENER"
    regime_name: str        # e.g. "#2 BEAR STEEPENER"
    rank: int               # 1, 2, 3, or 4
    headline: str           # e.g. "Long end up more than short.. the bank engine still runs"
    short_rate_delta_bps: float
    long_rate_delta_bps: float
    slope_delta_bps: float  # (Long - Short) change
    current_slope_2s10s_bps: float
    current_slope_3m30y_bps: float
    curve_points: List[YieldCurvePoint]
    comparison_period: str  # e.g. "1 Mese fa (30d)"
    impact_credit_engine: str # "Alimenta il motore del credito (Espansivo)" vs "Strozza il credito (Restrittivo)"
    favored_assets: List[str]
    unfavored_assets: List[str]
    macro_description: str
    color_hex: str


REGIMES_INFO = {
    "BULL_STEEPENER": {
        "name": "#1 BULL STEEPENER",
        "rank": 1,
        "headline": "Short end drops, curve steepens.. the Trump/Bessent dream",
        "credit_engine": "🟢 Alimenta vigorosamente il credito bancario (Margini di intermediazione in rapido aumento)",
        "color": "#10B981", # Emerald green
        "favored": ["Azioni Cicliche & Small Cap (Russell 2000)", "Banche & Finanziari", "Real Estate / REITs", "Oro / Asset Scarsi"],
        "unfavored": ["Liquidità monetaria / T-Bills (rendimento in calo)", "Dollaro USA (DXY)"],
        "desc": "La parte a breve della curva crolla a seguito di tagli aggressivi dei tassi della Fed, mentre la parte a lunga tiene o scende meno. È il regime più favorevole alla crescita economica, agli investimenti e al credito."
    },
    "BEAR_STEEPENER": {
        "name": "#2 BEAR STEEPENER",
        "rank": 2,
        "headline": "Long end up more than short.. the bank engine still runs",
        "credit_engine": "🟡 Il motore bancario funziona ancora (Curve ripida favorisce il margine di interesse)",
        "color": "#3B82F6", # Bright Blue
        "favored": ["Bancari & Assicurativi (Margine d'interesse)", "Materie Prime (Commodities)", "Energy & Materiali", "Titoli Value"],
        "unfavored": ["TLT / Titoli di Stato a Lunga Duration (Perdite in conto capitale)", "Growth ad alta duration", "REITs ad alto indebitamento"],
        "desc": "I tassi a lungo termine salgono più velocemente di quelli a breve, trainati da maggiori aspettative di inflazione, emissioni di debito pubblico elevate e premio a termine. La curva si irripidisce: ottimo per i margini bancari, ma punitivo per il reddito fisso a lunga scadenza (TLT)."
    },
    "BULL_FLATTENER": {
        "name": "#3 BULL FLATTENER",
        "rank": 3,
        "headline": "Term premia compress, long end drops.. scarce-asset heaven",
        "credit_engine": "🟠 Compressione dei margini di credito.. il motore rallenta",
        "color": "#F59E0B", # Amber
        "favored": ["TLT / Long Treasuries (Forte rally dei prezzi)", "Titoli Growth / Big Tech ad alta duration", "Asset scarsi / Oro", "Defensive / Quality"],
        "unfavored": ["Banche regionali e commerciali (Margini compressi)", "Materie prime cicliche", "Small Cap"],
        "desc": "I tassi a lunga scadenza scendono sensibilmente mentre i tassi a breve rimangono stabili o scendono meno. Segnala disinflazione e rallentamento economico. I premi al termine si comprimono e gli asset a lunga duration registrano forti guadagni."
    },
    "BEAR_FLATTENER": {
        "name": "#4 BEAR FLATTENER",
        "rank": 4,
        "headline": "Short end up more than long.. the cycle KILLER",
        "credit_engine": "🔴 Strozza il motore del credito bancario (Inversione della curva, contrazione del credito)",
        "color": "#EF4444", # Red
        "favored": ["Cash / T-Bills a 1-3 mesi (Rendimento elevato senza duration)", "Dollaro USA (DXY)", "Titoli difensivi a basso debito"],
        "unfavored": ["Azioni Cicliche e Azionario Globale", "High Yield & Corporate Debito", "Real Estate", "TLT & Obbligazioni a lungo termine"],
        "desc": "La Fed e le banche centrali alzano aggressivamente i tassi a breve per soffocare l'inflazione, mentre la parte a lunga non segue per paura di recessione. La curva si appiattisce o si inverte: è il classico 'Cycle Killer' che precede le recessioni."
    }
}


class YieldCurveEngine:
    """Engine for fetching yield curve tenors and classifying the yield curve regime."""

    def __init__(self):
        # Default fallback historical curve if network fails
        self.tenor_defs = [
            {"tenor": "3M", "years": 0.25, "yahoo": "^IRX", "fred": "DGS3MO", "default": 4.07, "default_prev": 4.25},
            {"tenor": "2Y", "years": 2.0,  "yahoo": "2YY=F", "fred": "DGS2",   "default": 4.75, "default_prev": 4.60},
            {"tenor": "5Y", "years": 5.0,  "yahoo": "^FVX", "fred": "DGS5",   "default": 4.90, "default_prev": 4.70},
            {"tenor": "10Y", "years": 10.0, "yahoo": "^TNX", "fred": "DGS10",  "default": 5.10, "default_prev": 4.85},
            {"tenor": "30Y", "years": 30.0, "yahoo": "^TYX", "fred": "DGS30",  "default": 5.44, "default_prev": 5.15},
        ]

    def get_curve_regime_state(self, lookback_days: int = 30) -> YieldCurveRegimeState:
        """
        Fetches current and lookback yield points, computes deltas, and identifies regime.
        """
        points: List[YieldCurvePoint] = []

        # Attempt to pull real data from Yahoo Finance
        yahoo_symbols = [d["yahoo"] for d in self.tenor_defs]
        data_hist = {}
        try:
            df = yf.download(yahoo_symbols, period="3mo", progress=False)["Close"]
            if not df.empty:
                for d in self.tenor_defs:
                    sym = d["yahoo"]
                    if sym in df.columns:
                        s = df[sym].dropna()
                        if len(s) >= 2:
                            curr_v = float(s.iloc[-1])
                            idx_prev = max(0, len(s) - 22) # ~1 month trading days
                            prev_v = float(s.iloc[idx_prev])
                            data_hist[d["tenor"]] = (curr_v, prev_v)
        except Exception:
            pass

        # Check SQLite db for 2Y and 10Y if missing
        if "2Y" not in data_hist:
            s2 = db.load_series("YIELD_US02")
            if s2 is not None and len(s2) >= 2:
                idx_prev = max(0, len(s2) - 22)
                data_hist["2Y"] = (float(s2.iloc[-1]), float(s2.iloc[idx_prev]))
        if "10Y" not in data_hist:
            s10 = db.load_series("YIELD_US10")
            if s10 is not None and len(s10) >= 2:
                idx_prev = max(0, len(s10) - 22)
                data_hist["10Y"] = (float(s10.iloc[-1]), float(s10.iloc[idx_prev]))

        # Build YieldCurvePoint list
        for d in self.tenor_defs:
            tenor = d["tenor"]
            if tenor in data_hist:
                curr, prev = data_hist[tenor]
            else:
                curr, prev = d["default"], d["default_prev"]

            delta_bps = (curr - prev) * 100.0
            points.append(
                YieldCurvePoint(
                    tenor=tenor,
                    maturity_years=d["years"],
                    current_yield=curr,
                    prev_yield=prev,
                    delta_bps=delta_bps,
                    source="Yahoo Finance / CBOE / FRED",
                )
            )

        # Calculate key spreads and deltas
        # Short rate: 3M or 2Y
        # Long rate: 10Y or 30Y
        p_3m = next((p for p in points if p.tenor == "3M"), points[0])
        p_2y = next((p for p in points if p.tenor == "2Y"), points[1])
        p_10y = next((p for p in points if p.tenor == "10Y"), points[3])
        p_30y = next((p for p in points if p.tenor == "30Y"), points[4])

        delta_short = p_2y.delta_bps # standard 2Y short rate delta
        delta_long = p_10y.delta_bps  # standard 10Y long rate delta
        delta_slope = delta_long - delta_short # > 0 -> Steepening, < 0 -> Flattening

        current_2s10s_bps = (p_10y.current_yield - p_2y.current_yield) * 100.0
        current_3m30y_bps = (p_30y.current_yield - p_3m.current_yield) * 100.0

        # Classify Regime according to Steno Research 4 regimes:
        # Steepener vs Flattener: delta_slope > 0 vs delta_slope <= 0
        # Bull vs Bear: overall rate direction.
        # Specifically:
        # Bull Steepener: Short end drops, curve steepens (delta_short < 0 and delta_slope > 0)
        # Bear Steepener: Long end up more than short (delta_long > delta_short and delta_long > 0)
        # Bull Flattener: Long end drops more than short (delta_long < 0 and delta_slope <= 0)
        # Bear Flattener: Short end up more than long (delta_short > delta_long and delta_short > 0)

        if delta_slope > 0:
            # Steepening
            if delta_short < 0:
                reg_id = "BULL_STEEPENER"
            else:
                reg_id = "BEAR_STEEPENER"
        else:
            # Flattening
            if delta_long < 0:
                reg_id = "BULL_FLATTENER"
            else:
                reg_id = "BEAR_FLATTENER"

        info = REGIMES_INFO[reg_id]

        return YieldCurveRegimeState(
            regime_id=reg_id,
            regime_name=info["name"],
            rank=info["rank"],
            headline=info["headline"],
            short_rate_delta_bps=delta_short,
            long_rate_delta_bps=delta_long,
            slope_delta_bps=delta_slope,
            current_slope_2s10s_bps=current_2s10s_bps,
            current_slope_3m30y_bps=current_3m30y_bps,
            curve_points=points,
            comparison_period="Ultimi 30 Giorni",
            impact_credit_engine=info["credit_engine"],
            favored_assets=info["favored"],
            unfavored_assets=info["unfavored"],
            macro_description=info["desc"],
            color_hex=info["color"],
        )


def build_yield_curve_comparison_figure(state: YieldCurveRegimeState, dark_mode: bool = True) -> go.Figure:
    """
    Renders an institutional yield curve chart comparing 'Before' (Dashed Gray) and 'After/Current' (Solid Colored).
    Directly replicates the visual grammar of the Steno Research 4-regime framework.
    """
    bg_color = "#0B0F19" if dark_mode else "#FFFFFF"
    paper_color = "#0F172A" if dark_mode else "#F8FAFC"
    text_color = "#F8FAFC" if dark_mode else "#0F172A"
    grid_color = "rgba(255, 255, 255, 0.08)" if dark_mode else "rgba(0, 0, 0, 0.08)"

    tenors = [p.tenor for p in state.curve_points]
    curr_yields = [p.current_yield for p in state.curve_points]
    prev_yields = [p.prev_yield for p in state.curve_points]

    fig = go.Figure()

    # Trace 1: Before (Dashed Gray)
    fig.add_trace(go.Scatter(
        x=tenors,
        y=prev_yields,
        mode="lines+markers",
        name="Before (1 Mese fa)",
        line=dict(color="#94A3B8", width=3, dash="dash"),
        marker=dict(size=8, color="#94A3B8"),
        hovertemplate="<b>%{x} Prima</b>: %{y:.2f}%<extra></extra>",
    ))

    # Trace 2: After (Current Solid colored line)
    fig.add_trace(go.Scatter(
        x=tenors,
        y=curr_yields,
        mode="lines+markers",
        name=f"After / Attuale ({state.regime_name})",
        line=dict(color=state.color_hex, width=4),
        marker=dict(size=11, color=state.color_hex, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>%{x} Attuale</b>: %{y:.2f}% (Var: %{customdata:+.1f} bps)<extra></extra>",
        customdata=[p.delta_bps for p in state.curve_points]
    ))

    # Add vertical delta arrows for short end and long end
    p_3m = state.curve_points[0]
    p_30y = state.curve_points[-1]

    # Arrow for 3M
    fig.add_annotation(
        x=p_3m.tenor,
        y=p_3m.current_yield,
        ax=p_3m.tenor,
        ay=p_3m.prev_yield,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#38BDF8" if p_3m.delta_bps < 0 else "#EF4444",
    )

    # Arrow for 30Y
    fig.add_annotation(
        x=p_30y.tenor,
        y=p_30y.current_yield,
        ax=p_30y.tenor,
        ay=p_30y.prev_yield,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#EF4444" if p_30y.delta_bps > 0 else "#38BDF8",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Curva dei Rendimenti USA: {state.regime_name}</b><br><span style='font-size: 13px; color: #94A3B8;'>{state.headline}</span>",
            font=dict(size=18, color=text_color),
            x=0.02,
            y=0.95
        ),
        xaxis=dict(
            title=dict(text="Scadenza (Maturity)", font=dict(size=13, color=text_color)),
            gridcolor=grid_color,
            zeroline=False,
            tickfont=dict(size=13, color=text_color),
        ),
        yaxis=dict(
            title=dict(text="Rendimento Nominale (%)", font=dict(size=13, color=text_color)),
            gridcolor=grid_color,
            zeroline=False,
            ticksuffix="%",
            tickfont=dict(size=13, color=text_color),
        ),
        plot_bgcolor=paper_color,
        paper_bgcolor=bg_color,
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=text_color)
        ),
        hovermode="x unified",
        height=430,
    )

    return fig


def build_regimes_matrix_preview_figure(current_regime_id: str, dark_mode: bool = True) -> go.Figure:
    """
    Renders a 2x2 grid representing the 4 yield curve regimes (Bull Steepener, Bear Steepener, Bull Flattener, Bear Flattener)
    highlighting the currently detected active regime.
    """
    fig = go.Figure()

    # Define quadrant boxes
    boxes = [
        # x0, x1, y0, y1, regime_id, rank, title, subtitle, color
        (-1, 0, 0, 1, "BULL_STEEPENER", "#1", "BULL STEEPENER", "Short end drops, curve steepens\nTrump/Bessent dream", "#10B981"),
        (0, 1, 0, 1, "BEAR_STEEPENER", "#2", "BEAR STEEPENER", "Long end up more than short\nBank engine runs", "#3B82F6"),
        (-1, 0, -1, 0, "BULL_FLATTENER", "#3", "BULL FLATTENER", "Term premia compress, long drops\nScarce-asset heaven", "#F59E0B"),
        (0, 1, -1, 0, "BEAR_FLATTENER", "#4", "BEAR FLATTENER", "Short end up more than long\nThe cycle KILLER", "#EF4444"),
    ]

    for x0, x1, y0, y1, r_id, rank, title, sub, col in boxes:
        is_active = (r_id == current_regime_id)
        fill_alpha = "0.28" if is_active else "0.07"
        border_col = col if is_active else "rgba(255, 255, 255, 0.2)"
        border_w = 3.5 if is_active else 1.0

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=f"rgba({int(col[1:3], 16)}, {int(col[3:5], 16)}, {int(col[5:7], 16)}, {fill_alpha})",
            line=dict(color=border_col, width=border_w),
        )

        badge_text = "<b>★ REGIME ATTUALE ★</b><br>" if is_active else ""
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=f"{badge_text}<span style='font-size: 15px; font-weight: bold; color: {col};'>{rank} {title}</span><br><br><span style='font-size: 11px; color: #94A3B8;'>{sub}</span>",
            showarrow=False,
            align="center",
            font=dict(color="#F8FAFC", size=12),
        )

    fig.update_layout(
        title=dict(
            text="<b>Mappa Sinottica dei 4 Regimi della Curva (Steno Research Framework)</b>",
            font=dict(size=15, color="#F8FAFC"),
            x=0.02,
        ),
        xaxis=dict(
            title=dict(text="← Bull (Rendimenti in Calo)  |  Bear (Rendimenti in Rialzo) →", font=dict(size=12, color="#94A3B8")),
            range=[-1.05, 1.05],
            showgrid=False,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.25)",
            showticklabels=False,
        ),
        yaxis=dict(
            title=dict(text="← Flattener (Curva Piatta)  |  Steepener (Curva Ripida) →", font=dict(size=12, color="#94A3B8")),
            range=[-1.05, 1.05],
            showgrid=False,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.25)",
            showticklabels=False,
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0B0F19",
        margin=dict(l=60, r=40, t=50, b=50),
        height=320,
    )

    return fig


yield_curve_engine = YieldCurveEngine()
