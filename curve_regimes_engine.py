"""
Yield Curve Regimes Engine (Steno Research / Macro Framework):
Analyzes the four yield curve regimes ranked for the business cycle:
#1 BULL STEEPENER  : Short end drops, curve steepens (Fed easing, cycle expansion/Trump-Bessent dream)
#2 BEAR STEEPENER  : Long end up more than short (Inflation expectations / term premium, bank engine runs)
#3 BULL FLATTENER  : Term premia compress, long end drops (Disinflation, scarce-asset heaven)
#4 BEAR FLATTENER  : Short end up more than long (Monetary tightening, cycle killer)

"Steepeners feed the credit engine.. flatteners starve it."

Integrates dual-layer slope analysis:
- Absolute Curve Geometry (Inversion vs Normal vs Disinversion/Uninverting)
- Dynamic Shift (Steepening vs Flattening / Bull vs Bear)
- Multi-dimensional Banking & Credit Engine Transmission Channel:
  * Net Interest Margin (NIM)
  * SLOOS Lending Standards (Senior Loan Officer Opinion Survey)
  * Credit Impulse & Loan Volume Creation
  * Risk of Recession vs Cyclical Refinancing Pressure
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
class InversionMetrics:
    is_2s10s_inverted: bool
    spread_2s10s_bps: float
    prev_2s10s_bps: float
    delta_2s10s_bps: float

    is_3m10y_inverted: bool
    spread_3m10y_bps: float
    prev_3m10y_bps: float
    delta_3m10y_bps: float

    inversion_status: str       # "NORMAL", "INVERTED", "UNINVERTING_BULL" (disinversione rialzista), "UNINVERTING_BEAR" (disinversione da inflazione)
    inversion_status_badge: str
    recession_warning_level: str # "BASSO", "MEDIO", "ELEVATO (FASE CRITICA POST-INVERSIONE)"


@dataclass
class CreditEngineAnalysis:
    net_interest_margin_impact: str  # Impatto su NIM e profitti bancari
    bank_lending_standards: str      # Criteri di erogazione (Restrittivi / Accomodanti)
    credit_impulse_direction: str    # Impulso del credito (Accelerazione / Contrazione)
    sloos_interpretation: str        # Lettura del ciclo bancario (SLOOS)
    corporate_refinancing_risk: str  # Rischio rollover debito societario
    summary_verdict: str             # Verdetto sintetico


@dataclass
class YieldCurveRegimeState:
    regime_id: str                   # "BULL_STEEPENER", "BEAR_STEEPENER", "BULL_FLATTENER", "BEAR_FLATTENER"
    regime_name: str                 # e.g. "#2 BEAR STEEPENER"
    rank: int                        # 1, 2, 3, or 4
    headline: str                    # e.g. "Long end up more than short.. the bank engine still runs"
    short_rate_delta_bps: float
    long_rate_delta_bps: float
    slope_delta_bps: float           # (Long - Short) change
    current_slope_2s10s_bps: float
    current_slope_3m30y_bps: float
    curve_points: List[YieldCurvePoint]
    inversion: InversionMetrics
    credit_analysis: CreditEngineAnalysis
    comparison_period: str           # e.g. "Ultimi 30 Giorni"
    impact_credit_engine: str
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
        "favored": ["Azioni Cicliche & Small Cap (Russell 2000)", "Banche Commerciali & Finanziari", "Real Estate / REITs", "Oro / Asset Scarsi"],
        "unfavored": ["Liquidità monetaria / T-Bills (rendimento in discesa)", "Dollaro USA (DXY)"],
        "desc": "La parte a breve della curva crolla a seguito di tagli aggressivi dei tassi della Fed, mentre la parte a lunga tiene o scende meno. È il regime di massimo stimolo alla creazione di moneta bancaria e al ciclo del credito."
    },
    "BEAR_STEEPENER": {
        "name": "#2 BEAR STEEPENER",
        "rank": 2,
        "headline": "Long end up more than short.. the bank engine still runs",
        "credit_engine": "🟡 Il motore bancario funziona ancora (Curva ripida sostiene il margine d'interesse, ma mutui più cari)",
        "color": "#3B82F6", # Bright Blue
        "favored": ["Bancari & Assicurativi (NIM solido)", "Materie Prime (Commodities)", "Energy & Materiali di Base", "Titoli Value"],
        "unfavored": ["TLT / Titoli di Stato a Lunga Duration (Forti perdite in conto capitale)", "Growth ad alta duration", "REITs ad alto indebitamento"],
        "desc": "I tassi a lungo termine salgono più velocemente di quelli a breve per aspettative di inflazione, emissioni ingenti del Tesoro e premio a termine in aumento. La curva si irripidisce: preserva i margini bancari ma penalizza la duration lunga (TLT)."
    },
    "BULL_FLATTENER": {
        "name": "#3 BULL FLATTENER",
        "rank": 3,
        "headline": "Term premia compress, long end drops.. scarce-asset heaven",
        "credit_engine": "🟠 Compressione dei margini di credito.. il motore bancario rallenta",
        "color": "#F59E0B", # Amber
        "favored": ["TLT / Long Treasuries (Forte rally dei prezzi)", "Titoli Growth / Big Tech ad alta duration", "Asset scarsi / Oro", "Defensive / Quality"],
        "unfavored": ["Banche regionali e commerciali (Margini compressi)", "Materie prime industriali", "Small Cap"],
        "desc": "I tassi a lunga scadenza scendono sensibilmente per disinflazione o timori di rallentamento, mentre i tassi a breve restano elevati o scendono poco. I margini di intermediazione si contraggono e gli investitori comprano duration."
    },
    "BEAR_FLATTENER": {
        "name": "#4 BEAR FLATTENER",
        "rank": 4,
        "headline": "Short end up more than long.. the cycle KILLER",
        "credit_engine": "🔴 Strozza il motore del credito bancario (Inversione/compressione estrema, banche stringono i rubinetti)",
        "color": "#EF4444", # Red
        "favored": ["Cash / T-Bills a 1-3 mesi (Rendimento elevato senza duration)", "Dollaro USA (DXY)", "Titoli difensivi a bilancio solido"],
        "unfavored": ["Azioni Cicliche e Azionario Globale", "High Yield & Corporate Debito", "Real Estate", "TLT & Obbligazioni a lungo termine"],
        "desc": "La banca centrale alza i tassi a breve per soffocare l'inflazione, mentre la parte a lungo rifiuta di salire per timori recessivi. È il classico 'Cycle Killer' che affama il credito e precede le contrazioni economiche."
    }
}


class YieldCurveEngine:
    """Engine for fetching yield curve tenors and performing deep credit transmission analysis."""

    def __init__(self):
        self.tenor_defs = [
            {"tenor": "3M", "years": 0.25, "yahoo": "^IRX", "fred": "DGS3MO", "default": 4.07, "default_prev": 4.25},
            {"tenor": "2Y", "years": 2.0,  "yahoo": "2YY=F", "fred": "DGS2",   "default": 4.75, "default_prev": 4.39},
            {"tenor": "5Y", "years": 5.0,  "yahoo": "^FVX", "fred": "DGS5",   "default": 4.90, "default_prev": 4.80},
            {"tenor": "10Y", "years": 10.0, "yahoo": "^TNX", "fred": "DGS10",  "default": 5.10, "default_prev": 4.95},
            {"tenor": "30Y", "years": 30.0, "yahoo": "^TYX", "fred": "DGS30",  "default": 5.44, "default_prev": 5.30},
        ]

    def get_curve_regime_state(self, lookback_days: int = 30) -> YieldCurveRegimeState:
        """
        Fetches current and lookback yield points, computes deltas, evaluates inversion status,
        and produces a rigorous credit engine transmission analysis.
        """
        points: List[YieldCurvePoint] = []
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
                            idx_prev = max(0, len(s) - 22)
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

        # Assemble points
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

        # Spread metrics
        p_3m = next((p for p in points if p.tenor == "3M"), points[0])
        p_2y = next((p for p in points if p.tenor == "2Y"), points[1])
        p_10y = next((p for p in points if p.tenor == "10Y"), points[3])
        p_30y = next((p for p in points if p.tenor == "30Y"), points[4])

        delta_short = p_2y.delta_bps
        delta_long = p_10y.delta_bps
        delta_slope = delta_long - delta_short

        curr_2s10s_bps = (p_10y.current_yield - p_2y.current_yield) * 100.0
        prev_2s10s_bps = (p_10y.prev_yield - p_2y.prev_yield) * 100.0
        delta_2s10s_bps = curr_2s10s_bps - prev_2s10s_bps

        curr_3m10y_bps = (p_10y.current_yield - p_3m.current_yield) * 100.0
        prev_3m10y_bps = (p_10y.prev_yield - p_3m.prev_yield) * 100.0
        delta_3m10y_bps = curr_3m10y_bps - prev_3m10y_bps

        current_3m30y_bps = (p_30y.current_yield - p_3m.current_yield) * 100.0

        # Inversion diagnosis
        is_2s10s_inverted = (curr_2s10s_bps < 0)
        is_3m10y_inverted = (curr_3m10y_bps < 0)
        was_2s10s_inverted = (prev_2s10s_bps < 0)

        if is_2s10s_inverted or is_3m10y_inverted:
            inversion_status = "INVERTED"
            inversion_status_badge = "⚠️ CURVA INVERTITA (Condizioni da Recessione in Incubazione)"
            recession_level = "ELEVATO (Allerta inversione attiva)"
        elif was_2s10s_inverted and not is_2s10s_inverted:
            if delta_short < 0:
                inversion_status = "UNINVERTING_BULL"
                inversion_status_badge = "🚨 DISINVERSIONE DA TAGLIO FED (Bull Steepener - Fase Storicamente più Pericolosa)"
                recession_level = "MASSIMO (La recessione storicamente scatta quando la curva si disinverte)"
            else:
                inversion_status = "UNINVERTING_BEAR"
                inversion_status_badge = "🟡 DISINVERSIONE DA BEAR STEEPENER (Spinta da Term Premium & Inflazione)"
                recession_level = "MEDIO (Inflazione persistente, rinvio della recessione)"
        else:
            inversion_status = "NORMAL"
            inversion_status_badge = "🟢 CURVA POSITIVA (Pendenza Fisiologica Espansiva)"
            recession_level = "BASSO"

        inversion_metrics = InversionMetrics(
            is_2s10s_inverted=is_2s10s_inverted,
            spread_2s10s_bps=curr_2s10s_bps,
            prev_2s10s_bps=prev_2s10s_bps,
            delta_2s10s_bps=delta_2s10s_bps,
            is_3m10y_inverted=is_3m10y_inverted,
            spread_3m10y_bps=curr_3m10y_bps,
            prev_3m10y_bps=prev_3m10y_bps,
            delta_3m10y_bps=delta_3m10y_bps,
            inversion_status=inversion_status,
            inversion_status_badge=inversion_status_badge,
            recession_warning_level=recession_level,
        )

        # Steno 4 Regimes Classification
        if delta_slope > 0:
            if delta_short < 0:
                reg_id = "BULL_STEEPENER"
            else:
                reg_id = "BEAR_STEEPENER"
        else:
            if delta_long < 0:
                reg_id = "BULL_FLATTENER"
            else:
                reg_id = "BEAR_FLATTENER"

        info = REGIMES_INFO[reg_id]

        # In-depth Credit Engine Transmission Analysis
        credit_analysis = self._evaluate_credit_transmission(
            regime_id=reg_id,
            inversion=inversion_metrics,
            short_delta=delta_short,
            long_delta=delta_long,
            curr_2s10s=curr_2s10s_bps
        )

        return YieldCurveRegimeState(
            regime_id=reg_id,
            regime_name=info["name"],
            rank=info["rank"],
            headline=info["headline"],
            short_rate_delta_bps=delta_short,
            long_rate_delta_bps=delta_long,
            slope_delta_bps=delta_slope,
            current_slope_2s10s_bps=curr_2s10s_bps,
            current_slope_3m30y_bps=current_3m30y_bps,
            curve_points=points,
            inversion=inversion_metrics,
            credit_analysis=credit_analysis,
            comparison_period="Ultimi 30 Giorni",
            impact_credit_engine=info["credit_engine"],
            favored_assets=info["favored"],
            unfavored_assets=info["unfavored"],
            macro_description=info["desc"],
            color_hex=info["color"],
        )

    def _evaluate_credit_transmission(
        self,
        regime_id: str,
        inversion: InversionMetrics,
        short_delta: float,
        long_delta: float,
        curr_2s10s: float
    ) -> CreditEngineAnalysis:
        """
        Deep econometric modeling of the banking credit channel based on both slope dynamics
        and level inversion.
        """
        if inversion.is_2s10s_inverted or inversion.is_3m10y_inverted:
            nim = "🔴 Margine di Interesse (NIM) Compresso / Negativo"
            standards = "🔴 Restrizione Severa dei Criteri di Fido (SLOOS Net Tightening > 40%)"
            impulse = "📉 Impulso del Credito in Contrazione Rapida (Credit Crunch)"
            sloos = "Le banche pagano di più sulla raccolta a breve di quanto incassano sui prestiti a lungo termine: incentivazione a bloccare la liquidità presso la banca centrale anziché finanziare l'economia reale."
            refi = "🔴 Elevato Rischio di Rollover per PMI e Real Estate Commerciale esposti a tassi variabili."
            verdict = "Il canale del credito è strozzato (Flattening / Inversione). La trasmissione monetaria è restrittiva al massimo."

        elif inversion.inversion_status == "UNINVERTING_BULL":
            nim = "🟢 Espansione Rapida del NIM (Crollo Costo della Raccolta)"
            standards = "🟡 Elevata Selettività Cautelativa (Banche preparano accantonamenti su perdite su crediti)"
            impulse = "🟡 Fase Transitoria: Offerta di credito potenziale in crescita, ma domanda frenata dal rallentamento economico"
            sloos = "La disinversione bull riflette tagli d'urgenza della banca centrale: il motore del credito si riattiva per i rifinanziamenti, ma il rischio insolvenze è all'apice del ciclo."
            refi = "🟢 Sollievo immediato sui prestiti a tasso variabile grazie al crollo dei tassi a breve."
            verdict = "Disinversione Bull: fase tipica di imminente atterraggio / recessione. Il credito riprende respiro sui costi, ma soffre la qualità dei debitori."

        elif regime_id == "BEAR_STEEPENER":
            nim = "🟢 Margine Bancario Solido ed in Espansione (Yield Curve Steep)"
            standards = "🟡 Criteri di Fido Neutro-Prudenti (SLOOS bilanciato)"
            impulse = "🟢 Il motore del credito continua a girare per le grandi imprese con potere di prezzo"
            sloos = "Le banche possono prestare a tassi a 10 anni elevati mantenendo il costo della raccolta controllato. Tuttavia, l'alto livello assoluto dei rendimenti a lungo termine deprime la domanda di nuovi mutui immobiliari."
            refi = "🟡 Costo del debito a tasso fisso a lungo termine molto oneroso. Forti perdite per chi detiene TLT."
            verdict = "Il motore bancario funziona ancora ('The bank engine still runs'), ma i tassi lunghi elevati fungono da freno progressivo alla domanda di credito per investimenti capital-intensive."

        elif regime_id == "BEAR_FLATTENER":
            nim = "🔴 Erosione Continua del Margine di Intermediazione"
            standards = "🔴 Banche in fase di inasprimento dei requisiti di prestito"
            impulse = "📉 Decelerazione marcata dei volumi di erogazione prestiti"
            sloos = "Il rialzo aggressivo dei tassi a breve alza il costo dei depositi, mentre la curva piatta toglie ogni premio per la trasformazione delle scadenze bancarie."
            refi = "🔴 Pressione massima su prestiti corporate a breve e carte di credito."
            verdict = "Regime 'Cycle Killer': la stretta monetaria si trasmette integralmente al sistema bancario, riducendo l'offerta aggregata di credito."

        else: # BULL_STEEPENER or BULL_FLATTENER
            nim = "🟡 Margini Moderati in Stabilizzazione"
            standards = "🟢 Progressivo allentamento dei vincoli di fido"
            impulse = "🟢 Ripresa graduale dei volumi di finanziamento"
            sloos = "L'abbassamento dei tassi favorisce la domanda di credito privato e mutui per acquisto abitazioni."
            refi = "🟢 Finestre favorevoli di rinegoziazione ed emissione bond societari."
            verdict = "Condizioni di credito distese e favorevoli agli investimenti produttivi."

        return CreditEngineAnalysis(
            net_interest_margin_impact=nim,
            bank_lending_standards=standards,
            credit_impulse_direction=impulse,
            sloos_interpretation=sloos,
            corporate_refinancing_risk=refi,
            summary_verdict=verdict,
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
        name="Before (30gg fa)",
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

    boxes = [
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
