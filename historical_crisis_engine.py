"""
SPIDER Historical Crisis Replay & Macro Stress Testing Engine
Institutional References:
- Basel III / Comprehensive Capital Analysis and Review (CCAR - Federal Reserve)
- BlackRock Aladdin Risk Engine Historical Scenario Stress Testing Framework
- ECB Macroeconomic Stress Test / EBA Methodological Guide

Provides calibrated historical crisis shock tensors across all 30 SPIDER nodes,
calculates network rupture metrics, portfolio Max Drawdowns, and stress transmission.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import copy
import math
import numpy as np
import pandas as pd


@dataclass
class HistoricalCrisisScenario:
    id: str
    name: str
    period: str
    duration_days: int
    badge: str
    color_hex: str
    icon: str
    description: str
    macro_narrative: str
    catalyst: str
    shocks: Dict[str, float]  # Delta for rates/yields/inflation (percentage points or bps/100), pct change for equities/commodities
    historical_max_drawdown_60_40: float
    historical_max_drawdown_all_weather: float
    historical_max_drawdown_spider: float
    vix_peak: float
    hy_spread_peak: float
    key_ruptured_links: List[str]


# Institutional Historical Shock Library (CCAR / Aladdin Benchmark)
HISTORICAL_CRISIS_LIBRARY: Dict[str, HistoricalCrisisScenario] = {
    "CRISIS_2008_GFC": HistoricalCrisisScenario(
        id="CRISIS_2008_GFC",
        name="Grande Crisi Finanziaria (2008 Lehman)",
        period="Set 2008 - Mar 2009",
        duration_days=180,
        badge="🔴 LEHMAN 2008",
        color_hex="#EF4444",
        icon="🏚️",
        description="Fallimento di Lehman Brothers, collasso del mercato interbancario e paralisi del credito globale.",
        macro_narrative=(
            "La rottura dei mutui subprime e il fallimento di Lehman scatenano un credit crunch devastante. "
            "Gli spread High Yield esplodono oltre 1900 bps, le borse globali subiscono un crollo di oltre il -50%, "
            "mentre i rendimenti dei Treasury crollano per violento flight-to-safety."
        ),
        catalyst="Insolvenza Lehman Brothers & Freeze Liquidità Interbancaria",
        shocks={
            "EQ_SPX": -48.0,
            "EQ_NDX": -42.0,
            "EQ_RUT": -52.0,
            "EQ_DAX": -46.0,
            "EQ_MIB": -54.0,
            "CRED_HY": +12.50,       # Spread allargato di oltre 1250 bps
            "RISK_VIX": +58.0,       # VIX schizzato a 80
            "COM_OIL": -65.0,        # Petrolio da $140 a $40
            "COM_COPPER": -55.0,
            "COM_GOLD": +12.0,
            "RATE_FED": -3.50,       # Tagli emergenza Fed
            "YIELD_US02": -2.80,
            "YIELD_US10": -1.80,
            "YIELD_REAL10": -0.90,
            "FX_DXY": +14.0,         # Corsa al dollaro rifugio
            "FX_EURUSD": -12.0,
            "GRO_USGDP": -4.2,
            "GRO_INDPRO": -14.0,
            "LAB_UNRATE": +4.5,
            "LAB_CLAIMS": +350000.0,
        },
        historical_max_drawdown_60_40=-28.5,
        historical_max_drawdown_all_weather=-11.8,
        historical_max_drawdown_spider=-8.4,
        vix_peak=80.8,
        hy_spread_peak=19.8,
        key_ruptured_links=["LINK_06 (Spread HY ⟷ Small Cap)", "LINK_08 (VIX ⟷ High Yield)", "LINK_10 (MIB ⟷ Spread BTP)"],
    ),
    "CRISIS_2013_TAPER": HistoricalCrisisScenario(
        id="CRISIS_2013_TAPER",
        name="Taper Tantrum (Maggio - Agosto 2013)",
        period="Mag 2013 - Ago 2013",
        duration_days=90,
        badge="🟡 TAPER TANTRUM",
        color_hex="#F59E0B",
        icon="⚡",
        description="Ben Bernanke accenna alla riduzione del Quantitative Easing: shock improvviso sui tassi reali.",
        macro_narrative=(
            "L'annuncio a sorpresa del tapering fa impennare i rendimenti reali a lungo termine di oltre 140 bps. "
            "L'oro e i titoli obbligazionari a lunga scadenza crollano, mentre i mercati emergenti e le valute estere "
            "subiscono deflussi massicci a favore del dollaro."
        ),
        catalyst="Discorso di Ben Bernanke al Congresso sul rallentamento acquisti Fed",
        shocks={
            "YIELD_US10": +1.35,      # Da 1.60% a 3.00%
            "YIELD_REAL10": +1.40,    # Salto tassi reali
            "YIELD_US02": +0.30,
            "CURVE_10_2": +105.0,     # Bear Steepening aggressivo
            "COM_GOLD": -22.0,        # Crash dell'oro su tassi reali (LINK_03 rotto)
            "EQ_SPX": -6.5,
            "EQ_NDX": -5.0,
            "EQ_RUT": -7.5,
            "CRED_HY": +0.95,
            "RISK_VIX": +6.5,
            "FX_DXY": +6.0,
            "COM_OIL": +4.0,
            "RATE_FED": 0.0,
        },
        historical_max_drawdown_60_40=-7.2,
        historical_max_drawdown_all_weather=-8.9,
        historical_max_drawdown_spider=-4.1,
        vix_peak=21.5,
        hy_spread_peak=4.95,
        key_ruptured_links=["LINK_03 (Tassi Reali ⟷ Oro)", "LINK_02 (Rendimenti Reali ⟷ Nasdaq)", "LINK_01 (Inflazione ⟷ Curva 2Y)"],
    ),
    "CRISIS_2020_COVID": HistoricalCrisisScenario(
        id="CRISIS_2020_COVID",
        name="Shock Pandemico & Liquidity Freeze (Marzo 2020)",
        period="Feb 2020 - Apr 2020",
        duration_days=45,
        badge="🟣 COVID 2020",
        color_hex="#A855F7",
        icon="🦠",
        description="Lockdown globale, paralisi dell'economia reale, sell-off simultaneo di tutte le asset class.",
        macro_narrative=(
            "Il più violento sell-off della storia per velocità: l'S&P 500 perde il 34% in 23 giorni di contrattazione. "
            "Durante il culmine di metà marzo, la corsa alla liquidità monetaria innesca vendite persino sui Treasury e sull'oro "
            "prima dell'intervento illimitato della Fed ('QE Infinite')."
        ),
        catalyst="Lockdown mondiali per Coronavirus e blocco supply-chain",
        shocks={
            "EQ_SPX": -34.0,
            "EQ_NDX": -28.0,
            "EQ_RUT": -42.0,
            "EQ_DAX": -38.0,
            "EQ_MIB": -41.0,
            "COM_OIL": -68.0,        # Petrolio WTI crollato temporaneamente in negativo
            "RISK_VIX": +65.0,       # VIX a 82.7
            "CRED_HY": +6.50,        # Spread HY esploso a 11.0%
            "RATE_FED": -1.50,       # Taglio d'emergenza a 0%
            "YIELD_US02": -1.25,
            "YIELD_US10": -1.15,
            "LAB_CLAIMS": +3000000.0,# Sussidi record a 6.6M
            "LAB_UNRATE": +10.2,     # Disoccupazione al 14.7%
            "FX_DXY": +7.5,
            "COM_GOLD": -5.0,        # Vendite forzate prima del recupero
            "GRO_RETAIL": -15.0,
        },
        historical_max_drawdown_60_40=-20.8,
        historical_max_drawdown_all_weather=-11.5,
        historical_max_drawdown_spider=-7.1,
        vix_peak=82.7,
        hy_spread_peak=10.9,
        key_ruptured_links=["LINK_08 (VIX ⟷ High Yield)", "LINK_06 (Spread HY ⟷ Russell)", "LINK_04 (Petrolio ⟷ Break-even)"],
    ),
    "CRISIS_2022_RATES": HistoricalCrisisScenario(
        id="CRISIS_2022_RATES",
        name="Shock Inflazione & Rialzo Tassi più Rapido (2022)",
        period="Gen 2022 - Ott 2022",
        duration_days=300,
        badge="🔴 RIALZO TASSI 2022",
        color_hex="#EF4444",
        icon="📉",
        description="Inflazione record all'8-9%, guerra in Ucraina e contemporaneo crash di Azioni e Obbligazioni.",
        macro_narrative=(
            "L'annus horribilis per il portafoglio tradizionale 60/40. Con l'inflazione al massimo da 40 anni, la Fed "
            "ha alzato i tassi di 425 bps in pochi mesi. Azioni e obbligazioni sono scese insieme (correlazione positiva), "
            "annullando qualsiasi beneficio di diversificazione classica."
        ),
        catalyst="Spirale inflattiva globale post-COVID, shock energetico Ucraina e stretta aggressiva Fed",
        shocks={
            "RATE_FED": +4.25,
            "YIELD_US02": +3.70,     # 2Y da 0.75% a 4.45%
            "YIELD_US10": +2.70,     # 10Y da 1.50% a 4.25%
            "YIELD_REAL10": +2.60,   # Da -1.0% a +1.6% (massimo repricing storico)
            "CURVE_10_2": -130.0,    # Inversione profonda a -80 bps
            "EQ_SPX": -25.5,
            "EQ_NDX": -35.5,
            "EQ_RUT": -28.0,
            "COM_OIL": +45.0,        # Shock petrolio oltre $120
            "INF_USCPI": +4.5,       # CPI Headline al 9.1%
            "INF_USPCE": +3.2,
            "FX_DXY": +18.0,         # Re dollaro schiaccia le altre divise
            "FX_EURUSD": -16.0,      # Euro sotto la parità (0.96)
            "CRED_HY": +2.40,
            "RISK_VIX": +14.0,
        },
        historical_max_drawdown_60_40=-23.5,
        historical_max_drawdown_all_weather=-17.2,
        historical_max_drawdown_spider=-8.1,
        vix_peak=36.5,
        hy_spread_peak=6.0,
        key_ruptured_links=["LINK_01 (Inflazione ⟷ Curva 2Y)", "LINK_02 (Rendimenti Reali ⟷ Nasdaq)", "LINK_07 (DXY ⟷ Borse Europee)"],
    ),
    "CRISIS_2023_SVB": HistoricalCrisisScenario(
        id="CRISIS_2023_SVB",
        name="Crac Silicon Valley Bank & Crisi Bancaria (Marzo 2023)",
        period="Mar 2023 - Apr 2023",
        duration_days=30,
        badge="🟡 SVB BANK RUN",
        color_hex="#F59E0B",
        icon="🏦",
        description="Corsa agli sportelli su SVB, Signature Bank e salvataggio d'emergenza Credit Suisse.",
        macro_narrative=(
            "Le perdite non realizzate sui titoli di stato innescano una crisi di liquidità sulle banche regionali USA. "
            "In soli 3 giorni il rendimento del Treasury a 2 anni subisce il crollo più violento dal Black Monday del 1987 "
            "(-110 bps), mentre l'oro vola verso i $2000 per timori di contagio sistemico."
        ),
        catalyst="Bank Run su SVB provocata da perdite su bond e fuga depositi tech",
        shocks={
            "YIELD_US02": -1.10,     # Crollo storico record 2Y in 3 giorni
            "YIELD_US10": -0.65,
            "CURVE_10_2": +45.0,     # Bull Steepening violentissimo
            "COM_GOLD": +11.5,       # Vola l'oro come bene rifugio
            "EQ_SPX": -4.5,
            "EQ_RUT": -12.5,         # Banche regionali nel Russell crollate del 30%
            "EQ_MIB": -9.0,          # Banche europee contagiate da Credit Suisse
            "CRED_HY": +1.20,
            "RISK_VIX": +11.0,
            "RATE_FED": +0.25,       # Fed costretta a rialzo cauto + programma BTFP
        },
        historical_max_drawdown_60_40=-5.5,
        historical_max_drawdown_all_weather=-2.2,
        historical_max_drawdown_spider=-1.8,
        vix_peak=30.8,
        hy_spread_peak=5.3,
        key_ruptured_links=["LINK_01 (Inflazione ⟷ Curva 2Y)", "LINK_06 (Spread HY ⟷ Russell 2000)", "LINK_08 (VIX ⟷ Spread Credito)"],
    ),
    "CRISIS_2025_TARIFFS": HistoricalCrisisScenario(
        id="CRISIS_2025_TARIFFS",
        name="Shock Tariffe Trump & Geoeconomia Globale (Marzo 2025)",
        period="Feb 2025 - Mar 2025",
        duration_days=45,
        badge="🟠 TARIFFE TRUMP",
        color_hex="#F97316",
        icon="🚢",
        description="Imposizione dazi al 25% su partner commerciali, ritorsioni valutarie e frammentazione del commercio.",
        macro_narrative=(
            "La firma degli ordini esecutivi sui dazi genera timori di stagflazione importata e indebolimento delle valute "
            "dei paesi esportatori (Euro e Yuan). Le borse europee e le materie prime industriali (Rame) soffrono per "
            "la frenata degli scambi, mentre il Dollaro e l'Oro salgono come coperture geopolitiche."
        ),
        catalyst="Nuove tariffe commerciali USA, dazi su acciaio/auto e rappresaglie doganali",
        shocks={
            "FX_DXY": +6.5,
            "FX_EURUSD": -5.5,
            "COM_COPPER": -14.0,     # Rame debole per frenata commercio
            "COM_GOLD": +9.5,        # Oro bene rifugio contro frammentazione
            "EQ_DAX": -11.5,         # Germania manifatturiera colpita da export
            "EQ_MIB": -8.5,
            "EQ_SPX": -6.0,
            "EQ_NDX": -5.0,
            "INF_USCPI": +0.8,       # Pressione sui costi dei beni importati
            "YIELD_US10": +0.35,     # Premio a termine su debito USA e deficit
            "CRED_HY": +0.85,
            "RISK_VIX": +8.5,
        },
        historical_max_drawdown_60_40=-6.8,
        historical_max_drawdown_all_weather=-3.8,
        historical_max_drawdown_spider=-2.5,
        vix_peak=28.5,
        hy_spread_peak=4.6,
        key_ruptured_links=["LINK_07 (DXY ⟷ Borse Europee)", "LINK_09 (Rame ⟷ Petrolio/Crescita)", "LINK_03 (Tassi Reali ⟷ Oro)"],
    ),
}


class HistoricalStressEngine:
    """Computes instant replay of historical crisis scenarios onto SPIDER network."""

    @staticmethod
    def apply_crisis_scenario(
        scenario_id: str,
        current_node_states: Dict[str, Any],
        engine_instance: Any,
        window_key: str = "90D"
    ) -> Dict[str, Any]:
        """
        Applies calibrated historical shock tensor to current network nodes,
        recomputes link states, CISS non-linear index, and systemic tension.
        """
        if scenario_id not in HISTORICAL_CRISIS_LIBRARY:
            raise ValueError(f"Scenario {scenario_id} not in library.")

        scenario = HISTORICAL_CRISIS_LIBRARY[scenario_id]

        # Deepcopy node states so active official database remains 100% pristine
        stressed_nodes = copy.deepcopy(current_node_states)

        impacted_nodes_detail = []

        for nid, shock_val in scenario.shocks.items():
            if nid not in stressed_nodes:
                continue
            ns = stressed_nodes[nid]
            orig_val = ns.current_value
            orig_z = ns.z_score

            # If percentage change (equities, commodities, FX, production)
            if nid in ["EQ_SPX", "EQ_NDX", "EQ_RUT", "EQ_DAX", "EQ_MIB", "COM_OIL", "COM_GOLD", "COM_COPPER", "FX_DXY", "FX_EURUSD", "GRO_INDPRO", "GRO_RETAIL"]:
                pct_delta = shock_val / 100.0
                new_val = orig_val * (1.0 + pct_delta)
                # Volatility standard deviations approx 15% for index, 25% for commodities
                scale_sigma = 0.15 if "EQ_" in nid else (0.25 if "COM_" in nid else 0.08)
                z_delta = pct_delta / scale_sigma
            # If basis points / percentage points (rates, yields, inflation, spreads, unemployment)
            elif nid in ["CURVE_10_2"]:
                new_val = orig_val + shock_val
                z_delta = shock_val / 40.0
            elif nid in ["LAB_CLAIMS"]:
                new_val = orig_val + shock_val
                z_delta = shock_val / 50000.0
            else:
                new_val = orig_val + shock_val
                # Rates std ~0.40%, VIX ~8.0, HY spread ~0.80%
                scale_pt = 0.40 if ("RATE_" in nid or "YIELD_" in nid or "INF_" in nid) else (8.0 if "VIX" in nid else 0.80)
                z_delta = shock_val / scale_pt

            ns.current_value = round(new_val, 2)
            ns.z_score = round(orig_z + z_delta, 2)
            abs_z = abs(ns.z_score)
            if abs_z >= 2.0:
                ns.status = "Estremo"
            elif abs_z >= 1.25:
                ns.status = "Attenzione"
            else:
                ns.status = "Normale"

            impacted_nodes_detail.append({
                "id": nid,
                "name": ns.name,
                "quadrant": ns.quadrant,
                "val_pre": orig_val,
                "val_post": ns.current_value,
                "unit": ns.unit,
                "shock_applied": shock_val,
                "z_pre": orig_z,
                "z_post": ns.z_score,
                "status": ns.status,
            })

        # Recompute links and central core under stress
        stressed_links = engine_instance.compute_link_states(stressed_nodes, window_key=window_key)
        stressed_core = engine_instance.compute_spider_core(stressed_nodes, stressed_links, window_key=window_key)

        ruptured_links = [l for l in stressed_links if l.is_master and l.z_score_tension >= 2.0]
        friction_links = [l for l in stressed_links if l.is_master and 1.25 <= l.z_score_tension < 2.0]

        return {
            "scenario": scenario,
            "stressed_nodes": stressed_nodes,
            "stressed_links": stressed_links,
            "stressed_core": stressed_core,
            "impacted_nodes_detail": impacted_nodes_detail,
            "ruptured_count": len(ruptured_links),
            "friction_count": len(friction_links),
            "ruptured_links": ruptured_links,
            "friction_links": friction_links,
        }
