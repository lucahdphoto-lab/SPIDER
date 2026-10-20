"""
Unit test suite for SPIDER (System for Portfolio, Index, and Data Economic Relationships).
Verifies configuration integrity, econometric calculations, SQLite caching, and UI generators.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import MASTER_LINKS, NODES, QUADRANTS, RINGS, TIME_WINDOWS
from data_fetcher import db, fetcher
from quant_engine import engine
from spider_ui import (
    build_macro_regime_matrix_figure,
    build_si_correlation_heatmap,
    build_si_subsector_bar_figure,
    build_spider_si_network_figure,
    build_spider_si_network_3d_figure,
    build_spider_web_figure,
    generate_links_dataframe,
    generate_nodes_dataframe,
)
from super_intelligence_engine import SuperIntelligenceState, si_engine



class TestSpiderArchitecture(unittest.TestCase):

    def test_01_config_integrity(self):
        """Verify that all 30 peripheral nodes, 10 master links, 3 rings and 4 quadrants exist."""
        self.assertEqual(len(NODES), 30, f"Expected 30 nodes, found {len(NODES)}")
        self.assertEqual(len(MASTER_LINKS), 10, f"Expected 10 master links, found {len(MASTER_LINKS)}")
        self.assertEqual(len(RINGS), 3, "Expected 3 concentric rings")
        self.assertEqual(len(QUADRANTS), 4, "Expected 4 macro quadrants")

        for l_id, l_def in MASTER_LINKS.items():
            self.assertIn(l_def.node_a, NODES, f"Link {l_id}: source {l_def.node_a} not in NODES")
            self.assertIn(l_def.node_b, NODES, f"Link {l_id}: target {l_def.node_b} not in NODES")
            self.assertIn(l_def.theoretical_corr, [-1, 1], f"Link {l_id}: invalid theoretical correlation")

    def test_02_database_initialization(self):
        """Verify SQLite cache initialization."""
        meta = db.get_all_metadata()
        self.assertIsInstance(meta, dict)

    def test_03_quant_engine_alignment_and_states(self):
        """Test mixed-frequency temporal alignment and state calculation."""
        df = engine.load_and_align_data(force_refresh=False)
        self.assertFalse(df.empty, "Aligned dataframe should not be empty")
        self.assertGreaterEqual(len(df.columns), 25, "At least 25 nodes should be aligned")

        node_states = engine.compute_node_states(window_key="90D")
        self.assertGreaterEqual(len(node_states), 25, "Node states should be computed")

        spx = node_states.get("EQ_SPX")
        self.assertIsNotNone(spx)
        self.assertGreater(spx.current_value, 0)
        self.assertIsInstance(spx.z_score, float)
        self.assertIn(spx.status, ["Normale", "Attenzione", "Estremo"])

        link_states = engine.compute_link_states(node_states, window_key="90D", use_spearman=False)
        self.assertGreaterEqual(len(link_states), 10, "At least 10 master links must be evaluated")

        core = engine.compute_spider_core(node_states, link_states)
        self.assertIsNotNone(core.macro_regime)
        self.assertGreater(core.systemic_web_tension, 0.0)
        self.assertIn(core.web_tension_status, [
            "Bassa Tensione (Equilibrio Fisiologico)",
            "Tensione Moderata (Attrito Macro)",
            "CRITICA (Rottura Correlazioni Sistemiche)"
        ])
        # ECB CISS Model Assertions
        self.assertIsNotNone(core.ciss)
        self.assertGreaterEqual(core.ciss.ciss_score, 0.0)
        self.assertLessEqual(core.ciss.ciss_score, 1.0)
        self.assertEqual(len(core.ciss.quadrant_stress), 4)
        self.assertEqual(len(core.ciss.corr_matrix), 4)
        self.assertGreaterEqual(core.ciss.contagion_contribution_pct, 0.0)

    def test_04_ui_components(self):
        """Test Plotly spider web figure and dataframe generators."""
        node_states = engine.compute_node_states(window_key="90D")
        link_states = engine.compute_link_states(node_states, window_key="90D")
        core = engine.compute_spider_core(node_states, link_states)

        fig_web = build_spider_web_figure(node_states, link_states, core, selected_node_id=None)
        self.assertIsNotNone(fig_web)
        self.assertGreater(len(fig_web.data), 5)

        fig_matrix = build_macro_regime_matrix_figure(core)
        self.assertIsNotNone(fig_matrix)

        df_nodes = generate_nodes_dataframe(node_states)
        self.assertFalse(df_nodes.empty)
        self.assertIn("ID Nodo", df_nodes.columns)
        self.assertIn("Z-Score", df_nodes.columns)

        df_links = generate_links_dataframe(link_states)
        self.assertFalse(df_links.empty)
        self.assertIn("ID Filo", df_links.columns)
        self.assertIn("Tensione (|Z-Score|)", df_links.columns)

    def test_05_historical_crisis_engine(self):
        """Verify the 6 CCAR/Aladdin historical crisis scenarios and shock tensor calculations."""
        from historical_crisis_engine import HISTORICAL_CRISIS_LIBRARY, HistoricalStressEngine

        self.assertEqual(len(HISTORICAL_CRISIS_LIBRARY), 6)
        expected_crises = [
            "CRISIS_2008_GFC",
            "CRISIS_2013_TAPER",
            "CRISIS_2020_COVID",
            "CRISIS_2022_RATES",
            "CRISIS_2023_SVB",
            "CRISIS_2025_TARIFFS",
        ]
        for cid in expected_crises:
            self.assertIn(cid, HISTORICAL_CRISIS_LIBRARY)

        # Test applying 2008 GFC shock
        node_states = engine.compute_node_states(window_key="90D")
        res = HistoricalStressEngine.apply_crisis_scenario("CRISIS_2008_GFC", node_states, engine)
        self.assertGreater(res["ruptured_count"], 0)
        self.assertGreater(res["stressed_core"].systemic_web_tension, 1.5)
        self.assertEqual(res["scenario"].id, "CRISIS_2008_GFC")
        self.assertLess(res["scenario"].historical_max_drawdown_60_40, -20.0)

    def test_06_net_liquidity_engine(self):
        """Verify Global Net Liquidity calculation, Fed H.4.1 formula and 4-8W asset correlations."""
        from net_liquidity_engine import net_liquidity_engine, NetLiquidityState

        aligned_df = engine.load_and_align_data(force_refresh=False)
        liq = net_liquidity_engine.compute_net_liquidity(aligned_df)

        self.assertIsInstance(liq, NetLiquidityState)
        self.assertGreater(liq.us_net_liquidity_b, 5000.0)  # > $5,000B
        self.assertGreater(liq.fed_total_assets_b, 6000.0)
        self.assertGreater(liq.tga_balance_b, 0.0)
        self.assertGreater(liq.rrp_balance_b, 0.0)
        # Check formula: Net Liquidity = Fed Total Assets - TGA - RRP
        computed_liq = liq.fed_total_assets_b - liq.tga_balance_b - liq.rrp_balance_b
        self.assertAlmostEqual(liq.us_net_liquidity_b, computed_liq, delta=1.0)
        # Check forward correlations with risk assets
        self.assertGreater(liq.corr_ndx_4w, 0.4)
        self.assertGreater(liq.corr_spx_4w, 0.4)
        self.assertFalse(liq.historical_timeseries.empty)
        self.assertIn("US_Net_Liquidity_B", liq.historical_timeseries.columns)
        self.assertGreater(liq.us_net_liquidity_t, 5.0)
        self.assertTrue(len(liq.macro_interpretation) > 0)

    def test_07_risk_parity_and_volatility_targeting(self):
        """Verify Equal Risk Contribution, Markowitz Efficient Frontier, and Systemic Tension De-risking."""
        from risk_parity_engine import risk_parity_engine, RiskParityState

        aligned_df = engine.load_and_align_data(force_refresh=False)

        # 1. Test in Equilibrium (|Z| = 0.8 < 1.25): Full Cyclical Beta
        rp_eq = risk_parity_engine.optimize_risk_parity_portfolio(aligned_df, systemic_tension_z=0.8)
        self.assertIsInstance(rp_eq, RiskParityState)
        self.assertGreater(rp_eq.cyclical_exposure_pct, 25.0)
        self.assertEqual(rp_eq.vol_scale_factor, 1.00)
        self.assertEqual(len(rp_eq.items), 6)
        # Verify Equal Risk Contribution among the core assets (~20% each)
        for it in rp_eq.items[:5]:
            self.assertAlmostEqual(it.risk_contribution_pct, 20.0, delta=2.5)

        # 2. Test in Systemic Rupture (|Z| = 2.4 >= 2.0): Volatility Targeting De-risking
        rp_rup = risk_parity_engine.optimize_risk_parity_portfolio(aligned_df, systemic_tension_z=2.4)
        self.assertLess(rp_rup.cyclical_exposure_pct, rp_eq.cyclical_exposure_pct)
        self.assertGreater(rp_rup.defensive_exposure_pct, 70.0)
        self.assertLess(rp_rup.vol_scale_factor, 0.65)
        self.assertLess(rp_rup.portfolio_volatility_pct, rp_eq.portfolio_volatility_pct)

        # Verify Efficient Frontier Curves
        self.assertGreater(len(rp_eq.frontier_curve), 5)
        self.assertGreater(len(rp_eq.unconstrained_frontier), 5)

    def test_08_diebold_yilmaz_spillover(self):
        """Verify Diebold-Yilmaz Spillover VAR(p)/GFEVD directional network, TCI, and net transmitters."""
        from diebold_yilmaz_engine import diebold_yilmaz_engine, DieboldYilmazState

        aligned_df = engine.load_and_align_data(force_refresh=False)
        dy = diebold_yilmaz_engine.calculate_spillover_state(aligned_df)

        self.assertIsInstance(dy, DieboldYilmazState)
        # TCI is non-negative and bounded (typically between 10% and 90%)
        self.assertGreater(dy.total_connectedness_index, 0.0)
        self.assertLess(dy.total_connectedness_index, 100.0)

        # Spillover matrix dimensions
        self.assertEqual(dy.spillover_matrix.shape[0], 8)
        self.assertEqual(dy.spillover_matrix.shape[1], 8)

        # Row sums of GFEVD matrix must sum to ~100%
        row_sums = dy.spillover_matrix.sum(axis=1)
        for r_sum in row_sums:
            self.assertAlmostEqual(r_sum, 100.0, delta=1.0)

        # Profile list and top transmitter/receiver logic
        self.assertEqual(len(dy.nodes_profile), 8)
        self.assertGreater(len(dy.top_transmitters), 0)
        self.assertGreater(len(dy.top_receivers), 0)

        # Sum of Net Spillovers across all nodes must be identically 0 (conservation of spillover)
        net_sum = sum(p.net_spillover_pct for p in dy.nodes_profile)
        self.assertAlmostEqual(net_sum, 0.0, delta=0.5)

        # Verify is_net_transmitter property consistency
        for p in dy.nodes_profile:
            self.assertEqual(p.is_net_transmitter, p.net_spillover_pct > 0)

        # Links for directional graph
        self.assertGreater(len(dy.pairwise_links), 5)
        for lnk in dy.pairwise_links:
            self.assertGreater(lnk.gross_spillover, 0.0)

    def test_09_super_intelligence_engine(self):
        """Verify the SPIDER SI (Super Intelligence) engine, fundamental cockpit, and network topology."""
        si_state = si_engine.compute_si_state()
        self.assertIsInstance(si_state, SuperIntelligenceState)

        # 10 Top Capitalized AI Stocks
        self.assertEqual(len(si_state.stocks), 10, "Expected exactly 10 AI megacap stocks")
        tickers = {s.ticker for s in si_state.stocks}
        expected_tickers = {"NVDA", "TSM", "AVGO", "AMD", "MU", "MSFT", "GOOGL", "AMZN", "ORCL", "META"}
        self.assertEqual(tickers, expected_tickers)

        # Check Aggregates
        self.assertGreater(si_state.total_market_cap_t, 10.0, "Total AI Market Cap should exceed $10T")
        self.assertGreater(si_state.total_annual_capex_b, 200.0, "Total AI Capex should exceed $200B")
        self.assertGreater(si_state.composite_index, 0.0, "Composite SI index must be positive")
        self.assertGreaterEqual(si_state.ai_momentum_score, 0.0)
        self.assertLessEqual(si_state.ai_momentum_score, 100.0)
        self.assertIsNotNone(si_state.top_performer)

        # Correlation Matrix Dimensions (10x10)
        self.assertEqual(si_state.correlation_matrix.shape, (10, 10))

        # Check Supply-Chain & Radial Spider Links
        self.assertGreaterEqual(len(si_state.spider_links), 15, "Expected at least 15 supply chain / radial links")

        # Test Plotly Visual Builders
        fig_net = build_spider_si_network_figure(si_state, dark_mode=True)
        self.assertIsNotNone(fig_net)
        self.assertGreater(len(fig_net.data), 5)

        fig_net_3d = build_spider_si_network_3d_figure(si_state, dark_mode=True)
        self.assertIsNotNone(fig_net_3d)
        self.assertGreater(len(fig_net_3d.data), 5)

        fig_corr = build_si_correlation_heatmap(si_state, dark_mode=True)
        self.assertIsNotNone(fig_corr)
        self.assertGreater(len(fig_corr.data), 0)

        fig_bars = build_si_subsector_bar_figure(si_state, dark_mode=True)
        self.assertIsNotNone(fig_bars)
        self.assertGreater(len(fig_bars.data), 0)


if __name__ == "__main__":
    unittest.main()



