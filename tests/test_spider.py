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
    build_spider_web_figure,
    generate_links_dataframe,
    generate_nodes_dataframe,
)


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


if __name__ == "__main__":
    unittest.main()
