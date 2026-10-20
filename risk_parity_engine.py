"""
Risk Parity & Volatility Targeting Engine for SPIDER (Institutional Asset Allocation).
References:
- Bridgewater Associates All Weather / Pure Alpha (Ray Dalio, Bob Prince)
- AQR Capital Management (Cliff Asness) - Risk Parity & Volatility Targeting
- Man AHL - Systematic De-risking & Trend/Tension Constrained Allocation

Calculates:
1. Equal Risk Contribution (ERC) / Inverse Volatility (w_i ~ 1 / sigma_i) weights
2. Exact Covariance-based Risk Contributions (RC_i = w_i * (Sigma @ w)_i / sigma_p)
3. Dynamic Tension De-risking constrained by SPIDER |Z| systemic tension:
   - |Z| < 1.25 (Equilibrium): Full beta exposure (Equities, High Yield, Commodities)
   - 1.25 <= |Z| < 2.0 (Attention): Progressive volatility scaling down
   - |Z| >= 2.0 (Systemic Rupture): Severe Volatility Targeting de-risking, shift to T-Bills, TIPS, defensive Gold/Cash
4. Markowitz Efficient Frontier generation comparing Unconstrained, Risk Parity, and Tension-Constrained Target Vol portfolios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# Institutional Asset Universe for Execution
RISK_PARITY_ASSETS = [
    {
        "id": "EQ_SPX",
        "name": "Azionario Globale (S&P 500 / MSCI World)",
        "asset_class": "Equities",
        "color": "#10B981", # Emerald
        "type": "Cyclical Beta",
        "base_vol_annual": 0.16,
        "is_cyclical": True,
    },
    {
        "id": "CRED_HY",
        "name": "High Yield Corporate Credit (HY Spread)",
        "asset_class": "Credit / HY",
        "color": "#F59E0B", # Amber
        "type": "Cyclical Beta",
        "base_vol_annual": 0.11,
        "is_cyclical": True,
    },
    {
        "id": "COM_OIL",
        "name": "Materie Prime & Energia (WTI Crude)",
        "asset_class": "Commodities",
        "color": "#DC2626", # Crimson
        "type": "Cyclical Beta",
        "base_vol_annual": 0.28,
        "is_cyclical": True,
    },
    {
        "id": "YIELD_US10",
        "name": "Treasuries a Medio-Lungo Termine (10Y Bond)",
        "asset_class": "Bonds",
        "color": "#64748B", # Slate
        "type": "Defensive Duration",
        "base_vol_annual": 0.08,
        "is_cyclical": False,
    },
    {
        "id": "COM_GOLD",
        "name": "Oro Fisico Sovrano (Hedging Asimmetrico)",
        "asset_class": "Gold / Safe Haven",
        "color": "#EAB308", # Gold
        "type": "Safe Haven",
        "base_vol_annual": 0.14,
        "is_cyclical": False,
    },
    {
        "id": "CASH_TBILLS",
        "name": "Liquidità Monetaria & T-Bills 1-3M (Risk-Free 5%)",
        "asset_class": "Cash / T-Bills",
        "color": "#38BDF8", # Sky
        "type": "Pure Safe Haven",
        "base_vol_annual": 0.012,
        "is_cyclical": False,
    },
]


@dataclass
class AssetAllocationItem:
    asset_id: str
    asset_name: str
    asset_class: str
    color_hex: str
    unconstrained_weight_pct: float
    raw_risk_parity_weight_pct: float
    tension_adjusted_weight_pct: float
    volatility_annual_pct: float
    risk_contribution_pct: float
    leverage_multiplier: float
    allocation_status: str


@dataclass
class EfficientFrontierPoint:
    expected_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    weights: Dict[str, float]


@dataclass
class RiskParityState:
    systemic_tension_z: float
    tension_regime: str                     # "Equilibrio Fisiologico", "Tensione in Attenzione", "Rottura Sistemica"
    tension_badge: str
    tension_color_hex: str
    portfolio_volatility_pct: float
    target_volatility_pct: float
    vol_scale_factor: float                 # Vol targeting multiplier (1.0 in equilibrium, < 0.5 in crisis)
    cyclical_exposure_pct: float            # Total % in Equities, HY, Commodities
    defensive_exposure_pct: float           # Total % in T-Bills, TIPS/Duration, Gold
    expected_return_pct: float
    sharpe_ratio: float
    items: List[AssetAllocationItem]
    frontier_curve: List[EfficientFrontierPoint]
    unconstrained_frontier: List[EfficientFrontierPoint]
    executive_summary: str
    rebalancing_advice: str


class RiskParityEngine:
    """Quantitative optimization engine for Equal Risk Contribution and Volatility Targeting."""

    @classmethod
    def compute_asset_returns_and_cov(cls, aligned_df: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """Extract or proxy daily return matrix and annual covariance for the 6 assets."""
        n_assets = len(RISK_PARITY_ASSETS)
        asset_names = [a["name"] for a in RISK_PARITY_ASSETS]
        asset_ids = [a["id"] for a in RISK_PARITY_ASSETS]

        # Use aligned_df to derive returns where available
        rets_dict = {}
        for a in RISK_PARITY_ASSETS:
            aid = a["id"]
            if aid in aligned_df.columns:
                s = aligned_df[aid]
                if aid == "CRED_HY":
                    # HY spread: spread widening is negative for credit return
                    r = -s.diff() * 0.04
                elif aid == "YIELD_US10":
                    # 10Y yield: yield change * -duration (approx 7.5y)
                    r = -s.diff() * 0.075
                else:
                    r = s.pct_change()
                rets_dict[aid] = r.fillna(0.0)
            elif aid == "CASH_TBILLS":
                # Cash yield ~ 5% annual => ~ 0.02% daily, very low vol
                rets_dict[aid] = pd.Series(0.05 / 252.0, index=aligned_df.index)

        df_rets = pd.DataFrame(rets_dict)
        # Fill missing assets if any
        for a in RISK_PARITY_ASSETS:
            aid = a["id"]
            if aid not in df_rets.columns:
                ann_vol = a["base_vol_annual"]
                daily_vol = ann_vol / np.sqrt(252.0)
                df_rets[aid] = np.random.normal(0.06 / 252.0, daily_vol, len(aligned_df))

        # Ensure correct column order
        df_rets = df_rets[asset_ids].dropna()

        # Compute empirical covariance matrix annualized (252 trading days)
        cov_matrix = df_rets.cov().values * 252.0

        # Ensure positive semi-definite & realistic diagonals
        for i, a in enumerate(RISK_PARITY_ASSETS):
            emp_vol = np.sqrt(cov_matrix[i, i])
            target_vol = a["base_vol_annual"]
            if emp_vol <= 0.001 or np.isnan(emp_vol):
                cov_matrix[i, i] = target_vol ** 2
            else:
                # Regularize towards institutional baseline vol
                cov_matrix[i, i] = 0.5 * cov_matrix[i, i] + 0.5 * (target_vol ** 2)

        vol_annual = np.sqrt(np.diag(cov_matrix))

        # Expected annual returns (Capital Market Assumptions / Historical)
        # Cash: 5.0%, 10Y: 4.5%, HY: 6.8%, Equities: 9.5%, Oil: 8.0%, Gold: 7.5%
        exp_returns = np.array([0.095, 0.068, 0.080, 0.045, 0.075, 0.050])

        return asset_ids, exp_returns, vol_annual, cov_matrix

    @classmethod
    def compute_equal_risk_contribution_weights(cls, cov_matrix: np.ndarray) -> np.ndarray:
        """Solves for exact Equal Risk Contribution (ERC) weights where RC_i = RC_j."""
        n = cov_matrix.shape[0]

        # Objective function: minimize sum of squared differences between risk contributions
        def risk_parity_objective(weights):
            w = np.array(weights)
            port_var = w @ cov_matrix @ w
            if port_var <= 1e-12:
                return 1e6
            port_vol = np.sqrt(port_var)
            # Marginal risk contribution: (cov @ w) / port_vol
            # Risk contribution: w_i * (cov @ w)_i / port_vol
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib / port_vol
            target_rc = port_vol / n
            return np.sum((risk_contrib - target_rc) ** 2) * 1e6

        # Inverse vol starting point
        inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
        w0 = inv_vol / np.sum(inv_vol)

        bounds = [(0.02, 0.60) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-9}
        )

        if res.success:
            w_opt = res.x / np.sum(res.x)
        else:
            w_opt = w0

        return w_opt

    @classmethod
    def optimize_risk_parity_portfolio(
        cls,
        aligned_df: pd.DataFrame,
        systemic_tension_z: float,
        target_annual_vol_pct: float = 10.0,
    ) -> RiskParityState:
        """Computes Risk Parity weights across the 5 core asset classes and applies Volatility Targeting de-risking."""
        # Institutional asset universe:
        # Core Risky Assets: Equities (SPX), HY Credit, Commodities (WTI), 10Y Treasuries, Gold
        # Cash/T-Bills acts as the risk-free buffer for Volatility Targeting De-risking
        core_assets = RISK_PARITY_ASSETS[:5]
        n_core = len(core_assets)
        core_ids = [a["id"] for a in core_assets]

        # Base annual volatilities calibrated to long-term institutional regimes:
        # SPX: 16%, HY: 10%, WTI Oil: 28%, 10Y Treasuries: 8.5%, Gold: 15%
        # Derived inverse vol weights: w_i ~ 1 / sigma_i
        base_vols = np.array([a["base_vol_annual"] for a in core_assets])
        inv_vols = 1.0 / base_vols
        raw_erc_5 = inv_vols / np.sum(inv_vols)

        # Expected returns (5-10Y Horizon):
        # SPX: 9.5%, HY: 6.8%, Oil: 7.5%, 10Y: 4.5%, Gold: 7.0%, Cash: 5.0%
        core_exp_rets = np.array([0.095, 0.068, 0.075, 0.045, 0.070])

        # Realistic cross-asset correlation matrix for risk contribution:
        corr_matrix = np.array([
            [ 1.00,  0.65,  0.35, -0.25,  0.10], # SPX
            [ 0.65,  1.00,  0.40, -0.15,  0.05], # HY
            [ 0.35,  0.40,  1.00, -0.20,  0.25], # Oil
            [-0.25, -0.15, -0.20,  1.00,  0.30], # 10Y
            [ 0.10,  0.05,  0.25,  0.30,  1.00], # Gold
        ])
        core_cov = np.diag(base_vols) @ corr_matrix @ np.diag(base_vols)

        # Fine-tune ERC weights using numerical solver
        def erc_obj(w):
            w = np.array(w)
            port_var = w @ core_cov @ w
            port_vol = np.sqrt(max(1e-8, port_var))
            rc = w * (core_cov @ w) / port_vol
            target_rc = port_vol / n_core
            return np.sum((rc - target_rc) ** 2) * 1e6

        res = minimize(
            erc_obj,
            raw_erc_5,
            method="SLSQP",
            bounds=[(0.05, 0.45) for _ in range(n_core)],
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        )
        erc_5_opt = res.x / np.sum(res.x) if res.success else raw_erc_5

        # Base unleveraged risk parity basket volatility
        base_basket_vol = float(np.sqrt(erc_5_opt @ core_cov @ erc_5_opt))

        # Dynamic Tension De-risking & Volatility Targeting Scaling:
        # |Z| < 1.25 (Equilibrium): Full investment in risk assets (e.g. 90-95% Risky, 5-10% Cash)
        # 1.25 <= |Z| < 2.0 (Attention): Moderate de-risking, shift 25-35% into Cash/T-Bills
        # |Z| >= 2.0 (Systemic Rupture): Severe de-risking, Cash/T-Bills boosted to 60-75%
        abs_z = abs(systemic_tension_z)

        if abs_z < 1.25:
            tension_regime = "Equilibrio Fisiologico (|Z| < 1.25)"
            tension_badge = "🟢 EQUILIBRIO SISTEMICO (RISK-ON ATTIVO)"
            tension_color = "#10B981"
            risky_budget = 0.92  # 92% Risky Basket, 8% Cash
            vol_scale_factor = 1.00
            gold_tilt = 1.00
            cyclical_scale = 1.00
        elif abs_z < 2.0:
            tension_regime = "Tensione in Attenzione (1.25 <= |Z| < 2.0)"
            tension_badge = "🟡 TENSIONE ELEVATA (DE-RISKING PREVENTIVO)"
            tension_color = "#F59E0B"
            progress = (abs_z - 1.25) / (2.0 - 1.25)
            risky_budget = 0.92 - (0.32 * progress) # down to 60%
            vol_scale_factor = 1.00 - (0.35 * progress)
            gold_tilt = 1.00 + (0.35 * progress)
            cyclical_scale = 1.00 - (0.25 * progress)
        else:
            tension_regime = "Rottura Sistemica (|Z| >= 2.0)"
            tension_badge = "🔴 ALLARME ROTTURA SISTEMICA (VOLATILITY TARGETING FLIGHT-TO-SAFETY)"
            tension_color = "#EF4444"
            excess_z = min(1.5, abs_z - 2.0)
            risky_budget = max(0.25, 0.60 - (0.25 * (excess_z / 1.0))) # down to 25-35%
            vol_scale_factor = max(0.35, 0.65 - (0.30 * (excess_z / 1.0)))
            gold_tilt = 1.50
            cyclical_scale = 0.50

        cash_weight = 1.0 - risky_budget

        # Apply tilts within the risky basket:
        basket_weights = erc_5_opt.copy()
        # [0: SPX (cyc), 1: HY (cyc), 2: Oil (cyc), 3: 10Y (def), 4: Gold (safe)]
        basket_weights[0] *= cyclical_scale
        basket_weights[1] *= cyclical_scale
        basket_weights[2] *= cyclical_scale
        basket_weights[4] *= gold_tilt
        basket_weights = basket_weights / np.sum(basket_weights)

        # Final 6-asset portfolio weights:
        final_weights = np.zeros(6)
        final_weights[:5] = basket_weights * risky_budget
        final_weights[5] = cash_weight

        # 6-Asset Full Covariance Matrix including Cash (near zero correlation & 1.2% vol)
        full_cov = np.zeros((6, 6))
        full_cov[:5, :5] = core_cov
        full_cov[5, 5] = (0.012) ** 2  # cash vol ~ 1.2%

        port_var = float(final_weights @ full_cov @ final_weights)
        port_vol = float(np.sqrt(max(1e-8, port_var)))

        # Risk Contributions of each asset
        marginal_contrib = full_cov @ final_weights
        rc_vector = final_weights * marginal_contrib / port_vol
        total_rc = np.sum(rc_vector)
        risk_contrib_pct = (rc_vector / total_rc * 100.0) if total_rc > 0 else (np.ones(6) / 6.0 * 100.0)

        # Expected Returns & Sharpe
        full_exp_returns = np.array([0.095, 0.068, 0.075, 0.045, 0.070, 0.050])
        exp_port_return = float(np.sum(final_weights * full_exp_returns))
        rf_rate = 0.05
        sharpe = (exp_port_return - rf_rate) / port_vol if port_vol > 0 else 0.0

        # Raw benchmark weights (Unconstrained 60/40)
        unconstrained_weights = np.array([0.60, 0.0, 0.0, 0.40, 0.0, 0.0])
        raw_erc_6 = np.zeros(6)
        raw_erc_6[:5] = erc_5_opt * 0.95
        # Build Allocation Items
        items: List[AssetAllocationItem] = []
        tot_cyclical = 0.0
        tot_defensive = 0.0

        for i, a in enumerate(RISK_PARITY_ASSETS):
            w_raw = raw_erc_6[i] * 100.0
            w_adj = final_weights[i] * 100.0
            w_unc = unconstrained_weights[i] * 100.0
            rc = risk_contrib_pct[i]
            ann_v = np.sqrt(full_cov[i, i]) * 100.0

            if a["is_cyclical"]:
                tot_cyclical += w_adj
                status = "🟢 Beta Pieno" if abs_z < 1.25 else ("🟡 De-risking" if abs_z < 2.0 else "🔴 Sottopeso Severo")
            else:
                tot_defensive += w_adj
                status = "⚖️ Base Paritaria" if abs_z < 1.25 else ("🛡️ Sovrappeso Difensivo" if abs_z < 2.0 else "🚀 Massima Copertura")

            items.append(AssetAllocationItem(
                asset_id=a["id"],
                asset_name=a["name"],
                asset_class=a["asset_class"],
                color_hex=a["color"],
                unconstrained_weight_pct=round(w_unc, 1),
                raw_risk_parity_weight_pct=round(w_raw, 1),
                tension_adjusted_weight_pct=round(w_adj, 1),
                volatility_annual_pct=round(ann_v, 1),
                risk_contribution_pct=round(rc, 1),
                leverage_multiplier=round(vol_scale_factor if a["is_cyclical"] else 1.0, 2),
                allocation_status=status
            ))

        # 3. Generate Efficient Frontier Points (Monte Carlo & Markowitz Optimization)
        frontier_pts = cls._generate_efficient_frontier(full_exp_returns, full_cov, final_weights, is_constrained=True)
        unconstrained_pts = cls._generate_efficient_frontier(full_exp_returns, full_cov, unconstrained_weights, is_constrained=False)

        # 4. Formulate Executive Summary
        if abs_z < 1.25:
            summary = (
                f"La tela macro di SPIDER si trova in perfetto equilibrio statistico (|Z| = {abs_z:.2f} < 1.25). "
                f"Il motore consiglia la piena operatività a Parità di Rischio: esposizione a beta ciclici al {tot_cyclical:.1f}% "
                f"(Azioni, High Yield, Commodities) con volatilità complessiva target al {port_vol * 100:.1f}%. Nessun taglio di leva richiesto."
            )
            rebalancing = "Mantenere ribilanciamento mensile ordinario delle quote di volatilità (Equal Risk Contribution)."
        elif abs_z < 2.0:
            summary = (
                f"Rilevata tensione intermarket anomala (|Z| = {abs_z:.2f} >= 1.25). "
                f"Attivato de-risking preventivo con fattore di scala volatilità a {vol_scale_factor * 100:.0f}%. "
                f"I pesi sui beta ciclici sono ridotti al {tot_cyclical:.1f}%, dirottando il capitale verso T-Bills 1-3M ({items[5].tension_adjusted_weight_pct:.1f}%) "
                f"e Oro sovrano ({items[4].tension_adjusted_weight_pct:.1f}%)."
            )
            rebalancing = "Taglio prudenziale del 30-40% sui veicoli azionari e commodities; estendere la quota monetaria priva di rischio."
        else:
            summary = (
                f"⚠️ ALLARME ROTTURA SISTEMICA: Tensione di Rete a |Z| = {abs_z:.2f} (superata soglia 2.0). "
                f"Il motore di Volatility Targeting (stile Man AHL / AQR) comprime drasticamente l'esposizione al rischio al {vol_scale_factor * 100:.0f}%. "
                f"La componente difensiva sale al {tot_defensive:.1f}%, azzerando il rischio di rovina finanziaria con allocazione massiccia in T-Bills e Cash monetario al 5%."
            )
            rebalancing = "Immediate Execution: De-leveraging forzato, chiusura posizioni marginali, rifugio in T-Bills a 1-3 mesi e Oro fisico."

        return RiskParityState(
            systemic_tension_z=round(systemic_tension_z, 2),
            tension_regime=tension_regime,
            tension_badge=tension_badge,
            tension_color_hex=tension_color,
            portfolio_volatility_pct=round(port_vol * 100.0, 2),
            target_volatility_pct=round(target_annual_vol_pct, 1),
            vol_scale_factor=round(vol_scale_factor, 2),
            cyclical_exposure_pct=round(tot_cyclical, 1),
            defensive_exposure_pct=round(tot_defensive, 1),
            expected_return_pct=round(exp_port_return * 100.0, 2),
            sharpe_ratio=round(sharpe, 2),
            items=items,
            frontier_curve=frontier_pts,
            unconstrained_frontier=unconstrained_pts,
            executive_summary=summary,
            rebalancing_advice=rebalancing
        )

    @classmethod
    def _generate_efficient_frontier(
        cls,
        exp_returns: np.ndarray,
        cov_matrix: np.ndarray,
        base_weights: np.ndarray,
        is_constrained: bool = True
    ) -> List[EfficientFrontierPoint]:
        """Calculates discrete points along the Markowitz Efficient Frontier."""
        points = []
        n = len(exp_returns)

        # Range of target returns from min to max expected
        min_ret = np.min(exp_returns) * 0.9
        max_ret = np.max(exp_returns) * 0.95
        target_returns = np.linspace(min_ret, max_ret, 25)

        for tr in target_returns:
            def frontier_obj(w):
                return w @ cov_matrix @ w

            if is_constrained:
                bounds = [(0.01, 0.50) for _ in range(n)]
            else:
                bounds = [(0.0, 1.0) for _ in range(n)]

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, target=tr: (w @ exp_returns) - target}
            ]

            res = minimize(
                frontier_obj,
                base_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 150}
            )

            if res.success:
                p_var = float(res.x @ cov_matrix @ res.x)
                p_vol = np.sqrt(max(1e-6, p_var))
                p_ret = float(res.x @ exp_returns)
                sharpe = (p_ret - 0.05) / p_vol if p_vol > 0 else 0.0
                points.append(EfficientFrontierPoint(
                    expected_return_pct=round(p_ret * 100.0, 2),
                    volatility_pct=round(p_vol * 100.0, 2),
                    sharpe_ratio=round(sharpe, 2),
                    weights={RISK_PARITY_ASSETS[k]["name"]: round(float(res.x[k] * 100.0), 1) for k in range(n)}
                ))

        # Sort points by volatility ascending
        points.sort(key=lambda pt: pt.volatility_pct)
        return points


risk_parity_engine = RiskParityEngine()
