"""
SPIDER UI Module:
Radial Concentric Spider-Web Network Graph in Plotly,
Interactive Macro Regime Matrix, Control Hub Cards, Dynamic Tabular Output,
What-If Stress Simulator, and Multi-Format Exporters.
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    NODES,
    QUADRANTS,
    RINGS,
    TIME_WINDOWS,
    Z_SCORE_CRITICAL,
    Z_SCORE_NORMAL,
    NodeDef,
)
from quant_engine import LinkState, NodeState, SpiderCoreState


def build_spider_web_figure(
    node_states: Dict[str, NodeState],
    link_states: List[LinkState],
    core_state: SpiderCoreState,
    selected_node_id: Optional[str] = None,
    filter_quadrant: str = "TUTTI",
    filter_ring: int = 0,
    show_labels: bool = True,
    dark_mode: bool = True,
) -> go.Figure:
    """Constructs the concentric radial Spider-Web network graph."""
    fig = go.Figure()

    bg_color = "#0B0F19" if dark_mode else "#F8FAFC"
    grid_color = "rgba(255, 255, 255, 0.12)" if dark_mode else "rgba(15, 23, 42, 0.12)"
    axis_color = "rgba(255, 255, 255, 0.20)" if dark_mode else "rgba(15, 23, 42, 0.20)"
    text_color = "#F1F5F9" if dark_mode else "#0F172A"

    # 1. Concentric Geometric Rings
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    for ring_num, r_info in RINGS.items():
        r = r_info["radius"]
        x_c = r * np.cos(theta_circle)
        y_c = r * np.sin(theta_circle)
        fig.add_trace(go.Scatter(
            x=x_c,
            y=y_c,
            mode="lines",
            line=dict(color=grid_color, width=1.4, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
            name=f"Ring_{ring_num}",
        ))
        fig.add_annotation(
            x=-r * 0.707,
            y=-r * 0.707 - 0.10,
            text=f"<b>{r_info['name']}</b>",
            showarrow=False,
            font=dict(size=9.5, color="#94A3B8" if dark_mode else "#64748B"),
            bgcolor="rgba(11, 15, 25, 0.75)" if dark_mode else "rgba(248, 250, 252, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            borderpad=3,
        )

    # 2. Quadrant Axes & Styled Headers
    max_r = 3.4
    fig.add_trace(go.Scatter(
        x=[-max_r, max_r, None, 0, 0],
        y=[0, 0, None, -max_r, max_r],
        mode="lines",
        line=dict(color=axis_color, width=1.2, dash="dash"),
        hoverinfo="skip",
        showlegend=False,
    ))

    quad_watermarks = [
        (2.45, 2.45, "INFLAZIONE & COMMODITIES", "#F59E0B"),
        (-2.45, 2.45, "CRESCITA & PRODUZIONE", "#10B981"),
        (-2.45, -2.45, "LAVORO & OCCUPAZIONE", "#3B82F6"),
        (2.45, -2.45, "LIQUIDITÀ & RISCHIO", "#EC4899"),
    ]
    for qx, qy, qtxt, qcol in quad_watermarks:
        fig.add_annotation(
            x=qx,
            y=qy,
            text=f"<b>{qtxt}</b>",
            showarrow=False,
            font=dict(size=11, color=qcol),
            bgcolor="rgba(15, 23, 42, 0.65)" if dark_mode else "rgba(255, 255, 255, 0.65)",
            bordercolor=qcol,
            borderwidth=1,
            borderpad=4,
            opacity=0.90,
        )

    # Polar to cartesian
    coords: Dict[str, Tuple[float, float]] = {"SPIDER_CORE": (0.0, 0.0)}
    node_textpositions: Dict[str, str] = {}
    for n_id, n_def in NODES.items():
        rad = RINGS[n_def.ring]["radius"]
        angle_rad = math.radians(n_def.angle_deg)
        coords[n_id] = (rad * math.cos(angle_rad), rad * math.sin(angle_rad))
        
        # Determine dynamic radial textposition to prevent labels from overlapping nodes and links
        # Radial projection: push label away from center
        deg = n_def.angle_deg % 360
        if 22.5 <= deg < 67.5:
            pos = "top right"
        elif 67.5 <= deg < 112.5:
            pos = "top center"
        elif 112.5 <= deg < 157.5:
            pos = "top left"
        elif 157.5 <= deg < 202.5:
            pos = "middle left"
        elif 202.5 <= deg < 247.5:
            pos = "bottom left"
        elif 247.5 <= deg < 292.5:
            pos = "bottom center"
        elif 292.5 <= deg < 337.5:
            pos = "bottom right"
        else:
            pos = "middle right"
        node_textpositions[n_id] = pos

    # Focus connections
    active_connected_nodes = set()
    if selected_node_id and selected_node_id != "SPIDER_CORE":
        active_connected_nodes.add(selected_node_id)
        for link in link_states:
            if link.node_a_id == selected_node_id:
                active_connected_nodes.add(link.node_b_id)
            elif link.node_b_id == selected_node_id:
                active_connected_nodes.add(link.node_a_id)

    # 3. Draw Links
    for link in link_states:
        if link.node_a_id not in coords or link.node_b_id not in coords:
            continue

        xA, yA = coords[link.node_a_id]
        xB, yB = coords[link.node_b_id]

        is_focused = True
        if selected_node_id and selected_node_id != "SPIDER_CORE":
            if link.node_a_id != selected_node_id and link.node_b_id != selected_node_id:
                is_focused = False

        if filter_ring > 0:
            ringA = NODES[link.node_a_id].ring if link.node_a_id in NODES else 0
            ringB = NODES[link.node_b_id].ring if link.node_b_id in NODES else 0
            if ringA != filter_ring and ringB != filter_ring:
                is_focused = False

        if filter_quadrant != "TUTTI":
            quadA = NODES[link.node_a_id].quadrant if link.node_a_id in NODES else ""
            quadB = NODES[link.node_b_id].quadrant if link.node_b_id in NODES else ""
            if quadA != filter_quadrant and quadB != filter_quadrant:
                is_focused = False

        if selected_node_id and selected_node_id != "SPIDER_CORE":
            line_color = link.color_hex if is_focused else ("rgba(255,255,255,0.03)" if dark_mode else "rgba(0,0,0,0.03)")
            line_width = (link.line_width * 1.5) if is_focused else 0.5
            dash_style = link.line_dash if is_focused else "dot"
        else:
            line_color = link.color_hex if is_focused else ("rgba(255,255,255,0.06)" if dark_mode else "rgba(0,0,0,0.06)")
            line_width = link.line_width if is_focused else 0.8
            dash_style = link.line_dash if is_focused else "dot"

        fig.add_trace(go.Scatter(
            x=[xA, xB],
            y=[yA, yB],
            mode="lines",
            line=dict(color=line_color, width=line_width, dash=dash_style),
            hoverinfo="none",
            showlegend=False,
        ))

        if is_focused and (link.is_master or (selected_node_id and is_focused)):
            x_mid = (xA + xB) / 2.0
            y_mid = (yA + yB) / 2.0
            badge_text = f"{link.active_corr:+.2f}" if link.is_master else f"|Z|={link.z_score_tension:.1f}"

            corr_dir_label = "🟢 Positiva (Coerente)" if link.active_corr > 0 else "🔴 Negativa / Inversa"
            hover_html = (
                f"<b>🕸️ FILO SPIDER: {link.id}</b><br>"
                f"<b>{link.node_a_name} ↔ {link.node_b_name}</b><br>"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
                f"• Stato Tensione: <b>{link.status}</b> (|Z| = {link.z_score_tension:.2f})<br>"
                f"• Relazione Teorica: <b>{'+1 (Diretta / Positiva)' if link.theoretical_corr > 0 else '-1 (Inversa)'}</b><br>"
                f"• Correlazione Rolling: <b>{link.active_corr:+.2f}</b> ({corr_dir_label})<br>"
                f"• Valore A: {link.node_a_val:,.2f} (Δ: {link.node_a_delta:+,.2f})<br>"
                f"• Valore B: {link.node_b_val:,.2f} (Δ: {link.node_b_delta:+,.2f})<br>"
                f"• Meccanismo Macro: {link.mechanism}<br>"
                f"• Diagnosi SPIDER: <i>{link.alert_description}</i>"
            )

            fig.add_trace(go.Scatter(
                x=[x_mid],
                y=[y_mid],
                mode="text",
                text=[badge_text if link.is_master else ""],
                textfont=dict(size=9.5, color=link.color_hex, family="Courier New, monospace"),
                hovertext=[hover_html],
                hoverinfo="text",
                showlegend=False,
            ))

    # 4. Peripheral Nodes
    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_labels = []
    node_hovertexts = []
    node_border_colors = []

    node_customdata = []
    for n_id, n_def in NODES.items():
        if n_id not in coords:
            continue
        x, y = coords[n_id]
        n_state = node_states.get(n_id)

        if filter_ring > 0 and n_def.ring != filter_ring:
            continue
        if filter_quadrant != "TUTTI" and n_def.quadrant != filter_quadrant:
            continue

        is_dimmed = False
        if selected_node_id and selected_node_id != "SPIDER_CORE":
            if n_id not in active_connected_nodes:
                is_dimmed = True

        quad_info = QUADRANTS.get(n_def.quadrant, {"color": "#38BDF8"})
        base_color = quad_info["color"]

        if is_dimmed:
            # Ultra-low opacity so focus nodes and lines pop out with maximum contrast
            node_fill = "rgba(51, 65, 85, 0.12)"
            node_border = "rgba(255, 255, 255, 0.03)"
            size = 9
        elif n_state and abs(n_state.z_score) >= Z_SCORE_CRITICAL:
            node_fill = "#D946EF"
            node_border = "#FFFFFF"
            size = 22
        elif n_state and abs(n_state.z_score) >= Z_SCORE_NORMAL:
            node_fill = "#F59E0B"
            node_border = "#FCD34D"
            size = 18
        else:
            node_fill = base_color
            node_border = "#FFFFFF" if (selected_node_id == n_id) else "rgba(255,255,255,0.4)"
            size = 16

        if selected_node_id == n_id:
            size = 28
            node_border = "#38BDF8"

        node_x.append(x)
        node_y.append(y)
        node_colors.append(node_fill)
        node_border_colors.append(node_border)
        node_sizes.append(size)
        node_labels.append(n_id if (show_labels and (not is_dimmed or selected_node_id == n_id)) else "")
        node_customdata.append(n_id)

        val_str = f"{n_state.current_value:,.2f} {n_state.unit}" if n_state else "N/D"
        delta_str = f"{n_state.delta_period:+,.2f} ({n_state.delta_period_pct:+.2f}%)" if n_state else "N/D"
        z_str = f"{n_state.z_score:+.2f}" if n_state else "0.00"

        yoy_str = f"{n_state.delta_yoy:+,.2f}" if n_state else "N/D"
        status_str = n_state.status if n_state else "Normale"
        ts_str = n_state.last_timestamp if n_state else ""

        hover_card = (
            f"<b>🏛️ NODO: {n_def.name} [{n_id}]</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"• Categoria: <b>{n_def.category}</b> (Anello {n_def.ring})<br>"
            f"• Valore Attuale: <b>{val_str}</b><br>"
            f"• Variazione Periodo: <b>{delta_str}</b><br>"
            f"• Variazione YoY (1 Anno): <b>{yoy_str}</b><br>"
            f"• Z-Score Dinamico: <b>{z_str}</b> ({status_str})<br>"
            f"• Ruolo Intermarket: {n_def.role}<br>"
            f"• Fonte Certificata: <b>{n_def.source}</b> ({n_def.official_ticker})<br>"
            f"• Ultimo Rilascio: {ts_str}<br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"<i>👉 Clicca sul nodo per bloccare il focus sui collegamenti</i>"
        )
        node_hovertexts.append(hover_card)

    node_pos_list = [node_textpositions.get(nid, "top center") for nid in node_customdata]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color=node_border_colors, width=2.0),
        ),
        text=node_labels,
        textposition=node_pos_list,
        textfont=dict(size=9.5, color=text_color, family="Inter, sans-serif"),
        hovertext=node_hovertexts,
        customdata=node_customdata,
        hoverinfo="text",
        name="Peripheral_Nodes",
        showlegend=False,
    ))

    # 5. SPIDER_CORE Central Node
    core_color = "#10B981" if core_state.systemic_web_tension < Z_SCORE_NORMAL else (
        "#F59E0B" if core_state.systemic_web_tension < Z_SCORE_CRITICAL else "#EF4444"
    )

    core_hover = (
        f"<b>🕷️ SPIDER_CORE: CENTRO DI CONTROLLO</b><br>"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        f"• Regime Macro: <b>{core_state.macro_regime}</b><br>"
        f"• Tensione della Tela: <b>{core_state.systemic_web_tension:.2f}</b> ({core_state.web_tension_status})<br>"
        f"• Bias Banche Centrali: <b>{core_state.central_bank_bias}</b><br>"
        f"• Tasso Reale FED: <b>{core_state.fed_real_rate:+.2f}%</b> | BCE: <b>{core_state.ecb_real_rate:+.2f}%</b><br>"
        f"• Taylor Rate Gap: <b>{core_state.taylor_rate_gap_bps:+.0f} bps</b><br>"
        f"• Allarmi Critici Attivi: <b>{len(core_state.active_critical_alerts)}</b>"
    )

    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(size=56, color=core_color, opacity=0.25),
        hoverinfo="none",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers+text",
        marker=dict(size=36, color=core_color, line=dict(color="#FFFFFF", width=3.0)),
        text=["SPIDER_CORE"],
        textposition="bottom center",
        textfont=dict(size=11, color="#FFFFFF", family="Inter, sans-serif"),
        hovertext=[core_hover],
        hoverinfo="text",
        name="Spider_Core",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        dragmode="lasso",  # Default freeform lasso selection enabled
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.6], scaleanchor="x", scaleratio=1),
        height=680,
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, monospace",
            bordercolor="#475569"
        ),
    )
    return fig



def build_macro_regime_matrix_figure(core_state: SpiderCoreState, dark_mode: bool = True) -> go.Figure:
    """Renders 2D Macro Regime Phase-Space Matrix (Growth vs Inflation)."""
    fig = go.Figure()

    bg_color = "#0B0F19" if dark_mode else "#F8FAFC"
    axis_color = "rgba(255, 255, 255, 0.25)" if dark_mode else "rgba(15, 23, 42, 0.25)"

    fig.add_shape(type="rect", x0=0, y0=-4, x1=4, y1=0, fillcolor="rgba(16, 185, 129, 0.12)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0, y0=0, x1=4, y1=4, fillcolor="rgba(245, 158, 11, 0.12)", line=dict(width=0))
    fig.add_shape(type="rect", x0=-4, y0=0, x1=0, y1=4, fillcolor="rgba(239, 68, 68, 0.12)", line=dict(width=0))
    fig.add_shape(type="rect", x0=-4, y0=-4, x1=0, y1=0, fillcolor="rgba(59, 130, 246, 0.12)", line=dict(width=0))

    fig.add_annotation(x=2.0, y=-2.0, text="<b>GOLDILOCKS</b><br>Disinflazione & Crescita<br>(Multipli Equity Risk-On)", showarrow=False, font=dict(color="#10B981", size=11))
    fig.add_annotation(x=2.0, y=2.0, text="<b>SURRISCALDAMENTO CICLICO</b><br>Late-Cycle Inflationary Boom<br>(Commodities & Value)", showarrow=False, font=dict(color="#F59E0B", size=11))
    fig.add_annotation(x=-2.0, y=2.0, text="<b>STAGFLAZIONE</b><br>Rallentamento & Inflazione<br>(Oro, Cash, Difensivi)", showarrow=False, font=dict(color="#EF4444", size=11))
    fig.add_annotation(x=-2.0, y=-2.0, text="<b>DEFLAZIONE / RECESSIONE</b><br>Contrazione Generale<br>(Treasury Bonds & Bunds)", showarrow=False, font=dict(color="#3B82F6", size=11))

    traj = core_state.trajectory_coords
    if traj and len(traj) > 1:
        tx = [t[1] for t in traj]
        ty = [t[2] for t in traj]
        tlabels = [t[0] for t in traj]

        fig.add_trace(go.Scatter(
            x=tx,
            y=ty,
            mode="lines+markers",
            line=dict(color="#38BDF8", width=2.5, dash="dash"),
            marker=dict(size=8, color="#38BDF8"),
            text=tlabels,
            hoverinfo="text+x+y",
            name="Traiettoria 12M",
        ))

    curr_x, curr_y = core_state.regime_coords
    fig.add_trace(go.Scatter(
        x=[curr_x],
        y=[curr_y],
        mode="markers+text",
        marker=dict(size=22, color="#FFFFFF", line=dict(color="#D946EF", width=4)),
        text=["<b>OGGI</b>"],
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=12),
        name="Regime Attuale",
        hovertext=f"<b>Punto Attuale</b><br>Growth Score: {curr_x:+.2f}<br>Inflation Score: {curr_y:+.2f}<br>Regime: {core_state.macro_regime}",
        hoverinfo="text",
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Score Composito Crescita Economica (Z-Score)", zeroline=True, zerolinecolor=axis_color, range=[-3.5, 3.5], gridcolor=axis_color),
        yaxis=dict(title="Score Composito Inflazione & Pressioni Prezzi (Z-Score)", zeroline=True, zerolinecolor=axis_color, range=[-3.5, 3.5], gridcolor=axis_color),
        height=380,
        showlegend=False,
    )
    return fig


def generate_nodes_dataframe(node_states: Dict[str, NodeState]) -> pd.DataFrame:
    rows = []
    for n_id, s in node_states.items():
        rows.append({
            "ID Nodo": s.id,
            "Nome Indicatore / Asset": s.name,
            "Anello": f"Anello {s.ring}",
            "Quadrante": s.quadrant,
            "Categoria": s.category,
            "Valore Attuale": f"{s.current_value:,.2f} {s.unit}",
            "Valore Precedente": f"{s.prev_value:,.2f}",
            "Delta Periodo": f"{s.delta_period:+,.2f} ({s.delta_period_pct:+.2f}%)",
            "Delta YoY (1A)": f"{s.delta_yoy:+,.2f}",
            "Z-Score": round(s.z_score, 2),
            "Stato": s.status,
            "Fonte Certificata": s.source,
            "Ticker Ufficiale": s.official_ticker,
            "Ultimo Rilascio": s.last_timestamp,
        })
    df = pd.DataFrame(rows)
    return df


def generate_links_dataframe(link_states: List[LinkState]) -> pd.DataFrame:
    rows = []
    for l in link_states:
        rows.append({
            "ID Filo": l.id,
            "Relazione (Nodo A ↔ Nodo B)": f"{l.node_a_name} ↔ {l.node_b_name}",
            "Tipo": "Filo Maestro" if l.is_master else "Raggio Radiale / Spoke",
            "Valore A": f"{l.node_a_val:,.2f}",
            "Valore B": f"{l.node_b_val:,.2f}",
            "Correlazione Attuale": round(l.active_corr, 2),
            "Correlazione Teorica": f"{l.theoretical_corr:+d}",
            "Tensione (|Z-Score|)": round(l.z_score_tension, 2),
            "Stato del Filo": l.status,
            "Meccanismo Trasmissione": l.mechanism,
            "Allarme Anomalia Macro": l.alert_description,
        })
    df = pd.DataFrame(rows)
    return df
