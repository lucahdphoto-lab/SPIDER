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

    # 0. Pointillism Radial Micro-Particle Constellation Field (Trama a Micro-Punti Sfumata ad Alta Definizione)
    # Crea una matrice radiale di micro-punti luminosi che sfumano verso l'esterno, accendendosi nei quadranti dominanti
    g_score = core_state.growth_score
    inf_score = core_state.inflation_score
    t_val = core_state.systemic_web_tension

    # Griglia polare di micro-punti (36 raggi concentrici x 48 passi angolari)
    dot_radii = np.linspace(0.4, 3.25, 24)
    dot_angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    
    mesh_r, mesh_theta = np.meshgrid(dot_radii, dot_angles)
    pts_x = (mesh_r * np.cos(mesh_theta)).flatten()
    pts_y = (mesh_r * np.sin(mesh_theta)).flatten()
    pts_r = mesh_r.flatten()
    pts_th = (mesh_theta.flatten() % (2 * np.pi))

    # Calcola il colore e l'intensità di ciascun micro-punto in base al quadrante e alla gravità
    pts_colors = []
    pts_sizes = []
    for px, py, pr, pth in zip(pts_x, pts_y, pts_r, pts_th):
        # Gradiente radiale: la densità/luminosità cresce dal centro verso gli anelli esterni
        dist_factor = (pr / 3.25) ** 1.3
        
        # Quadrante 1: Inflazione & Commodities (0° - 90°) -> Giallo/Ambra Oro Vivace
        if 0 <= pth < np.pi / 2:
            base_rgb = (245, 158, 11)   # #F59E0B
            tension_boost = max(0.0, inf_score) * 0.4
            alpha = min(0.65, max(0.08, (0.15 + tension_boost) * dist_factor))
            sz = 2.0 + 1.8 * dist_factor
        # Quadrante 2: Crescita & Produzione (90° - 180°) -> Verde Smeraldo / Ciano Vivace
        elif np.pi / 2 <= pth < np.pi:
            base_rgb = (16, 185, 129)   # #10B981
            tension_boost = max(0.0, g_score) * 0.4
            alpha = min(0.65, max(0.08, (0.15 + tension_boost) * dist_factor))
            sz = 2.0 + 1.8 * dist_factor
        # Quadrante 3: Lavoro & Occupazione (180° - 270°) -> Blu Elettrico
        elif np.pi <= pth < 3 * np.pi / 2:
            base_rgb = (59, 130, 246)   # #3B82F6
            unrate_z = node_states["LAB_UNRATE"].z_score if "LAB_UNRATE" in node_states else 0.0
            alpha = min(0.60, max(0.08, (0.14 + max(0.0, unrate_z) * 0.25) * dist_factor))
            sz = 2.0 + 1.5 * dist_factor
        # Quadrante 4: Liquidità & Rischio (270° - 360°) -> Magenta / Fucsia Neon
        else:
            base_rgb = (236, 72, 153)   # #EC4899
            hy_z = node_states["CRED_HY"].z_score if "CRED_HY" in node_states else 0.0
            alpha = min(0.65, max(0.08, (0.15 + max(0.0, hy_z) * 0.3) * dist_factor))
            sz = 2.0 + 1.8 * dist_factor

        pts_colors.append(f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {alpha:.3f})")
        pts_sizes.append(sz)

    fig.add_trace(go.Scatter(
        x=pts_x,
        y=pts_y,
        mode="markers",
        marker=dict(
            size=pts_sizes,
            color=pts_colors,
        ),
        hoverinfo="skip",
        showlegend=False,
        name="Macro_Pointillism_Field",
    ))

    # 1. Concentric Geometric Rings con sfumature delicate di anello (Causa-Effetto: Anello 1 -> Anello 2 -> Anello 3)
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
            x=-r,
            y=0.10,
            text=f"<b>{r_info['name'].split(' ')[0]} {r_info['name'].split(' ')[1]}</b>",
            showarrow=False,
            font=dict(size=9, color="#94A3B8" if dark_mode else "#64748B"),
            bgcolor="rgba(11, 15, 25, 0.85)" if dark_mode else "rgba(248, 250, 252, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.15)",
            borderwidth=1,
            borderpad=2,
        )

    # 2. Quadrant Axes & Styled Headers in Outer Margin
    max_r = 3.5
    fig.add_trace(go.Scatter(
        x=[-max_r, max_r, None, 0, 0],
        y=[0, 0, None, -max_r, max_r],
        mode="lines",
        line=dict(color=axis_color, width=1.0, dash="dash"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Place quadrant labels at the absolute outer corners so they never touch nodes or links
    quad_watermarks = [
        (2.35, 3.35, "🟡 INFLAZIONE & COMMODITIES", "#F59E0B"),
        (-2.35, 3.35, "🟢 CRESCITA & PRODUZIONE", "#10B981"),
        (-2.35, -3.35, "🔵 LAVORO & OCCUPAZIONE", "#3B82F6"),
        (2.35, -3.35, "🟣 LIQUIDITÀ & RISCHIO", "#EC4899"),
    ]
    for qx, qy, qtxt, qcol in quad_watermarks:
        fig.add_annotation(
            x=qx,
            y=qy,
            text=f"<b>{qtxt}</b>",
            showarrow=False,
            font=dict(size=10.5, color=qcol),
            bgcolor="rgba(15, 23, 42, 0.85)" if dark_mode else "rgba(255, 255, 255, 0.85)",
            bordercolor=qcol,
            borderwidth=1,
            borderpad=4,
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

        # Se il filo è in rottura o divergenza critica (|Z| >= 2.0), aggiungi un alone di sottofondo sfumato
        if link.z_score_tension >= Z_SCORE_CRITICAL:
            fig.add_trace(go.Scatter(
                x=[xA, xB],
                y=[yA, yB],
                mode="lines",
                line=dict(color="rgba(217, 70, 239, 0.18)", width=line_width * 3.5),
                hoverinfo="none",
                showlegend=False,
            ))

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


def build_spider_web_3d_figure(
    node_states: Dict[str, NodeState],
    link_states: List[LinkState],
    core_state: CoreSystemicState,
    selected_node_id: Optional[str] = None,
    filter_quadrant: str = "TUTTI",
    filter_ring: int = 0,
    show_labels: bool = True,
    dark_mode: bool = True,
    elevation_boost: float = 1.35,
) -> go.Figure:
    """
    Costruisce la visualizzazione tridimensionale interattiva (WebGL) della tela di ragno SPIDER.
    I nodi poggiano su una topologia radiale 3D:
    - Livello Z base determinato dalla tensione / Z-Score fisiologico
    - Quando un nodo viene selezionato, si solleva (elevazione positiva sull'asse Z)
      e si ingrandisce con effetto 'floating pedestal', mettendo in risalto le linee di trazione 3D.
    """
    fig = go.Figure()
    bg_color = "rgba(11, 15, 25, 1.0)" if dark_mode else "rgba(248, 250, 252, 1.0)"
    text_color = "#F8FAFC" if dark_mode else "#0F172A"

    # Calcolo coordinate 3D dei nodi
    # x, y in base a raggio polare e angolo
    # z: default calcolato da z_score * 0.15 (curvatura parabolica della tela a imbuto)
    coords_3d: Dict[str, Tuple[float, float, float]] = {
        "SPIDER_CORE": (0.0, 0.0, 0.3 if selected_node_id == "SPIDER_CORE" else 0.0)
    }

    # Focus e nodi connessi
    active_connected_nodes = set()
    if selected_node_id and selected_node_id != "SPIDER_CORE":
        active_connected_nodes.add(selected_node_id)
        for link in link_states:
            if link.node_a_id == selected_node_id:
                active_connected_nodes.add(link.node_b_id)
            elif link.node_b_id == selected_node_id:
                active_connected_nodes.add(link.node_a_id)

    # 1. Cerchi concentrici dei 3 anelli nel piano 3D
    ring_theta = [i * (2 * math.pi / 72) for i in range(73)]
    for r_num, r_data in RINGS.items():
        if filter_ring > 0 and r_num != filter_ring:
            continue
        rad = r_data["radius"]
        c_x = [rad * math.cos(t) for t in ring_theta]
        c_y = [rad * math.sin(t) for t in ring_theta]
        c_z = [-0.15 * (rad ** 0.8) for _ in ring_theta]  # Leggera curvatura a imbuto della tela
        fig.add_trace(go.Scatter3d(
            x=c_x,
            y=c_y,
            z=c_z,
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.22)" if dark_mode else "rgba(100, 116, 139, 0.25)", width=1.5),
            hoverinfo="none",
            showlegend=False,
        ))

    # Calcolo coordinate di ogni nodo
    for n_id, n_def in NODES.items():
        rad = RINGS[n_def.ring]["radius"]
        angle_rad = math.radians(n_def.angle_deg)
        x = rad * math.cos(angle_rad)
        y = rad * math.sin(angle_rad)
        
        # Livello base Z parabolico
        base_z = -0.15 * (rad ** 0.8)
        
        # Elevazione speciale: se il nodo è selezionato, esce marcatamente in alto!
        if selected_node_id == n_id:
            z = base_z + elevation_boost
        elif n_id in active_connected_nodes:
            z = base_z + (elevation_boost * 0.35)  # I nodi partner collegati si sollevano lievemente
        else:
            n_st = node_states.get(n_id)
            z_dev = (n_st.z_score * 0.08) if n_st else 0.0
            z = base_z + z_dev

        coords_3d[n_id] = (x, y, z)

    # 2. Disegno dei collegamenti 3D (Fili della tela)
    for link in link_states:
        if link.node_a_id not in coords_3d or link.node_b_id not in coords_3d:
            continue

        xA, yA, zA = coords_3d[link.node_a_id]
        xB, yB, zB = coords_3d[link.node_b_id]

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
            line_color = link.color_hex if is_focused else ("rgba(255,255,255,0.02)" if dark_mode else "rgba(0,0,0,0.02)")
            line_width = (link.line_width * 2.2) if is_focused else 0.8
        else:
            line_color = link.color_hex if is_focused else ("rgba(255,255,255,0.05)" if dark_mode else "rgba(0,0,0,0.05)")
            line_width = (link.line_width * 1.4) if is_focused else 0.8

        fig.add_trace(go.Scatter3d(
            x=[xA, xB],
            y=[yA, yB],
            z=[zA, zB],
            mode="lines",
            line=dict(color=line_color, width=line_width),
            hoverinfo="none",
            showlegend=False,
        ))

    # Se un nodo è selezionato, disegna un pilastro / raggio laser verticale verso il piano base
    if selected_node_id and selected_node_id in coords_3d and selected_node_id != "SPIDER_CORE":
        sx, sy, sz = coords_3d[selected_node_id]
        rad_sel = RINGS[NODES[selected_node_id].ring]["radius"]
        base_floor_z = -0.15 * (rad_sel ** 0.8)
        fig.add_trace(go.Scatter3d(
            x=[sx, sx],
            y=[sy, sy],
            z=[base_floor_z, sz],
            mode="lines",
            line=dict(color="#38BDF8", width=4.5),
            hoverinfo="none",
            showlegend=False,
        ))

    # 3. Nodi Periferici 3D
    node_x, node_y, node_z = [], [], []
    node_colors = []
    node_sizes = []
    node_labels = []
    node_hovertexts = []
    node_customdata = []

    for n_id, n_def in NODES.items():
        if n_id not in coords_3d:
            continue
        x, y, z = coords_3d[n_id]
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
            node_fill = "rgba(51, 65, 85, 0.2)"
            size = 4
        elif selected_node_id == n_id:
            node_fill = "#38BDF8"  # Neon blue spotlight
            size = 14
        elif n_state and abs(n_state.z_score) >= Z_SCORE_CRITICAL:
            node_fill = "#D946EF"
            size = 10
        elif n_state and abs(n_state.z_score) >= Z_SCORE_NORMAL:
            node_fill = "#F59E0B"
            size = 8
        else:
            node_fill = base_color
            size = 7

        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(node_fill)
        node_sizes.append(size)
        node_labels.append(n_id if (show_labels and (not is_dimmed or selected_node_id == n_id)) else "")
        node_customdata.append(n_id)

        val_str = f"{n_state.current_value:,.2f} {n_state.unit}" if n_state else "N/D"
        delta_str = f"{n_state.delta_period:+,.2f} ({n_state.delta_period_pct:+.2f}%)" if n_state else "N/D"
        z_str = f"{n_state.z_score:+.2f}" if n_state else "0.00"
        yoy_str = f"{n_state.delta_yoy:+,.2f}" if n_state else "N/D"
        status_str = n_state.status if n_state else "Normale"

        hover_card = (
            f"<b>🏛️ NODO 3D: {n_def.name} [{n_id}]</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"• Quota Z (Altezza): <b>{z:+.2f}</b><br>"
            f"• Categoria: <b>{n_def.category}</b> (Anello {n_def.ring})<br>"
            f"• Valore Attuale: <b>{val_str}</b><br>"
            f"• Variazione Periodo: <b>{delta_str}</b><br>"
            f"• Variazione YoY: <b>{yoy_str}</b><br>"
            f"• Z-Score Dinamico: <b>{z_str}</b> ({status_str})<br>"
            f"• Ruolo Macro: {n_def.role}"
        )
        node_hovertexts.append(hover_card)

    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.95,
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9.5, color=text_color, family="Inter, sans-serif"),
        hovertext=node_hovertexts,
        customdata=node_customdata,
        hoverinfo="text",
        name="Nodes_3D",
        showlegend=False,
    ))

    # 4. SPIDER_CORE Central Node 3D
    core_x, core_y, core_z = coords_3d["SPIDER_CORE"]
    core_color = "#10B981" if core_state.systemic_web_tension < Z_SCORE_NORMAL else (
        "#F59E0B" if core_state.systemic_web_tension < Z_SCORE_CRITICAL else "#EF4444"
    )

    core_hover = (
        f"<b>🕷️ SPIDER_CORE: CENTRO DI CONTROLLO 3D</b><br>"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        f"• Regime Macro: <b>{core_state.macro_regime}</b><br>"
        f"• Tensione della Tela: <b>{core_state.systemic_web_tension:.2f}</b> ({core_state.web_tension_status})<br>"
        f"• Bias Banche Centrali: <b>{core_state.central_bank_bias}</b>"
    )

    fig.add_trace(go.Scatter3d(
        x=[core_x],
        y=[core_y],
        z=[core_z],
        mode="markers+text",
        marker=dict(size=16, color=core_color, opacity=0.9),
        text=["SPIDER_CORE"],
        textposition="bottom center",
        textfont=dict(size=11, color="#FFFFFF", family="Inter, sans-serif"),
        hovertext=[core_hover],
        hoverinfo="text",
        name="Spider_Core_3D",
        showlegend=False,
    ))

    # Camera 3D angolata per una prospettiva dinamica a volo d'uccello
    camera = dict(
        eye=dict(x=1.35, y=-1.65, z=1.2),
        center=dict(x=0, y=0, z=-0.1),
        up=dict(x=0, y=0, z=1)
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=True, zeroline=False, showticklabels=False, title="", gridcolor="rgba(255,255,255,0.06)"),
            camera=camera,
            bgcolor=bg_color,
        ),
        margin=dict(l=0, r=0, t=10, b=10),
        height=720,
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="#38BDF8",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#F8FAFC",
        ),
    )

    return fig


def build_zscore_distribution_figure(
    node_states: Dict[str, NodeState],
    dark_mode: bool = True,
) -> go.Figure:
    """
    Costruisce il grafico a barre orizzontali degli Z-Score dei 30 Nodi Macro-Finanziari,
    con visualizzazione standardizzata (-3 a +3), asse centrale allo zero,
    e codifica a 3 colori istituzionali:
    - 🟦 Estremo (|Z| >= 2.0)
    - 🟥 Attenzione (1.25 <= |Z| < 2.0)
    - 🟨 Normale (|Z| < 1.25)
    """
    sorted_nodes = sorted(node_states.values(), key=lambda x: x.z_score)

    names = [ns.name for ns in sorted_nodes]
    z_scores = [ns.z_score for ns in sorted_nodes]
    values_str = [f"{ns.current_value:,.2f} {ns.unit}" for ns in sorted_nodes]
    statuses = [ns.status for ns in sorted_nodes]

    colors = []
    for z in z_scores:
        abs_z = abs(z)
        if abs_z >= 2.0:
            colors.append("#3B82F6")  # Estremo (Blu)
        elif abs_z >= 1.25:
            colors.append("#EF4444")  # Attenzione (Rosso)
        else:
            colors.append("#F59E0B")  # Normale (Giallo/Oro)

    bg_color = "#0B0F19" if dark_mode else "#F8FAFC"
    card_bg = "rgba(15, 23, 42, 0.7)" if dark_mode else "#FFFFFF"
    text_color = "#F8FAFC" if dark_mode else "#0F172A"
    grid_color = "rgba(255, 255, 255, 0.08)" if dark_mode else "rgba(15, 23, 42, 0.08)"

    fig = go.Figure()

    # Barre orizzontali principali
    customdata = list(zip(values_str, statuses))
    fig.add_trace(go.Bar(
        y=names,
        x=z_scores,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate="<b>%{y}</b><br>Z-Score Standardizzato: <b>%{x:+.2f}</b><br>Valore Attuale: %{customdata[0]}<br>Stato: %{customdata[1]}<extra></extra>",
        showlegend=False,
    ))

    # Tracce fittizie per la legenda a 3 voci (Estremo, Attenzione, Normale) identica allo screenshot
    fig.add_trace(go.Bar(
        x=[None], y=[None], orientation="h",
        name="Estremo",
        marker=dict(color="#3B82F6"),
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None], orientation="h",
        name="Attenzione",
        marker=dict(color="#EF4444"),
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None], orientation="h",
        name="Normale",
        marker=dict(color="#F59E0B"),
        showlegend=True,
    ))

    # Forme geometriche (linea zero centrale + soglie di allerta tratteggiate)
    shapes = [
        # Linea dello zero centrale
        dict(
            type="line",
            x0=0, x1=0,
            y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="#94A3B8" if dark_mode else "#334155", width=2),
        ),
        # Soglia Estremo -2.0
        dict(
            type="line",
            x0=-2.0, x1=-2.0,
            y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="rgba(59, 130, 246, 0.45)", width=1.2, dash="dash"),
        ),
        # Soglia Estremo +2.0
        dict(
            type="line",
            x0=2.0, x1=2.0,
            y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="rgba(59, 130, 246, 0.45)", width=1.2, dash="dash"),
        ),
        # Soglia Attenzione -1.25
        dict(
            type="line",
            x0=-1.25, x1=-1.25,
            y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="rgba(239, 68, 68, 0.3)", width=1, dash="dot"),
        ),
        # Soglia Attenzione +1.25
        dict(
            type="line",
            x0=1.25, x1=1.25,
            y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="rgba(239, 68, 68, 0.3)", width=1, dash="dot"),
        ),
    ]

    fig.update_layout(
        title=dict(
            text="<b>Z-Score degli Indicatori Macro-Finanziari (30 Nodi)</b>",
            font=dict(size=17, color=text_color, family="Inter, sans-serif"),
            x=0.03,
            y=0.98,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=card_bg,
        shapes=shapes,
        xaxis=dict(
            title=dict(text="Z-Score Standardizzato", font=dict(size=12, color="#94A3B8")),
            range=[-3.0, 3.0],
            tickmode="linear",
            tick0=-3,
            dtick=1,
            gridcolor=grid_color,
            zeroline=False,
            tickfont=dict(color="#CBD5E1", size=11),
        ),
        yaxis=dict(
            title=dict(text="Indicatore", font=dict(size=12, color="#94A3B8")),
            autorange="reversed",  # Mette i negativi in alto esattamente come nello screenshot!
            tickfont=dict(color="#CBD5E1", size=11),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=190, r=30, t=50, b=65),
        height=740,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=text_color),
        ),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="#38BDF8",
            font_size=12,
            font_color="#F8FAFC",
        ),
    )

    return fig


def generate_zscore_mini_svg(node_states: Dict[str, NodeState]) -> str:
    """Genera una rappresentazione grafica vettoriale SVG compatta (riepilogo) dello Z-Score per il box KPI."""
    sorted_nodes = sorted(node_states.values(), key=lambda x: x.z_score)
    svg_bars = []

    center_x = 110.0
    scale = 30.0  # 30px per unità Z (da -3 a +3 => 110 - 90 = 20, 110 + 90 = 200)

    y_start = 5.0
    bar_height = 1.9
    step_y = 2.45  # 30 * 2.45 = ~74px

    for i, ns in enumerate(sorted_nodes):
        z = ns.z_score
        y = y_start + i * step_y
        abs_z = abs(z)
        if abs_z >= 2.0:
            color = "#3B82F6"
        elif abs_z >= 1.25:
            color = "#EF4444"
        else:
            color = "#F59E0B"

        bar_w = min(max(abs_z * scale, 1.0), 92.0)
        if z < 0:
            bx = center_x - bar_w
        else:
            bx = center_x

        svg_bars.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_height:.1f}" rx="0.5" fill="{color}" opacity="0.9"/>')

    bars_html = "".join(svg_bars)

    svg = (
        f'<div style="width: 100%; display: flex; justify-content: center; margin: 2px 0;">'
        f'<svg viewBox="0 0 220 86" style="width: 100%; max-height: 78px; overflow: visible;">'
        f'<line x1="20" y1="3" x2="20" y2="78" stroke="rgba(255,255,255,0.08)" stroke-width="0.7" stroke-dasharray="2,2"/>'
        f'<line x1="50" y1="3" x2="50" y2="78" stroke="rgba(59,130,246,0.25)" stroke-width="0.8" stroke-dasharray="2,2"/>'
        f'<line x1="170" y1="3" x2="170" y2="78" stroke="rgba(59,130,246,0.25)" stroke-width="0.8" stroke-dasharray="2,2"/>'
        f'<line x1="200" y1="3" x2="200" y2="78" stroke="rgba(255,255,255,0.08)" stroke-width="0.7" stroke-dasharray="2,2"/>'
        f'<line x1="{center_x:.1f}" y1="1" x2="{center_x:.1f}" y2="79" stroke="#94A3B8" stroke-width="1.2"/>'
        f'{bars_html}'
        f'<text x="20" y="85" fill="#64748B" font-size="7" font-family="monospace" text-anchor="middle">-3</text>'
        f'<text x="50" y="85" fill="#3B82F6" font-size="7" font-family="monospace" text-anchor="middle">-2</text>'
        f'<text x="{center_x:.1f}" y="85" fill="#94A3B8" font-size="7" font-family="monospace" text-anchor="middle">0</text>'
        f'<text x="170" y="85" fill="#3B82F6" font-size="7" font-family="monospace" text-anchor="middle">+2</text>'
        f'<text x="200" y="85" fill="#64748B" font-size="7" font-family="monospace" text-anchor="middle">+3</text>'
        f'</svg></div>'
    )
    return svg


