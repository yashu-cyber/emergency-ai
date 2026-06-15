import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import time
from datetime import datetime

from agent.simulator import EvacuationSimulator
from agent.graph import NODE_COORDS
from agent.reasoner import get_path_cost

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Command Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #020617 !important;
    color: #F1F5F9 !important;
}

/* ── Remove default streamlit chrome ── */
.stApp { background: #020617 !important; }
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
.block-container { padding: 0 1.5rem 2rem 1.5rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020617; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ── Command Header ── */
.cmd-header {
    background: linear-gradient(135deg, #0F172A 0%, #0d1f3c 100%);
    border-bottom: 1px solid rgba(59,130,246,0.2);
    padding: 0.85rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -1.5rem 1.5rem -1.5rem;
}
.cmd-logo {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.cmd-logo-icon { font-size: 1.5rem; }
.cmd-logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #F1F5F9;
    text-transform: uppercase;
}
.cmd-logo-sub {
    font-size: 0.62rem;
    color: #64748B;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1px;
}

/* ── Metric Pills ── */
.metric-strip {
    display: flex;
    gap: 0.75rem;
    align-items: center;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.45rem 1rem;
    text-align: center;
    min-width: 120px;
    backdrop-filter: blur(8px);
}
.pill-label {
    font-size: 0.6rem;
    color: #64748B;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.pill-value {
    font-size: 0.88rem;
    font-weight: 600;
    color: #F1F5F9;
}
.pill-green  { border-color: rgba(34,197,94,0.35);  background: rgba(34,197,94,0.08); }
.pill-yellow { border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.08); }
.pill-orange { border-color: rgba(249,115,22,0.35); background: rgba(249,115,22,0.08); }
.pill-red    { border-color: rgba(239,68,68,0.35);  background: rgba(239,68,68,0.08); }
.pill-blue   { border-color: rgba(59,130,246,0.35); background: rgba(59,130,246,0.08); }
.pill-green  .pill-value { color: #22C55E; }
.pill-yellow .pill-value { color: #F59E0B; }
.pill-orange .pill-value { color: #F97316; }
.pill-red    .pill-value { color: #EF4444; }
.pill-blue   .pill-value { color: #60A5FA; }

/* ── Sidebar elements ── */
.sidebar-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 1.2rem 0 0.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 0.5rem;
}
.sidebar-section-title:first-of-type { border-top: none; padding-top: 0.5rem; }

/* ── Incident preset cards ── */
.preset-card {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.8rem;
    color: #CBD5E1;
}
.preset-card:hover { background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.12); }
.preset-card.active { border-color: rgba(59,130,246,0.6); background: rgba(59,130,246,0.1); color: #F1F5F9; }

/* ── Panel cards ── */
.panel-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    backdrop-filter: blur(8px);
}
.panel-card-title {
    font-size: 0.68rem;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Algorithm cards ── */
.algo-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s;
}
.algo-card.selected {
    border-color: rgba(59,130,246,0.5);
    background: rgba(59,130,246,0.07);
    box-shadow: 0 0 16px rgba(59,130,246,0.12);
}
.algo-card.winner {
    border-color: rgba(34,197,94,0.5);
    background: rgba(34,197,94,0.06);
}
.algo-name {
    font-size: 0.78rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.algo-badge {
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 4px;
}
.badge-active  { background: rgba(59,130,246,0.25); color: #60A5FA; }
.badge-winner  { background: rgba(34,197,94,0.25);  color: #22C55E; }
.badge-nopath  { background: rgba(239,68,68,0.2);   color: #F87171; }
.algo-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.3rem 0.6rem;
    font-size: 0.72rem;
}
.algo-metric-item { color: #94A3B8; }
.algo-metric-item span { color: #CBD5E1; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

/* ── Stat row ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.78rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-row-label { color: #64748B; }
.stat-row-value { color: #E2E8F0; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

/* ── Route path display ── */
.route-path {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #22C55E;
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    word-break: break-all;
    margin-top: 0.4rem;
}
.route-path.nopath {
    color: #EF4444;
    background: rgba(239,68,68,0.06);
    border-color: rgba(239,68,68,0.15);
}

/* ── Event timeline ── */
.timeline-outer {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
}
.timeline-title {
    font-size: 0.62rem;
    font-weight: 700;
    color: #475569;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.timeline-scroll {
    display: flex;
    gap: 0.6rem;
    overflow-x: auto;
    padding-bottom: 4px;
}
.timeline-scroll::-webkit-scrollbar { height: 3px; }
.evt-chip {
    flex-shrink: 0;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.4rem 0.65rem;
    min-width: 130px;
}
.evt-time { font-size: 0.6rem; color: #475569; font-family: 'JetBrains Mono', monospace; margin-bottom: 2px; }
.evt-text { font-size: 0.72rem; color: #94A3B8; }
.evt-chip.fire    { border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.07);  }
.evt-chip.gas     { border-color: rgba(249,115,22,0.3); background: rgba(249,115,22,0.07); }
.evt-chip.crowd   { border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.07); }
.evt-chip.move    { border-color: rgba(59,130,246,0.3); background: rgba(59,130,246,0.07); }
.evt-chip.success { border-color: rgba(34,197,94,0.3);  background: rgba(34,197,94,0.07);  }
.evt-chip.block   { border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.07);  }
.evt-chip.route   { border-color: rgba(34,197,94,0.25); background: rgba(34,197,94,0.05);  }
.evt-chip.fire .evt-text    { color: #FCA5A5; }
.evt-chip.gas .evt-text     { color: #FED7AA; }
.evt-chip.crowd .evt-text   { color: #FDE68A; }
.evt-chip.move .evt-text    { color: #93C5FD; }
.evt-chip.success .evt-text { color: #86EFAC; }
.evt-chip.block .evt-text   { color: #FCA5A5; }
.evt-chip.route .evt-text   { color: #86EFAC; }

/* ── Bottom algo comparison ── */
.compare-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    flex: 1;
}
.compare-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.compare-opt-tag {
    font-size: 0.62rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
}
.compare-complexity {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    margin: 0.3rem 0 0.6rem;
}
.compare-desc {
    font-size: 0.72rem;
    color: #64748B;
    line-height: 1.6;
}

/* ── Streamlit widget overrides ── */
.stSlider > div > div { background: rgba(255,255,255,0.06) !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: #3B82F6 !important; }
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}
label { color: #94A3B8 !important; font-size: 0.78rem !important; }
.stButton > button {
    width: 100% !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary { color: #94A3B8 !important; font-size: 0.8rem !important; }

/* ── Status banner ── */
.status-banner {
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    text-align: center;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.status-banner.success {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22C55E;
}
.status-banner.failed {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #EF4444;
}

/* ── Legend row inside map ── */
.legend-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
    font-size: 0.7rem;
    color: #64748B;
}
.legend-item { display: flex; align-items: center; gap: 0.35rem; }
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "simulator" not in st.session_state:
    st.session_state.simulator      = EvacuationSimulator()
    st.session_state.initialized    = False
    st.session_state.step_result    = None
    st.session_state.timeline       = []   # list of (ts, text, kind)
    st.session_state.incident_preset = "Normal Scenario"


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
SEVERITY_COLOR = {"GREEN": "green", "YELLOW": "yellow", "ORANGE": "orange", "RED": "red"}
SEVERITY_EMOJI = {"GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴"}
INCIDENT_EMOJI = {
    "NONE": "✅", "FIRE": "🔥", "GAS_LEAK": "☣️", "BLOCKED_EXIT": "🚧", "CROWD_SURGE": "👥"
}

def now_ts():
    return datetime.now().strftime("%H:%M:%S")

def add_event(text: str, kind: str = "move"):
    st.session_state.timeline.append((now_ts(), text, kind))

def build_sensors(temperature, smoke, gas_ppm, blocked_exits, corridor_occupancies):
    return {
        "temperature": temperature,
        "smoke": smoke,
        "gas_ppm": gas_ppm,
        "exit_status": {ex: (0 if ex in blocked_exits else 1)
                        for ex in ["EXIT-A", "EXIT-B", "EXIT-C", "EXIT-D"]},
        "corridor_occupancies": corridor_occupancies,
    }


# ─────────────────────────────────────────────
#  BUILDING MAP RENDERING
# ─────────────────────────────────────────────
# Visual layout overrides – keep the same logical graph but
# re-position nodes for a clear 2-floor floorplan look.
MAP_POS = {
    # Floor 1 – bottom band  (y = 0..2)
    "R1":     (0.0,  0.0),
    "R2":     (2.0,  0.0),
    "R3":     (4.0,  0.0),
    "R4":     (6.0,  0.0),
    "R5":     (8.0,  0.0),
    "C1":     (2.5,  1.4),
    "C2":     (5.5,  1.4),
    "EXIT-A": (-1.5, 0.0),
    "EXIT-B": ( 9.5, 0.0),
    # Stairwell – middle
    "C5":     (4.0,  3.0),
    # Floor 2 – top band  (y = 4.5..6.5)
    "R6":     (0.0,  6.0),
    "R7":     (2.0,  6.0),
    "R8":     (4.0,  6.0),
    "R9":     (6.0,  6.0),
    "R10":    (8.0,  6.0),
    "C3":     (2.5,  4.6),
    "C4":     (5.5,  4.6),
    "EXIT-C": (-1.5, 6.0),
    "EXIT-D": ( 9.5, 6.0),
}

NODE_TYPE_COLOR = {
    "room":     "#1E3A5F",
    "corridor": "#1E2A3A",
    "exit":     "#052E16",
}

def draw_building_map(simulator, step_data, initialized):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    ax.set_xlim(-3.0, 11.5)
    ax.set_ylim(-1.8, 8.0)
    ax.axis("off")

    # ── Floor separator lines ──
    for y, label in [(3.5, "FLOOR  1"), (2.5, "FLOOR  2")]:
        pass
    # Floor 1 label
    ax.text(-2.8, -1.4, "FLOOR 1", fontsize=7, color="#1E3A5F",
            fontfamily="monospace", fontweight="bold", va="bottom")
    ax.axhline(y=-0.7, xmin=0.02, xmax=0.98,
               color="#1E3A5F", linewidth=0.6, linestyle="--", alpha=0.5)
    # Floor 2 label
    ax.text(-2.8, 6.5, "FLOOR 2", fontsize=7, color="#1E3A5F",
            fontfamily="monospace", fontweight="bold", va="bottom")
    ax.axhline(y=6.7, xmin=0.02, xmax=0.98,
               color="#1E3A5F", linewidth=0.6, linestyle="--", alpha=0.5)
    # Stairwell band
    ax.axhspan(2.3, 3.7, alpha=0.04, color="#6366F1")
    ax.text(4.0, 3.05, "STAIRWELL / C5", fontsize=6, color="#6366F1",
            ha="center", va="bottom", fontfamily="monospace", alpha=0.8)

    # ── Build networkx graph for edge drawing ──
    G = nx.DiGraph()
    graph_state = simulator.graph if initialized else EvacuationSimulator().graph
    for node in graph_state.nodes:
        G.add_node(node)
    for u, nbrs in graph_state.edges.items():
        for v in nbrs:
            G.add_edge(u, v)

    active_path  = (step_data["active_path"] if step_data else []) if initialized else []
    path_edges   = set()
    if active_path and len(active_path) > 1:
        for i in range(len(active_path) - 1):
            path_edges.add((active_path[i], active_path[i + 1]))
            path_edges.add((active_path[i + 1], active_path[i]))

    # ── Draw edges ──
    for u, v in G.edges():
        if u not in MAP_POS or v not in MAP_POS:
            continue
        x1, y1 = MAP_POS[u]
        x2, y2 = MAP_POS[v]
        if (u, v) in path_edges:
            # Active route – bright green
            ax.plot([x1, x2], [y1, y2], color="#22C55E", linewidth=3.5,
                    zorder=2, solid_capstyle="round",
                    path_effects=[pe.SimpleLineShadow(shadow_color="#22C55E", alpha=0.25, offset=(0, 0)),
                                  pe.Normal()])
        elif initialized and (graph_state.nodes.get(u, {}).get("blocked") or
                               graph_state.nodes.get(v, {}).get("blocked")):
            # Blocked edge – dashed crimson
            ax.plot([x1, x2], [y1, y2], color="#7F1D1D", linewidth=1.2,
                    linestyle="--", zorder=1, alpha=0.7)
        else:
            ax.plot([x1, x2], [y1, y2], color="#1E293B", linewidth=1.3,
                    zorder=1, alpha=0.9)

    # ── Draw nodes ──
    agent_pos = simulator.agent_position if initialized else None

    for node, (nx_, ny_) in MAP_POS.items():
        if not initialized:
            nd = {"type": "room" if "R" in node else ("corridor" if "C" in node else "exit"),
                  "hazard_score": 0.0, "crowd_density": 0.0, "blocked": False}
        else:
            nd = graph_state.nodes.get(node, {})

        ntype = nd.get("type", "room")
        hazard   = nd.get("hazard_score", 0.0)
        crowd    = nd.get("crowd_density", 0.0)
        blocked  = nd.get("blocked", False)
        is_agent = (node == agent_pos)
        is_path  = (node in active_path) if active_path else False

        # ── Glow layers ──
        if is_agent:
            for r, a in [(0.38, 0.08), (0.26, 0.14), (0.18, 0.20)]:
                ax.scatter(nx_, ny_, s=(r * 800) ** 1.1, color="#3B82F6",
                           alpha=a, zorder=3, linewidths=0)
        elif blocked:
            for r, a in [(0.32, 0.10), (0.22, 0.18)]:
                ax.scatter(nx_, ny_, s=(r * 800) ** 1.1, color="#EF4444",
                           alpha=a, zorder=3, linewidths=0)
        elif hazard > 0.5:
            for r, a in [(0.30, 0.12), (0.20, 0.20)]:
                ax.scatter(nx_, ny_, s=(r * 800) ** 1.1, color="#EF4444",
                           alpha=a * hazard, zorder=3, linewidths=0)
        elif hazard > 0.0:
            ax.scatter(nx_, ny_, s=15000 * hazard, color="#F97316",
                       alpha=0.18, zorder=3, linewidths=0)
        elif crowd > 0.5:
            ax.scatter(nx_, ny_, s=12000 * crowd, color="#F59E0B",
                       alpha=0.22, zorder=3, linewidths=0)
        elif is_path and ntype == "exit":
            ax.scatter(nx_, ny_, s=8000, color="#22C55E", alpha=0.15, zorder=3, linewidths=0)

        # ── Node body ──
        if ntype == "exit":
            base_size = 320
            if blocked:
                fc = "#3B0000"
                ec = "#EF4444"
            elif is_path:
                fc = "#052E16"
                ec = "#22C55E"
            else:
                fc = "#052E16"
                ec = "#166534"
        elif ntype == "corridor":
            base_size = 220
            fc = "#0F1A2B"
            ec = "#334155"
        else:  # room
            base_size = 280
            if is_agent:
                fc = "#1E3A6E"
                ec = "#3B82F6"
            elif blocked:
                fc = "#3B0000"
                ec = "#EF4444"
            elif hazard > 0.5:
                fc = "#3B0A00"
                ec = "#EF4444"
            elif hazard > 0.0:
                fc = "#2D1B00"
                ec = "#F97316"
            elif crowd > 0.5:
                fc = "#2D2000"
                ec = "#F59E0B"
            elif is_path:
                fc = "#0A2A1A"
                ec = "#22C55E"
            else:
                fc = "#0F1E33"
                ec = "#1E3A5F"

        ax.scatter(nx_, ny_, s=base_size, color=fc, edgecolors=ec,
                   linewidths=1.8, zorder=5, marker="s" if ntype == "exit" else "o")

        # ── Blocked X marker ──
        if blocked:
            ax.text(nx_, ny_, "✕", fontsize=9, ha="center", va="center",
                    color="#EF4444", fontweight="bold", zorder=7)
        elif is_agent:
            ax.text(nx_, ny_, "●", fontsize=7, ha="center", va="center",
                    color="#3B82F6", zorder=7)

        # ── Node label ──
        label_color = "#F1F5F9" if is_agent else \
                      "#EF4444" if blocked else \
                      "#22C55E" if ntype == "exit" else \
                      "#94A3B8" if ntype == "corridor" else \
                      "#CBD5E1"
        offset_y = -0.45 if ntype != "exit" else 0.50
        ax.text(nx_, ny_ + offset_y, node, fontsize=7.5, ha="center", va="center",
                color=label_color, fontweight="bold",
                fontfamily="monospace", zorder=8)

    # ── Embedded legend ──
    legend_items = [
        (mpatches.Patch(facecolor="#1E3A6E", edgecolor="#3B82F6", label="Agent")),
        (mpatches.Patch(facecolor="#052E16", edgecolor="#22C55E", label="Safe Exit")),
        (mpatches.Patch(facecolor="#3B0000", edgecolor="#EF4444", label="Blocked")),
        (mpatches.Patch(facecolor="#3B0A00", edgecolor="#EF4444", label="Fire Hazard")),
        (mpatches.Patch(facecolor="#2D1B00", edgecolor="#F97316", label="Gas Hazard")),
        (mpatches.Patch(facecolor="#2D2000", edgecolor="#F59E0B", label="Crowd Surge")),
    ]
    leg = ax.legend(handles=legend_items, loc="lower center",
                    ncol=6, frameon=True, fancybox=True,
                    framealpha=0.85, edgecolor="#1E293B",
                    fontsize=6.5, labelcolor="#94A3B8",
                    facecolor="#0F172A",
                    bbox_to_anchor=(0.5, -0.22))

    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────
#  COMMAND HEADER
# ─────────────────────────────────────────────
initialized = st.session_state.initialized
simulator   = st.session_state.simulator
step_data   = st.session_state.step_result

inc_type, inc_sev, inc_affected, inc_action = (
    simulator.current_incident if initialized
    else ("NONE", "GREEN", [], "System ready")
)

sev_cls    = SEVERITY_COLOR.get(inc_sev, "green")
sev_emoji  = SEVERITY_EMOJI.get(inc_sev, "🟢")
inc_emoji  = INCIDENT_EMOJI.get(inc_type, "✅")
agent_node = simulator.agent_position if initialized else "—"
algo_label = st.session_state.get("active_algo", "—")
step_num   = simulator.step_index if initialized else 0

st.markdown(f"""
<div class="cmd-header">
  <div class="cmd-logo">
    <div class="cmd-logo-icon">🚨</div>
    <div>
      <div class="cmd-logo-text">Emergency Command Center</div>
      <div class="cmd-logo-sub">Hybrid Incident Response Agent · Real-time Evacuation Simulation</div>
    </div>
  </div>
  <div class="metric-strip">
    <div class="pill pill-{sev_cls}">
      <div class="pill-label">Incident</div>
      <div class="pill-value">{inc_emoji} {inc_type if inc_type != 'NONE' else 'NORMAL'}</div>
    </div>
    <div class="pill pill-{sev_cls}">
      <div class="pill-label">Severity</div>
      <div class="pill-value">{sev_emoji} {inc_sev}</div>
    </div>
    <div class="pill pill-blue">
      <div class="pill-label">Algorithm</div>
      <div class="pill-value">⚡ {algo_label if algo_label != '—' else 'Not Set'}</div>
    </div>
    <div class="pill pill-blue">
      <div class="pill-label">Agent Position</div>
      <div class="pill-value">📍 {agent_node}</div>
    </div>
    <div class="pill">
      <div class="pill-label">Simulation Step</div>
      <div class="pill-value">{"🔴 LIVE" if initialized and not simulator.finished else ("✅ DONE" if initialized else "⏸ IDLE")} · {step_num}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR – CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem; text-align:center;">
      <div style="font-size:0.65rem; font-weight:700; color:#475569; letter-spacing:0.15em; text-transform:uppercase;">
        ⚙️ Simulation Configuration
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Algorithm picker ──
    st.markdown('<div class="sidebar-section-title">ROUTING ALGORITHM</div>', unsafe_allow_html=True)
    algo_choice = st.selectbox("Active Algorithm", ["A*", "BFS", "DFS"], label_visibility="collapsed")

    # ── Start room ──
    st.markdown('<div class="sidebar-section-title">AGENT START LOCATION</div>', unsafe_allow_html=True)
    start_room = st.selectbox("Starting Room", [f"R{i}" for i in range(1, 11)], label_visibility="collapsed")

    # ── Scenario presets ──
    st.markdown('<div class="sidebar-section-title">INCIDENT SCENARIO PRESET</div>', unsafe_allow_html=True)

    PRESETS = [
        ("Normal Scenario",            "🟢", "All systems operational"),
        ("Fire in East Wing",          "🔥", "R5 / EXIT-B affected"),
        ("Gas Leak in West Wing",      "☣️",  "C1 & C3 at risk"),
        ("Blocked Corridor/Exits",     "🚧", "EXIT-B obstructed"),
        ("High Occupancy Crowd Surge", "👥", "C2 & C4 congested"),
    ]

    preset_names = [p[0] for p in PRESETS]
    current_preset = st.session_state.incident_preset

    selected_preset_idx = st.radio(
        "Scenario",
        options=range(len(PRESETS)),
        format_func=lambda i: f"{PRESETS[i][1]}  {PRESETS[i][0]}",
        index=preset_names.index(current_preset) if current_preset in preset_names else 0,
        label_visibility="collapsed",
    )
    incident_selection = PRESETS[selected_preset_idx][0]
    st.session_state.incident_preset = incident_selection

    # ── Sensor defaults from preset ──
    temp_default  = 22.0;  smoke_default = 0.05; gas_default = 15.0
    blocked_exits_default = []; high_occ = {}

    if incident_selection == "Fire in East Wing":
        temp_default = 72.0; smoke_default = 0.88
    elif incident_selection == "Gas Leak in West Wing":
        gas_default = 480.0
    elif incident_selection == "Blocked Corridor/Exits":
        blocked_exits_default = ["EXIT-B"]
    elif incident_selection == "High Occupancy Crowd Surge":
        high_occ = {"C2": 95.0, "C4": 88.0}

    # ── Fine sensor controls ──
    st.markdown('<div class="sidebar-section-title">SENSOR OVERRIDE</div>', unsafe_allow_html=True)

    with st.expander("🔥 Fire Sensors"):
        temperature = st.slider("Temperature (°C)", 15.0, 100.0, float(temp_default), 1.0)
        smoke       = st.slider("Smoke Density",     0.0,   1.0, float(smoke_default), 0.05)

    with st.expander("☣️ Gas Sensor"):
        gas_ppm = st.slider("Gas Concentration (PPM)", 10.0, 600.0, float(gas_default), 10.0)

    with st.expander("🚧 Exit Blockages"):
        blocked_exits = st.multiselect(
            "Blocked Exits",
            ["EXIT-A", "EXIT-B", "EXIT-C", "EXIT-D"],
            default=blocked_exits_default,
        )

    with st.expander("👥 Corridor Occupancy"):
        corridor_occupancies = {}
        for c in ["C1", "C2", "C3", "C4", "C5"]:
            corridor_occupancies[c] = st.slider(
                f"{c} (%)", 0.0, 100.0, float(high_occ.get(c, 25.0)), 5.0
            )

    sensors = build_sensors(temperature, smoke, gas_ppm, blocked_exits, corridor_occupancies)

    # ── Action buttons ──
    st.markdown('<div class="sidebar-section-title">SIMULATION CONTROL</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    run_btn  = col_b1.button("▶ Start / Reset", use_container_width=True,
                              type="primary")
    step_btn = col_b2.button("Next Step ➡", use_container_width=True,
                              disabled=not initialized or simulator.finished)

    # ── Handle button clicks ──
    if run_btn:
        st.session_state.simulator = EvacuationSimulator()
        sim2 = st.session_state.simulator
        sim2.initialize_simulation(start_room, sensors)
        st.session_state.initialized = True
        st.session_state.active_algo = algo_choice
        st.session_state.timeline    = []
        step_res = sim2.run_step(algo_choice, sensors)
        st.session_state.step_result = step_res
        # Seed timeline
        add_event(f"Simulation started at {start_room}", "move")
        inc2 = sim2.current_incident
        if inc2[0] != "NONE":
            kind2 = {"FIRE": "fire", "GAS_LEAK": "gas",
                     "BLOCKED_EXIT": "block", "CROWD_SURGE": "crowd"}.get(inc2[0], "move")
            add_event(f"{inc2[0].replace('_', ' ').title()} detected · {inc2[1]}", kind2)
        for n in inc2[2]:
            add_event(f"Node {n} affected", "block")
        if step_res and step_res["active_path"]:
            add_event(f"Route computed → {step_res['active_path'][-1]}", "route")
        st.rerun()

    if step_btn and initialized and not simulator.finished:
        step_res2 = simulator.run_step(algo_choice, sensors)
        st.session_state.step_result = step_res2
        st.session_state.active_algo = algo_choice
        inc3 = simulator.current_incident
        if inc3[0] != "NONE":
            kind3 = {"FIRE": "fire", "GAS_LEAK": "gas",
                     "BLOCKED_EXIT": "block", "CROWD_SURGE": "crowd"}.get(inc3[0], "move")
            add_event(f"{inc3[0].replace('_', ' ').title()} active · Severity {inc3[1]}", kind3)
        add_event(f"Agent moved to {simulator.agent_position}", "move")
        if step_res2 and step_res2["active_path"]:
            add_event(f"Route → {step_res2['active_path'][-1]}", "route")
        if simulator.finished:
            if simulator.agent_position in simulator.graph.get_exits():
                add_event(f"✅ Evacuated via {simulator.agent_position}", "success")
            else:
                add_event("❌ Evacuation failed — all exits blocked", "block")
        st.rerun()


# ─────────────────────────────────────────────
#  REFRESH LIVE STATE (after possible rerun)
# ─────────────────────────────────────────────
initialized = st.session_state.initialized
simulator   = st.session_state.simulator
step_data   = st.session_state.step_result

inc_type, inc_sev, inc_affected, inc_action = (
    simulator.current_incident if initialized
    else ("NONE", "GREEN", [], "System ready")
)
algo_label = st.session_state.get("active_algo", algo_choice)


# ─────────────────────────────────────────────
#  MAIN 3-COLUMN LAYOUT
# ─────────────────────────────────────────────
left_col, map_col, right_col = st.columns([1, 3.2, 1.1])

# ───────────── LEFT: Route & Sim Info ──────────
with left_col:
    # Route summary card
    st.markdown("""
    <div class="panel-card">
      <div class="panel-card-title">🗺 ROUTE SUMMARY</div>
    """, unsafe_allow_html=True)

    if initialized and step_data:
        active_path = step_data.get("active_path", [])
        if active_path and len(active_path) > 1:
            target = active_path[-1]
            hops   = len(active_path) - 1
            cost   = get_path_cost(simulator.graph, active_path)
            path_str = " → ".join(active_path)
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-row-label">Recommended Exit</span>
              <span class="stat-row-value" style="color:#22C55E">{target}</span>
            </div>
            <div class="stat-row">
              <span class="stat-row-label">Remaining Hops</span>
              <span class="stat-row-value">{hops}</span>
            </div>
            <div class="stat-row">
              <span class="stat-row-label">Dynamic Cost</span>
              <span class="stat-row-value">{cost:.2f}</span>
            </div>
            <div style="margin-top:0.6rem; font-size:0.65rem; color:#64748B; margin-bottom:3px;">ROUTE PATH</div>
            <div class="route-path">{path_str}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="route-path nopath">No path available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569; font-size:0.78rem;">Initialize simulation to compute route.</div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Simulation state card
    st.markdown("""
    <div class="panel-card">
      <div class="panel-card-title">⚡ SIMULATION STATE</div>
    """, unsafe_allow_html=True)

    finished_str = "COMPLETE" if (initialized and simulator.finished) else \
                   "RUNNING"  if initialized else "IDLE"
    fin_color    = "#22C55E" if (initialized and simulator.finished and
                                  simulator.agent_position in simulator.graph.get_exits()) else \
                   "#EF4444" if (initialized and simulator.finished) else \
                   "#3B82F6" if initialized else "#475569"

    blocked_count = sum(1 for nd in (simulator.graph.nodes.values() if initialized else [])
                        if nd.get("blocked"))
    hazard_count  = sum(1 for nd in (simulator.graph.nodes.values() if initialized else [])
                        if nd.get("hazard_score", 0) > 0.0)

    st.markdown(f"""
    <div class="stat-row">
      <span class="stat-row-label">Status</span>
      <span class="stat-row-value" style="color:{fin_color}">{finished_str}</span>
    </div>
    <div class="stat-row">
      <span class="stat-row-label">Current Step</span>
      <span class="stat-row-value">{step_num}</span>
    </div>
    <div class="stat-row">
      <span class="stat-row-label">Nodes Blocked</span>
      <span class="stat-row-value" style="color:{'#EF4444' if blocked_count else '#22C55E'}">{blocked_count}</span>
    </div>
    <div class="stat-row">
      <span class="stat-row-label">Hazard Nodes</span>
      <span class="stat-row-value" style="color:{'#F97316' if hazard_count else '#22C55E'}">{hazard_count}</span>
    </div>
    <div class="stat-row">
      <span class="stat-row-label">Path History</span>
      <span class="stat-row-value">{len(simulator.history_paths) if initialized else 0} nodes</span>
    </div>
    """, unsafe_allow_html=True)

    if initialized and simulator.finished:
        if simulator.agent_position in simulator.graph.get_exits():
            st.markdown(f'<div class="status-banner success">🎉 Evacuated via {simulator.agent_position}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-banner failed">❌ All exits unreachable</div>',
                        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Incident info card
    st.markdown("""
    <div class="panel-card">
      <div class="panel-card-title">🚨 ACTIVE INCIDENT</div>
    """, unsafe_allow_html=True)

    sev_color_map = {"GREEN": "#22C55E", "YELLOW": "#F59E0B", "ORANGE": "#F97316", "RED": "#EF4444"}
    sev_c = sev_color_map.get(inc_sev, "#22C55E")
    st.markdown(f"""
    <div class="stat-row">
      <span class="stat-row-label">Type</span>
      <span class="stat-row-value">{INCIDENT_EMOJI.get(inc_type, '')} {inc_type}</span>
    </div>
    <div class="stat-row">
      <span class="stat-row-label">Severity</span>
      <span class="stat-row-value" style="color:{sev_c}">{inc_sev}</span>
    </div>
    <div class="stat-row">
      <span class="stat-row-label">Affected</span>
      <span class="stat-row-value">{', '.join(inc_affected) if inc_affected else 'None'}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────── CENTER: Building Map ──────────
with map_col:
    st.markdown("""
    <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06);
                border-radius:14px; padding:1rem 1.25rem 0.5rem; margin-bottom:0.75rem;">
      <div style="font-size:0.65rem; font-weight:700; color:#475569; letter-spacing:0.15em;
                  text-transform:uppercase; margin-bottom:0.75rem;">
        🏢 BUILDING FLOORPLAN  ·  LIVE EVACUATION MAP
      </div>
    """, unsafe_allow_html=True)

    fig = draw_building_map(simulator, step_data, initialized)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if not initialized:
        st.markdown("""
        <div style="text-align:center; color:#334155; font-size:0.8rem; padding:0.5rem 0;">
          ← Configure scenario and press <strong style="color:#3B82F6">Start / Reset</strong> to begin simulation
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Event Timeline ──
    timeline = st.session_state.timeline
    if timeline:
        chips_html = ""
        for ts, text, kind in timeline:
            chips_html += f'<div class="evt-chip {kind}"><div class="evt-time">{ts}</div><div class="evt-text">{text}</div></div>'
        st.markdown(f"""
        <div class="timeline-outer">
          <div class="timeline-title">📡 EVENT TIMELINE</div>
          <div class="timeline-scroll">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="timeline-outer">
          <div class="timeline-title">📡 EVENT TIMELINE</div>
          <div style="color:#334155; font-size:0.75rem; padding:0.4rem 0;">
            Events will appear here as the simulation progresses…
          </div>
        </div>
        """, unsafe_allow_html=True)


# ────────────── RIGHT: Algorithm Cards ────────
with right_col:
    st.markdown("""
    <div class="panel-card">
      <div class="panel-card-title">🧠 ALGORITHM COMPARISON</div>
    """, unsafe_allow_html=True)

    ALGO_META = {
        "A*":  {"full": "A* Search",             "color": "#3B82F6", "opt": "OPTIMAL"},
        "BFS": {"full": "Breadth-First Search",  "color": "#8B5CF6", "opt": "HOP-OPT"},
        "DFS": {"full": "Depth-First Search",    "color": "#F59E0B", "opt": "NON-OPT"},
    }

    if initialized and step_data:
        alts  = step_data.get("alternatives", {})
        costs = {}
        for name, (path, _, _) in alts.items():
            if path:
                costs[name] = get_path_cost(simulator.graph, path)

        winner = min(costs, key=costs.get) if costs else None

        for name in ["A*", "BFS", "DFS"]:
            meta = ALGO_META[name]
            if name not in alts:
                continue
            path, explored, exec_time = alts[name]
            cost    = costs.get(name, None)
            hops    = len(path) - 1 if path else 0
            target  = path[-1] if path else "N/A"

            is_selected = (name == algo_label)
            is_winner   = (name == winner and not is_selected)

            card_class = "algo-card selected" if is_selected else \
                         "algo-card winner"   if is_winner   else "algo-card"

            badges = ""
            if is_selected:
                badges += '<span class="algo-badge badge-active">ACTIVE</span>'
            if name == winner:
                badges += '<span class="algo-badge badge-winner">★ BEST COST</span>'
            if not path:
                badges += '<span class="algo-badge badge-nopath">NO PATH</span>'

            cost_str = f"{cost:.2f}" if cost is not None else "—"
            time_str = f"{exec_time:.3f}ms"

            st.markdown(f"""
            <div class="{card_class}">
              <div class="algo-name">
                <span style="color:{meta['color']}">{name}</span>
                <span style="display:flex; gap:3px">{badges}</span>
              </div>
              <div style="font-size:0.65rem; color:#475569; margin-bottom:0.5rem;">
                {meta['full']}
              </div>
              <div class="algo-metrics">
                <div class="algo-metric-item">Target Exit<br><span>{target}</span></div>
                <div class="algo-metric-item">Hops<br><span>{hops}</span></div>
                <div class="algo-metric-item">Cost<br><span>{cost_str}</span></div>
                <div class="algo-metric-item">Explored<br><span>{explored}</span></div>
                <div class="algo-metric-item">Time<br><span>{time_str}</span></div>
                <div class="algo-metric-item">Optimality<br><span style="color:#94A3B8">{meta['opt']}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#334155; font-size:0.78rem; padding:0.5rem 0;">
          Algorithm metrics will populate after simulation starts.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Reasoning trace (collapsed)
    if initialized and simulator.traces:
        with st.expander("📋 Step-by-Step Trace Log", expanded=False):
            last_trace = simulator.traces[-1]
            sev_trace_colors = {
                "RED":    "#FCA5A5", "ORANGE": "#FED7AA",
                "YELLOW": "#FDE68A", "GREEN":  "#86EFAC"
            }
            tc = sev_trace_colors.get(inc_sev, "#86EFAC")
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                        color:{tc}; background:rgba(0,0,0,0.3); border-radius:6px;
                        padding:0.75rem; white-space:pre-wrap; max-height:320px; overflow-y:auto;">
{last_trace}
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BOTTOM: ALGORITHM COMPARISON CARDS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.65rem; font-weight:700; color:#475569; letter-spacing:0.15em;
            text-transform:uppercase; margin-bottom:1rem;">
  🎓 SEARCH ALGORITHM TRADE-OFF ANALYSIS
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

compare_cards = [
    (col_a, "A* Search", "#3B82F6", "OPTIMAL", "O(E log V)", "O(V)",
     "Computes the globally safest and shortest route by incorporating dynamic edge weights "
     "(distance + hazard × 3 + crowd × 2) into an admissible heuristic. Best choice for "
     "real emergencies where both physical distance and threat severity must be minimized."),
    (col_b, "Breadth-First Search", "#8B5CF6", "HOP-OPT", "O(V + E)", "O(V)",
     "Finds the path crossing the fewest intermediate nodes. Optimal only for hop count — "
     "completely ignores fire hazard scores, gas leak penalties, and crowd congestion. "
     "Suitable only when all corridors are threat-free and uniformly traversable."),
    (col_c, "Depth-First Search", "#F59E0B", "NON-OPT", "O(V + E)", "O(V)",
     "Explores deep branches and returns the first reachable exit without evaluating "
     "path costs. Highly volatile — may route the agent through active fire or "
     "gas zones. Useful only for basic connectivity checks, not safe evacuation."),
]

for col, title, color, opt_label, time_c, space_c, desc in compare_cards:
    with col:
        opt_bg  = {"OPTIMAL": "rgba(34,197,94,0.15)",  "HOP-OPT": "rgba(139,92,246,0.15)",
                   "NON-OPT": "rgba(245,158,11,0.15)"}[opt_label]
        opt_col = {"OPTIMAL": "#22C55E", "HOP-OPT": "#A78BFA", "NON-OPT": "#F59E0B"}[opt_label]
        st.markdown(f"""
        <div class="compare-card" style="border-color:rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.25);">
          <div class="compare-card-title">
            <span style="color:{color}; font-size:0.85rem; font-weight:700;">{title}</span>
            <span class="compare-opt-tag" style="background:{opt_bg}; color:{opt_col};">{opt_label}</span>
          </div>
          <div class="compare-complexity">
            Time: {time_c} &nbsp;&nbsp; Space: {space_c}
          </div>
          <div class="compare-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="margin:1.5rem 0 0.5rem; padding:1rem 1.25rem;
            background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.05);
            border-radius:10px; font-size:0.75rem; color:#64748B; line-height:1.8;">
  <strong style="color:#94A3B8;">Evacuation Context:</strong>
  In a real-world emergency response system, routing must simultaneously optimize for
  <em>physical safety</em> (avoiding hazard zones), <em>speed</em> (minimizing travel time),
  and <em>congestion</em> (bypassing overcrowded corridors). A* Search integrates all three
  into its cost function, guaranteeing the optimal safe evacuation route.
  BFS is appropriate only for uniform-hazard environments.
  DFS is unsuitable for human evacuation routing.
</div>
""", unsafe_allow_html=True)
