import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import time

from agent.simulator import EvacuationSimulator
from agent.graph import NODE_COORDS
from agent.reasoner import get_path_cost

# Page configuration
st.set_page_config(
    page_title="Hybrid Incident Response Agent",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium dark theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-title {
        font-size: 1.1rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Panel Cards */
    .card-panel {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .panel-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Reasoning traces styles */
    .trace-container {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        max-height: 480px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }
    
    .trace-RED {
        background-color: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        color: #fca5a5;
    }
    
    .trace-ORANGE {
        background-color: rgba(249, 115, 22, 0.12);
        border-left: 4px solid #f97316;
        color: #fdbb2d;
    }
    
    .trace-YELLOW {
        background-color: rgba(234, 179, 8, 0.1);
        border-left: 4px solid #eab308;
        color: #fef08a;
    }
    
    .trace-GREEN {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        color: #a7f3d0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "simulator" not in st.session_state:
    st.session_state.simulator = EvacuationSimulator()
    st.session_state.initialized = False
    st.session_state.selected_start = "R3"
    st.session_state.selected_algo = "A*"
    st.session_state.step_result = None

# Title Header
st.markdown("<div class='main-title'>🚨 Hybrid Incident Response Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Intelligent multi-floor evacuation routing using Rule-Based Expert Systems & Dynamic Search Algorithms</div>", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### ⚙️ Simulation Configuration")

# 1. Routing Settings
algo_choice = st.sidebar.selectbox("Evacuation Search Algorithm", ["A*", "BFS", "DFS"])
start_room = st.sidebar.selectbox("Starting Room Location", [f"R{i}" for i in range(1, 11)])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Real-Time Sensors")

incident_selection = st.sidebar.selectbox(
    "Active Sensor Incident Preset",
    ["Normal Scenario", "Fire in East Wing", "Gas Leak in West Wing", "Blocked Corridor/Exits", "High Occupancy Crowd Surge"]
)

# Preset configuration values
temp_default = 22.0
smoke_default = 0.05
gas_default = 15.0
blocked_exits_default = []
high_corridor_occupancies = {}

if incident_selection == "Fire in East Wing":
    temp_default = 72.0
    smoke_default = 0.88
    # Fire near EXIT-B / R5
elif incident_selection == "Gas Leak in West Wing":
    gas_default = 480.0
    # Blocks C1 & C3
elif incident_selection == "Blocked Corridor/Exits":
    blocked_exits_default = ["EXIT-B"]
elif incident_selection == "High Occupancy Crowd Surge":
    high_corridor_occupancies = {"C2": 95.0, "C4": 88.0}

# Sensor detail inputs (sidebar)
st.sidebar.markdown("**Fire Sensors**")
temperature = st.sidebar.slider("Ambient Temperature (°C)", 15.0, 100.0, float(temp_default), step=1.0)
smoke = st.sidebar.slider("Smoke Density (obscuration)", 0.0, 1.0, float(smoke_default), step=0.05)

st.sidebar.markdown("**Gas Sensors**")
gas_ppm = st.sidebar.slider("Gas Concentration (PPM)", 10.0, 600.0, float(gas_default), step=10.0)

st.sidebar.markdown("**Manual Exit Blockages**")
blocked_exits = st.sidebar.multiselect(
    "Mark Exits as Blocked",
    ["EXIT-A", "EXIT-B", "EXIT-C", "EXIT-D"],
    default=blocked_exits_default
)

st.sidebar.markdown("**Corridor Occupancy**")
corridors = ["C1", "C2", "C3", "C4", "C5"]
corridor_occupancies = {}
for c in corridors:
    default_occ = high_corridor_occupancies.get(c, 25.0)
    corridor_occupancies[c] = st.sidebar.slider(f"{c} Occupancy (%)", 0.0, 100.0, float(default_occ), step=5.0)

# Build sensor data dictionary
sensors = {
    "temperature": temperature,
    "smoke": smoke,
    "gas_ppm": gas_ppm,
    "exit_status": {ex: (0 if ex in blocked_exits else 1) for ex in ["EXIT-A", "EXIT-B", "EXIT-C", "EXIT-D"]},
    "corridor_occupancies": corridor_occupancies
}

st.sidebar.markdown("---")

# Simulation execution buttons
col_sim1, col_sim2 = st.sidebar.columns(2)
run_btn = col_sim1.button("Start / Reset", use_container_width=True)
step_btn = col_sim2.button("Next Step ➡️", use_container_width=True, disabled=not st.session_state.initialized or (st.session_state.simulator.finished))

# Handle Button Clicks
if run_btn:
    st.session_state.simulator.initialize_simulation(start_room, sensors)
    st.session_state.initialized = True
    st.session_state.selected_start = start_room
    st.session_state.selected_algo = algo_choice
    
    # Run the initial path evaluation
    st.session_state.step_result = st.session_state.simulator.run_step(algo_choice, sensors)

if step_btn and st.session_state.initialized and not st.session_state.simulator.finished:
    st.session_state.step_result = st.session_state.simulator.run_step(algo_choice, sensors)

# ----------------- MAIN PANEL LAYOUT -----------------
col1, col2, col3 = st.columns([1.1, 1.1, 0.8])

# Render default placeholders if not initialized
if not st.session_state.initialized:
    st.info("👈 Set your parameters and click **Start / Reset** in the sidebar to initialize the evacuation simulation.")
    
    # Static rendering of building map
    with col1:
        st.markdown("<div class='card-panel'><div class='panel-header'>🗺️ Building Layout Map</div>", unsafe_allow_html=True)
        # Dummy draw
        sim = EvacuationSimulator()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        G = nx.DiGraph()
        for node, neighbors in sim.graph.edges.items():
            for n in neighbors:
                G.add_edge(node, n)
        pos = NODE_COORDS
        nx.draw(G, pos, with_labels=True, node_color="#e2e8f0", edge_color="#cbd5e1", node_size=500, font_size=8, font_weight="bold", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card-panel'><div class='panel-header'>🧠 Reasoning Trace</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='trace-container trace-GREEN'>"
            "Waiting for simulation to start...\n"
            "════════════════════════════════════\n"
            "Systems operational.\n"
            "All sensors polling.\n"
            "Ready for start command.\n"
            "════════════════════════════════════"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='card-panel'><div class='panel-header'>📊 Algorithm Statistics</div>", unsafe_allow_html=True)
        st.markdown("No active data. Initialize simulation to run calculations.")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Simulator initialized - Active rendering
    simulator = st.session_state.simulator
    step_data = st.session_state.step_result
    
    # Extract incident details
    inc_type, inc_severity, inc_affected, inc_action = simulator.current_incident

    # LEFT COLUMN: Building Map
    with col1:
        st.markdown("<div class='card-panel'><div class='panel-header'>🗺️ Building Layout Map</div>", unsafe_allow_html=True)
        
        # Plotting the Graph
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        
        G = nx.DiGraph()
        for node in simulator.graph.nodes:
            G.add_node(node)
        for u, neighbors in simulator.graph.edges.items():
            for v in neighbors:
                G.add_edge(u, v)
                
        pos = NODE_COORDS
        
        # Determine node colors
        node_colors = []
        for node in G.nodes:
            node_data = simulator.graph.nodes[node]
            if node == simulator.agent_position:
                node_colors.append("#3b82f6")  # Blue: Agent
            elif node_data["blocked"]:
                node_colors.append("#ef4444")  # Red: Blocked
            elif node_data["hazard_score"] > 0.0:
                node_colors.append("#eab308")  # Yellow: Hazard
            elif node_data["type"] == "exit":
                node_colors.append("#10b981")  # Green: Exit
            elif node_data["type"] == "corridor":
                node_colors.append("#64748b")  # Dark Grey: Corridor
            else:
                node_colors.append("#94a3b8")  # Light Grey: Room

        # Active path highlighting
        edge_colors = []
        edge_widths = []
        active_path = step_data["active_path"] if step_data else []
        path_edges = set()
        if active_path and len(active_path) > 1:
            for i in range(len(active_path) - 1):
                path_edges.add((active_path[i], active_path[i+1]))
                path_edges.add((active_path[i+1], active_path[i])) # handle both directions for visuals

        for u, v in G.edges:
            if (u, v) in path_edges:
                edge_colors.append("#10b981")  # Green route
                edge_widths.append(3.5)
            else:
                if simulator.graph.nodes[u]["blocked"] or simulator.graph.nodes[v]["blocked"]:
                    edge_colors.append("#7f1d1d")
                    edge_widths.append(0.8)
                else:
                    edge_colors.append("#475569")
                    edge_widths.append(1.2)

        # Draw nodes & labels
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=12, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", font_color="#ffffff", ax=ax)
        
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Legend key
        st.markdown(
            "<div style='display:flex; justify-content:space-around; font-size:0.8rem; margin-top:0.5rem;'>"
            "<span>🔵 Agent</span>"
            "<span>🟢 Safe Exit</span>"
            "<span>🔴 Blocked</span>"
            "<span>🟡 Hazard</span>"
            "<span>⚪ Normal Room</span>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # MIDDLE COLUMN: Reasoning Trace
    with col2:
        st.markdown("<div class='card-panel'><div class='panel-header'>🧠 Step-by-Step Reasoning</div>", unsafe_allow_html=True)
        
        # Style reasoning trace container by active severity
        trace_class = f"trace-{inc_severity}"
        
        all_traces = "\n\n".join(simulator.traces)
        st.markdown(f"<div class='trace-container {trace_class}'>{all_traces}</div>", unsafe_allow_html=True)
        
        if simulator.finished:
            if simulator.agent_position in simulator.graph.get_exits():
                st.success(f"🎉 Evacuation Complete! Agent safely exited via **{simulator.agent_position}**.")
            else:
                st.error("❌ Evacuation Failed! All exits are unreachable or blocked.")
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN: Stats Panel
    with col3:
        st.markdown("<div class='card-panel'><div class='panel-header'>📊 Live Run Performance</div>", unsafe_allow_html=True)
        
        if step_data:
            stats = step_data["stats"]
            col_m1, col_m2 = st.columns(2)
            col_m1.markdown(f"<div class='metric-label'>Algorithm</div><div class='metric-value'>{algo_choice}</div>", unsafe_allow_html=True)
            col_m2.markdown(f"<div class='metric-label'>Dynamic Cost</div><div class='metric-value'>{stats['cost']:.2f}</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            
            col_m3, col_m4 = st.columns(2)
            col_m3.markdown(f"<div class='metric-label'>Explored Nodes</div><div class='metric-value'>{stats['explored']}</div>", unsafe_allow_html=True)
            col_m4.markdown(f"<div class='metric-label'>Exec Time</div><div class='metric-value'>{stats['time']:.3f} ms</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("##### 🔍 Path Comparisons (Current Step)")
            
            comparison_rows = []
            alts = step_data["alternatives"]
            for name, (path, explored, exec_time) in alts.items():
                cost = get_path_cost(simulator.graph, path) if path else float('inf')
                path_hops = len(path) - 1 if path else 0
                path_str = f"{path[0]}→...→{path[-1]}" if path else "No Path"
                comparison_rows.append({
                    "Algorithm": name,
                    "Target Exit": path[-1] if path else "N/A",
                    "Hops": path_hops,
                    "Dynamic Cost": round(cost, 2) if cost != float('inf') else "N/A",
                    "Explored": explored,
                    "Time (ms)": round(exec_time, 3)
                })
                
            df_comp = pd.DataFrame(comparison_rows)
            st.dataframe(df_comp.set_index("Algorithm"), use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- BOTTOM SECTION: ALGORITHMIC TRADE-OFF ANALYSIS -----------------
st.markdown("---")
st.markdown("### 🎓 Evacuation Search Algorithmic Trade-off Analysis")

trade_off_data = [
    {
        "Algorithm": "A* Search",
        "Time Complexity": "O(E log V)",
        "Space Complexity": "O(V)",
        "Optimality": "Guaranteed Optimal (Dynamic Cost)",
        "Evacuation Scenario Fit": "Excellent. Accounts for distance, gas hazard scores, and crowd densities using admissible heuristics to find the safest, fastest route."
    },
    {
        "Algorithm": "Breadth-First Search (BFS)",
        "Time Complexity": "O(V + E)",
        "Space Complexity": "O(V)",
        "Optimality": "Optimal for Hop Count only",
        "Evacuation Scenario Fit": "Moderate. Minimizes the number of corridors traversed (hops) but completely ignores fire/smoke hazard zones and crowd congestion weights."
    },
    {
        "Algorithm": "Depth-First Search (DFS)",
        "Time Complexity": "O(V + E)",
        "Space Complexity": "O(V)",
        "Optimality": "Non-optimal",
        "Evacuation Scenario Fit": "Poor. Explores paths deeply and returns the first exit found, which can lead the agent directly into high-threat hazard zones unnecessarily."
    }
]

df_trade_off = pd.DataFrame(trade_off_data)
st.table(df_trade_off.set_index("Algorithm"))

st.markdown("""
**Evacuation Context Summary:**
In a real-life emergency response system, paths must optimize for safety (minimum hazard) and speed (minimum congestion and distance) rather than just the number of corridors crossed. 
* **A* Search** integrates these dynamic weight variables directly into its heuristic and cost calculations, guaranteeing the safest route is picked.
* **BFS** is ideal if all corridors have uniform hazards and we only care about minimum transit nodes.
* **DFS** is highly volatile and dangerous for human evacuation routing since it lacks path-cost optimization.
""")
