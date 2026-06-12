def get_path_cost(building_graph, path):
    """Compute the total cost of a path by summing dynamic edge weights."""
    if not path or len(path) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        cost += building_graph.get_edge_weight(path[i], path[i + 1])
    return cost

def generate_step_trace(
    step_index,
    agent_loc,
    incident_type,
    severity,
    blocked_nodes,
    algo_name,
    computed_path,
    building_graph,
    alternatives
):
    """
    Generates a step-by-step reasoning trace block explaining the routing decision.
    
    Arguments:
    - step_index: Current iteration of the simulation (1-based)
    - agent_loc: Node name where the agent is currently located
    - incident_type: Active incident ("FIRE", "GAS_LEAK", etc.)
    - severity: Alarm level ("RED", "ORANGE", "YELLOW", "GREEN")
    - blocked_nodes: List of nodes marked blocked
    - algo_name: Currently selected algorithm ("A*", "BFS", "DFS")
    - computed_path: Path list returned by the active algorithm
    - building_graph: The current BuildingGraph state
    - alternatives: Dict containing paths/stats of other algorithms, e.g.:
                    {'BFS': (bfs_path, bfs_explored, bfs_time), 'DFS': (dfs_path, dfs_explored, dfs_time)}
    """
    trace = []
    trace.append("════════════════════════════════════")
    trace.append(f"[STEP {step_index}] Agent location: {agent_loc}")
    
    if incident_type != "NONE":
        trace.append(f"[INCIDENT] {incident_type} detected (severity: {severity})")
    else:
        trace.append("[INCIDENT] Normal state (severity: GREEN)")
        
    if blocked_nodes:
        trace.append(f"[BLOCKED] Nodes bypassed/removed from graph: {blocked_nodes}")
    else:
        trace.append("[BLOCKED] All exits and corridors open")
        
    goal_name = computed_path[-1] if computed_path else "NONE"
    trace.append(f"[RECOMPUTING] Running {algo_name} from {agent_loc} → nearest exit ({goal_name})")
    
    if computed_path:
        path_str = " → ".join(computed_path)
        trace.append(f"[RESULT] Selected Path: {path_str}")
        
        # Calculate details for WHY section
        total_cost = get_path_cost(building_graph, computed_path)
        
        # For the direct next hop
        next_hop = computed_path[1] if len(computed_path) > 1 else None
        if next_hop:
            v_data = building_graph.nodes[next_hop]
            dist = building_graph.get_edge_weight(agent_loc, next_hop) - (v_data["hazard_score"] * 3.0) - (v_data["crowd_density"] * 2.0)
            hazard = "HIGH" if v_data["hazard_score"] > 0.6 else "MED" if v_data["hazard_score"] > 0.3 else "LOW"
            crowd = "HIGH" if v_data["crowd_density"] > 0.6 else "MED" if v_data["crowd_density"] > 0.3 else "LOW"
            trace.append(
                f"[WHY] Goal Exit {goal_name} is optimal. Next hop {next_hop}: hazard={hazard}({v_data['hazard_score']}), "
                f"crowd={crowd}({v_data['crowd_density']}), edge_dist={dist:.2f} → step_cost={building_graph.get_edge_weight(agent_loc, next_hop):.2f} (Total Path Cost: {total_cost:.2f})"
            )
        else:
            trace.append(f"[WHY] Already at goal exit {goal_name}.")
    else:
        trace.append("[RESULT] NO PATH FOUND! All exits are blocked or unreachable.")
        trace.append("[WHY] The graph search exhausted all non-blocked routes and found no open path to an exit.")

    # Add comparison section
    comparison_lines = []
    for alt_name, (alt_path, _, _) in alternatives.items():
        if alt_name == algo_name:
            continue
        if alt_path:
            alt_cost = get_path_cost(building_graph, alt_path)
            alt_str = " → ".join(alt_path)
            comparison_lines.append(f"{alt_name} finds: {alt_path[0]} → ... → {alt_path[-1]} (cost={alt_cost:.2f}, hops={len(alt_path)-1})")
        else:
            comparison_lines.append(f"{alt_name} finds: NO PATH")
            
    if comparison_lines:
        trace.append(f"[COMPARISON] " + " | ".join(comparison_lines))
        
    next_step = computed_path[1] if (computed_path and len(computed_path) > 1) else "STAY"
    trace.append(f"[ACTION] Moving to {next_step}")
    trace.append("════════════════════════════════════")
    
    return "\n".join(trace)
