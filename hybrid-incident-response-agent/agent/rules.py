def evaluate_rules(sensors, building_graph):
    """
    Evaluates sensor inputs and applies rules to alter the graph.
    Sensors dict should contain:
    - 'temperature': float (degrees C)
    - 'smoke': float (0.0 to 1.0)
    - 'gas_ppm': float (ppm)
    - 'exit_status': dict (e.g. {'EXIT-A': 1, 'EXIT-B': 0})
    - 'corridor_occupancies': dict (e.g. {'C1': 85.0})
    
    Returns:
    - incident_type: str ("FIRE", "GAS_LEAK", "BLOCKED_EXIT", "CROWD_SURGE", "NONE")
    - severity: str ("RED", "ORANGE", "YELLOW", "GREEN")
    - affected_nodes: list of str
    - recommended_action: str
    """
    incident_type = "NONE"
    severity = "GREEN"
    affected_nodes = []
    actions = []

    # Reset any temporary dynamic modifications (simulator handles overall reset, but we apply updates)
    
    # 1. FIRE Rule: temp > 60 or smoke > 0.7
    # Action: block the nearest exit to the fire (we can specify EXIT-B as the default fire zone or search for the closest exit to R5/R10)
    temp = sensors.get("temperature", 20.0)
    smoke = sensors.get("smoke", 0.0)
    if temp > 60.0 or smoke > 0.7:
        incident_type = "FIRE"
        severity = "RED"
        # We block EXIT-B as it represents the main risk area in this scenario, 
        # or we block the exit closest to where the fire sensor is triggered (R5 is near EXIT-B).
        blocked_exit = "EXIT-B"
        building_graph.block_node(blocked_exit)
        affected_nodes.append(blocked_exit)
        
        # Also let's add hazard scores to neighboring nodes (R5 and C2)
        building_graph.update_hazard("R5", 0.9)
        building_graph.update_hazard("C2", 0.6)
        affected_nodes.extend(["R5", "C2"])
        
        actions.append(f"Block nearest exit: {blocked_exit}. Set hazard at R5 to 0.9 and C2 to 0.6.")

    # 2. GAS_LEAK Rule: gas_ppm > 300
    # Action: block ventilation corridors (C1 and C3) to prevent propagation, set alarm level ORANGE
    gas = sensors.get("gas_ppm", 0.0)
    if gas > 300.0:
        if incident_type == "NONE":
            incident_type = "GAS_LEAK"
            severity = "ORANGE"
        elif severity != "RED":
            severity = "ORANGE"
            
        vent_nodes = ["C1", "C3"]
        for node in vent_nodes:
            building_graph.block_node(node)
            affected_nodes.append(node)
        actions.append(f"Block ventilation corridors to contain leak: {', '.join(vent_nodes)}.")

    # 3. BLOCKED_EXIT Rule: exit_sensor == 0
    # Action: block specific exit, trigger reroute
    exit_status = sensors.get("exit_status", {})
    for exit_node, status in exit_status.items():
        if status == 0:
            building_graph.block_node(exit_node)
            if exit_node not in affected_nodes:
                affected_nodes.append(exit_node)
            if incident_type == "NONE":
                incident_type = "BLOCKED_EXIT"
                severity = "ORANGE"
            actions.append(f"Mark exit blocked by sensor: {exit_node}.")

    # 4. CROWD_SURGE Rule: occupancy > 80%
    # Action: increase edge weights for that corridor (crowd density = 1.0)
    corridor_occs = sensors.get("corridor_occupancies", {})
    for corridor, occupancy in corridor_occs.items():
        if occupancy > 80.0:
            building_graph.update_crowd_density(corridor, 1.0)
            if corridor not in affected_nodes:
                affected_nodes.append(corridor)
            if incident_type == "NONE":
                incident_type = "CROWD_SURGE"
                severity = "YELLOW"
            actions.append(f"Crowd surge at {corridor} (>80%). Increase dynamic traversal cost.")

    if not actions:
        recommended_action = "No incidents active. All paths safe."
    else:
        recommended_action = " | ".join(actions)

    return incident_type, severity, list(set(affected_nodes)), recommended_action
