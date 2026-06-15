def evaluate_rules(sensors, building_graph):
    """
    Evaluates sensor inputs and applies rules to alter the graph.
    Sensors dict should contain:
    - 'temperature': float (degrees C)
    - 'smoke': float (0.0 to 1.0)
    - 'gas_ppm': float (ppm)
    - 'exit_status': dict (e.g. {'EXIT-A': 1, 'EXIT-B': 0})
    - 'corridor_occupancies': dict (e.g. {'C1': 85.0})
    - 'security_threat': bool
    - 'threat_zones': list of str
    
    Returns:
    - incident_type: str ("FIRE", "GAS_LEAK", "BLOCKED_EXIT", "CROWD_SURGE", "TERROR_ATTACK", "NONE")
    - severity: str ("RED", "ORANGE", "YELLOW", "GREEN")
    - affected_nodes: list of str
    - recommended_action: str
    """
    incident_type = "NONE"
    severity = "GREEN"
    affected_nodes = []
    actions = []

    # Reset any temporary dynamic modifications (simulator handles overall reset, but we apply updates)
    
    # 0. TERROR_ATTACK Rule: security_threat is True
    # Action: Establish security perimeter, block routes near incident, force evacuation away.
    if sensors.get("security_threat", False):
        incident_type = "TERROR_ATTACK"
        severity = "RED"
        threat_zones = sensors.get("threat_zones", ["C5", "R7", "R8"])
        for node in threat_zones:
            building_graph.block_node(node)
            affected_nodes.append(node)
            # Add hazard to adjacent nodes to simulate restricted perimeter
            for neighbor in building_graph.edges.get(node, []):
                building_graph.update_hazard(neighbor, 0.9)
                if neighbor not in affected_nodes:
                    affected_nodes.append(neighbor)
        
        actions.append(f"Security lockdown activated. Perimeter established at: {', '.join(threat_zones)}. Evacuate away from threat zones.")

    # 1. FIRE Rule: temp > 60 or smoke > 0.7
    temp = sensors.get("temperature", 20.0)
    smoke = sensors.get("smoke", 0.0)
    if temp > 60.0 or smoke > 0.7:
        if severity != "RED":
            incident_type = "FIRE"
            severity = "RED"
        blocked_exit = "EXIT-B"
        building_graph.block_node(blocked_exit)
        if blocked_exit not in affected_nodes:
            affected_nodes.append(blocked_exit)
        
        building_graph.update_hazard("R5", 0.9)
        building_graph.update_hazard("C2", 0.6)
        affected_nodes.extend([n for n in ["R5", "C2"] if n not in affected_nodes])
        
        actions.append(f"Fire isolated near {blocked_exit}. Hazard elevated at R5 and C2.")

    # 2. GAS_LEAK Rule: gas_ppm > 300
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
            if node not in affected_nodes:
                affected_nodes.append(node)
        actions.append(f"Gas leak detected. Ventilation corridors restricted: {', '.join(vent_nodes)}.")

    # 3. BLOCKED_EXIT Rule: exit_sensor == 0
    exit_status = sensors.get("exit_status", {})
    for exit_node, status in exit_status.items():
        if status == 0:
            building_graph.block_node(exit_node)
            if exit_node not in affected_nodes:
                affected_nodes.append(exit_node)
            if incident_type == "NONE":
                incident_type = "BLOCKED_EXIT"
                severity = "ORANGE"
            actions.append(f"Exit structure compromised: {exit_node}.")

    # 4. CROWD_SURGE Rule: occupancy > 80%
    corridor_occs = sensors.get("corridor_occupancies", {})
    for corridor, occupancy in corridor_occs.items():
        if occupancy > 80.0:
            building_graph.update_crowd_density(corridor, 1.0)
            if corridor not in affected_nodes:
                affected_nodes.append(corridor)
            if incident_type == "NONE":
                incident_type = "CROWD_SURGE"
                severity = "YELLOW"
            actions.append(f"High congestion reported at {corridor}. Dynamic traversal cost increased.")

    if not actions:
        recommended_action = "System operational. Standard evacuation routes active."
    else:
        recommended_action = " | ".join(actions)

    return incident_type, severity, list(set(affected_nodes)), recommended_action
