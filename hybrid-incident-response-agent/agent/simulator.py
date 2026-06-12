from agent.graph import BuildingGraph
from agent.algorithms import astar, bfs, dfs
from agent.rules import evaluate_rules
from agent.reasoner import generate_step_trace, get_path_cost

class EvacuationSimulator:
    def __init__(self):
        self.graph = BuildingGraph()
        self.agent_position = "R1"
        self.step_index = 0
        self.traces = []
        self.finished = False
        self.history_paths = [] # Track path taken by agent
        self.current_incident = ("NONE", "GREEN", [], "No incident active")

    def initialize_simulation(self, start_node, sensors):
        """Reset simulator state and set start node."""
        self.graph.reset_graph()
        self.agent_position = start_node
        self.step_index = 0
        self.traces = []
        self.finished = False
        self.history_paths = [start_node]
        
        # Apply rules initially
        self.current_incident = evaluate_rules(sensors, self.graph)

    def run_step(self, selected_algorithm, sensors):
        """
        Executes a single step of the simulation.
        1. Evaluate rules based on current sensors.
        2. Run A*, BFS, and DFS from agent's current position to compare.
        3. Step the agent to the next node according to the selected algorithm.
        4. Generate step trace.
        """
        if self.finished:
            return
            
        self.step_index += 1
        
        # 1. Reset dynamic flags, then re-evaluate rules based on new sensor inputs
        self.graph.reset_graph()
        incident_type, severity, affected_nodes, recommended_action = evaluate_rules(sensors, self.graph)
        self.current_incident = (incident_type, severity, affected_nodes, recommended_action)
        
        # Determine available goal exits (non-blocked)
        active_exits = self.graph.get_active_exits()
        
        # 2. Run all three algorithms from agent position for comparison
        # A* Path
        astar_path, astar_explored, astar_time = astar(self.graph, self.agent_position, active_exits)
        # BFS Path
        bfs_path, bfs_explored, bfs_time = bfs(self.graph, self.agent_position, active_exits)
        # DFS Path
        dfs_path, dfs_explored, dfs_time = dfs(self.graph, self.agent_position, active_exits)
        
        # Select active path and statistics based on user configuration
        if selected_algorithm == "A*":
            active_path = astar_path
            stats = {"explored": astar_explored, "time": astar_time, "cost": get_path_cost(self.graph, astar_path)}
        elif selected_algorithm == "BFS":
            active_path = bfs_path
            stats = {"explored": bfs_explored, "time": bfs_time, "cost": get_path_cost(self.graph, bfs_path)}
        else: # DFS
            active_path = dfs_path
            stats = {"explored": dfs_explored, "time": dfs_time, "cost": get_path_cost(self.graph, dfs_path)}
            
        # Compile alternatives for reasoner comparison block
        alternatives = {
            "A*": (astar_path, astar_explored, astar_time),
            "BFS": (bfs_path, bfs_explored, bfs_time),
            "DFS": (dfs_path, dfs_explored, dfs_time)
        }
        
        # Generate trace
        trace = generate_step_trace(
            step_index=self.step_index,
            agent_loc=self.agent_position,
            incident_type=incident_type,
            severity=severity,
            blocked_nodes=affected_nodes,
            algo_name=selected_algorithm,
            computed_path=active_path,
            building_graph=self.graph,
            alternatives=alternatives
        )
        self.traces.append(trace)
        
        # Move the agent to the next node in the active path
        if active_path and len(active_path) > 1:
            self.agent_position = active_path[1]
            self.history_paths.append(self.agent_position)
        else:
            # Cannot move (no path or already at exit)
            self.finished = True
            
        # Check if agent has reached an exit
        if self.agent_position in self.graph.get_exits():
            self.finished = True
            
        return {
            "agent_position": self.agent_position,
            "active_path": active_path,
            "finished": self.finished,
            "trace": trace,
            "stats": stats,
            "alternatives": alternatives
        }
