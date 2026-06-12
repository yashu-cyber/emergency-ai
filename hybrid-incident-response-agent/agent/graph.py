import math

# Node coordinates representing layout of the 2-floor building
NODE_COORDS = {
    # Floor 1 Rooms
    "R1": (0.0, 0.0),
    "R2": (2.0, 0.0),
    "R3": (4.0, 0.0),
    "R4": (6.0, 0.0),
    "R5": (8.0, 0.0),
    # Floor 1 Corridors
    "C1": (2.0, 2.0),
    "C2": (6.0, 2.0),
    # Connecting Stairwell
    "C5": (4.0, 3.0),
    # Floor 2 Rooms
    "R6": (0.0, 6.0),
    "R7": (2.0, 6.0),
    "R8": (4.0, 6.0),
    "R9": (6.0, 6.0),
    "R10": (8.0, 6.0),
    # Floor 2 Corridors
    "C3": (2.0, 4.0),
    "C4": (6.0, 4.0),
    # Exits
    "EXIT-A": (-2.0, 0.0),
    "EXIT-B": (10.0, 0.0),
    "EXIT-C": (-2.0, 6.0),
    "EXIT-D": (10.0, 6.0),
}

# Default connections (bidirectional list of edge tuples)
DEFAULT_EDGES = [
    # Floor 1 connections
    ("R1", "EXIT-A"),
    ("R1", "C1"),
    ("R2", "C1"),
    ("R3", "C1"),
    ("R3", "C5"),
    ("R4", "C2"),
    ("R4", "C5"),
    ("R5", "C2"),
    ("R5", "EXIT-B"),
    ("C1", "EXIT-A"),
    ("C1", "C5"),
    ("C2", "EXIT-B"),
    ("C2", "C5"),
    # Floor 2 connections
    ("R6", "EXIT-C"),
    ("R6", "C3"),
    ("R7", "C3"),
    ("R8", "C3"),
    ("R8", "C5"),
    ("R9", "C4"),
    ("R9", "C5"),
    ("R10", "C4"),
    ("R10", "EXIT-D"),
    ("C3", "EXIT-C"),
    ("C4", "EXIT-D"),
]

def calculate_distance(node1, node2):
    """Calculate Euclidean distance between two nodes."""
    x1, y1 = NODE_COORDS[node1]
    x2, y2 = NODE_COORDS[node2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class BuildingGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.reset_graph()

    def reset_graph(self):
        """Restore the graph to its default structure and clean state."""
        self.nodes = {}
        for node, coords in NODE_COORDS.items():
            # Classify node type
            if "R" in node:
                node_type = "room"
            elif "C" in node:
                node_type = "corridor"
            else:
                node_type = "exit"

            self.nodes[node] = {
                "coords": coords,
                "type": node_type,
                "hazard_score": 0.0,
                "crowd_density": 0.0,
                "blocked": False,
            }

        # Initialize adjacency list
        self.edges = {node: [] for node in NODE_COORDS}
        for u, v in DEFAULT_EDGES:
            # Bidirectional graph representation
            self.edges[u].append(v)
            self.edges[v].append(u)

    def block_node(self, node):
        """Mark a node as blocked so it cannot be traversed."""
        if node in self.nodes:
            self.nodes[node]["blocked"] = True

    def unblock_node(self, node):
        """Unblock a node."""
        if node in self.nodes:
            self.nodes[node]["blocked"] = False

    def update_hazard(self, node, score):
        """Update hazard score for a node (bound between 0.0 and 1.0)."""
        if node in self.nodes:
            self.nodes[node]["hazard_score"] = max(0.0, min(1.0, float(score)))

    def update_crowd_density(self, node, density):
        """Update crowd density for a node (bound between 0.0 and 1.0)."""
        if node in self.nodes:
            self.nodes[node]["crowd_density"] = max(0.0, min(1.0, float(density)))

    def get_neighbors(self, node):
        """Return accessible neighbors of a node (skipping blocked nodes)."""
        if node not in self.edges or self.nodes[node]["blocked"]:
            return []
        return [neighbor for neighbor in self.edges[node] if not self.nodes[neighbor]["blocked"]]

    def get_edge_weight(self, u, v):
        """Calculate weight of traversing from u to v.
        Weight = distance + hazard_score(v) * 3 + crowd_density(v) * 2
        """
        dist = calculate_distance(u, v)
        v_data = self.nodes[v]
        weight = dist + (v_data["hazard_score"] * 3.0) + (v_data["crowd_density"] * 2.0)
        return weight

    def get_exits(self):
        """Return list of all exits."""
        return [node for node, data in self.nodes.items() if data["type"] == "exit"]

    def get_active_exits(self):
        """Return exits that are currently not blocked."""
        return [node for node, data in self.nodes.items() if data["type"] == "exit" and not data["blocked"]]
