import heapq
import time
from collections import deque
from agent.graph import calculate_distance

def get_heuristic(node, goals):
    """Admissible heuristic: minimum Euclidean distance to any of the goal exits."""
    if not goals:
        return 0.0
    return min(calculate_distance(node, g) for g in goals)

def astar(building_graph, start, goals):
    """
    Custom A* search from start node to the closest goal exit.
    Returns: (path, nodes_explored, execution_time_ms)
    """
    start_time = time.perf_counter()
    
    if not goals or start in goals:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ([start] if start in goals else [], 0, elapsed)

    # Priority queue: (f_score, g_score, current_node, path)
    # Using a counter to avoid comparison on paths/nodes if f_scores are equal
    counter = 0
    pq = []
    
    h_start = get_heuristic(start, goals)
    heapq.heappush(pq, (h_start, 0.0, start, [start]))
    
    g_scores = {start: 0.0}
    explored_count = 0
    closed_set = set()

    while pq:
        f, g, current, path = heapq.heappop(pq)
        
        if current in closed_set:
            continue
        closed_set.add(current)
        explored_count += 1

        if current in goals:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return (path, explored_count, elapsed)

        for neighbor in building_graph.get_neighbors(current):
            weight = building_graph.get_edge_weight(current, neighbor)
            tentative_g = g + weight
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                h = get_heuristic(neighbor, goals)
                f_score = tentative_g + h
                heapq.heappush(pq, (f_score, tentative_g, neighbor, path + [neighbor]))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return ([], explored_count, elapsed)

def bfs(building_graph, start, goals):
    """
    Custom BFS search from start node to any of the goals.
    Returns: (path, nodes_explored, execution_time_ms)
    """
    start_time = time.perf_counter()
    
    if not goals or start in goals:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ([start] if start in goals else [], 0, elapsed)

    queue = deque([(start, [start])])
    visited = {start}
    explored_count = 0

    while queue:
        current, path = queue.popleft()
        explored_count += 1

        if current in goals:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return (path, explored_count, elapsed)

        for neighbor in building_graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return ([], explored_count, elapsed)

def dfs(building_graph, start, goals):
    """
    Custom DFS search from start node to any of the goals.
    Returns: (path, nodes_explored, execution_time_ms)
    """
    start_time = time.perf_counter()
    
    if not goals or start in goals:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ([start] if start in goals else [], 0, elapsed)

    stack = [(start, [start])]
    visited = set()
    explored_count = 0

    while stack:
        current, path = stack.pop()
        
        if current in visited:
            continue
        visited.add(current)
        explored_count += 1

        if current in goals:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return (path, explored_count, elapsed)

        # Iterating neighbors in reverse to match BFS expansion ordering preference
        for neighbor in reversed(building_graph.get_neighbors(current)):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return ([], explored_count, elapsed)
