from __future__ import annotations

import functools
import heapq
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import osmnx as ox

ox.settings.log_console = False
ox.settings.use_cache = True


@dataclass
class PathResult:
    nodes: List[int]
    cost: float


@functools.lru_cache(maxsize=12)
def load_drive_graph(place_name: str, network_type: str = "drive") -> nx.MultiDiGraph:
    """Download and prepare a drivable graph for the given place."""
    graph = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def geocode(query: str) -> Tuple[float, float]:
    """Geocode an address or place name to latitude/longitude."""
    return ox.geocode(query)


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the nearest graph node to the provided coordinate."""
    return ox.nearest_nodes(graph, lon, lat)


def compute_max_speed_mps(graph: nx.MultiDiGraph) -> float:
    """Return the maximum edge speed (in m/s) available in the graph."""
    speeds_kph = nx.get_edge_attributes(graph, "speed_kph").values()
    max_speed_kph = max((speed for speed in speeds_kph if speed is not None), default=50.0)
    return max_speed_kph / 3.6


def build_distance_heuristic(graph: nx.MultiDiGraph, goal_node: int) -> Callable[[int], float]:
    """Create a heuristic that estimates meters remaining using great-circle distance."""
    goal_lat = graph.nodes[goal_node]["y"]
    goal_lon = graph.nodes[goal_node]["x"]

    def heuristic(node: int) -> float:
        node_lat = graph.nodes[node]["y"]
        node_lon = graph.nodes[node]["x"]
        return ox.distance.great_circle_vec(node_lat, node_lon, goal_lat, goal_lon)

    return heuristic


def build_time_heuristic(
    graph: nx.MultiDiGraph, goal_node: int, max_speed_mps: float
) -> Callable[[int], float]:
    """Create a heuristic that estimates seconds remaining (distance / max speed)."""
    max_speed = max(max_speed_mps, 1.0)
    distance_heuristic = build_distance_heuristic(graph, goal_node)

    def heuristic(node: int) -> float:
        return distance_heuristic(node) / max_speed

    return heuristic


def reconstruct_path(came_from: Dict[int, int], current: int) -> List[int]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def mi_astar(
    graph: nx.MultiDiGraph,
    start_node: int,
    goal_node: int,
    weight_attr: str,
    heuristic: Callable[[int], float],
) -> PathResult:
    """Manual A* implementation for MultiDiGraph instances."""
    open_heap: List[Tuple[float, int]] = []
    heapq.heappush(open_heap, (heuristic(start_node), start_node))

    came_from: Dict[int, int] = {}
    g_score: Dict[int, float] = {start_node: 0.0}
    visited: Dict[int, float] = {}

    while open_heap:
        current_f, current_node = heapq.heappop(open_heap)
        if current_node == goal_node:
            path_nodes = reconstruct_path(came_from, current_node)
            return PathResult(nodes=path_nodes, cost=g_score[current_node])

        if current_node in visited and current_f >= visited[current_node]:
            continue
        visited[current_node] = current_f

        for neighbor in graph.successors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor)
            if not edge_data:
                continue

            best_weight = None
            for edge_attrs in edge_data.values():
                weight_val = edge_attrs.get(weight_attr)
                if weight_val is None:
                    continue
                if best_weight is None or weight_val < best_weight:
                    best_weight = weight_val

            if best_weight is None:
                continue

            tentative_g = g_score[current_node] + best_weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, neighbor))

    raise ValueError("No path could be found between the selected nodes.")


def compute_route_metrics(
    graph: nx.MultiDiGraph, route_nodes: List[int], minimize_key: str
) -> Dict[str, float]:
    """Return cumulative length (meters) and travel time (seconds) for the route."""
    if len(route_nodes) < 2:
        return {"length_m": 0.0, "travel_time_s": 0.0}

    lengths = ox.utils_graph.get_route_edge_attributes(
        graph, route_nodes, "length", minimize_key=minimize_key
    )
    times = ox.utils_graph.get_route_edge_attributes(
        graph, route_nodes, "travel_time", minimize_key=minimize_key
    )
    total_length = float(sum(lengths)) if lengths else 0.0
    total_time = float(sum(times)) if times else 0.0
    return {"length_m": total_length, "travel_time_s": total_time}


def route_nodes_to_coordinates(
    graph: nx.MultiDiGraph, route_nodes: Iterable[int]
) -> List[Tuple[float, float]]:
    """Map a list of node ids to latitude/longitude coordinate pairs."""
    return [(graph.nodes[node]["y"], graph.nodes[node]["x"]) for node in route_nodes]


def concatenate_routes(routes: List[List[int]]) -> List[int]:
    """Merge multiple node routes into a single continuous list without duplicates."""
    merged: List[int] = []
    for segment in routes:
        if not segment:
            continue
        if not merged:
            merged.extend(segment)
        else:
            merged.extend(segment[1:])
    return merged


def solve_nearest_neighbor_tour(
    graph: nx.MultiDiGraph,
    stop_nodes: List[int],
    weight_attr: str,
    max_speed_mps: float,
    return_to_start: bool = True,
) -> List[List[int]]:
    """Compute a greedy multi-stop tour using repeated A* calls."""
    if len(stop_nodes) < 2:
        raise ValueError("At least two stops are required to compute a multi-stop route.")

    remaining = list(stop_nodes[1:])
    order = [stop_nodes[0]]
    segments: List[List[int]] = []
    cache: Dict[Tuple[int, int], PathResult] = {}

    def build_heuristic(goal: int) -> Callable[[int], float]:
        if weight_attr == "travel_time":
            return build_time_heuristic(graph, goal, max_speed_mps)
        return build_distance_heuristic(graph, goal)

    current = stop_nodes[0]
    while remaining:
        best_idx = None
        best_result: Optional[PathResult] = None
        for idx, candidate in enumerate(remaining):
            cache_key = (current, candidate)
            result = cache.get(cache_key)
            if result is None:
                heuristic = build_heuristic(candidate)
                result = mi_astar(graph, current, candidate, weight_attr, heuristic)
                cache[cache_key] = result
            if best_result is None or result.cost < best_result.cost:
                best_idx = idx
                best_result = result

        assert best_result is not None and best_idx is not None
        segments.append(best_result.nodes)
        next_stop = remaining.pop(best_idx)
        order.append(next_stop)
        current = next_stop

    if return_to_start:
        start_node = stop_nodes[0]
        cache_key = (current, start_node)
        result = cache.get(cache_key)
        if result is None:
            heuristic = build_heuristic(start_node)
            result = mi_astar(graph, current, start_node, weight_attr, heuristic)
            cache[cache_key] = result
        segments.append(result.nodes)
        order.append(start_node)

    return segments
