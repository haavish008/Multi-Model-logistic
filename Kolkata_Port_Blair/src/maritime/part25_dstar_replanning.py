
import heapq
import itertools
import json
import math

import networkx as nx
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString
from config.port_config import OUTPUT_DIR
from config.route_config import (
    SOURCE_NODE,
    TARGET_NODE,
    DYNAMIC_CHANGES_FILE
)

EDGE_FILE = OUTPUT_DIR / "environmental_edge_assessment.gpkg"
NODE_FILE = OUTPUT_DIR / "real_maritime_nodes.gpkg"

ROUTE_FILE = OUTPUT_DIR / "dstar_updated_route.gpkg"
WAYPOINT_FILE = OUTPUT_DIR / "dstar_updated_waypoints.csv"
REPORT_FILE = OUTPUT_DIR / "dstar_replanning_report.json"

def calculate_key(node, start, g, rhs, heuristic):
    value = min(g[node], rhs[node])

    return (
        value + heuristic(start, node),
        value
    )

class DStarLite:

    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.last_start = start
        self.goal = goal
        self.km = 0

        self.g = {
            node: math.inf
            for node in graph.nodes
        }

        self.rhs = {
            node: math.inf
            for node in graph.nodes
        }

        self.rhs[goal] = 0

        self.queue = []
        self.counter = itertools.count()

        self.insert(goal)

    def heuristic(self, first, second):
        return 0

    def key(self, node):
        return calculate_key(
            node,
            self.start,
            self.g,
            self.rhs,
            self.heuristic
        )

    def insert(self, node):
        heapq.heappush(
            self.queue,
            (
                self.key(node),
                next(self.counter),
                node
            )
        )

    def top_key(self):
        while self.queue:
            key, _, node = self.queue[0]

            if key != self.key(node):
                heapq.heappop(self.queue)
                continue

            return key

        return (math.inf, math.inf)

    def update_vertex(self, node):
        if node != self.goal:
            self.rhs[node] = min(
                (
                    data["weight"] + self.g[neighbor]
                    for neighbor, data in self.graph[node].items()
                ),
                default=math.inf
            )

        if self.rhs[node] != self.g[node]:
            self.insert(node)

    def compute_shortest_path(self):
        iterations = 0

        while (
            self.top_key() < self.key(self.start)
            or self.rhs[self.start] != self.g[self.start]
        ):
            if not self.queue:
                break

            old_key, _, node = heapq.heappop(self.queue)
            new_key = self.key(node)

            if old_key < new_key:
                self.insert(node)

            elif self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]

                for predecessor in self.graph.predecessors(node):
                    self.update_vertex(predecessor)

            else:
                self.g[node] = math.inf
                self.update_vertex(node)

                for predecessor in self.graph.predecessors(node):
                    self.update_vertex(predecessor)

            iterations += 1

            if iterations > 10000000:
                raise RuntimeError(
                    "D* Lite exceeded its iteration limit."
                )

        return iterations

    def update_edge(self, source, target, weight):
        if self.graph.has_edge(source, target):
            self.graph[source][target]["weight"] = weight

        if self.graph.has_edge(target, source):
            self.graph[target][source]["weight"] = weight

        self.update_vertex(source)
        self.update_vertex(target)

    def get_path(self):
        if math.isinf(self.g[self.start]):
            return []

        current = self.start
        path = [current]
        visited = {current}

        while current != self.goal:
            candidates = []

            for neighbor, data in self.graph[current].items():
                score = data["weight"] + self.g[neighbor]

                if math.isfinite(score):
                    candidates.append((score, neighbor))

            if not candidates:
                return []

            _, next_node = min(candidates)

            if next_node in visited:
                return []

            current = next_node
            path.append(current)
            visited.add(current)

        return path

def load_network():
    if not EDGE_FILE.exists():
        raise FileNotFoundError(
            "Environmental edge dataset is missing."
        )

    edges = gpd.read_file(EDGE_FILE)

    required = ["source", "target", "distance_km"]

    if any(column not in edges.columns for column in required):
        raise ValueError("Required graph columns are missing.")

    graph = nx.DiGraph()

    for _, row in edges.iterrows():
        distance = float(row["distance_km"])

        if not math.isfinite(distance) or distance <= 0:
            continue

        source = str(row["source"])
        target = str(row["target"])

        graph.add_edge(
            source,
            target,
            weight=distance,
            distance_km=distance
        )

        graph.add_edge(
            target,
            source,
            weight=distance,
            distance_km=distance
        )

    return graph, edges

def apply_dynamic_changes(planner):
    changes_path = OUTPUT_DIR.parent.parent / DYNAMIC_CHANGES_FILE

    if not changes_path.exists():
        print("No dynamic closure or cost updates supplied.")
        return 0

    changes = pd.read_csv(changes_path)

    required = ["source", "target", "blocked", "updated_cost"]

    if any(column not in changes.columns for column in required):
        raise ValueError("Dynamic changes CSV has invalid columns.")

    applied = 0

    for _, row in changes.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        if not planner.graph.has_edge(source, target):
            print(f"Unknown edge: {source} -> {target}")
            continue

        blocked = str(row["blocked"]).strip().lower() in [
            "true", "1", "yes"
        ]

        if blocked:
            weight = math.inf
        else:
            weight = float(row["updated_cost"])

            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Invalid updated cost for {source} -> {target}"
                )

        planner.update_edge(source, target, weight)
        applied += 1

    return applied

def run_dstar_replanning():
    graph, edges = load_network()

    if SOURCE_NODE is None or TARGET_NODE is None:
        raise ValueError(
            "Configure SOURCE_NODE and TARGET_NODE "
            "in config/route_config.py."
        )

    start = str(SOURCE_NODE)
    goal = str(TARGET_NODE)

    if start not in graph or goal not in graph:
        raise ValueError("Source or destination node is not in the graph.")

    planner = DStarLite(graph, start, goal)

    initial_iterations = planner.compute_shortest_path()
    initial_path = planner.get_path()

    initial_cost = (
        planner.g[start]
        if initial_path else math.inf
    )

    changes_applied = apply_dynamic_changes(planner)

    updated_iterations = planner.compute_shortest_path()
    updated_path = planner.get_path()

    updated_cost = (
        planner.g[start]
        if updated_path else math.inf
    )

    nodes = gpd.read_file(NODE_FILE)

    if nodes.crs is None:
        raise ValueError("Node dataset has no CRS.")

    node_lookup = {
        str(row["node_id"]): row.geometry
        for _, row in nodes.iterrows()
    }

    route_records = []

    if updated_path:
        coordinates = []

        for node in updated_path:
            if node not in node_lookup:
                raise ValueError(
                    f"Missing geometry for node {node}"
                )

            point = node_lookup[node]
            coordinates.append((point.x, point.y))

            route_records.append({
                "node_id": node,
                "sequence": len(route_records) + 1,
                "geometry": point
            })

        route_geometry = LineString(coordinates)

        route_gdf = gpd.GeoDataFrame(
            [{
                "route_type": "DSTAR_LITE",
                "total_cost": updated_cost,
                "node_count": len(updated_path),
                "geometry": route_geometry
            }],
            geometry="geometry",
            crs=nodes.crs
        )

        route_gdf.to_file(
            ROUTE_FILE,
            driver="GPKG"
        )

        waypoint_gdf = gpd.GeoDataFrame(
            route_records,
            geometry="geometry",
            crs=nodes.crs
        )

        waypoint_gdf.to_file(
            WAYPOINT_FILE.replace(".csv", ".gpkg"),
            driver="GPKG"
        )

    report = {
        "algorithm": "D* Lite",
        "source": start,
        "destination": goal,
        "initial_path": initial_path,
        "updated_path": updated_path,
        "initial_cost": (
            initial_cost if math.isfinite(initial_cost) else None
        ),
        "updated_cost": (
            updated_cost if math.isfinite(updated_cost) else None
        ),
        "dynamic_changes_applied": changes_applied,
        "initial_iterations": initial_iterations,
        "updated_iterations": updated_iterations,
        "route_available": bool(updated_path),
        "limitations": [
            "Dynamic changes must come from verified observations.",
            "This implementation uses distance-based costs.",
            "Environmental and navigational constraints require separate validation.",
            "No real-time maritime data feed is connected."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_dstar_replanning()
