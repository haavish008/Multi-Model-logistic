
from pathlib import Path
import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString
from config.port_config import SOURCE, DESTINATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "maritime"
RESULTS_DIR = PROJECT_ROOT / "results" / "maritime"

GEOD = Geod(ellps="WGS84")

def heuristic(graph, current, destination):
    start = graph.nodes[current]
    end = graph.nodes[destination]

    _, _, distance = GEOD.inv(
        start["longitude"],
        start["latitude"],
        end["longitude"],
        end["latitude"],
    )

    return distance / 1000 * 0.70

def build_graph():
    node_file = DATA_DIR / "maritime_nodes.gpkg"
    edge_file = DATA_DIR / "maritime_edges_costed.gpkg"

    nodes = gpd.read_file(node_file)
    edges = gpd.read_file(edge_file)

    graph = nx.Graph()

    for _, node in nodes.iterrows():
        graph.add_node(
            node["node_id"],
            latitude=float(node["latitude"]),
            longitude=float(node["longitude"]),
        )

    for _, edge in edges.iterrows():
        graph.add_edge(
            edge["from_node"],
            edge["to_node"],
            total_cost=float(edge["total_cost"]),
            distance_km=float(edge["distance_km"]),
        )

    return graph

def calculate_route():
    graph = build_graph()

    if SOURCE not in graph or DESTINATION not in graph:
        raise ValueError("Source or destination is missing from the graph.")

    route = nx.astar_path(
        graph,
        SOURCE,
        DESTINATION,
        heuristic=lambda current, destination: heuristic(
            graph, current, destination
        ),
        weight="total_cost",
    )

    route_edges = []

    for start, end in zip(route[:-1], route[1:]):
        route_edges.append(graph[start][end])

    total_distance = sum(
        edge["distance_km"] for edge in route_edges
    )

    total_cost = sum(
        edge["total_cost"] for edge in route_edges
    )

    node_file = DATA_DIR / "maritime_nodes.gpkg"
    nodes = gpd.read_file(node_file)

    route_nodes = nodes.set_index("node_id").loc[route].reset_index()

    route_geometry = LineString([
        (row.geometry.x, row.geometry.y)
        for _, row in route_nodes.iterrows()
    ])

    route_gdf = gpd.GeoDataFrame(
        [{
            "route_name": "A* Best Route",
            "total_distance_km": total_distance,
            "total_cost": total_cost,
            "node_count": len(route),
            "geometry": route_geometry,
        }],
        geometry="geometry",
        crs="EPSG:4326",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    route_gdf.to_file(
        RESULTS_DIR / "astar_best_route.gpkg",
        driver="GPKG",
    )

    pd.DataFrame({
        "sequence": range(1, len(route) + 1),
        "waypoint": route,
    }).to_csv(
        RESULTS_DIR / "astar_route_waypoints.csv",
        index=False,
    )

    pd.DataFrame([{
        "algorithm": "A*",
        "distance_km": total_distance,
        "total_cost": total_cost,
        "waypoint_count": len(route),
    }]).to_csv(
        RESULTS_DIR / "astar_route_metrics.csv",
        index=False,
    )

    print("A* route:", " -> ".join(route))
    print("Distance:", round(total_distance, 2), "km")
    print("Total cost:", round(total_cost, 2))

    return route

if __name__ == "__main__":
    calculate_route()
