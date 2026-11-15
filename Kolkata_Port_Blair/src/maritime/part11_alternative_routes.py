
from pathlib import Path
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from config.port_config import SOURCE, DESTINATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "maritime"
RESULTS_DIR = PROJECT_ROOT / "results" / "maritime"

def build_graph():
    nodes = gpd.read_file(DATA_DIR / "maritime_nodes.gpkg")
    edges = gpd.read_file(DATA_DIR / "maritime_edges_costed.gpkg")

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

    return graph, nodes

def generate_alternatives(number_of_routes=5):
    graph, nodes = build_graph()

    if SOURCE not in graph or DESTINATION not in graph:
        raise ValueError("Source or destination is missing.")

    if not nx.has_path(graph, SOURCE, DESTINATION):
        raise nx.NetworkXNoPath(
            "No maritime route connects the selected ports."
        )

    paths = nx.shortest_simple_paths(
        graph,
        SOURCE,
        DESTINATION,
        weight="total_cost",
    )

    records = []
    geometries = []

    for route_number, route in enumerate(paths, start=1):
        if route_number > number_of_routes:
            break

        distance = sum(
            graph[start][end]["distance_km"]
            for start, end in zip(route[:-1], route[1:])
        )

        cost = sum(
            graph[start][end]["total_cost"]
            for start, end in zip(route[:-1], route[1:])
        )

        route_nodes = nodes.set_index("node_id").loc[route]

        geometry = LineString([
            (row.geometry.x, row.geometry.y)
            for _, row in route_nodes.iterrows()
        ])

        records.append({
            "route_id": route_number,
            "waypoints": " -> ".join(route),
            "distance_km": distance,
            "total_cost": cost,
            "node_count": len(route),
        })

        geometries.append(geometry)

    if not records:
        raise ValueError("No routes were generated.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        RESULTS_DIR / "alternative_route_metrics.csv",
        index=False,
    )

    routes_gdf = gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs="EPSG:4326",
    )

    routes_gdf.to_file(
        RESULTS_DIR / "alternative_routes.gpkg",
        driver="GPKG",
    )

    print("Routes generated:", len(records))

    for record in records:
        print(
            record["route_id"],
            record["distance_km"],
            record["total_cost"],
        )

    if len(records) == 1:
        print(
            "Only one path exists. Add validated maritime network "
            "segments before attempting alternative route analysis."
        )

    return routes_gdf

if __name__ == "__main__":
    generate_alternatives()
