
from pathlib import Path
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Geod
from config.port_config import WAYPOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "maritime"
RESULTS_DIR = PROJECT_ROOT / "results" / "maritime"

GEOD = Geod(ellps="WGS84")

def calculate_distance_km(start, end):
    _, _, distance = GEOD.inv(
        start["longitude"],
        start["latitude"],
        end["longitude"],
        end["latitude"],
    )
    return distance / 1000

def build_waypoint_graph():
    graph = nx.Graph()
    node_records = []
    edge_records = []

    for index, waypoint in enumerate(WAYPOINTS):
        node_id = waypoint["name"]

        graph.add_node(
            node_id,
            latitude=waypoint["latitude"],
            longitude=waypoint["longitude"],
        )

        node_records.append({
            "node_id": node_id,
            "name": waypoint["name"],
            "latitude": waypoint["latitude"],
            "longitude": waypoint["longitude"],
            "geometry": Point(
                waypoint["longitude"],
                waypoint["latitude"],
            ),
        })

    for index in range(len(WAYPOINTS) - 1):
        start = WAYPOINTS[index]
        end = WAYPOINTS[index + 1]

        distance_km = calculate_distance_km(start, end)

        coordinates = [
            (start["longitude"], start["latitude"]),
            (end["longitude"], end["latitude"]),
        ]

        geometry = LineString(coordinates)

        graph.add_edge(
            start["name"],
            end["name"],
            distance_km=distance_km,
            geometry=geometry,
        )

        edge_records.append({
            "from_node": start["name"],
            "to_node": end["name"],
            "distance_km": distance_km,
            "geometry": geometry,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nodes = gpd.GeoDataFrame(
        node_records,
        geometry="geometry",
        crs="EPSG:4326",
    )

    edges = gpd.GeoDataFrame(
        edge_records,
        geometry="geometry",
        crs="EPSG:4326",
    )

    nodes.to_file(
        OUTPUT_DIR / "maritime_nodes.gpkg",
        driver="GPKG",
    )

    edges.to_file(
        OUTPUT_DIR / "maritime_edges.gpkg",
        driver="GPKG",
    )

    route_distance = sum(
        data["distance_km"]
        for _, _, data in graph.edges(data=True)
    )

    summary = pd.DataFrame([{
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "corridor_distance_km": route_distance,
    }])

    summary.to_csv(
        RESULTS_DIR / "waypoint_network_summary.csv",
        index=False,
    )

    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Waypoint corridor distance:", round(route_distance, 2), "km")

    return graph

if __name__ == "__main__":
    build_waypoint_graph()
