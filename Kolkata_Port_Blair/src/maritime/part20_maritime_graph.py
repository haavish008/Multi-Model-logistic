
from pathlib import Path
import json
import math
import geopandas as gpd
import networkx as nx

from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import Geod

from config.port_config import PROJECT_ROOT

SEA_LANE_DIR = PROJECT_ROOT / "data" / "raw" / "maritime" / "sea_lanes"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"

NODES_FILE = OUTPUT_DIR / "real_maritime_nodes.gpkg"
EDGES_FILE = OUTPUT_DIR / "real_maritime_edges.gpkg"
REPORT_FILE = OUTPUT_DIR / "real_maritime_graph_report.json"

SUPPORTED_FILES = [".shp", ".gpkg", ".geojson"]

GEOD = Geod(ellps="WGS84")

def node_identifier(coordinate, precision=6):
    return f"{round(coordinate[0], precision)}_{round(coordinate[1], precision)}"

def calculate_distance(start, end):
    _, _, distance = GEOD.inv(
        start[0],
        start[1],
        end[0],
        end[1]
    )
    return distance / 1000

def run_maritime_graph():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in SEA_LANE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FILES
    ]

    if not files:
        raise FileNotFoundError(
            f"No sea-lane datasets found in {SEA_LANE_DIR}"
        )

    graph = nx.Graph()
    source_records = []

    for path in files:
        gdf = gpd.read_file(path)

        if gdf.empty:
            continue

        if gdf.crs is None:
            raise ValueError(f"Missing CRS in {path.name}")

        gdf = gdf.to_crs("EPSG:4326")

        for _, row in gdf.iterrows():
            geometry = row.geometry

            if geometry is None or geometry.is_empty:
                continue

            if geometry.geom_type == "MultiLineString":
                geometries = list(geometry.geoms)
            elif geometry.geom_type == "LineString":
                geometries = [geometry]
            else:
                continue

            for line in geometries:
                coordinates = list(line.coords)

                for start, end in zip(
                    coordinates[:-1],
                    coordinates[1:]
                ):
                    start = (start[0], start[1])
                    end = (end[0], end[1])

                    if start == end:
                        continue

                    start_id = node_identifier(start)
                    end_id = node_identifier(end)

                    distance = calculate_distance(start, end)

                    if not math.isfinite(distance) or distance <= 0:
                        continue

                    graph.add_node(
                        start_id,
                        longitude=start[0],
                        latitude=start[1]
                    )

                    graph.add_node(
                        end_id,
                        longitude=end[0],
                        latitude=end[1]
                    )

                    if graph.has_edge(start_id, end_id):
                        continue

                    graph.add_edge(
                        start_id,
                        end_id,
                        distance_km=distance,
                        weight=distance,
                        source_file=path.name
                    )

                    source_records.append({
                        "source": start_id,
                        "target": end_id,
                        "distance_km": distance,
                        "source_file": path.name,
                        "geometry": LineString([start, end])
                    })

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "No usable LineString maritime network features found."
        )

    node_records = []

    for node_id, attributes in graph.nodes(data=True):
        node_records.append({
            "node_id": node_id,
            "longitude": attributes["longitude"],
            "latitude": attributes["latitude"],
            "geometry": Point(
                attributes["longitude"],
                attributes["latitude"]
            )
        })

    nodes = gpd.GeoDataFrame(
        node_records,
        geometry="geometry",
        crs="EPSG:4326"
    )

    edges = gpd.GeoDataFrame(
        source_records,
        geometry="geometry",
        crs="EPSG:4326"
    )

    nodes.to_file(NODES_FILE, driver="GPKG")
    edges.to_file(EDGES_FILE, driver="GPKG")

    components = list(nx.connected_components(graph))

    report = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "connected_component_count": len(components),
        "largest_component_nodes": max(
            [len(component) for component in components],
            default=0
        ),
        "average_degree": (
            sum(dict(graph.degree()).values())
            / graph.number_of_nodes()
        ),
        "network_files": [path.name for path in files],
        "nodes_output": str(NODES_FILE),
        "edges_output": str(EDGES_FILE)
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_maritime_graph()
