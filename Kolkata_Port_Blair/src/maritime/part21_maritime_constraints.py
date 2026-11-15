
from pathlib import Path
import json
import geopandas as gpd
import networkx as nx

from shapely.ops import unary_union

from config.port_config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "maritime"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"

EDGE_FILE = OUTPUT_DIR / "real_maritime_edges.gpkg"
NODE_FILE = OUTPUT_DIR / "real_maritime_nodes.gpkg"

NAVIGABLE_DIR = DATA_DIR / "navigable_waters"
RESTRICTED_DIR = DATA_DIR / "restricted_areas"

OUTPUT_EDGES = OUTPUT_DIR / "constrained_maritime_edges.gpkg"
OUTPUT_NODES = OUTPUT_DIR / "constrained_maritime_nodes.gpkg"
REPORT_FILE = OUTPUT_DIR / "maritime_constraint_report.json"

SUPPORTED_FILES = {".shp", ".gpkg", ".geojson"}

def load_polygon_data(folder, target_crs):
    files = [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FILES
    ]

    geometries = []

    for path in files:
        gdf = gpd.read_file(path)

        if gdf.empty:
            continue

        if gdf.crs is None:
            raise ValueError(f"Missing CRS in {path.name}")

        gdf = gdf.to_crs(target_crs)

        for geometry in gdf.geometry:
            if geometry is None or geometry.is_empty:
                continue

            if geometry.geom_type in ["Polygon", "MultiPolygon"]:
                geometries.append(geometry)

    if not geometries:
        return None, files

    return unary_union(geometries), files

def run_maritime_constraints():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EDGE_FILE.exists() or not NODE_FILE.exists():
        raise FileNotFoundError(
            "Run Part 20 before Part 21."
        )

    edges = gpd.read_file(EDGE_FILE)
    nodes = gpd.read_file(NODE_FILE)

    if edges.crs is None or nodes.crs is None:
        raise ValueError("Network datasets must have a CRS.")

    if edges.empty or nodes.empty:
        raise ValueError("Network datasets are empty.")

    navigable_union, navigable_files = load_polygon_data(
        NAVIGABLE_DIR,
        edges.crs
    )

    restricted_union, restricted_files = load_polygon_data(
        RESTRICTED_DIR,
        edges.crs
    )

    original_count = len(edges)

    if navigable_union is not None:
        edges["inside_navigable_area"] = edges.geometry.apply(
            lambda geometry: navigable_union.covers(geometry)
        )
    else:
        edges["inside_navigable_area"] = True

    if restricted_union is not None:
        edges["intersects_restricted_area"] = edges.geometry.apply(
            lambda geometry: geometry.intersects(restricted_union)
        )
    else:
        edges["intersects_restricted_area"] = False

    edges["constraint_status"] = "RETAINED"

    edges.loc[
        ~edges["inside_navigable_area"],
        "constraint_status"
    ] = "OUTSIDE_NAVIGABLE_AREA"

    edges.loc[
        edges["intersects_restricted_area"],
        "constraint_status"
    ] = "INTERSECTS_RESTRICTED_AREA"

    filtered_edges = edges[
        edges["inside_navigable_area"]
        & ~edges["intersects_restricted_area"]
    ].copy()

    retained_nodes = set(filtered_edges["source"]).union(
        set(filtered_edges["target"])
    )

    filtered_nodes = nodes[
        nodes["node_id"].isin(retained_nodes)
    ].copy()

    filtered_edges.to_file(
        OUTPUT_EDGES,
        driver="GPKG"
    )

    filtered_nodes.to_file(
        OUTPUT_NODES,
        driver="GPKG"
    )

    report = {
        "original_edges": original_count,
        "retained_edges": len(filtered_edges),
        "removed_edges": original_count - len(filtered_edges),
        "original_nodes": len(nodes),
        "retained_nodes": len(filtered_nodes),
        "navigable_area_data_available": navigable_union is not None,
        "restricted_area_data_available": restricted_union is not None,
        "navigable_files": [path.name for path in navigable_files],
        "restricted_files": [path.name for path in restricted_files],
        "output_edges": str(OUTPUT_EDGES),
        "output_nodes": str(OUTPUT_NODES),
        "limitations": [
            "Polygon coverage does not certify navigational safety.",
            "Dataset completeness and coordinate accuracy require verification.",
            "Missing navigable-water data means no navigable-area filtering was applied.",
            "Only supplied restricted-area datasets are considered."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_maritime_constraints()
