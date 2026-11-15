
from pathlib import Path
import pandas as pd
import geopandas as gpd

from config.port_config import OUTPUT_DIR

REQUIRED_OUTPUTS = [
    "real_maritime_nodes.gpkg",
    "real_maritime_edges.gpkg",
    "constrained_maritime_edges.gpkg",
    "bathymetry_assessed_edges.gpkg",
    "environmental_edge_assessment.gpkg"
]

OPTIONAL_OUTPUTS = [
    "astar_best_route.gpkg",
    "genetic_best_route.gpkg",
    "dstar_updated_route.gpkg",
    "advanced_topsis_ranking.csv",
    "final_route_comparison.csv",
    "performance_evaluation.csv"
]

def test_required_outputs_exist():
    missing = [
        name for name in REQUIRED_OUTPUTS
        if not (OUTPUT_DIR / name).exists()
    ]

    assert not missing, f"Missing outputs: {missing}"

def test_network_connectivity():
    nodes_path = OUTPUT_DIR / "real_maritime_nodes.gpkg"
    edges_path = OUTPUT_DIR / "real_maritime_edges.gpkg"

    if not nodes_path.exists() or not edges_path.exists():
        return

    nodes = gpd.read_file(nodes_path)
    edges = gpd.read_file(edges_path)

    assert not nodes.empty
    assert not edges.empty

    node_ids = set(nodes["node_id"])

    assert set(edges["source"]).issubset(node_ids)
    assert set(edges["target"]).issubset(node_ids)

def test_route_comparison_schema():
    path = OUTPUT_DIR / "final_route_comparison.csv"

    if not path.exists():
        return

    data = pd.read_csv(path)

    assert "algorithm" in data.columns
    assert "distance_km" in data.columns

def test_performance_evaluation_schema():
    path = OUTPUT_DIR / "performance_evaluation.csv"

    if not path.exists():
        return

    data = pd.read_csv(path)

    assert not data.empty
    assert "algorithm" in data.columns

def test_optional_outputs_are_readable():
    for filename in OPTIONAL_OUTPUTS:
        path = OUTPUT_DIR / filename

        if not path.exists():
            continue

        if path.suffix == ".gpkg":
            data = gpd.read_file(path)
            assert data.crs is not None

        elif path.suffix == ".csv":
            data = pd.read_csv(path)
            assert len(data.columns) > 0
