
import json
import networkx as nx
import geopandas as gpd
import pandas as pd

from config.port_config import OUTPUT_DIR, WAYPOINTS

REPORT_FILE = OUTPUT_DIR / "system_validation_report.json"

def run_validation():
    checks = {}

    required_files = [
        "maritime_nodes.gpkg",
        "maritime_edges.gpkg",
        "maritime_edges_costed.gpkg",
        "astar_best_route.gpkg",
        "astar_route_waypoints.csv",
        "astar_route_metrics.csv"
    ]

    for filename in required_files:
        checks[f"file_{filename}"] = (
            OUTPUT_DIR / filename
        ).exists()

    node_file = OUTPUT_DIR / "maritime_nodes.gpkg"
    edge_file = OUTPUT_DIR / "maritime_edges_costed.gpkg"
    route_file = OUTPUT_DIR / "astar_best_route.gpkg"

    if node_file.exists() and edge_file.exists():
        nodes = gpd.read_file(node_file)
        edges = gpd.read_file(edge_file)

        checks["nodes_not_empty"] = not nodes.empty
        checks["edges_not_empty"] = not edges.empty
        checks["nodes_have_crs"] = nodes.crs is not None
        checks["edges_have_crs"] = edges.crs is not None
        checks["node_geometries_valid"] = (
            nodes.geometry.notna().all()
            and nodes.geometry.is_valid.all()
        )
        checks["edge_geometries_valid"] = (
            edges.geometry.notna().all()
            and edges.geometry.is_valid.all()
        )

        if "node_id" in nodes.columns:
            graph = nx.Graph()

            for node_id in nodes["node_id"]:
                graph.add_node(node_id)

            if "source" in edges.columns and "target" in edges.columns:
                for _, edge in edges.iterrows():
                    graph.add_edge(
                        edge["source"],
                        edge["target"]
                    )

                checks["graph_connected"] = (
                    nx.is_connected(graph)
                    if graph.number_of_nodes() > 0
                    else False
                )

                checks["source_present"] = (
                    WAYPOINTS[0]["name"] in graph.nodes
                )

                checks["destination_present"] = (
                    WAYPOINTS[-1]["name"] in graph.nodes
                )

        if "total_cost" in edges.columns:
            checks["edge_costs_nonnegative"] = (
                pd.to_numeric(
                    edges["total_cost"],
                    errors="coerce"
                ).dropna().ge(0).all()
            )

    if route_file.exists():
        route = gpd.read_file(route_file)

        checks["route_not_empty"] = not route.empty
        checks["route_has_crs"] = route.crs is not None
        checks["route_geometry_valid"] = (
            route.geometry.notna().all()
            and route.geometry.is_valid.all()
        )

    bathymetry_file = OUTPUT_DIR / "waypoint_bathymetry.csv"

    if bathymetry_file.exists():
        bathymetry = pd.read_csv(bathymetry_file)
        checks["bathymetry_file_not_empty"] = not bathymetry.empty

    risk_file = OUTPUT_DIR / "maritime_risk_indicators.csv"

    if risk_file.exists():
        risk = pd.read_csv(risk_file)
        checks["risk_file_not_empty"] = not risk.empty

    report = {
        "project": "Kolkata Port to Port Blair Maritime Logistics",
        "checks": checks,
        "passed": sum(value is True for value in checks.values()),
        "failed": sum(value is False for value in checks.values()),
        "status": "PASSED" if all(checks.values()) else "REVIEW REQUIRED",
        "limitations": [
            "The waypoint network is a simplified conceptual corridor.",
            "Route connectivity does not establish navigational safety.",
            "Bathymetric and environmental data require domain validation.",
            "Alternative routes require additional validated network edges."
        ]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(f"Saved: {REPORT_FILE}")

if __name__ == "__main__":
    run_validation()
