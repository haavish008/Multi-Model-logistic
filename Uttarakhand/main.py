
from pathlib import Path
import sys
import traceback

import networkx as nx
import pandas as pd

from src.part17_config import load_project_config
from src.part18_topsis_integration import (
    rank_route_alternatives
)
from src.part22_qgis_export import (
    export_route,
    export_graph_layers
)


def load_existing_graph(config):
    """
    Connect this function to your completed Part 1-7
    graph-loading module.
    """

    graph_path = (
        config.processed_dir / "road_graph.graphml"
    )

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_path}"
        )

    return nx.read_graphml(graph_path)


def get_source_destination(graph):
    """
    Replace these values with your actual snapped
    Dehradun and Joshimath graph node IDs.
    """

    source_node = "SOURCE_NODE_ID"
    destination_node = "DESTINATION_NODE_ID"

    if source_node not in graph:
        raise ValueError(
            "Replace SOURCE_NODE_ID with the actual "
            "Dehradun graph node ID."
        )

    if destination_node not in graph:
        raise ValueError(
            "Replace DESTINATION_NODE_ID with the actual "
            "Joshimath graph node ID."
        )

    return source_node, destination_node


def run_final_project():

    config = load_project_config()

    print("\nPROJECT CONFIGURATION")
    print(config.validate())

    print("\nLOADING ROAD GRAPH")

    graph = load_existing_graph(config)

    print(
        "Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Edges:",
        graph.number_of_edges()
    )

    source, destination = (
        get_source_destination(graph)
    )

    if not nx.has_path(
        graph,
        source,
        destination
    ):
        raise nx.NetworkXNoPath(
            "Destination is unreachable."
        )

    print("\nRUNNING A*")

    astar_route = nx.astar_path(
        graph,
        source,
        destination,
        weight="dynamic_cost"
    )

    print(
        "A* route nodes:",
        len(astar_route)
    )

    # Save the best route
    export_route(
        graph,
        astar_route,
        config.results_dir / "best_route.gpkg",
        route_name="Dehradun-Joshimath A*"
    )

    # Export graph layers
    export_graph_layers(
        graph,
        config.results_dir
    )

    print("\nCHECKING TOPSIS INPUT")

    performance_path = (
        config.results_dir
        / "integrated_route_comparison.csv"
    )

    if performance_path.exists():

        performance = pd.read_csv(
            performance_path
        )

        ranking = rank_route_alternatives(
            performance
        )

        ranking.to_csv(
            config.results_dir
            / "topsis_route_ranking.csv",
            index=False
        )

        print(ranking)

    else:

        print(
            "Route performance file missing. "
            "Run Part 16 integration first."
        )

    print("\nPROJECT EXECUTION COMPLETED")


if __name__ == "__main__":

    try:
        run_final_project()

    except Exception as error:

        print(
            "\nPROJECT EXECUTION FAILED:",
            error
        )

        traceback.print_exc()

        sys.exit(1)
