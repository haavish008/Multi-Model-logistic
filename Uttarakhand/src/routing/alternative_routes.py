
from pathlib import Path

import networkx as nx
import pandas as pd


def generate_alternative_routes(
    graph,
    source,
    destination,
    number_of_routes=5,
    weight="dynamic_cost"
):

    if source not in graph or destination not in graph:
        raise ValueError(
            "Source or destination is not in graph."
        )

    if not nx.has_path(graph, source, destination):
        raise nx.NetworkXNoPath(
            "No route exists between source and destination."
        )

    if graph.is_multigraph():
        raise TypeError(
            "Convert the MultiGraph into a Graph with a defined "
            "edge aggregation policy before using this function."
        )

    route_generator = nx.shortest_simple_paths(
        graph,
        source,
        destination,
        weight=weight
    )

    routes = []

    for index, route in enumerate(
        route_generator,
        start=1
    ):

        routes.append({
            "route_id": f"ROUTE_{index}",
            "route": route
        })

        if len(routes) >= number_of_routes:
            break

    return routes


def evaluate_alternative_routes(
    graph,
    routes
):

    rows = []

    for item in routes:

        route = item["route"]

        total_distance_km = 0.0
        total_time_hours = 0.0
        risk_values = []
        total_cost = 0.0

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            edge = graph[u][v]

            distance = float(
                edge.get(
                    "distance_km",
                    edge.get("length_m", 0.0) / 1000.0
                )
            )

            travel_time = float(
                edge.get("travel_hours", 0.0)
            )

            risk = float(
                edge.get(
                    "dynamic_risk",
                    edge.get("risk_score", 0.0)
                )
            )

            cost = float(
                edge.get("dynamic_cost", 0.0)
            )

            total_distance_km += distance
            total_time_hours += travel_time
            total_cost += cost

            risk_values.append(risk)

        rows.append({

            "route_id": item["route_id"],

            "distance_km": total_distance_km,

            "travel_hours": total_time_hours,

            "mean_risk": (
                sum(risk_values) / len(risk_values)
                if risk_values else 0.0
            ),

            "maximum_edge_risk": (
                max(risk_values)
                if risk_values else 0.0
            ),

            "total_dynamic_cost": total_cost,

            "number_of_nodes": len(route),

            "number_of_edges": len(route) - 1,

            "route_nodes": " -> ".join(
                map(str, route)
            )
        })

    return pd.DataFrame(rows)


def save_alternative_routes(
    routes,
    metrics,
    output_dir
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    route_records = []

    for item in routes:

        for position, node in enumerate(item["route"]):

            route_records.append({
                "route_id": item["route_id"],
                "position": position,
                "node": str(node)
            })

    pd.DataFrame(route_records).to_csv(
        output_dir / "alternative_routes_nodes.csv",
        index=False
    )

    metrics.to_csv(
        output_dir / "alternative_routes.csv",
        index=False
    )

    print("\nALTERNATIVE ROUTE COMPARISON")

    print(
        metrics[
            [
                "route_id",
                "distance_km",
                "travel_hours",
                "mean_risk",
                "total_dynamic_cost"
            ]
        ].to_string(index=False)
    )


def run_alternative_route_pipeline(
    graph,
    source,
    destination,
    output_dir,
    number_of_routes=5
):

    routes = generate_alternative_routes(
        graph,
        source,
        destination,
        number_of_routes
    )

    metrics = evaluate_alternative_routes(
        graph,
        routes
    )

    save_alternative_routes(
        routes,
        metrics,
        output_dir
    )

    return routes, metrics


if __name__ == "__main__":

    print(
        "Import your existing graph loader and snapped "
        "source/destination nodes before running Part 30."
    )
