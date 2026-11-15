import time
import json
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path


def evaluate_route_performance(
    graph,
    route,
    vehicle_type="diesel_heavy",
    cargo_tonnes=10.0
):

    if not route or len(route) < 2:
        raise ValueError("Invalid route supplied.")

    fuel_data = FUEL[vehicle_type]

    total_distance = 0.0
    total_time = 0.0
    total_fuel = 0.0
    total_fuel_cost = 0.0
    risk_values = []
    elevation_values = []
    slope_values = []

    for u, v in zip(route[:-1], route[1:]):

        if not graph.has_edge(u, v):
            raise ValueError(
                f"Route contains a missing edge: {u} -> {v}"
            )

        edge = graph[u][v]

        distance = float(
            edge.get("distance_km", edge["length_m"] / 1000)
        )

        travel_time = float(
            edge.get("travel_hours", 0.0)
        )

        fuel_used = distance / fuel_data["mileage"]

        fuel_cost = fuel_used * fuel_data["rate"]

        total_distance += distance
        total_time += travel_time
        total_fuel += fuel_used
        total_fuel_cost += fuel_cost

        risk = edge.get("risk_score", np.nan)

        if np.isfinite(risk):
            risk_values.append(float(risk))

        elevation = edge.get("elevation_m", np.nan)

        if np.isfinite(elevation):
            elevation_values.append(float(elevation))

        slope = edge.get("slope_deg", np.nan)

        if np.isfinite(slope):
            slope_values.append(float(slope))

    mean_risk = (
        float(np.mean(risk_values))
        if risk_values else np.nan
    )

    emission_factor = 0.062

    carbon_emissions = (
        total_distance
        * cargo_tonnes
        * emission_factor
    )

    return {
        "distance_km": total_distance,
        "travel_hours": total_time,
        "fuel_consumption": total_fuel,
        "fuel_unit": fuel_data["unit"],
        "fuel_cost_inr": total_fuel_cost,
        "mean_risk": mean_risk,
        "carbon_emissions_kg": carbon_emissions,
        "mean_elevation_m": (
            float(np.mean(elevation_values))
            if elevation_values else np.nan
        ),
        "mean_slope_deg": (
            float(np.mean(slope_values))
            if slope_values else np.nan
        ),
        "number_of_nodes": len(route),
        "number_of_edges": len(route) - 1
    }


def benchmark_routing_algorithms(
    graph,
    source,
    destination
):

    algorithms = {}

    start_time = time.perf_counter()

    astar_route = nx.astar_path(
        graph,
        source,
        destination,
        weight="dynamic_cost"
    )

    astar_runtime = time.perf_counter() - start_time

    algorithms["A*"] = {
        "route": astar_route,
        "runtime_seconds": astar_runtime
    }

    start_time = time.perf_counter()

    dstar = DStarLite(
        graph,
        source,
        destination
    )

    dstar_route = dstar.extract_path()

    dstar_runtime = time.perf_counter() - start_time

    algorithms["D* Lite"] = {
        "route": dstar_route,
        "runtime_seconds": dstar_runtime
    }

    start_time = time.perf_counter()

    ga_route, ga_history = genetic_algorithm(
        graph,
        source,
        destination,
        population_size=30,
        generations=40
    )

    ga_runtime = time.perf_counter() - start_time

    algorithms["Genetic Algorithm"] = {
        "route": ga_route,
        "runtime_seconds": ga_runtime
    }

    start_time = time.perf_counter()

    q_agent = QLearningRouteOptimizer(
        graph,
        source,
        destination
    )

    q_history = q_agent.train(
        episodes=300
    )

    q_route = q_agent.get_route()

    q_runtime = time.perf_counter() - start_time

    algorithms["Q-learning"] = {
        "route": q_route,
        "runtime_seconds": q_runtime
    }

    return algorithms, ga_history, q_history


def create_algorithm_performance_table(
    graph,
    algorithm_results,
    vehicle_type="diesel_heavy",
    cargo_tonnes=10.0
):

    rows = []

    for algorithm, result in algorithm_results.items():

        route = result["route"]

        metrics = evaluate_route_performance(
            graph,
            route,
            vehicle_type,
            cargo_tonnes
        )

        metrics["algorithm"] = algorithm

        metrics["runtime_seconds"] = (
            result["runtime_seconds"]
        )

        metrics["route_cost"] = calculate_route_cost(
            graph,
            route
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


def evaluate_blockage_scenario(
    graph,
    source,
    destination,
    route
):

    if len(route) < 2:
        raise ValueError("Route does not contain a road segment.")

    index = len(route) // 2

    u = route[index - 1]
    v = route[index]

    if not graph.has_edge(u, v):
        raise ValueError("Selected road segment is unavailable.")

    original_cost = edge_cost(graph, u, v)

    graph[u][v]["dynamic_cost"] = 1e15

    try:

        start_time = time.perf_counter()

        if nx.has_path(graph, source, destination):

            alternate_route = nx.astar_path(
                graph,
                source,
                destination,
                weight="dynamic_cost"
            )

            runtime = time.perf_counter() - start_time

            alternate_metrics = evaluate_route_performance(
                graph,
                alternate_route
            )

            result = {
                "blocked_edge_u": str(u),
                "blocked_edge_v": str(v),
                "alternative_route_found": True,
                "replanning_time_seconds": runtime,
                "alternative_distance_km": (
                    alternate_metrics["distance_km"]
                ),
                "alternative_travel_hours": (
                    alternate_metrics["travel_hours"]
                )
            }

        else:

            result = {
                "blocked_edge_u": str(u),
                "blocked_edge_v": str(v),
                "alternative_route_found": False,
                "replanning_time_seconds": np.nan,
                "alternative_distance_km": np.nan,
                "alternative_travel_hours": np.nan
            }

    finally:

        graph[u][v]["dynamic_cost"] = original_cost

    return result


def validate_route(
    graph,
    route,
    source,
    destination
):

    checks = {

        "route_not_empty": bool(route),

        "correct_start": (
            bool(route) and route[0] == source
        ),

        "correct_destination": (
            bool(route) and route[-1] == destination
        ),

        "all_edges_exist": all(
            graph.has_edge(u, v)
            for u, v in zip(route[:-1], route[1:])
        ),

        "no_repeated_nodes": (
            len(route) == len(set(route))
        ),

        "nonnegative_edge_costs": all(
            edge_cost(graph, u, v) >= 0
            for u, v in zip(route[:-1], route[1:])
        )
    }

    checks["route_valid"] = all(
        checks.values()
    )

    return checks


def generate_research_evaluation_report(
    graph,
    source,
    destination,
    config,
    vehicle_type="diesel_heavy",
    cargo_tonnes=10.0
):

    print("\nStarting algorithm evaluation...")

    algorithm_results, ga_history, q_history = (
        benchmark_routing_algorithms(
            graph,
            source,
            destination
        )
    )

    performance = create_algorithm_performance_table(
        graph,
        algorithm_results,
        vehicle_type,
        cargo_tonnes
    )

    performance.to_csv(
        config.results_dir / "algorithm_performance.csv",
        index=False
    )

    ga_history.to_csv(
        config.results_dir / "ga_convergence.csv",
        index=False
    )

    q_history.to_csv(
        config.results_dir / "q_learning_rewards.csv",
        index=False
    )

    validation_rows = []

    for algorithm, result in algorithm_results.items():

        checks = validate_route(
            graph,
            result["route"],
            source,
            destination
        )

        validation_rows.append({
            "algorithm": algorithm,
            **checks
        })

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_df.to_csv(
        config.results_dir / "route_validation.csv",
        index=False
    )

    blockage = evaluate_blockage_scenario(
        graph,
        source,
        destination,
        algorithm_results["A*"]["route"]
    )

    pd.DataFrame([blockage]).to_csv(
        config.results_dir / "road_blockage_evaluation.csv",
        index=False
    )

    report = {
        "project": "Uttarakhand Logistics Optimization",
        "study_corridor": "Dehradun-Joshimath via NH-7",
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "algorithm_count": len(algorithm_results),
        "valid_routes": int(
            validation_df["route_valid"].sum()
        ),
        "total_routes_tested": len(validation_df),
        "blockage_scenario": blockage,
        "results_directory": str(config.results_dir)
    }

    with open(
        config.results_dir / "research_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )

    print("\nResearch evaluation completed.")

    return performance, validation_df, report


def generate_final_project_summary(config):

    expected_files = [

        "mission_summary.csv",

        "alternative_routes.csv",

        "algorithm_comparison.csv",

        "algorithm_performance.csv",

        "route_validation.csv",

        "road_blockage_evaluation.csv",

        "ga_convergence.csv",

        "q_learning_rewards.csv",

        "research_evaluation.json",

        "uttarakhand_logistics_map.html"

    ]

    rows = []

    for filename in expected_files:

        path = config.results_dir / filename

        rows.append({

            "file": filename,

            "available": path.exists(),

            "file_size_kb": (
                round(path.stat().st_size / 1024, 2)
                if path.exists()
                else 0
            )

        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        config.results_dir / "final_project_output_check.csv",
        index=False
    )

    print("\nFINAL PROJECT OUTPUT CHECK")
    print(summary.to_string(index=False))

    return summary


def run_final_evaluation(
    graph,
    source,
    destination,
    config,
    vehicle_type="diesel_heavy",
    cargo_tonnes=10.0
):

    performance, validation, report = (
        generate_research_evaluation_report(
            graph,
            source,
            destination,
            config,
            vehicle_type,
            cargo_tonnes
        )
    )

    output_summary = generate_final_project_summary(
        config
    )

    return {
        "performance": performance,
        "validation": validation,
        "report": report,
        "output_summary": output_summary
    }