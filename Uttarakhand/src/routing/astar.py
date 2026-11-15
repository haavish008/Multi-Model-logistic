def run_streamlit_dashboard(config):

    try:

        import streamlit as st
        import streamlit.components.v1 as components

    except ImportError:

        raise SystemExit(
            "Streamlit is missing. Install it using: "
            "pip install streamlit"
        )

    st.set_page_config(
        page_title="Uttarakhand Logistics AI",
        page_icon="🗺️",
        layout="wide"
    )

    st.title(
        "AI-Based Uttarakhand Logistics Decision Support System"
    )

    st.caption(
        "Dehradun → Joshimath | NH-7 | Weather and Terrain Risk"
    )

    results = config.results_dir

    summary_path = (
        results / "mission_summary.csv"
    )

    risk_path = (
        results / "mission_risk.csv"
    )

    alternative_path = (
        results / "alternative_routes.csv"
    )

    ga_path = (
        results / "genetic_algorithm_best_route.csv"
    )

    q_path = (
        results / "q_learning_training_history.csv"
    )

    def load_csv(path):

        if path.exists():

            return pd.read_csv(path)

        return None

    summary = load_csv(
        summary_path
    )

    risk = load_csv(
        risk_path
    )

    if summary is not None and not summary.empty:

        row = summary.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Route Distance",
            (
                f"{row['calculated_distance_km']:.2f} km"
                if pd.notna(row["calculated_distance_km"])
                else "Not calculated"
            )
        )

        col2.metric(
            "Travel Time",
            (
                f"{row['calculated_travel_time_hours']:.2f} hours"
                if pd.notna(row["calculated_travel_time_hours"])
                else "Not calculated"
            )
        )

        col3.metric(
            "Fuel Cost",
            (
                f"₹{row['fuel_cost_inr']:.2f}"
                if pd.notna(row["fuel_cost_inr"])
                else "Not calculated"
            )
        )

        col4.metric(
            "Risk Level",
            str(row["risk_level"])
        )

        st.subheader(
            "Mission Summary"
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

    else:

        st.warning(
            "Mission summary is not available. Run the main pipeline first."
        )

    st.subheader(
        "Uttarakhand Route Map"
    )

    map_path = (
        results /
        "uttarakhand_logistics_map.html"
    )

    if map_path.exists():

        html = map_path.read_text(
            encoding="utf-8"
        )

        components.html(
            html,
            height=650,
            scrolling=True
        )

    else:

        st.info(
            "Interactive map has not been generated."
        )

    st.subheader(
        "Route Risk Analysis"
    )

    if risk is not None:

        st.dataframe(
            risk,
            use_container_width=True
        )

    risk_table_path = (
        results /
        "route_risk_analysis.csv"
    )

    route_risk = load_csv(
        risk_table_path
    )

    if route_risk is not None:

        st.dataframe(
            route_risk,
            use_container_width=True
        )

    st.subheader(
        "Alternative Route Comparison"
    )

    alternatives = load_csv(
        alternative_path
    )

    if alternatives is not None:

        st.dataframe(
            alternatives,
            use_container_width=True
        )

    else:

        st.info(
            "Alternative route results are not available yet."
        )

    st.subheader(
        "Genetic Algorithm Results"
    )

    ga_results = load_csv(
        ga_path
    )

    if ga_results is not None:

        st.dataframe(
            ga_results,
            use_container_width=True
        )

    ga_history = load_csv(
        results /
        "genetic_algorithm_history.csv"
    )

    if ga_history is not None:

        st.line_chart(
            ga_history.set_index(
                "generation"
            )["best_fitness"]
        )

    st.subheader(
        "Reinforcement Learning Training"
    )

    q_history = load_csv(
        q_path
    )

    if q_history is not None:

        st.line_chart(
            q_history.set_index(
                "episode"
            )["total_reward"]
        )

    st.subheader(
        "Graph Neural Network Results"
    )

    gnn_history = load_csv(
        results /
        "gnn_training_history.csv"
    )

    if gnn_history is not None:

        st.line_chart(
            gnn_history.set_index(
                "epoch"
            )["training_loss"]
        )

    gnn_predictions = load_csv(
        results /
        "gnn_node_risk_predictions.csv"
    )

    if gnn_predictions is not None:

        st.dataframe(
            gnn_predictions.head(100),
            use_container_width=True
        )

    st.subheader(
        "Emergency Air-Road Logistics"
    )

    emergency_path = (
        results /
        "emergency_air_road_route.csv"
    )

    emergency = load_csv(
        emergency_path
    )

    if emergency is not None:

        st.dataframe(
            emergency,
            use_container_width=True
        )

    st.subheader(
        "Report-Based Route Comparison"
    )

    comparison = load_csv(
        results /
        "report_mode_comparison.csv"
    )

    if comparison is not None:

        st.dataframe(
            comparison,
            use_container_width=True
        )

    st.divider()

    st.caption(
        "This dashboard displays the outputs saved by the Python pipeline. "
        "It does not independently validate the underlying GIS datasets."
    )


def run_advanced_algorithms(
    graph,
    source,
    target,
    config,
    landslide_path=None,
    run_gnn=False,
    q_learning_episodes=200
):

    logging.info(
        "Starting advanced algorithm pipeline..."
    )

    landslides = preprocess_landslide_inventory(
        landslide_path,
        config
    )

    graph = calculate_landslide_edge_risk(
        graph,
        landslides,
        buffer_distance_m=500
    )

    export_landslide_enriched_graph(
        graph,
        config
    )

    alternative_routes = generate_alternative_routes(
        graph,
        source,
        target,
        number_of_routes=5
    )

    export_alternative_routes(
        alternative_routes,
        config
    )

    export_alternative_route_geometries(
        graph,
        alternative_routes,
        config
    )

    ga_best_route, ga_history = run_genetic_algorithm(
        graph,
        source,
        target,
        alternative_routes
    )

    ga_metrics = export_genetic_algorithm_results(
        graph,
        ga_best_route,
        ga_history,
        config
    )

    logging.info(
        "GA best cost: %.4f",
        ga_metrics["dynamic_cost"]
    )

    dstar_planner, dstar_route = run_dstar_lite(
        graph,
        source,
        target
    )

    export_dstar_results(
        graph,
        dstar_route,
        config,
        scenario_name="normal"
    )

    q_agent, q_history = train_q_learning(
        graph,
        source,
        target,
        episodes=q_learning_episodes
    )

    q_route = extract_q_learning_route(
        graph,
        q_agent,
        source,
        target
    )

    export_q_learning_results(
        q_history,
        q_route,
        graph,
        config
    )

    gnn_result = None

    if run_gnn:

        gnn_result = train_graph_neural_network(
            graph,
            config,
            epochs=100
        )

    logging.info(
        "Advanced algorithms completed."
    )

    return {
        "graph": graph,
        "landslides": landslides,
        "alternative_routes": alternative_routes,
        "ga_route": ga_best_route,
        "dstar_planner": dstar_planner,
        "dstar_route": dstar_route,
        "q_learning_agent": q_agent,
        "q_learning_route": q_route,
        "gnn_result": gnn_result
    }


def compare_algorithm_routes(
    graph,
    route_dictionary,
    config
):

    rows = []

    for algorithm, route in route_dictionary.items():

        if not route:

            continue

        distance = 0.0
        time = 0.0
        fuel_cost = 0.0
        risk_sum = 0.0
        risk_distance = 0.0
        dynamic_cost = 0.0

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            edge = graph[u][v]

            distance += edge.get(
                "distance_km",
                edge["length_m"] / 1000.0
            )

            time += edge.get(
                "travel_hours",
                0.0
            )

            fuel_cost += edge.get(
                "fuel_cost_inr",
                0.0
            )

            dynamic_cost += edge.get(
                "dynamic_cost",
                0.0
            )

            edge_risk = edge.get(
                "risk_score",
                np.nan
            )

            if np.isfinite(edge_risk):

                edge_distance = edge.get(
                    "distance_km",
                    edge["length_m"] / 1000.0
                )

                risk_sum += (
                    edge_risk * edge_distance
                )

                risk_distance += edge_distance

        average_risk = (
            risk_sum / risk_distance
            if risk_distance > 0
            else np.nan
        )

        rows.append({
            "algorithm": algorithm,
            "distance_km": distance,
            "travel_hours": time,
            "fuel_cost_inr": fuel_cost,
            "average_risk": average_risk,
            "dynamic_cost": dynamic_cost,
            "number_of_nodes": len(route)
        })

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        config.results_dir /
        "algorithm_comparison.csv",
        index=False
    )

    return comparison


def execute_complete_advanced_workflow(
    graph,
    source,
    target,
    config,
    landslide_path=None,
    run_gnn=False
):

    advanced = run_advanced_algorithms(
        graph,
        source,
        target,
        config,
        landslide_path=landslide_path,
        run_gnn=run_gnn
    )

    routes = {
        "A*": nx.astar_path(
            graph,
            source,
            target,
            heuristic=lambda a, b: 0.0,
            weight="dynamic_cost"
        ),
        "Genetic Algorithm": advanced["ga_route"],
        "D* Lite": advanced["dstar_route"],
        "Q-learning": advanced["q_learning_route"]
    }

    comparison = compare_algorithm_routes(
        graph,
        routes,
        config
    )

    return advanced, comparison


def launch_dashboard_from_arguments():

    arguments = parse_arguments()

    config = ProjectConfig(
        project_root=arguments.project_root.expanduser().resolve(),
        case_date=arguments.date,
        vehicle_type=arguments.vehicle,
        cargo_tonnes=arguments.cargo_tonnes,
        max_snap_distance_m=arguments.max_snap_distance,
        generate_maps=not arguments.no_maps
    )

    run_streamlit_dashboard(
        config
    )