def topsis_normalize(
    decision_matrix
):

    matrix = np.asarray(
        decision_matrix,
        dtype=float
    )

    denominator = np.sqrt(
        np.sum(
            matrix ** 2,
            axis=0
        )
    )

    denominator[
        denominator == 0
    ] = 1.0

    normalized_matrix = (
        matrix / denominator
    )

    return normalized_matrix


def topsis_apply_weights(
    normalized_matrix,
    weights
):

    weights = np.asarray(
        weights,
        dtype=float
    )

    if np.any(weights < 0):

        raise ValueError(
            "TOPSIS weights cannot be negative."
        )

    if weights.sum() <= 0:

        raise ValueError(
            "TOPSIS weights must have a positive sum."
        )

    weights = (
        weights / weights.sum()
    )

    weighted_matrix = (
        normalized_matrix * weights
    )

    return weighted_matrix, weights


def topsis_ideal_solutions(
    weighted_matrix,
    benefit_criteria
):

    ideal_best = []

    ideal_worst = []

    number_of_criteria = (
        weighted_matrix.shape[1]
    )

    for column in range(
        number_of_criteria
    ):

        values = weighted_matrix[
            :, column
        ]

        if column in benefit_criteria:

            ideal_best.append(
                np.max(values)
            )

            ideal_worst.append(
                np.min(values)
            )

        else:

            ideal_best.append(
                np.min(values)
            )

            ideal_worst.append(
                np.max(values)
            )

    return (
        np.asarray(ideal_best),
        np.asarray(ideal_worst)
    )


def calculate_topsis_scores(
    weighted_matrix,
    ideal_best,
    ideal_worst
):

    distance_from_best = np.sqrt(
        np.sum(
            (
                weighted_matrix -
                ideal_best
            ) ** 2,
            axis=1
        )
    )

    distance_from_worst = np.sqrt(
        np.sum(
            (
                weighted_matrix -
                ideal_worst
            ) ** 2,
            axis=1
        )
    )

    denominator = (
        distance_from_best +
        distance_from_worst
    )

    denominator[
        denominator == 0
    ] = 1.0

    scores = (
        distance_from_worst /
        denominator
    )

    return scores


def run_topsis(
    alternatives,
    criteria,
    weights,
    benefit_criteria=None
):

    if benefit_criteria is None:

        benefit_criteria = []

    missing = [
        criterion
        for criterion in criteria
        if criterion not in alternatives.columns
    ]

    if missing:

        raise ValueError(
            "Missing TOPSIS criteria: "
            + ", ".join(missing)
        )

    if alternatives.empty:

        raise ValueError(
            "TOPSIS requires at least one alternative."
        )

    if alternatives[criteria].isna().any().any():

        raise ValueError(
            "TOPSIS input contains missing values. "
            "Do not replace unknown report values with invented values."
        )

    matrix = alternatives[
        criteria
    ].to_numpy(dtype=float)

    normalized = topsis_normalize(
        matrix
    )

    weight_values = [
        weights[criterion]
        for criterion in criteria
    ]

    weighted, normalized_weights = (
        topsis_apply_weights(
            normalized,
            weight_values
        )
    )

    benefit_indices = [
        index
        for index, criterion in enumerate(criteria)
        if criterion in benefit_criteria
    ]

    ideal_best, ideal_worst = (
        topsis_ideal_solutions(
            weighted,
            benefit_indices
        )
    )

    scores = calculate_topsis_scores(
        weighted,
        ideal_best,
        ideal_worst
    )

    output = alternatives.copy()

    output["topsis_score"] = scores

    output["topsis_rank"] = (
        output["topsis_score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    return output.sort_values(
        "topsis_rank"
    )


def export_report_topsis_comparison(
    config
):

    report_df = pd.DataFrame(
        REPORT_ROUTE_COMPARISON
    )

    report_df.to_csv(
        config.results_dir /
        "report_mode_comparison.csv",
        index=False
    )

    missing_criteria = [
        "Numerical risk score for each mode",
        "Comparable monetary cost for each mode"
    ]

    status = pd.DataFrame([{
        "topsis_status": "NOT_COMPUTED_FROM_REPORT_TABLE",
        "reason": (
            "The report does not provide numerical risk "
            "and comparable monetary cost for both alternatives."
        ),
        "missing_inputs": "; ".join(
            missing_criteria
        ),
        "reported_weights_sum": sum(
            TOPSIS_WEIGHTS.values()
        ),
        "weights_normalized_by_topsis_function": True
    }])

    status.to_csv(
        config.results_dir /
        "topsis_report_comparison_status.csv",
        index=False
    )

    logging.info(
        "Report comparison exported. "
        "Full two-mode TOPSIS ranking requires missing numeric criteria."
    )

    return report_df


def topsis_for_candidate_routes(
    candidate_csv,
    config
):

    candidate_path = Path(
        candidate_csv
    )

    if not candidate_path.exists():

        logging.warning(
            "Candidate route table not supplied: %s",
            candidate_path
        )

        return None

    candidates = pd.read_csv(
        candidate_path
    )

    criteria = [
        "risk",
        "time",
        "cost",
        "distance",
        "carbon"
    ]

    missing = [
        column
        for column in criteria
        if column not in candidates.columns
    ]

    if missing:

        raise ValueError(
            "Candidate routes file must contain: "
            + ", ".join(criteria)
        )

    weights = {
        "risk": TOPSIS_WEIGHTS["risk"],
        "time": TOPSIS_WEIGHTS["time"],
        "cost": TOPSIS_WEIGHTS["cost"],
        "distance": TOPSIS_WEIGHTS["distance"],
        "carbon": TOPSIS_WEIGHTS["carbon"]
    }

    result = run_topsis(
        candidates,
        criteria,
        weights,
        benefit_criteria=[]
    )

    result.to_csv(
        config.results_dir /
        "topsis_candidate_route_ranking.csv",
        index=False
    )

    return result


def export_emergency_route_data(
    config
):

    emergency_df = pd.DataFrame(
        EMERGENCY_ROUTE_SEGMENTS
    )

    emergency_df[
        "cumulative_distance_km"
    ] = emergency_df[
        "distance_km"
    ].cumsum()

    emergency_df[
        "cumulative_time_minutes"
    ] = emergency_df[
        "travel_time_minutes"
    ].cumsum()

    emergency_df.to_csv(
        config.results_dir /
        "emergency_air_road_route.csv",
        index=False
    )

    route_locations = [
        "Dehradun",
        "Jolly Grant Airport",
        "Gauchar Airstrip",
        "Joshimath"
    ]

    records = []

    for name in route_locations:

        location = LOCATIONS[name]

        records.append({
            "name": name,
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "elevation_m": location["elevation_m"],
            "type": location["type"],
            "geometry": Point(
                location["longitude"],
                location["latitude"]
            )
        })

    emergency_points = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=SOURCE_CRS
    )

    emergency_points.to_file(
        config.results_dir /
        "emergency_route_points.gpkg",
        layer="emergency_route_points",
        driver="GPKG"
    )

    logging.info(
        "Emergency route reference data exported."
    )

    logging.info(
        "Report emergency segment distance total: %.2f km",
        emergency_df["distance_km"].sum()
    )

    logging.info(
        "Report emergency segment time total: %.2f hours",
        emergency_df["travel_time_minutes"].sum() / 60.0
    )

    return emergency_df, emergency_points


def process_airport_data(
    airport_gdf,
    config
):

    if airport_gdf is None:

        logging.warning(
            "Airport GIS layer not available. "
            "Using the report's tabulated airport coordinates."
        )

        airport_names = [
            "Jolly Grant Airport",
            "Gauchar Airstrip"
        ]

        records = []

        for name in airport_names:

            data = LOCATIONS[name]

            records.append({
                "name": name,
                "elevation_m": data["elevation_m"],
                "source": "Report coordinates",
                "geometry": Point(
                    data["longitude"],
                    data["latitude"]
                )
            })

        airport_gdf = gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=SOURCE_CRS
        )

    else:

        airport_gdf = airport_gdf.copy()

        airport_gdf = airport_gdf.to_crs(
            SOURCE_CRS
        )

    output = (
        config.processed_dir /
        "emergency_airports.gpkg"
    )

    airport_gdf.to_file(
        output,
        layer="emergency_airports",
        driver="GPKG"
    )

    return airport_gdf


def create_mission_summary(
    config,
    route_metrics
):

    vehicle = FUEL_CONFIG[
        config.vehicle_type
    ]

    if route_metrics is None:

        distance = np.nan

        travel_time = np.nan

        fuel_quantity = np.nan

        fuel_cost = np.nan

        risk_score = np.nan

        risk_level = "NOT_COMPUTED"

    else:

        distance = route_metrics[
            "distance_km"
        ]

        travel_time = route_metrics[
            "travel_hours"
        ]

        fuel_quantity = (
            distance / vehicle["mileage"]
        )

        fuel_cost = (
            fuel_quantity *
            vehicle["price_per_unit"]
        )

        risk_score = route_metrics[
            "risk_score"
        ]

        risk_level = route_metrics[
            "risk_level"
        ]

    summary = {
        "project": PROJECT_NAME,
        "study_area": STUDY_AREA,
        "case_date": config.case_date,
        "origin": ORIGIN_NAME,
        "destination": DESTINATION_NAME,
        "transport_mode": "Road - NH-7",
        "vehicle": vehicle["fuel_name"],
        "calculated_distance_km": distance,
        "calculated_travel_time_hours": travel_time,
        "fuel_quantity": fuel_quantity,
        "fuel_unit": vehicle["unit"],
        "fuel_rate": vehicle["price_per_unit"],
        "fuel_cost_inr": fuel_cost,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "report_distance_benchmark_km": 298.0,
        "report_time_benchmark_hours": 8.5,
        "report_air_road_distance_km": 222.0,
        "report_air_road_time_hours_approx": 5.0,
        "report_topsis_weight_sum": sum(
            TOPSIS_WEIGHTS.values()
        ),
        "topsis_weights_normalized": True
    }

    summary_df = pd.DataFrame([
        summary
    ])

    summary_df.to_csv(
        config.results_dir /
        "mission_summary.csv",
        index=False
    )

    risk_df = pd.DataFrame([{
        "case_date": config.case_date,
        "risk_score": risk_score,
        "risk_level": risk_level
    }])

    risk_df.to_csv(
        config.results_dir /
        "mission_risk.csv",
        index=False
    )

    logging.info(
        "Mission summary exported."
    )

    return summary_df


def export_route_risk_summary(
    route_gdf,
    config
):

    if route_gdf is None or route_gdf.empty:

        logging.warning(
            "No route risk data available."
        )

        return None

    columns = [
        "sequence",
        "distance_km",
        "rainfall_mm",
        "temperature_c",
        "wind_speed",
        "pressure_hpa",
        "cloud_cover",
        "elevation_m",
        "slope_deg",
        "weather_risk",
        "terrain_risk",
        "risk_score",
        "risk_level"
    ]

    available_columns = [
        column
        for column in columns
        if column in route_gdf.columns
    ]

    risk_table = route_gdf[
        available_columns
    ].copy()

    risk_table.to_csv(
        config.results_dir /
        "route_risk_analysis.csv",
        index=False
    )

    return risk_table


def export_qgis_layers(
    config,
    route_gdf,
    boundary_gdf,
    airport_gdf
):

    if boundary_gdf is not None:

        boundary_metric = boundary_gdf.to_crs(
            METRIC_CRS
        )

        boundary_metric.to_file(
            config.processed_dir /
            "uttarakhand_boundary_projected.gpkg",
            layer="uttarakhand_boundary",
            driver="GPKG"
        )

    if route_gdf is not None and not route_gdf.empty:

        route_gdf.to_file(
            config.results_dir /
            "rich_route_terrain.gpkg",
            layer="rich_route_terrain",
            driver="GPKG"
        )

    if airport_gdf is not None:

        airport_gdf.to_file(
            config.processed_dir /
            "emergency_airports.gpkg",
            layer="emergency_airports",
            driver="GPKG"
        )

    locations = create_location_geodataframe()

    locations.to_file(
        config.processed_dir /
        "report_locations.gpkg",
        layer="report_locations",
        driver="GPKG"
    )

    logging.info(
        "QGIS-ready layers exported."
    )


def create_interactive_map(
    config,
    route_gdf=None,
    boundary_gdf=None
):

    if not config.generate_maps:

        return

    try:

        import folium

        from folium.features import GeoJson

        map_object = folium.Map(
            location=[30.30, 78.80],
            zoom_start=8,
            tiles="OpenStreetMap",
            control_scale=True
        )

        if boundary_gdf is not None:

            boundary = boundary_gdf.to_crs(
                SOURCE_CRS
            )

            GeoJson(
                boundary.__geo_interface__,
                name="Uttarakhand Boundary",
                style_function=lambda feature: {
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 0.03
                }
            ).add_to(map_object)

        for name in CORRIDOR_ORDER:

            data = LOCATIONS[name]

            folium.CircleMarker(
                location=[
                    data["latitude"],
                    data["longitude"]
                ],
                radius=4,
                color="darkred",
                fill=True,
                tooltip=name,
                popup=(
                    f"{name}<br>"
                    f"Elevation: {data['elevation_m']} m"
                )
            ).add_to(map_object)

        for name in [
            "Jolly Grant Airport",
            "Gauchar Airstrip"
        ]:

            data = LOCATIONS[name]

            folium.Marker(
                location=[
                    data["latitude"],
                    data["longitude"]
                ],
                tooltip=name,
                icon=folium.Icon(
                    color="blue",
                    icon="plane",
                    prefix="fa"
                )
            ).add_to(map_object)

        if route_gdf is not None and not route_gdf.empty:

            route_wgs84 = route_gdf.to_crs(
                SOURCE_CRS
            )

            GeoJson(
                route_wgs84.__geo_interface__,
                name="A* Optimized Road Route",
                style_function=lambda feature: {
                    "color": "green",
                    "weight": 4,
                    "opacity": 0.9
                }
            ).add_to(map_object)

        emergency_order = [
            "Dehradun",
            "Jolly Grant Airport",
            "Gauchar Airstrip",
            "Joshimath"
        ]

        emergency_coordinates = [
            [
                LOCATIONS[name]["latitude"],
                LOCATIONS[name]["longitude"]
            ]
            for name in emergency_order
        ]

        folium.PolyLine(
            emergency_coordinates,
            color="blue",
            weight=3,
            dash_array="7,8",
            tooltip="Report emergency logistics schematic"
        ).add_to(map_object)

        folium.LayerControl().add_to(
            map_object
        )

        output = (
            config.results_dir /
            "uttarakhand_logistics_map.html"
        )

        map_object.save(
            str(output)
        )

        logging.info(
            "Interactive map saved: %s",
            output
        )

    except ImportError:

        logging.warning(
            "Folium is not installed. "
            "Install it using pip install folium."
        )

    except Exception:

        logging.exception(
            "Interactive map generation failed."
        )


def create_static_route_map(
    config,
    route_gdf=None,
    boundary_gdf=None
):

    if not config.generate_maps:

        return

    try:

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(
            figsize=(12, 8)
        )

        if boundary_gdf is not None:

            boundary_metric = boundary_gdf.to_crs(
                METRIC_CRS
            )

            boundary_metric.plot(
                ax=axis,
                facecolor="none",
                edgecolor="grey",
                linewidth=0.8
            )

        if route_gdf is not None and not route_gdf.empty:

            route_gdf.plot(
                ax=axis,
                color="green",
                linewidth=2.5,
                label="A* Optimized Road Route"
            )

        locations = create_location_geodataframe()

        locations_metric = locations.to_crs(
            METRIC_CRS
        )

        locations_metric.plot(
            ax=axis,
            color="darkred",
            markersize=20
        )

        for _, row in locations_metric.iterrows():

            axis.annotate(
                row["name"],
                (
                    row.geometry.x,
                    row.geometry.y
                ),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points"
            )

        axis.set_title(
            "Dehradun to Joshimath - NH-7 Logistics Route"
        )

        axis.set_axis_off()

        axis.legend(
            loc="lower left"
        )

        figure.tight_layout()

        output = (
            config.results_dir /
            "uttarakhand_optimal_route.png"
        )

        figure.savefig(
            output,
            dpi=220,
            bbox_inches="tight"
        )

        plt.close(figure)

        logging.info(
            "Static route map saved: %s",
            output
        )

    except ImportError:

        logging.warning(
            "Matplotlib is not installed."
        )

    except Exception:

        logging.exception(
            "Static map generation failed."
        )


def export_nearest_resource_reference(
    config
):

    rows = []

    for name in [
        "Jolly Grant Airport",
        "Gauchar Airstrip"
    ]:

        data = LOCATIONS[name]

        rows.append({
            "resource_name": name,
            "resource_type": data["type"],
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "elevation_m": data["elevation_m"]
        })

    pd.DataFrame(
        rows
    ).to_csv(
        config.results_dir /
        "emergency_resources.csv",
        index=False
    )


def validate_report_data():

    assert "Dehradun" in LOCATIONS

    assert "Joshimath" in LOCATIONS

    assert len(CORRIDOR_ORDER) == 8

    assert len(EMERGENCY_ROUTE_SEGMENTS) == 3

    assert math.isclose(
        sum(
            segment["distance_km"]
            for segment in EMERGENCY_ROUTE_SEGMENTS
        ),
        222.0,
        abs_tol=0.01
    )

    assert math.isclose(
        sum(
            segment["travel_time_minutes"]
            for segment in EMERGENCY_ROUTE_SEGMENTS
        ),
        273.0,
        abs_tol=0.01
    )

    logging.info(
        "Report reference data validation successful."
    )


def run_project(config):

    initialize_logging(config)

    validate_report_data()

    logging.info(
        "Project root: %s",
        config.project_root
    )

    export_report_reference_tables(
        config
    )

    save_location_layer(
        config
    )

    inputs = discover_project_inputs(
        config
    )

    roads_raw = load_vector_dataset(
        inputs["roads"],
        "Uttarakhand road network"
    )

    boundary = load_vector_dataset(
        inputs["boundary"],
        "Uttarakhand administrative boundary"
    )

    airports_raw = load_vector_dataset(
        inputs["airports"],
        "Uttarakhand airports"
    )

    airports = process_airport_data(
        airports_raw,
        config
    )

    export_nearest_resource_reference(
        config
    )

    export_emergency_route_data(
        config
    )

    if roads_raw is None:

        logging.error(
            "Road network file is missing. "
            "A* cannot run without the road geometry."
        )

        export_report_topsis_comparison(
            config
        )

        create_mission_summary(
            config,
            None
        )

        create_interactive_map(
            config,
            None,
            boundary
        )

        logging.error(
            "Place uttarakhand_roads.gpkg in data/raw "
            "and run the program again."
        )

        return

    roads = preprocess_road_network(
        roads_raw,
        config
    )

    inputs = prepare_terrain_inputs(
        inputs,
        config
    )

    graph, nodes_gdf, edges_gdf = build_road_graph(
        roads,
        config
    )

    source_node, destination_node = (
        snap_corridor_endpoints(
            graph,
            config
        )
    )

    weather_processor = IMDAAWeatherProcessor(
        inputs["weather"],
        config.case_date
    )

    try:

        graph = engineer_graph_features(
            graph,
            config,
            inputs,
            weather_processor
        )

        export_enriched_graph(
            graph,
            config
        )

        route_nodes = run_astar_search(
            graph,
            source_node,
            destination_node
        )

        route_gdf, route_metrics = (
            calculate_route_metrics(
                graph,
                route_nodes
            )
        )

        save_optimal_route(
            route_gdf,
            config
        )

        export_route_node_sequence(
            route_nodes,
            config
        )

        export_route_risk_summary(
            route_gdf,
            config
        )

        create_mission_summary(
            config,
            route_metrics
        )

        export_report_topsis_comparison(
            config
        )

        export_qgis_layers(
            config,
            route_gdf,
            boundary,
            airports
        )

        create_interactive_map(
            config,
            route_gdf,
            boundary
        )

        create_static_route_map(
            config,
            route_gdf,
            boundary
        )

        logging.info("=" * 65)

        logging.info(
            "UTTARAKHAND LOGISTICS PROJECT COMPLETED"
        )

        logging.info(
            "A* route distance: %.2f km",
            route_metrics["distance_km"]
        )

        logging.info(
            "Estimated travel time: %.2f hours",
            route_metrics["travel_hours"]
        )

        logging.info(
            "Fuel cost: Rs. %.2f",
            route_metrics["fuel_cost_inr"]
        )

        logging.info(
            "Route risk score: %s",
            route_metrics["risk_score"]
        )

        logging.info(
            "Route risk level: %s",
            route_metrics["risk_level"]
        )

        logging.info(
            "Results directory: %s",
            config.results_dir
        )

        logging.info("=" * 65)

    finally:

        weather_processor.close()


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "AI-Based Uttarakhand Logistics Route Optimization "
            "for Dehradun to Joshimath via NH-7"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=(
            Path.home() /
            "Downloads" /
            "Uttarakhand"
        ),
        help="Path to the Uttarakhand project folder"
    )

    parser.add_argument(
        "--date",
        type=str,
        default=CASE_STUDY_DATE,
        help="Weather scenario date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--vehicle",
        type=str,
        choices=list(FUEL_CONFIG.keys()),
        default="heavy_diesel",
        help="Vehicle and fuel type"
    )

    parser.add_argument(
        "--cargo-tonnes",
        type=float,
        default=1.0,
        help="Cargo weight in tonnes"
    )

    parser.add_argument(
        "--max-snap-distance",
        type=float,
        default=5000.0,
        help="Maximum allowed road graph snap distance in metres"
    )

    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Disable map generation"
    )

    return parser.parse_args()


def main():

    arguments = parse_arguments()

    config = ProjectConfig(
        project_root=arguments.project_root.expanduser().resolve(),
        case_date=arguments.date,
        vehicle_type=arguments.vehicle,
        cargo_tonnes=arguments.cargo_tonnes,
        max_snap_distance_m=arguments.max_snap_distance,
        generate_maps=not arguments.no_maps
    )

    run_project(
        config
    )


if __name__ == "__main__":

    main()