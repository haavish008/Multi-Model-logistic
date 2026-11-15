WEATHER_VARIABLE_ALIASES = {

    "rainfall_mm": [
        "rainfall",
        "rain",
        "precipitation",
        "precip",
        "rf",
        "rfp",
        "tp"
    ],

    "temperature_c": [
        "temperature",
        "temp",
        "t2",
        "t2m"
    ],

    "wind_speed": [
        "wind_speed",
        "windspeed",
        "wind",
        "ws"
    ],

    "pressure_hpa": [
        "pressure",
        "pres",
        "sp",
        "mslp"
    ],

    "cloud_cover": [
        "cloud_cover",
        "cloud",
        "tcc"
    ]

}


def detect_weather_variable(dataset, aliases):

    available_variables = list(
        dataset.data_vars
    )

    for alias in aliases:

        for variable in available_variables:

            if alias.lower() in variable.lower():

                return variable

    return None


def detect_weather_coordinate(dataset, aliases):

    available_names = list(
        dataset.coords
    ) + list(
        dataset.dims
    )

    for alias in aliases:

        for name in available_names:

            if alias.lower() in name.lower():

                return name

    return None


class IMDAAWeatherProcessor:

    def __init__(
        self,
        weather_path,
        case_date
    ):

        self.weather_path = weather_path

        self.case_date = pd.Timestamp(
            case_date
        )

        self.dataset = None

        self.variable_mapping = {}

        self.time_coordinate = None

        self.latitude_coordinate = None

        self.longitude_coordinate = None

        if weather_path is None:

            logging.warning(
                "IMDAA NetCDF file not found. "
                "Weather values will remain unavailable."
            )

            return

        self.open_dataset()

    def open_dataset(self):

        try:

            self.dataset = xr.open_dataset(
                self.weather_path
            )

            logging.info(
                "IMDAA dataset opened: %s",
                self.weather_path
            )

            logging.info(
                "Dataset dimensions: %s",
                dict(self.dataset.sizes)
            )

            self.time_coordinate = (
                detect_weather_coordinate(
                    self.dataset,
                    ["time", "valid_time", "date"]
                )
            )

            self.latitude_coordinate = (
                detect_weather_coordinate(
                    self.dataset,
                    ["latitude", "lat"]
                )
            )

            self.longitude_coordinate = (
                detect_weather_coordinate(
                    self.dataset,
                    ["longitude", "lon"]
                )
            )

            for feature, aliases in (
                WEATHER_VARIABLE_ALIASES.items()
            ):

                self.variable_mapping[feature] = (
                    detect_weather_variable(
                        self.dataset,
                        aliases
                    )
                )

            logging.info(
                "Detected weather coordinates: time=%s, lat=%s, lon=%s",
                self.time_coordinate,
                self.latitude_coordinate,
                self.longitude_coordinate
            )

            logging.info(
                "Detected weather variables: %s",
                self.variable_mapping
            )

        except Exception:

            logging.exception(
                "Unable to open or inspect the IMDAA NetCDF file."
            )

            self.dataset = None

    @staticmethod
    def convert_units(
        variable_name,
        value
    ):

        if not np.isfinite(value):

            return float("nan")

        name = variable_name.lower()

        if (
            "temp" in name or
            name.startswith("t")
        ) and value > 150:

            value = value - 273.15

        if (
            "pressure" in name or
            "pres" in name or
            name in ["sp", "mslp"]
        ) and value > 2000:

            value = value / 100.0

        return float(value)

    def get_nearest_time_value(self):

        if self.dataset is None:

            return None

        if self.time_coordinate is None:

            return None

        try:

            coordinate = self.dataset[
                self.time_coordinate
            ]

            values = pd.to_datetime(
                coordinate.values
            )

            if len(values) == 0:

                return None

            differences = np.abs(
                values -
                self.case_date.to_datetime64()
            )

            index = int(
                np.argmin(differences)
            )

            selected_time = coordinate.values[
                index
            ]

            logging.info(
                "Selected IMDAA time: %s",
                selected_time
            )

            return selected_time

        except Exception:

            logging.exception(
                "Could not select the nearest IMDAA time."
            )

            return None

    def sample_location(
        self,
        longitude,
        latitude
    ):

        output = {

            "rainfall_mm": np.nan,

            "temperature_c": np.nan,

            "wind_speed": np.nan,

            "pressure_hpa": np.nan,

            "cloud_cover": np.nan

        }

        if self.dataset is None:

            return output

        if (
            self.latitude_coordinate is None or
            self.longitude_coordinate is None
        ):

            logging.warning(
                "Latitude/longitude coordinates were not detected."
            )

            return output

        try:

            selection = {

                self.latitude_coordinate: latitude,

                self.longitude_coordinate: longitude

            }

            nearest_time = (
                self.get_nearest_time_value()
            )

            if nearest_time is not None:

                selection[
                    self.time_coordinate
                ] = nearest_time

            for feature, variable in (
                self.variable_mapping.items()
            ):

                if variable is None:

                    continue

                data_array = self.dataset[
                    variable
                ]

                selected = data_array.sel(
                    selection,
                    method="nearest"
                )

                values = np.asarray(
                    selected.values
                ).astype(float).ravel()

                if len(values) == 0:

                    continue

                value = float(
                    np.nanmean(values)
                )

                output[feature] = (
                    self.convert_units(
                        variable,
                        value
                    )
                )

        except Exception:

            logging.exception(
                "Weather sampling failed at %.5f, %.5f",
                longitude,
                latitude
            )

        return output

    def close(self):

        if self.dataset is not None:

            self.dataset.close()


def classify_rainfall_risk(rainfall):

    if not np.isfinite(rainfall):

        return {
            "rainfall_risk": np.nan,
            "rainfall_category": "UNKNOWN"
        }

    if rainfall > 50:

        return {
            "rainfall_risk": 1.0,
            "rainfall_category": "HIGH"
        }

    if rainfall >= 15:

        return {
            "rainfall_risk": 0.5,
            "rainfall_category": "MEDIUM"
        }

    return {
        "rainfall_risk": 0.1,
        "rainfall_category": "LOW"
    }


def normalize_risk(
    value,
    scale
):

    if not np.isfinite(value):

        return np.nan

    return float(
        np.clip(
            value / scale,
            0,
            1
        )
    )


def calculate_weather_risk(weather):

    rainfall = weather["rainfall_mm"]

    temperature = weather["temperature_c"]

    wind = weather["wind_speed"]

    pressure = weather["pressure_hpa"]

    cloud = weather["cloud_cover"]

    rainfall_result = classify_rainfall_risk(
        rainfall
    )

    rainfall_component = (
        rainfall_result["rainfall_risk"]
    )

    components = []

    if np.isfinite(rainfall_component):

        components.append(
            rainfall_component
        )

    if np.isfinite(temperature):

        components.append(
            normalize_risk(
                abs(temperature - 20),
                35
            )
        )

    if np.isfinite(wind):

        components.append(
            normalize_risk(
                wind,
                100
            )
        )

    if np.isfinite(pressure):

        components.append(
            normalize_risk(
                abs(pressure - 1013),
                80
            )
        )

    if np.isfinite(cloud):

        components.append(
            normalize_risk(
                cloud,
                100
            )
        )

    if not components:

        weather_score = np.nan

    else:

        weather_score = float(
            np.nanmean(components)
        )

    return {
        "weather_risk": weather_score,
        "rainfall_category": (
            rainfall_result["rainfall_category"]
        )
    }


def calculate_terrain_risk(terrain):

    slope = terrain["slope_deg"]

    elevation = terrain["elevation_m"]

    components = []

    if np.isfinite(slope):

        components.append(
            normalize_risk(
                slope,
                60
            )
        )

    if np.isfinite(elevation):

        components.append(
            normalize_risk(
                elevation,
                3000
            )
        )

    if not components:

        return np.nan

    return float(
        np.nanmean(components)
    )


def calculate_combined_risk(
    weather_risk,
    terrain_risk
):

    available = []

    if np.isfinite(weather_risk):

        available.append(
            (weather_risk, 0.60)
        )

    if np.isfinite(terrain_risk):

        available.append(
            (terrain_risk, 0.40)
        )

    if not available:

        return np.nan

    weighted_sum = sum(
        value * weight
        for value, weight in available
    )

    weight_sum = sum(
        weight
        for _, weight in available
    )

    return float(
        np.clip(
            weighted_sum / weight_sum,
            0,
            1
        )
    )


def risk_level_from_score(score):

    if not np.isfinite(score):

        return "UNKNOWN"

    if score >= 0.67:

        return "HIGH"

    if score >= 0.34:

        return "MEDIUM"

    return "LOW"


def calculate_fuel_consumption(
    distance_km,
    vehicle_type
):

    vehicle = FUEL_CONFIG[
        vehicle_type
    ]

    quantity = (
        distance_km /
        vehicle["mileage"]
    )

    cost = (
        quantity *
        vehicle["price_per_unit"]
    )

    return {
        "fuel_name": vehicle["fuel_name"],
        "fuel_quantity": quantity,
        "fuel_unit": vehicle["unit"],
        "fuel_rate": vehicle["price_per_unit"],
        "fuel_cost_inr": cost
    }


def calculate_travel_time(
    distance_km,
    speed_kmh,
    rainfall,
    wind_speed,
    slope
):

    base_speed = max(
        speed_kmh,
        1.0
    )

    delay_factor = 1.0

    if np.isfinite(rainfall):

        if rainfall > 50:

            delay_factor += 0.30

        elif rainfall >= 15:

            delay_factor += 0.15

    if np.isfinite(wind_speed):

        delay_factor += min(
            max(wind_speed, 0) / 200,
            0.20
        )

    if np.isfinite(slope):

        delay_factor += min(
            max(slope, 0) / 120,
            0.50
        )

    travel_hours = (
        distance_km /
        base_speed
    ) * delay_factor

    return travel_hours


def calculate_dynamic_edge_cost(
    distance_km,
    travel_hours,
    fuel_cost,
    weather_risk,
    terrain_risk
):

    weather_value = (
        weather_risk
        if np.isfinite(weather_risk)
        else 0.0
    )

    terrain_value = (
        terrain_risk
        if np.isfinite(terrain_risk)
        else 0.0
    )

    distance_component = (
        distance_km / 10.0
    )

    time_component = travel_hours

    fuel_component = (
        fuel_cost / 1000.0
    )

    weather_component = (
        weather_value * 5.0
    )

    terrain_component = (
        terrain_value * 5.0
    )

    total_cost = (
        distance_component +
        time_component +
        fuel_component +
        weather_component +
        terrain_component
    )

    return {
        "distance_component": distance_component,
        "time_component": time_component,
        "fuel_component": fuel_component,
        "weather_component": weather_component,
        "terrain_component": terrain_component,
        "dynamic_cost": total_cost
    }


def engineer_graph_features(
    graph,
    config,
    inputs,
    weather_processor
):

    logging.info(
        "Starting graph edge feature engineering..."
    )

    vehicle = FUEL_CONFIG[
        config.vehicle_type
    ]

    for u, v, data in graph.edges(
        data=True
    ):

        geometry = data["geometry"]

        midpoint = geometry.interpolate(
            0.5,
            normalized=True
        )

        midpoint_gdf = gpd.GeoSeries(
            [midpoint],
            crs=METRIC_CRS
        ).to_crs(
            SOURCE_CRS
        )

        longitude, latitude = (
            midpoint_gdf.iloc[0].x,
            midpoint_gdf.iloc[0].y
        )

        distance_km = (
            data["length_m"] / 1000.0
        )

        terrain = sample_terrain_features(
            longitude,
            latitude,
            inputs
        )

        weather = weather_processor.sample_location(
            longitude,
            latitude
        )

        weather_risk_data = calculate_weather_risk(
            weather
        )

        weather_risk = (
            weather_risk_data["weather_risk"]
        )

        terrain_risk = calculate_terrain_risk(
            terrain
        )

        combined_risk = calculate_combined_risk(
            weather_risk,
            terrain_risk
        )

        travel_hours = calculate_travel_time(
            distance_km,
            config.default_speed_kmh,
            weather["rainfall_mm"],
            weather["wind_speed"],
            terrain["slope_deg"]
        )

        fuel_quantity = (
            distance_km /
            vehicle["mileage"]
        )

        fuel_cost = (
            fuel_quantity *
            vehicle["price_per_unit"]
        )

        cost_components = calculate_dynamic_edge_cost(
            distance_km,
            travel_hours,
            fuel_cost,
            weather_risk,
            terrain_risk
        )

        data.update({
            "longitude": longitude,
            "latitude": latitude,
            "distance_km": distance_km,
            "travel_hours": travel_hours,
            "fuel_quantity": fuel_quantity,
            "fuel_unit": vehicle["unit"],
            "fuel_cost_inr": fuel_cost,
            "elevation_m": terrain["elevation_m"],
            "slope_deg": terrain["slope_deg"],
            "aspect_deg": terrain["aspect_deg"],
            "tri": terrain["tri"],
            "rainfall_mm": weather["rainfall_mm"],
            "temperature_c": weather["temperature_c"],
            "wind_speed": weather["wind_speed"],
            "pressure_hpa": weather["pressure_hpa"],
            "cloud_cover": weather["cloud_cover"],
            "rainfall_category": (
                weather_risk_data["rainfall_category"]
            ),
            "weather_risk": weather_risk,
            "terrain_risk": terrain_risk,
            "risk_score": combined_risk,
            "risk_level": risk_level_from_score(
                combined_risk
            ),
            **cost_components
        })

    logging.info(
        "Graph edge feature engineering completed."
    )

    return graph


def export_enriched_graph(
    graph,
    config
):

    records = []

    for edge_id, (u, v, data) in enumerate(
        graph.edges(data=True)
    ):

        record = {
            key: value
            for key, value in data.items()
            if key != "geometry"
        }

        record.update({
            "edge_id": edge_id,
            "u_x": u[0],
            "u_y": u[1],
            "v_x": v[0],
            "v_y": v[1],
            "geometry": data["geometry"]
        })

        records.append(record)

    edges_gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=METRIC_CRS
    )

    output = (
        config.processed_dir /
        "graph_edges_enriched.gpkg"
    )

    edges_gdf.to_file(
        output,
        layer="graph_edges_enriched",
        driver="GPKG"
    )

    logging.info(
        "Enriched graph saved: %s",
        output
    )

    return edges_gdf


def run_astar_search(
    graph,
    source_node,
    destination_node
):

    logging.info(
        "Running A* route optimization..."
    )

    if source_node not in graph:

        raise ValueError(
            "Source node is not in the road graph."
        )

    if destination_node not in graph:

        raise ValueError(
            "Destination node is not in the road graph."
        )

    if not nx.has_path(
        graph,
        source_node,
        destination_node
    ):

        raise nx.NetworkXNoPath(
            "No connected road route exists between "
            "Dehradun and Joshimath in the supplied graph. "
            "Inspect the NH-7 coverage and road intersections."
        )

    route = nx.astar_path(
        graph,
        source_node,
        destination_node,
        heuristic=lambda current, goal: 0.0,
        weight="dynamic_cost"
    )

    logging.info(
        "A* found a route with %d nodes.",
        len(route)
    )

    return route


def calculate_route_metrics(
    graph,
    route_nodes
):

    route_records = []

    totals = {
        "distance_km": 0.0,
        "travel_hours": 0.0,
        "fuel_quantity": 0.0,
        "fuel_cost_inr": 0.0,
        "dynamic_cost": 0.0
    }

    weighted_risk_sum = 0.0

    for sequence, (u, v) in enumerate(
        zip(
            route_nodes[:-1],
            route_nodes[1:]
        ),
        start=1
    ):

        edge = graph[u][v]

        row = {
            key: value
            for key, value in edge.items()
            if key != "geometry"
        }

        row.update({
            "sequence": sequence,
            "from_x": u[0],
            "from_y": u[1],
            "to_x": v[0],
            "to_y": v[1],
            "geometry": edge["geometry"]
        })

        route_records.append(row)

        for key in totals:

            totals[key] += float(
                edge.get(key, 0.0)
            )

        risk = edge.get(
            "risk_score",
            np.nan
        )

        if np.isfinite(risk):

            weighted_risk_sum += (
                risk *
                edge["distance_km"]
            )

    route_gdf = gpd.GeoDataFrame(
        route_records,
        geometry="geometry",
        crs=METRIC_CRS
    )

    total_distance = totals["distance_km"]

    if total_distance > 0:

        totals["risk_score"] = (
            weighted_risk_sum /
            total_distance
        )

    else:

        totals["risk_score"] = np.nan

    totals["risk_level"] = risk_level_from_score(
        totals["risk_score"]
    )

    return route_gdf, totals


def save_optimal_route(
    route_gdf,
    config
):

    output = (
        config.results_dir /
        "best_route.gpkg"
    )

    if route_gdf.empty:

        logging.warning(
            "No route edges available to save."
        )

        return None

    route_gdf.to_file(
        output,
        layer="best_route",
        driver="GPKG"
    )

    logging.info(
        "Optimal route saved: %s",
        output
    )

    return output


def export_route_node_sequence(
    route_nodes,
    config
):

    records = []

    for index, node in enumerate(
        route_nodes,
        start=1
    ):

        records.append({
            "sequence": index,
            "x_utm": node[0],
            "y_utm": node[1]
        })

    pd.DataFrame(
        records
    ).to_csv(
        config.results_dir /
        "astar_node_sequence.csv",
        index=False
    )