def build_road_graph(roads, config):

    logging.info(
        "Building road graph from cleaned road network..."
    )

    merged_roads = unary_union(
        list(roads.geometry)
    )

    if merged_roads.geom_type == "LineString":

        line_parts = [merged_roads]

    elif merged_roads.geom_type == "MultiLineString":

        line_parts = list(
            merged_roads.geoms
        )

    elif merged_roads.geom_type == "GeometryCollection":

        line_parts = [
            geom
            for geom in merged_roads.geoms
            if geom.geom_type == "LineString"
        ]

    else:

        raise ValueError(
            "Unexpected geometry type during graph construction: "
            + merged_roads.geom_type
        )

    graph = nx.Graph()

    node_coordinates = set()

    edge_records = []

    for line in line_parts:

        coordinates = list(
            line.coords
        )

        for start, end in zip(
            coordinates[:-1],
            coordinates[1:]
        ):

            u = (
                round(float(start[0]), 2),
                round(float(start[1]), 2)
            )

            v = (
                round(float(end[0]), 2),
                round(float(end[1]), 2)
            )

            if u == v:

                continue

            segment = LineString([
                u,
                v
            ])

            length_m = float(
                segment.length
            )

            if length_m <= 0:

                continue

            node_coordinates.add(u)

            node_coordinates.add(v)

            graph.add_node(
                u,
                x=u[0],
                y=u[1]
            )

            graph.add_node(
                v,
                x=v[0],
                y=v[1]
            )

            if graph.has_edge(u, v):

                continue

            graph.add_edge(
                u,
                v,
                length_m=length_m,
                geometry=segment
            )

            edge_records.append({

                "u_x": u[0],
                "u_y": u[1],
                "v_x": v[0],
                "v_y": v[1],
                "length_m": length_m,
                "geometry": segment

            })

    if graph.number_of_edges() == 0:

        raise RuntimeError(
            "No graph edges were generated. "
            "Check the road geometries."
        )

    node_records = []

    node_id_lookup = {}

    for index, coordinate in enumerate(
        sorted(node_coordinates)
    ):

        node_id_lookup[coordinate] = index

        node_records.append({

            "node_id": index,
            "x": coordinate[0],
            "y": coordinate[1],
            "geometry": Point(
                coordinate
            )

        })

    nodes_gdf = gpd.GeoDataFrame(
        node_records,
        geometry="geometry",
        crs=METRIC_CRS
    )

    for u, v, data in graph.edges(
        data=True
    ):

        data["u_id"] = node_id_lookup[u]

        data["v_id"] = node_id_lookup[v]

    edges_gdf = gpd.GeoDataFrame(
        edge_records,
        geometry="geometry",
        crs=METRIC_CRS
    )

    edges_gdf["edge_id"] = np.arange(
        len(edges_gdf)
    )

    nodes_gdf.to_file(
        config.processed_dir /
        "graph_nodes.gpkg",
        layer="graph_nodes",
        driver="GPKG"
    )

    edges_gdf.to_file(
        config.processed_dir /
        "graph_edges.gpkg",
        layer="graph_edges",
        driver="GPKG"
    )

    logging.info(
        "Graph construction completed."
    )

    logging.info(
        "Graph nodes: %d",
        graph.number_of_nodes()
    )

    logging.info(
        "Graph edges: %d",
        graph.number_of_edges()
    )

    logging.info(
        "Connected components: %d",
        nx.number_connected_components(graph)
    )

    return graph, nodes_gdf, edges_gdf


def find_nearest_graph_node(
    graph,
    point,
    maximum_distance_m
):

    nearest_node = None

    nearest_distance = float("inf")

    for node in graph.nodes:

        distance = math.hypot(
            node[0] - point.x,
            node[1] - point.y
        )

        if distance < nearest_distance:

            nearest_distance = distance

            nearest_node = node

    if nearest_node is None:

        raise RuntimeError(
            "Graph does not contain any nodes."
        )

    if nearest_distance > maximum_distance_m:

        raise ValueError(
            "Nearest road node is too far from the requested location. "
            f"Distance: {nearest_distance:.2f} metres. "
            "Check the road coverage and coordinate reference system."
        )

    return nearest_node, nearest_distance


def get_projected_location(name):

    location = LOCATIONS[name]

    transformer = Transformer.from_crs(
        SOURCE_CRS,
        METRIC_CRS,
        always_xy=True
    )

    x, y = transformer.transform(
        location["longitude"],
        location["latitude"]
    )

    return Point(x, y)


def snap_corridor_endpoints(graph, config):

    origin_point = get_projected_location(
        ORIGIN_NAME
    )

    destination_point = get_projected_location(
        DESTINATION_NAME
    )

    source_node, source_distance = find_nearest_graph_node(
        graph,
        origin_point,
        config.max_snap_distance_m
    )

    target_node, target_distance = find_nearest_graph_node(
        graph,
        destination_point,
        config.max_snap_distance_m
    )

    logging.info(
        "Dehradun snap distance: %.2f m",
        source_distance
    )

    logging.info(
        "Joshimath snap distance: %.2f m",
        target_distance
    )

    if source_node == target_node:

        raise RuntimeError(
            "Origin and destination snapped to the same graph node. "
            "Check the input road network and CRS."
        )

    return source_node, target_node


def load_raster_information(path):

    if path is None:

        return None

    try:

        with rasterio.open(path) as src:

            information = {
                "path": str(path),
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
                "bounds": tuple(src.bounds),
                "nodata": src.nodata
            }

        logging.info(
            "Raster information: %s",
            information
        )

        return information

    except Exception:

        logging.exception(
            "Could not inspect raster: %s",
            path
        )

        return None


def sample_raster_value(
    raster_path,
    longitude,
    latitude
):

    if raster_path is None:

        return float("nan")

    try:

        with rasterio.open(raster_path) as src:

            x = longitude

            y = latitude

            if src.crs is not None:

                transformer = Transformer.from_crs(
                    SOURCE_CRS,
                    src.crs,
                    always_xy=True
                )

                x, y = transformer.transform(
                    longitude,
                    latitude
                )

            sample = next(
                src.sample([(x, y)])
            )

            value = float(
                sample[0]
            )

            if src.nodata is not None:

                if value == src.nodata:

                    return float("nan")

            if not np.isfinite(value):

                return float("nan")

            return value

    except Exception:

        logging.exception(
            "Raster sampling failed: %s",
            raster_path
        )

        return float("nan")


def calculate_slope_from_dem(
    dem_path,
    output_path
):

    if dem_path is None:

        logging.warning(
            "DEM is unavailable. Slope cannot be derived."
        )

        return None

    logging.info(
        "Calculating slope from DEM..."
    )

    try:

        with rasterio.open(dem_path) as src:

            elevation = src.read(
                1,
                masked=True
            ).astype("float64")

            transform = src.transform

            x_resolution = abs(
                transform.a
            )

            y_resolution = abs(
                transform.e
            )

            elevation_array = np.asarray(
                elevation.filled(np.nan),
                dtype=float
            )

            gradient_y, gradient_x = np.gradient(
                elevation_array,
                y_resolution,
                x_resolution
            )

            slope = np.degrees(
                np.arctan(
                    np.sqrt(
                        gradient_x ** 2 +
                        gradient_y ** 2
                    )
                )
            )

            profile = src.profile.copy()

            profile.update({
                "dtype": "float32",
                "count": 1,
                "nodata": -9999,
                "compress": "lzw"
            })

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as destination:

                destination.write(
                    np.where(
                        np.isfinite(slope),
                        slope,
                        -9999
                    ).astype("float32"),
                    1
                )

        logging.info(
            "Slope raster saved: %s",
            output_path
        )

        return output_path

    except Exception:

        logging.exception(
            "Slope calculation failed."
        )

        return None


def calculate_aspect_from_dem(
    dem_path,
    output_path
):

    if dem_path is None:

        return None

    logging.info(
        "Calculating aspect from DEM..."
    )

    try:

        with rasterio.open(dem_path) as src:

            elevation = src.read(
                1,
                masked=True
            ).astype("float64")

            transform = src.transform

            x_resolution = abs(
                transform.a
            )

            y_resolution = abs(
                transform.e
            )

            z = np.asarray(
                elevation.filled(np.nan),
                dtype=float
            )

            gradient_y, gradient_x = np.gradient(
                z,
                y_resolution,
                x_resolution
            )

            aspect = (
                np.degrees(
                    np.arctan2(
                        -gradient_x,
                        gradient_y
                    )
                ) + 360
            ) % 360

            profile = src.profile.copy()

            profile.update({
                "dtype": "float32",
                "count": 1,
                "nodata": -9999,
                "compress": "lzw"
            })

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as destination:

                destination.write(
                    np.where(
                        np.isfinite(aspect),
                        aspect,
                        -9999
                    ).astype("float32"),
                    1
                )

        logging.info(
            "Aspect raster saved: %s",
            output_path
        )

        return output_path

    except Exception:

        logging.exception(
            "Aspect calculation failed."
        )

        return None


def calculate_tri_from_dem(
    dem_path,
    output_path
):

    if dem_path is None:

        return None

    logging.info(
        "Calculating Terrain Ruggedness Index..."
    )

    try:

        from scipy.ndimage import generic_filter

        with rasterio.open(dem_path) as src:

            elevation = src.read(
                1,
                masked=True
            ).astype("float64")

            z = np.asarray(
                elevation.filled(np.nan),
                dtype=float
            )

            def local_tri(values):

                center = values[len(values) // 2]

                differences = values - center

                differences = differences[
                    np.isfinite(differences)
                ]

                if len(differences) == 0:

                    return np.nan

                return np.sqrt(
                    np.sum(differences ** 2)
                )

            tri = generic_filter(
                z,
                local_tri,
                size=3,
                mode="nearest"
            )

            profile = src.profile.copy()

            profile.update({
                "dtype": "float32",
                "count": 1,
                "nodata": -9999,
                "compress": "lzw"
            })

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as destination:

                destination.write(
                    np.where(
                        np.isfinite(tri),
                        tri,
                        -9999
                    ).astype("float32"),
                    1
                )

        logging.info(
            "TRI raster saved: %s",
            output_path
        )

        return output_path

    except ImportError:

        logging.warning(
            "SciPy is required to derive TRI. "
            "Install scipy or supply tri.tif."
        )

        return None

    except Exception:

        logging.exception(
            "TRI calculation failed."
        )

        return None


def sample_terrain_features(
    longitude,
    latitude,
    inputs
):

    elevation = sample_raster_value(
        inputs.get("dem"),
        longitude,
        latitude
    )

    slope = sample_raster_value(
        inputs.get("slope"),
        longitude,
        latitude
    )

    aspect = sample_raster_value(
        inputs.get("aspect"),
        longitude,
        latitude
    )

    tri = sample_raster_value(
        inputs.get("tri"),
        longitude,
        latitude
    )

    if not np.isfinite(elevation):

        nearest_location = min(
            LOCATIONS.items(),
            key=lambda item: math.hypot(
                item[1]["longitude"] - longitude,
                item[1]["latitude"] - latitude
            )
        )

        elevation = float(
            nearest_location[1]["elevation_m"]
        )

    return {
        "elevation_m": elevation,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "tri": tri
    }


def prepare_terrain_inputs(
    inputs,
    config
):

    dem = inputs.get("dem")

    if dem is None:

        logging.warning(
            "DEM file is missing. "
            "Detailed terrain analysis will be incomplete."
        )

        return inputs

    load_raster_information(dem)

    if inputs.get("slope") is None:

        derived_slope = calculate_slope_from_dem(
            dem,
            config.processed_dir /
            "slope_derived.tif"
        )

        inputs["slope"] = derived_slope

    if inputs.get("aspect") is None:

        derived_aspect = calculate_aspect_from_dem(
            dem,
            config.processed_dir /
            "aspect_derived.tif"
        )

        inputs["aspect"] = derived_aspect

    if inputs.get("tri") is None:

        derived_tri = calculate_tri_from_dem(
            dem,
            config.processed_dir /
            "tri_derived.tif"
        )

        inputs["tri"] = derived_tri

    return inputs