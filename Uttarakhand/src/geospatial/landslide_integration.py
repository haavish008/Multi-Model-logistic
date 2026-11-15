def discover_landslide_file(config):

    possible_names = [
        "uttarakhand_landslides.gpkg",
        "landslides.gpkg",
        "landslide_inventory.gpkg",
        "landslides.shp",
        "BUFFER_UTTARAKHAND_LANDSLIDE.shp"
    ]

    for name in possible_names:

        path = find_input_file(config, name)

        if path is not None:

            logging.info(
                "Landslide dataset found: %s",
                path
            )

            return path

    logging.warning(
        "No landslide inventory found. "
        "Landslide risk will not be calculated from an inventory."
    )

    return None


def preprocess_landslide_inventory(
    landslide_path,
    config
):

    if landslide_path is None:

        return None

    try:

        landslides = gpd.read_file(
            landslide_path
        )

        if landslides.empty:

            logging.warning(
                "Landslide inventory is empty."
            )

            return None

        if landslides.crs is None:

            raise ValueError(
                "Landslide inventory has no CRS."
            )

        landslides = landslides[
            landslides.geometry.notna()
        ].copy()

        landslides = landslides[
            ~landslides.geometry.is_empty
        ].copy()

        landslides = landslides.to_crs(
            METRIC_CRS
        )

        landslides = landslides[
            landslides.geometry.is_valid
        ].copy()

        landslides = landslides.drop_duplicates(
            subset=["geometry"]
        )

        landslides["landslide_id"] = np.arange(
            len(landslides)
        )

        output = (
            config.processed_dir /
            "landslides_cleaned.gpkg"
        )

        landslides.to_file(
            output,
            layer="landslides_cleaned",
            driver="GPKG"
        )

        logging.info(
            "Cleaned landslide features: %d",
            len(landslides)
        )

        return landslides

    except Exception:

        logging.exception(
            "Landslide preprocessing failed."
        )

        return None


def calculate_landslide_edge_risk(
    graph,
    landslides,
    buffer_distance_m=500
):

    if landslides is None:

        logging.warning(
            "No landslide inventory. "
            "Existing graph risk values will be retained."
        )

        return graph

    logging.info(
        "Calculating landslide proximity risk..."
    )

    try:

        spatial_index = landslides.sindex

    except Exception:

        spatial_index = None

    for u, v, data in graph.edges(
        data=True
    ):

        geometry = data["geometry"]

        search_area = geometry.buffer(
            buffer_distance_m
        )

        if spatial_index is not None:

            possible_indices = list(
                spatial_index.intersection(
                    search_area.bounds
                )
            )

            candidates = landslides.iloc[
                possible_indices
            ]

        else:

            candidates = landslides

        if candidates.empty:

            count = 0

        else:

            intersects = candidates.geometry.intersects(
                search_area
            )

            count = int(
                intersects.sum()
            )

        landslide_risk = float(
            1.0 - math.exp(-count / 3.0)
        )

        data["landslide_count_nearby"] = count

        data["landslide_risk"] = landslide_risk

        existing_risk = data.get(
            "risk_score",
            np.nan
        )

        if np.isfinite(existing_risk):

            data["risk_score"] = float(
                np.clip(
                    0.75 * existing_risk +
                    0.25 * landslide_risk,
                    0,
                    1
                )
            )

        else:

            data["risk_score"] = landslide_risk

        data["risk_level"] = risk_level_from_score(
            data["risk_score"]
        )

        data["dynamic_cost"] = (
            data.get("dynamic_cost", 0.0) +
            5.0 * landslide_risk
        )

    logging.info(
        "Landslide proximity risk calculation completed."
    )

    return graph


def export_landslide_enriched_graph(
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

    result = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=METRIC_CRS
    )

    result.to_file(
        config.processed_dir /
        "graph_edges_landslide.gpkg",
        layer="graph_edges_landslide",
        driver="GPKG"
    )

    return result


def calculate_path_cost(
    graph,
    path
):

    total_cost = 0.0

    for u, v in zip(
        path[:-1],
        path[1:]
    ):

        total_cost += float(
            graph[u][v].get(
                "dynamic_cost",
                graph[u][v].get("length_m", 1.0)
            )
        )

    return total_cost


def generate_alternative_routes(
    graph,
    source,
    target,
    number_of_routes=5
):

    logging.info(
        "Generating alternative routes..."
    )

    if not nx.has_path(
        graph,
        source,
        target
    ):

        raise nx.NetworkXNoPath(
            "No route exists between the selected endpoints."
        )

    routes = []

    generator = nx.shortest_simple_paths(
        graph,
        source,
        target,
        weight="dynamic_cost"
    )

    for index, path in enumerate(
        generator
    ):

        if index >= number_of_routes:

            break

        routes.append({
            "route_id": index + 1,
            "nodes": path,
            "cost": calculate_path_cost(
                graph,
                path
            ),
            "distance_km": sum(
                graph[u][v].get(
                    "distance_km",
                    graph[u][v]["length_m"] / 1000.0
                )
                for u, v in zip(
                    path[:-1],
                    path[1:]
                )
            )
        })

    logging.info(
        "Generated %d alternative routes.",
        len(routes)
    )

    return routes


def export_alternative_routes(
    routes,
    config
):

    rows = []

    for route in routes:

        rows.append({
            "route_id": route["route_id"],
            "distance_km": route["distance_km"],
            "dynamic_cost": route["cost"],
            "number_of_nodes": len(
                route["nodes"]
            )
        })

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        config.results_dir /
        "alternative_routes.csv",
        index=False
    )

    return output


def export_alternative_route_geometries(
    graph,
    routes,
    config
):

    records = []

    for route in routes:

        geometries = [
            graph[u][v]["geometry"]
            for u, v in zip(
                route["nodes"][:-1],
                route["nodes"][1:]
            )
        ]

        if not geometries:

            continue

        merged_geometry = linemerge(
            unary_union(geometries)
        )

        records.append({
            "route_id": route["route_id"],
            "distance_km": route["distance_km"],
            "dynamic_cost": route["cost"],
            "geometry": merged_geometry
        })

    if not records:

        return None

    route_gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=METRIC_CRS
    )

    route_gdf.to_file(
        config.results_dir /
        "alternative_routes.gpkg",
        layer="alternative_routes",
        driver="GPKG"
    )

    return route_gdf


@dataclass
class GeneticAlgorithmConfig:

    population_size: int = 30

    generations: int = 40

    mutation_rate: float = 0.20

    elite_count: int = 4

    tournament_size: int = 3

    random_seed: int = 42


def initialize_ga_population(
    candidate_routes,
    population_size,
    random_generator
):

    if not candidate_routes:

        raise ValueError(
            "GA requires candidate routes."
        )

    population = []

    for _ in range(
        population_size
    ):

        selected = random_generator.choice(
            candidate_routes
        )

        population.append(
            list(selected["nodes"])
        )

    return population


def genetic_route_fitness(
    graph,
    route
):

    if len(route) < 2:

        return float("inf")

    if not all(
        graph.has_edge(u, v)
        for u, v in zip(
            route[:-1],
            route[1:]
        )
    ):

        return float("inf")

    return calculate_path_cost(
        graph,
        route
    )


def ga_tournament_selection(
    population,
    graph,
    tournament_size,
    random_generator
):

    selected = random_generator.sample(
        population,
        min(
            tournament_size,
            len(population)
        )
    )

    return min(
        selected,
        key=lambda route: genetic_route_fitness(
            graph,
            route
        )
    )


def ga_crossover(
    graph,
    parent_one,
    parent_two,
    random_generator
):

    shared_nodes = list(
        set(parent_one[1:-1]).intersection(
            parent_two[1:-1]
        )
    )

    if not shared_nodes:

        return list(parent_one)

    join_node = random_generator.choice(
        shared_nodes
    )

    first_index = parent_one.index(
        join_node
    )

    second_index = parent_two.index(
        join_node
    )

    child = (
        parent_one[:first_index] +
        parent_two[second_index:]
    )

    if not all(
        graph.has_edge(u, v)
        for u, v in zip(
            child[:-1],
            child[1:]
        )
    ):

        return list(parent_one)

    cleaned = []

    visited = set()

    for node in child:

        if node not in visited:

            cleaned.append(node)

            visited.add(node)

    if cleaned[0] != parent_one[0]:

        return list(parent_one)

    if cleaned[-1] != parent_one[-1]:

        return list(parent_one)

    return cleaned


def ga_mutation(
    graph,
    individual,
    candidate_routes,
    mutation_rate,
    random_generator
):

    if random_generator.random() > mutation_rate:

        return list(individual)

    if not candidate_routes:

        return list(individual)

    alternative = random_generator.choice(
        candidate_routes
    )

    mutated = list(
        alternative["nodes"]
    )

    if (
        mutated[0] == individual[0] and
        mutated[-1] == individual[-1]
    ):

        return mutated

    return list(individual)


def run_genetic_algorithm(
    graph,
    source,
    target,
    candidate_routes,
    ga_config=None
):

    if ga_config is None:

        ga_config = GeneticAlgorithmConfig()

    random_generator = np.random.default_rng(
        ga_config.random_seed
    )

    if not candidate_routes:

        raise ValueError(
            "No candidate routes supplied to Genetic Algorithm."
        )

    population = initialize_ga_population(
        candidate_routes,
        ga_config.population_size,
        random_generator
    )

    history = []

    for generation in range(
        ga_config.generations
    ):

        population = sorted(
            population,
            key=lambda route: genetic_route_fitness(
                graph,
                route
            )
        )

        best_fitness = genetic_route_fitness(
            graph,
            population[0]
        )

        history.append({
            "generation": generation + 1,
            "best_fitness": best_fitness
        })

        logging.info(
            "GA generation %d | Best cost %.4f",
            generation + 1,
            best_fitness
        )

        elites = [
            list(route)
            for route in population[
                :ga_config.elite_count
            ]
        ]

        new_population = elites

        while len(new_population) < ga_config.population_size:

            parent_one = ga_tournament_selection(
                population,
                graph,
                ga_config.tournament_size,
                random_generator
            )

            parent_two = ga_tournament_selection(
                population,
                graph,
                ga_config.tournament_size,
                random_generator
            )

            child = ga_crossover(
                graph,
                parent_one,
                parent_two,
                random_generator
            )

            child = ga_mutation(
                graph,
                child,
                candidate_routes,
                ga_config.mutation_rate,
                random_generator
            )

            if (
                child[0] == source and
                child[-1] == target
            ):

                new_population.append(
                    child
                )

            else:

                new_population.append(
                    list(parent_one)
                )

        population = new_population

    best_route = min(
        population,
        key=lambda route: genetic_route_fitness(
            graph,
            route
        )
    )

    history_df = pd.DataFrame(
        history
    )

    return best_route, history_df


def export_genetic_algorithm_results(
    graph,
    best_route,
    history,
    config
):

    history.to_csv(
        config.results_dir /
        "genetic_algorithm_history.csv",
        index=False
    )

    metrics = {
        "algorithm": "Genetic Algorithm",
        "distance_km": sum(
            graph[u][v].get(
                "distance_km",
                graph[u][v]["length_m"] / 1000.0
            )
            for u, v in zip(
                best_route[:-1],
                best_route[1:]
            )
        ),
        "dynamic_cost": calculate_path_cost(
            graph,
            best_route
        ),
        "number_of_nodes": len(
            best_route
        )
    }

    pd.DataFrame([
        metrics
    ]).to_csv(
        config.results_dir /
        "genetic_algorithm_best_route.csv",
        index=False
    )

    return metrics