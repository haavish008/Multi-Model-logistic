def calculate_route_cost(graph, route):

    if len(route) < 2:
        return float("inf")

    total = 0.0

    for u, v in zip(route[:-1], route[1:]):

        if not graph.has_edge(u, v):
            return float("inf")

        total += edge_cost(graph, u, v)

    return total


def generate_initial_population(
    graph,
    start,
    goal,
    population_size=30
):

    population = []

    try:
        paths = nx.shortest_simple_paths(
            graph,
            start,
            goal,
            weight="dynamic_cost"
        )

        for path in itertools.islice(paths, population_size):
            population.append(path)

    except nx.NetworkXNoPath:
        raise ValueError("No feasible path exists for GA.")

    while len(population) < population_size:
        population.append(random.choice(population).copy())

    return population


def mutate_route(graph, route, mutation_rate=0.20):

    if len(route) < 4:
        return route.copy()

    if random.random() > mutation_rate:
        return route.copy()

    i = random.randrange(0, len(route) - 2)
    j = random.randrange(i + 2, len(route))

    try:
        replacement = nx.shortest_path(
            graph,
            route[i],
            route[j],
            weight="dynamic_cost"
        )

        new_route = (
            route[:i]
            + replacement
            + route[j + 1:]
        )

        if all(
            graph.has_edge(u, v)
            for u, v in zip(new_route[:-1], new_route[1:])
        ):
            return new_route

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return route.copy()


def crossover_routes(parent1, parent2):

    common_nodes = list(
        set(parent1[1:-1]).intersection(parent2[1:-1])
    )

    if not common_nodes:
        return parent1.copy()

    pivot = random.choice(common_nodes)

    child = (
        parent1[:parent1.index(pivot)]
        + parent2[parent2.index(pivot):]
    )

    return child


def genetic_algorithm(
    graph,
    start,
    goal,
    population_size=30,
    generations=50,
    mutation_rate=0.20
):

    population = generate_initial_population(
        graph,
        start,
        goal,
        population_size
    )

    history = []

    for generation in range(generations):

        population.sort(
            key=lambda route: calculate_route_cost(graph, route)
        )

        best_route = population[0]

        history.append({
            "generation": generation + 1,
            "best_cost": calculate_route_cost(graph, best_route)
        })

        elite_count = max(2, population_size // 10)

        new_population = [
            route.copy()
            for route in population[:elite_count]
        ]

        while len(new_population) < population_size:

            parent1, parent2 = random.sample(
                population[:max(2, population_size // 2)],
                2
            )

            child = crossover_routes(parent1, parent2)

            child = mutate_route(
                graph,
                child,
                mutation_rate
            )

            if all(
                graph.has_edge(u, v)
                for u, v in zip(child[:-1], child[1:])
            ):
                new_population.append(child)

            else:
                new_population.append(parent1.copy())

        population = new_population

    population.sort(
        key=lambda route: calculate_route_cost(graph, route)
    )

    return population[0], pd.DataFrame(history)


