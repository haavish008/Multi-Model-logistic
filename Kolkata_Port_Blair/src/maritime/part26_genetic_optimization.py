
import json
import random
import math
import networkx as nx
import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString
from config.port_config import OUTPUT_DIR
from config.route_config import SOURCE_NODE, TARGET_NODE

EDGE_FILE = OUTPUT_DIR / "bathymetry_assessed_edges.gpkg"
NODE_FILE = OUTPUT_DIR / "real_maritime_nodes.gpkg"

OUTPUT_FILE = OUTPUT_DIR / "genetic_best_route.gpkg"
RANKING_FILE = OUTPUT_DIR / "genetic_route_candidates.csv"
REPORT_FILE = OUTPUT_DIR / "genetic_optimization_report.json"

POPULATION_SIZE = 40
GENERATIONS = 60
MUTATION_RATE = 0.25
ELITE_COUNT = 5

def build_graph(edges):
    graph = nx.Graph()

    for _, row in edges.iterrows():
        if row["bathymetry_status"] != "DEPTH_SCREEN_PASSED":
            continue

        distance = float(row["distance_km"])

        if not math.isfinite(distance) or distance <= 0:
            continue

        source = str(row["source"])
        target = str(row["target"])

        graph.add_edge(
            source,
            target,
            distance=distance
        )

    return graph

def route_distance(graph, route):
    return sum(
        graph[a][b]["distance"]
        for a, b in zip(route[:-1], route[1:])
    )

def generate_population(graph, source, target):
    population = []
    attempts = 0

    while len(population) < POPULATION_SIZE and attempts < 5000:
        attempts += 1

        randomized = nx.Graph()

        for u, v, data in graph.edges(data=True):
            multiplier = random.uniform(0.8, 1.3)

            randomized.add_edge(
                u,
                v,
                weight=data["distance"] * multiplier
            )

        try:
            route = nx.shortest_path(
                randomized,
                source,
                target,
                weight="weight"
            )

            if len(route) == len(set(route)):
                population.append(route)

        except nx.NetworkXNoPath:
            continue

    return population

def crossover(parent_a, parent_b):
    common = set(parent_a[1:-1]).intersection(parent_b[1:-1])

    if not common:
        return random.choice([parent_a, parent_b]).copy()

    common_node = random.choice(list(common))

    index_a = parent_a.index(common_node)
    index_b = parent_b.index(common_node)

    child = parent_a[:index_a] + parent_b[index_b:]

    if len(child) != len(set(child)):
        return random.choice([parent_a, parent_b]).copy()

    return child

def mutate(graph, route):
    if len(route) < 3:
        return route.copy()

    index = random.randint(1, len(route) - 2)

    start = route[index - 1]
    end = route[index + 1]

    temporary = graph.copy()

    for u, v in temporary.edges():
        temporary[u][v]["weight"] = (
            temporary[u][v]["distance"]
            * random.uniform(0.8, 1.3)
        )

    try:
        replacement = nx.shortest_path(
            temporary,
            start,
            end,
            weight="weight"
        )

        candidate = route[:index - 1] + replacement + route[index + 2:]

        if len(candidate) == len(set(candidate)):
            return candidate

    except nx.NetworkXNoPath:
        pass

    return route.copy()

def fitness(graph, route):
    try:
        distance = route_distance(graph, route)

        if not math.isfinite(distance) or distance <= 0:
            return math.inf

        return distance

    except (KeyError, ValueError):
        return math.inf

def run_genetic_optimization():
    if SOURCE_NODE is None or TARGET_NODE is None:
        raise ValueError(
            "Configure source and target node IDs in route_config.py."
        )

    edges = gpd.read_file(EDGE_FILE)
    nodes = gpd.read_file(NODE_FILE)

    if edges.empty:
        raise ValueError("Maritime edge dataset is empty.")

    graph = build_graph(edges)

    source = str(SOURCE_NODE)
    target = str(TARGET_NODE)

    if source not in graph or target not in graph:
        raise ValueError("Source or destination is not in the feasible graph.")

    if not nx.has_path(graph, source, target):
        raise ValueError("No depth-screen-passed route connects the ports.")

    population = generate_population(
        graph,
        source,
        target
    )

    if len(population) < 2:
        raise ValueError("Unable to generate multiple route candidates.")

    for generation in range(GENERATIONS):
        population = sorted(
            population,
            key=lambda route: fitness(graph, route)
        )

        next_population = [
            route.copy()
            for route in population[:ELITE_COUNT]
        ]

        while len(next_population) < POPULATION_SIZE:
            parent_a, parent_b = random.choices(
                population[:max(2, len(population) // 2)],
                k=2
            )

            child = crossover(parent_a, parent_b)

            if random.random() < MUTATION_RATE:
                child = mutate(graph, child)

            if (
                child[0] == source
                and child[-1] == target
                and len(child) == len(set(child))
                and all(
                    graph.has_edge(a, b)
                    for a, b in zip(child[:-1], child[1:])
                )
            ):
                next_population.append(child)

        population = next_population

        best = min(
            population,
            key=lambda route: fitness(graph, route)
        )

        print(
            f"Generation {generation + 1}: "
            f"Distance = {fitness(graph, best):.3f} km"
        )

    unique_routes = {}

    for route in population:
        key = tuple(route)
        unique_routes[key] = route

    ranked = sorted(
        unique_routes.values(),
        key=lambda route: fitness(graph, route)
    )

    candidate_records = []

    for index, route in enumerate(ranked, start=1):
        candidate_records.append({
            "route_id": index,
            "distance_km": fitness(graph, route),
            "node_count": len(route),
            "waypoints": " -> ".join(route)
        })

    ranking = pd.DataFrame(candidate_records)
    ranking.to_csv(RANKING_FILE, index=False)

    best_route = ranked[0]

    node_lookup = {
        str(row["node_id"]): row.geometry
        for _, row in nodes.iterrows()
    }

    coordinates = [
        (node_lookup[node].x, node_lookup[node].y)
        for node in best_route
    ]

    route_gdf = gpd.GeoDataFrame(
        [{
            "algorithm": "Genetic Algorithm",
            "distance_km": fitness(graph, best_route),
            "node_count": len(best_route),
            "geometry": LineString(coordinates)
        }],
        geometry="geometry",
        crs=nodes.crs
    )

    route_gdf.to_file(
        OUTPUT_FILE,
        driver="GPKG"
    )

    report = {
        "algorithm": "Genetic Algorithm",
        "generations": GENERATIONS,
        "population_size": POPULATION_SIZE,
        "candidate_count": len(ranked),
        "best_distance_km": fitness(graph, best_route),
        "best_route": best_route,
        "outputs": {
            "best_route": str(OUTPUT_FILE),
            "candidate_ranking": str(RANKING_FILE)
        },
        "limitations": [
            "Optimization uses distance as its fitness objective.",
            "Randomized candidate generation does not guarantee a global optimum.",
            "Bathymetry screening is preliminary and does not certify navigability.",
            "Environmental exposure and fuel costs are not included in fitness."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_genetic_optimization()
