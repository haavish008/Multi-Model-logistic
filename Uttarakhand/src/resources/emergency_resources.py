
import networkx as nx
import pandas as pd


def find_nearest_resources(
    graph,
    source_node,
    resources,
    resource_type=None,
    top_n=5
):

    if source_node not in graph:
        raise ValueError("Source node is missing from graph.")

    required = {
        "resource_id",
        "resource_type",
        "nearest_node"
    }

    missing = required - set(resources.columns)

    if missing:
        raise ValueError(
            f"Missing resource columns: {missing}"
        )

    selected = resources.copy()

    if resource_type:
        selected = selected[
            selected["resource_type"]
            .str.lower()
            == resource_type.lower()
        ]

    distances = nx.single_source_dijkstra_path_length(
        graph,
        source_node,
        weight="length_m"
    )

    results = []

    for _, resource in selected.iterrows():

        node = resource["nearest_node"]

        if node not in distances:
            continue

        results.append({
            "resource_id": resource["resource_id"],
            "resource_type": resource["resource_type"],
            "nearest_node": node,
            "road_distance_km": (
                distances[node] / 1000.0
            )
        })

    result = pd.DataFrame(results)

    if result.empty:
        return result

    return result.sort_values(
        "road_distance_km"
    ).head(top_n).reset_index(drop=True)


def calculate_resource_accessibility(
    graph,
    source_node,
    resources
):

    resource_types = (
        resources["resource_type"]
        .dropna()
        .unique()
    )

    results = []

    for resource_type in resource_types:

        nearest = find_nearest_resources(
            graph,
            source_node,
            resources,
            resource_type=resource_type,
            top_n=1
        )

        if not nearest.empty:

            row = nearest.iloc[0]

            results.append({
                "resource_type": resource_type,
                "nearest_resource_id": row["resource_id"],
                "distance_km": row["road_distance_km"]
            })

    return pd.DataFrame(results)
