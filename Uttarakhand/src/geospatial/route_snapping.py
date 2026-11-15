
import math
import networkx as nx


def validate_graph_coordinates(graph):

    missing = []

    for node, attributes in graph.nodes(data=True):

        if "x" not in attributes or "y" not in attributes:
            missing.append(node)
            continue

        try:
            x = float(attributes["x"])
            y = float(attributes["y"])

            if not math.isfinite(x) or not math.isfinite(y):
                missing.append(node)

        except (TypeError, ValueError):
            missing.append(node)

    if missing:
        raise ValueError(
            f"{len(missing)} graph nodes have invalid coordinates. "
            f"Examples: {missing[:5]}"
        )


def haversine_distance_km(
    lon1,
    lat1,
    lon2,
    lat2
):

    radius = 6371.0

    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    lon2 = math.radians(lon2)
    lat2 = math.radians(lat2)

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return 2 * radius * math.asin(
        math.sqrt(a)
    )


def nearest_graph_node(
    graph,
    longitude,
    latitude,
    coordinate_crs="EPSG:4326"
):

    validate_graph_coordinates(graph)

    if coordinate_crs == "EPSG:4326":

        # Graph coordinates must be projected metres.
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32644",
            always_xy=True
        )

        target_x, target_y = transformer.transform(
            longitude,
            latitude
        )

    elif coordinate_crs == "EPSG:32644":

        target_x = longitude
        target_y = latitude

    else:
        raise ValueError(
            "Unsupported coordinate CRS."
        )

    best_node = None
    best_distance = float("inf")

    for node, attributes in graph.nodes(data=True):

        x = float(attributes["x"])
        y = float(attributes["y"])

        distance = math.hypot(
            x - target_x,
            y - target_y
        )

        if distance < best_distance:
            best_distance = distance
            best_node = node

    return {
        "node": best_node,
        "distance_m": best_distance,
        "target_x": target_x,
        "target_y": target_y
    }


def snap_corridor_locations(
    graph,
    locations,
    maximum_snap_distance_m=2000
):

    results = {}

    for name, coordinates in locations.items():

        longitude, latitude = coordinates

        result = nearest_graph_node(
            graph,
            longitude,
            latitude
        )

        if result["distance_m"] > maximum_snap_distance_m:

            raise ValueError(
                f"{name} snap distance is "
                f"{result['distance_m']:.1f} metres. "
                "Check road coverage, CRS and coordinates."
            )

        results[name] = result

    return results


def validate_corridor_connectivity(
    graph,
    source_node,
    destination_node
):

    if source_node == destination_node:
        raise ValueError(
            "Source and destination snapped to the same node. "
            "Check the coordinates and graph coverage."
        )

    if source_node not in graph:
        raise ValueError("Source node not found.")

    if destination_node not in graph:
        raise ValueError("Destination node not found.")

    connected = nx.has_path(
        graph,
        source_node,
        destination_node
    )

    if not connected:
        raise nx.NetworkXNoPath(
            "The source and destination are disconnected."
        )

    return {
        "connected": True,
        "source_node": source_node,
        "destination_node": destination_node,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges()
    }


def print_snap_report(results):

    print("\nLOCATION SNAPPING REPORT")

    for name, result in results.items():

        print(
            f"{name}: node={result['node']}, "
            f"snap_distance={result['distance_m']:.2f} m"
        )


if __name__ == "__main__":

    # Example only: use your actual graph object.
    # Import your graph loader and create G before running.

    LOCATIONS = {
        "Dehradun": (78.0322, 30.3165),
        "Joshimath": (79.5644, 30.5552)
    }

    print(
        "Import your existing graph loader, load G, "
        "then call snap_corridor_locations(G, LOCATIONS)."
    )
