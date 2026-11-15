
import math
import networkx as nx


def calculate_environmental_risk(
    rainfall_mm,
    slope_deg,
    landslide_risk,
    rainfall_thresholds=(15.0, 50.0)
):

    rainfall_mm = max(0.0, float(rainfall_mm))
    slope_deg = max(0.0, float(slope_deg))
    landslide_risk = max(
        0.0,
        min(1.0, float(landslide_risk))
    )

    low, high = rainfall_thresholds

    if rainfall_mm < low:
        rainfall_score = 0.2
    elif rainfall_mm <= high:
        rainfall_score = 0.5
    else:
        rainfall_score = 1.0

    slope_score = min(
        1.0,
        slope_deg / 45.0
    )

    combined_risk = (
        0.40 * rainfall_score
        + 0.25 * slope_score
        + 0.35 * landslide_risk
    )

    return min(
        1.0,
        max(0.0, combined_risk)
    )


def update_graph_environment(
    graph,
    weather_by_edge,
    blocked_edges=None,
    risk_weight=1.0,
    time_weight=1.0
):

    blocked_edges = blocked_edges or set()

    for u, v, attributes in graph.edges(data=True):

        weather = weather_by_edge.get(
            (u, v),
            weather_by_edge.get((v, u), {})
        )

        rainfall = weather.get(
            "rainfall_mm",
            attributes.get("rainfall_mm", 0.0)
        )

        slope = attributes.get(
            "slope_deg",
            0.0
        )

        landslide_risk = attributes.get(
            "landslide_risk",
            0.0
        )

        risk = calculate_environmental_risk(
            rainfall,
            slope,
            landslide_risk
        )

        distance_m = float(
            attributes.get("length_m", 0.0)
        )

        base_speed = float(
            attributes.get("base_speed_kmh", 25.0)
        )

        if base_speed <= 0:
            raise ValueError(
                f"Invalid speed on edge {u}, {v}"
            )

        # Configurable slowdown based on environmental risk.
        adjusted_speed = base_speed * (
            1.0 - 0.40 * risk
        )

        adjusted_speed = max(
            5.0,
            adjusted_speed
        )

        travel_hours = (
            distance_m / 1000.0
        ) / adjusted_speed

        distance_km = distance_m / 1000.0

        # Risk penalty is expressed in equivalent km.
        dynamic_cost = (
            time_weight * travel_hours
            + risk_weight * risk * distance_km
        )

        is_blocked = (
            (u, v) in blocked_edges
            or (v, u) in blocked_edges
        )

        attributes["rainfall_mm"] = rainfall
        attributes["dynamic_risk"] = risk
        attributes["adjusted_speed_kmh"] = adjusted_speed
        attributes["travel_hours"] = travel_hours
        attributes["dynamic_cost"] = (
            math.inf if is_blocked else dynamic_cost
        )
        attributes["is_blocked"] = is_blocked

    return graph


def find_route_after_environment_update(
    graph,
    source,
    destination
):

    if source not in graph or destination not in graph:
        raise ValueError("Invalid source or destination node.")

    if not nx.has_path(graph, source, destination):
        return None

    return nx.astar_path(
        graph,
        source,
        destination,
        weight="dynamic_cost"
    )
