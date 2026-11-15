
import json
import pandas as pd
import geopandas as gpd

from config.port_config import OUTPUT_DIR
from config.vessel_config import VESSEL_CONFIG

CANDIDATE_FILE = OUTPUT_DIR / "genetic_route_candidates.csv"
EDGE_FILE = OUTPUT_DIR / "environmental_edge_assessment.gpkg"

OUTPUT_FILE = OUTPUT_DIR / "fuel_time_route_estimates.csv"
REPORT_FILE = OUTPUT_DIR / "fuel_time_report.json"

KNOT_TO_KMH = 1.852

def run_fuel_time_estimation():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError("Run Part 26 before Part 28.")

    if not EDGE_FILE.exists():
        raise FileNotFoundError("Environmental edge dataset is missing.")

    speed = VESSEL_CONFIG.get("cruising_speed_knots")
    fuel_rate = VESSEL_CONFIG.get(
        "fuel_consumption_tonnes_per_hour"
    )

    if speed is None or fuel_rate is None:
        raise ValueError(
            "Configure cruising_speed_knots and "
            "fuel_consumption_tonnes_per_hour in vessel_config.py."
        )

    if speed <= 0 or fuel_rate <= 0:
        raise ValueError("Vessel speed and fuel rate must be positive.")

    candidates = pd.read_csv(CANDIDATE_FILE)
    edges = gpd.read_file(EDGE_FILE)

    required = ["source", "target", "distance_km"]

    if any(column not in edges.columns for column in required):
        raise ValueError("Required edge columns are missing.")

    edge_lookup = {}

    for _, row in edges.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        distance = float(row["distance_km"])

        if distance <= 0:
            continue

        edge_lookup[(source, target)] = distance
        edge_lookup[(target, source)] = distance

    speed_kmh = float(speed) * KNOT_TO_KMH

    records = []

    for _, candidate in candidates.iterrows():
        route = str(candidate["waypoints"]).split(" -> ")

        route_distance = 0
        missing_edges = []

        for source, target in zip(route[:-1], route[1:]):
            distance = edge_lookup.get((source, target))

            if distance is None:
                missing_edges.append(f"{source}->{target}")
            else:
                route_distance += distance

        if missing_edges:
            records.append({
                "route_id": candidate["route_id"],
                "status": "INCOMPLETE_EDGE_DATA",
                "missing_edges": ";".join(missing_edges)
            })
            continue

        travel_hours = route_distance / speed_kmh
        fuel_tonnes = travel_hours * float(fuel_rate)

        records.append({
            "route_id": candidate["route_id"],
            "status": "ESTIMATED",
            "distance_km": route_distance,
            "speed_knots": float(speed),
            "speed_kmh": speed_kmh,
            "travel_time_hours": travel_hours,
            "travel_time_days": travel_hours / 24,
            "fuel_consumption_tonnes_per_hour": float(fuel_rate),
            "estimated_fuel_tonnes": fuel_tonnes,
            "node_count": len(route),
            "waypoints": " -> ".join(route)
        })

    results = pd.DataFrame(records)

    results.to_csv(OUTPUT_FILE, index=False)

    valid = results[
        results["status"] == "ESTIMATED"
    ]

    report = {
        "algorithm": "Vessel Fuel and Travel Time Estimation",
        "vessel_speed_knots": float(speed),
        "fuel_consumption_tonnes_per_hour": float(fuel_rate),
        "total_candidates": len(results),
        "successfully_estimated": len(valid),
        "incomplete_candidates": len(results) - len(valid),
        "outputs": {
            "route_estimates": str(OUTPUT_FILE)
        },
        "limitations": [
            "Assumes constant vessel speed.",
            "Assumes constant fuel consumption rate.",
            "Does not model currents, weather, maneuvering or waiting time.",
            "Does not include port loading or unloading time.",
            "Estimates are not operational fuel guarantees."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(results.to_string(index=False))

if __name__ == "__main__":
    run_fuel_time_estimation()
