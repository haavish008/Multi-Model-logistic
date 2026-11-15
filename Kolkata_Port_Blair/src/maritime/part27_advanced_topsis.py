
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from config.port_config import OUTPUT_DIR

CANDIDATE_FILE = OUTPUT_DIR / "genetic_route_candidates.csv"
EDGE_FILE = OUTPUT_DIR / "environmental_edge_assessment.gpkg"

OUTPUT_FILE = OUTPUT_DIR / "advanced_topsis_ranking.csv"
REPORT_FILE = OUTPUT_DIR / "advanced_topsis_report.json"

def calculate_route_metrics(candidates, edges):
    edge_lookup = {}

    for _, row in edges.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        edge_lookup[(source, target)] = row
        edge_lookup[(target, source)] = row

    records = []

    for _, candidate in candidates.iterrows():
        route = str(candidate["waypoints"]).split(" -> ")

        route_edges = []

        for source, target in zip(route[:-1], route[1:]):
            edge = edge_lookup.get((source, target))

            if edge is None:
                raise ValueError(
                    f"Missing environmental data for edge {source} -> {target}"
                )

            route_edges.append(edge)

        record = {
            "route_id": candidate["route_id"],
            "distance_km": float(candidate["distance_km"]),
            "node_count": len(route),
            "waypoints": candidate["waypoints"]
        }

        for category in ["wind", "wave"]:
            column = f"{category}_mean"

            if all(
                pd.notna(edge[column])
                for edge in route_edges
            ):
                record[f"average_{category}"] = float(
                    np.mean([
                        edge[column]
                        for edge in route_edges
                    ])
                )
            else:
                record[f"average_{category}"] = np.nan

        records.append(record)

    return pd.DataFrame(records)

def run_advanced_topsis():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            "Run Part 26 before Part 27."
        )

    if not EDGE_FILE.exists():
        raise FileNotFoundError(
            "Environmental edge assessment is missing."
        )

    candidates = pd.read_csv(CANDIDATE_FILE)
    edges = gpd.read_file(EDGE_FILE)

    if len(candidates) < 2:
        raise ValueError(
            "At least two genuine route candidates are required."
        )

    required = [
        "route_id",
        "distance_km",
        "waypoints"
    ]

    if any(column not in candidates.columns for column in required):
        raise ValueError("Candidate route columns are incomplete.")

    metrics = calculate_route_metrics(
        candidates,
        edges
    )

    criteria = [
        "distance_km",
        "average_wind",
        "average_wave"
    ]

    active_criteria = [
        column for column in criteria
        if metrics[column].notna().all()
    ]

    if len(active_criteria) < 2:
        raise ValueError(
            "TOPSIS requires at least two complete criteria. "
            "Check environmental coverage and route data."
        )

    matrix = metrics[active_criteria].to_numpy(dtype=float)

    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError(
            "TOPSIS criteria contain invalid or negative values."
        )

    denominators = np.sqrt(
        np.sum(matrix ** 2, axis=0)
    )

    if np.any(denominators == 0):
        raise ValueError(
            "One or more criteria have zero normalization denominators."
        )

    normalized = matrix / denominators

    weights = np.full(
        len(active_criteria),
        1 / len(active_criteria)
    )

    weighted = normalized * weights

    ideal_best = weighted.min(axis=0)
    ideal_worst = weighted.max(axis=0)

    distance_best = np.sqrt(
        np.sum((weighted - ideal_best) ** 2, axis=1)
    )

    distance_worst = np.sqrt(
        np.sum((weighted - ideal_worst) ** 2, axis=1)
    )

    denominator = distance_best + distance_worst

    scores = np.divide(
        distance_worst,
        denominator,
        out=np.full_like(distance_worst, 0.5),
        where=denominator != 0
    )

    metrics["topsis_score"] = scores

    metrics["rank"] = (
        metrics["topsis_score"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    metrics = metrics.sort_values(
        ["rank", "topsis_score"],
        ascending=[True, False]
    )

    metrics.to_csv(
        OUTPUT_FILE,
        index=False
    )

    report = {
        "algorithm": "Advanced TOPSIS",
        "candidate_count": len(metrics),
        "criteria_used": active_criteria,
        "criterion_weights": {
            column: float(weight)
            for column, weight in zip(active_criteria, weights)
        },
        "best_route_id": str(metrics.iloc[0]["route_id"]),
        "best_topsis_score": float(
            metrics.iloc[0]["topsis_score"]
        ),
        "ranking_file": str(OUTPUT_FILE),
        "limitations": [
            "Equal weights are illustrative model assumptions.",
            "All included criteria are treated as cost criteria.",
            "Environmental observations must be temporally representative.",
            "TOPSIS ranking does not establish navigational safety."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    run_advanced_topsis()
