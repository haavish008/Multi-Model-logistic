
import json
import pandas as pd

from config.port_config import OUTPUT_DIR

OUTPUT_FILE = OUTPUT_DIR / "final_route_comparison.csv"
REPORT_FILE = OUTPUT_DIR / "route_comparison_report.json"

def load_csv(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()

def load_json(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        with open(path) as file:
            return json.load(file)

    return {}

def run_route_comparison():
    records = []

    astar = load_csv("astar_route_metrics.csv")

    if not astar.empty:
        row = astar.iloc[0]

        records.append({
            "algorithm": "A*",
            "distance_km": row.get("distance_km"),
            "total_cost": row.get("total_cost"),
            "travel_time_hours": None,
            "fuel_tonnes": None,
            "node_count": row.get("node_count")
        })

    dstar = load_json("dstar_replanning_report.json")

    if dstar:
        records.append({
            "algorithm": "D* Lite",
            "distance_km": None,
            "total_cost": dstar.get("updated_cost"),
            "travel_time_hours": None,
            "fuel_tonnes": None,
            "node_count": (
                len(dstar.get("updated_path", []))
                if dstar.get("updated_path")
                else None
            )
        })

    genetic = load_json("genetic_optimization_report.json")

    if genetic:
        records.append({
            "algorithm": "Genetic Algorithm",
            "distance_km": genetic.get("best_distance_km"),
            "total_cost": None,
            "travel_time_hours": None,
            "fuel_tonnes": None,
            "node_count": genetic.get("candidate_count")
        })

    candidates = load_csv("genetic_route_candidates.csv")
    fuel = load_csv("fuel_time_route_estimates.csv")
    topsis = load_csv("advanced_topsis_ranking.csv")

    if not candidates.empty:
        for _, row in candidates.iterrows():
            route_id = row["route_id"]

            fuel_row = (
                fuel[fuel["route_id"] == route_id].iloc[0]
                if not fuel.empty
                and route_id in fuel["route_id"].values
                else None
            )

            topsis_row = (
                topsis[topsis["route_id"] == route_id].iloc[0]
                if not topsis.empty
                and route_id in topsis["route_id"].values
                else None
            )

            records.append({
                "algorithm": f"GA Candidate {route_id}",
                "distance_km": row.get("distance_km"),
                "total_cost": None,
                "travel_time_hours": (
                    fuel_row.get("travel_time_hours")
                    if fuel_row is not None else None
                ),
                "fuel_tonnes": (
                    fuel_row.get("estimated_fuel_tonnes")
                    if fuel_row is not None else None
                ),
                "node_count": row.get("node_count"),
                "topsis_score": (
                    topsis_row.get("topsis_score")
                    if topsis_row is not None else None
                ),
                "topsis_rank": (
                    topsis_row.get("rank")
                    if topsis_row is not None else None
                )
            })

    if not records:
        raise ValueError(
            "No route results found. Run the optimization modules first."
        )

    comparison = pd.DataFrame(records)

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    valid_distances = comparison.dropna(
        subset=["distance_km"]
    )

    best_distance_algorithm = None

    if not valid_distances.empty:
        best_distance_algorithm = (
            valid_distances.loc[
                valid_distances["distance_km"].idxmin(),
                "algorithm"
            ]
        )

    report = {
        "module": "Route Comparison",
        "algorithms_and_candidates": len(comparison),
        "best_distance_result": best_distance_algorithm,
        "comparison_file": str(OUTPUT_FILE),
        "limitations": [
            "Only available algorithm outputs are compared.",
            "D* Lite and A* costs may use different cost definitions.",
            "Missing metrics are not estimated or fabricated.",
            "Algorithm results are comparable only when their network, constraints and cost units match."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    run_route_comparison()
