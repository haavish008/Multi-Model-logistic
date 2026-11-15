
import json
import numpy as np
import pandas as pd

from config.port_config import OUTPUT_DIR

INPUT_FILE = OUTPUT_DIR / "final_route_comparison.csv"

OUTPUT_FILE = OUTPUT_DIR / "performance_evaluation.csv"
REPORT_FILE = OUTPUT_DIR / "performance_evaluation_report.json"

def run_performance_evaluation():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run Part 30 before Part 31."
        )

    data = pd.read_csv(INPUT_FILE)

    if data.empty:
        raise ValueError("Route comparison dataset is empty.")

    numeric_columns = [
        "distance_km",
        "total_cost",
        "travel_time_hours",
        "fuel_tonnes",
        "topsis_score"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    evaluation = []

    for algorithm, group in data.groupby("algorithm"):
        record = {
            "algorithm": algorithm,
            "route_count": len(group),
            "valid_distance_count": int(
                group["distance_km"].notna().sum()
            ),
            "average_distance_km": (
                group["distance_km"].mean()
            ),
            "minimum_distance_km": (
                group["distance_km"].min()
            ),
            "average_travel_time_hours": (
                group["travel_time_hours"].mean()
                if "travel_time_hours" in group.columns
                else np.nan
            ),
            "average_fuel_tonnes": (
                group["fuel_tonnes"].mean()
                if "fuel_tonnes" in group.columns
                else np.nan
            ),
            "average_topsis_score": (
                group["topsis_score"].mean()
                if "topsis_score" in group.columns
                else np.nan
            ),
            "average_total_cost": (
                group["total_cost"].mean()
                if "total_cost" in group.columns
                else np.nan
            )
        }

        evaluation.append(record)

    results = pd.DataFrame(evaluation)

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    valid_distances = data.dropna(
        subset=["distance_km"]
    )

    best_distance = None

    if not valid_distances.empty:
        best_row = valid_distances.loc[
            valid_distances["distance_km"].idxmin()
        ]

        best_distance = {
            "algorithm": str(best_row["algorithm"]),
            "distance_km": float(best_row["distance_km"])
        }

    valid_topsis = data.dropna(
        subset=["topsis_score"]
    ) if "topsis_score" in data.columns else pd.DataFrame()

    best_topsis = None

    if not valid_topsis.empty:
        best_row = valid_topsis.loc[
            valid_topsis["topsis_score"].idxmax()
        ]

        best_topsis = {
            "algorithm": str(best_row["algorithm"]),
            "score": float(best_row["topsis_score"])
        }

    report = {
        "module": "Maritime Performance Evaluation",
        "total_records": len(data),
        "algorithms_evaluated": int(
            data["algorithm"].nunique()
        ),
        "records_with_distance": len(valid_distances),
        "best_distance_result": best_distance,
        "best_topsis_result": best_topsis,
        "evaluation_file": str(OUTPUT_FILE),
        "limitations": [
            "Execution time was not recorded by earlier modules.",
            "Missing metrics are not estimated.",
            "Algorithm comparisons require identical network constraints.",
            "Distance alone does not establish route quality.",
            "No real-world navigation performance has been validated."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(results.to_string(index=False))

if __name__ == "__main__":
    run_performance_evaluation()
