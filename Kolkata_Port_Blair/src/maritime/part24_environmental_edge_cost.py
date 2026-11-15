
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from config.port_config import OUTPUT_DIR

EDGE_FILE = OUTPUT_DIR / "bathymetry_assessed_edges.gpkg"
ENVIRONMENT_FILE = OUTPUT_DIR / "vessel_environment_assessment.csv"

OUTPUT_FILE = OUTPUT_DIR / "environmental_edge_assessment.gpkg"
REPORT_FILE = OUTPUT_DIR / "environmental_edge_report.json"

def run_environmental_edge_assessment():
    if not EDGE_FILE.exists():
        raise FileNotFoundError("Run Part 22 before Part 24.")

    if not ENVIRONMENT_FILE.exists():
        raise FileNotFoundError("Run Part 23 before Part 24.")

    edges = gpd.read_file(EDGE_FILE)
    environment = pd.read_csv(ENVIRONMENT_FILE)

    required_edge_columns = [
        "source",
        "target",
        "distance_km"
    ]

    required_environment_columns = [
        "waypoint",
        "category",
        "converted_value",
        "status"
    ]

    missing_edges = [
        column for column in required_edge_columns
        if column not in edges.columns
    ]

    missing_environment = [
        column for column in required_environment_columns
        if column not in environment.columns
    ]

    if missing_edges or missing_environment:
        raise ValueError({
            "missing_edge_columns": missing_edges,
            "missing_environment_columns": missing_environment
        })

    if environment["waypoint"].isna().all():
        raise ValueError(
            "Environmental samples have no waypoint identifiers. "
            "Fix the Part 6 and Part 7 waypoint sampling outputs."
        )

    environment["converted_value"] = pd.to_numeric(
        environment["converted_value"],
        errors="coerce"
    )

    environment = environment.dropna(
        subset=["waypoint", "converted_value"]
    )

    summaries = (
        environment.groupby(
            ["waypoint", "category"]
        )["converted_value"]
        .agg(["mean", "max", "count"])
        .reset_index()
    )

    summary_lookup = {}

    for _, row in summaries.iterrows():
        summary_lookup[
            (str(row["waypoint"]), row["category"])
        ] = {
            "mean": float(row["mean"]),
            "maximum": float(row["max"]),
            "count": int(row["count"])
        }

    records = []

    for _, edge in edges.iterrows():
        source = str(edge["source"])
        target = str(edge["target"])

        record = {
            "source": source,
            "target": target,
            "distance_km": float(edge["distance_km"])
        }

        for category in ["WIND", "WAVE"]:
            first = summary_lookup.get((source, category))
            second = summary_lookup.get((target, category))

            if first and second:
                record[f"{category.lower()}_mean"] = (
                    first["mean"] + second["mean"]
                ) / 2

                record[f"{category.lower()}_maximum"] = max(
                    first["maximum"],
                    second["maximum"]
                )

                record[f"{category.lower()}_observations"] = (
                    first["count"] + second["count"]
                )

                record[f"{category.lower()}_status"] = "ASSESSED"
            else:
                record[f"{category.lower()}_mean"] = np.nan
                record[f"{category.lower()}_maximum"] = np.nan
                record[f"{category.lower()}_observations"] = 0
                record[f"{category.lower()}_status"] = "MISSING"

        records.append(record)

    assessment = pd.DataFrame(records)

    result = edges.merge(
        assessment,
        on=["source", "target", "distance_km"],
        how="left",
        suffixes=("", "_environment")
    )

    result.to_file(
        OUTPUT_FILE,
        driver="GPKG"
    )

    report = {
        "total_edges": len(result),
        "wind_assessed_edges": int(
            (result["wind_status"] == "ASSESSED").sum()
        ),
        "wave_assessed_edges": int(
            (result["wave_status"] == "ASSESSED").sum()
        ),
        "wind_missing_edges": int(
            (result["wind_status"] == "MISSING").sum()
        ),
        "wave_missing_edges": int(
            (result["wave_status"] == "MISSING").sum()
        ),
        "outputs": {
            "environmental_edges": str(OUTPUT_FILE)
        },
        "limitations": [
            "Wind and wave observations are summarized at edge endpoints.",
            "Temporal alignment is not performed.",
            "Ocean current direction and vessel heading are not modeled.",
            "No navigational safety thresholds are inferred.",
            "Environmental exposure is not yet incorporated into route costs."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_environmental_edge_assessment()
