
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "maritime"
RESULTS_DIR = PROJECT_ROOT / "results" / "maritime"

RISK_FILE = RESULTS_DIR / "maritime_risk_indicators.csv"

DISTANCE_WEIGHT = 0.70
RISK_WEIGHT = 0.30

def calculate_edge_costs():
    edge_file = DATA_DIR / "maritime_edges.gpkg"

    if not edge_file.exists():
        raise FileNotFoundError(edge_file)

    edges = gpd.read_file(edge_file)

    if edges.empty:
        raise ValueError("Maritime edge dataset is empty.")

    required = {"from_node", "to_node", "distance_km"}

    if not required.issubset(edges.columns):
        raise ValueError(f"Missing edge columns: {required - set(edges.columns)}")

    if not RISK_FILE.exists():
        raise FileNotFoundError(RISK_FILE)

    risk = pd.read_csv(RISK_FILE)

    required_risk = {"waypoint", "relative_percentile"}

    if not required_risk.issubset(risk.columns):
        raise ValueError("Risk indicator columns are missing.")

    risk["relative_percentile"] = pd.to_numeric(
        risk["relative_percentile"],
        errors="coerce",
    )

    risk_by_waypoint = (
        risk.groupby("waypoint")["relative_percentile"]
        .mean()
        .to_dict()
    )

    edge_risks = []

    for _, edge in edges.iterrows():
        start_risk = risk_by_waypoint.get(edge["from_node"])
        end_risk = risk_by_waypoint.get(edge["to_node"])

        available = [
            value for value in [start_risk, end_risk]
            if value is not None and np.isfinite(value)
        ]

        edge_risks.append(
            float(np.mean(available)) if available else np.nan
        )

    edges["risk_percentile"] = edge_risks

    if edges["risk_percentile"].isna().all():
        raise ValueError("No usable risk observations are available.")

    fallback_risk = edges["risk_percentile"].median()
    edges["risk_percentile"] = edges["risk_percentile"].fillna(fallback_risk)

    edges["distance_cost"] = edges["distance_km"] * DISTANCE_WEIGHT

    edges["risk_cost"] = (
        edges["distance_km"]
        * (edges["risk_percentile"] / 100)
        * RISK_WEIGHT
    )

    edges["total_cost"] = (
        edges["distance_cost"] + edges["risk_cost"]
    )

    output_file = DATA_DIR / "maritime_edges_costed.gpkg"

    edges.to_file(output_file, driver="GPKG")

    print("Costed edges:", len(edges))
    print("Total distance:", edges["distance_km"].sum())
    print("Total cost:", edges["total_cost"].sum())
    print("Saved:", output_file)

    return edges

if __name__ == "__main__":
    calculate_edge_costs()
