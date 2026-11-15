
from pathlib import Path
import numpy as np
import pandas as pd

from config.port_config import OUTPUT_DIR

INPUT_FILE = OUTPUT_DIR / "alternative_route_metrics.csv"
OUTPUT_FILE = OUTPUT_DIR / "topsis_route_ranking.csv"

def run_topsis():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = ["route_id", "distance_km", "total_cost"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)

    if len(df) < 2:
        print("TOPSIS requires at least two genuine alternative routes.")
        print("The current waypoint network may contain only one route.")
        print("Add validated maritime network connections before ranking.")
        return

    matrix = df[["distance_km", "total_cost"]].to_numpy(dtype=float)

    if np.any(matrix < 0):
        raise ValueError("TOPSIS input criteria cannot contain negative values.")

    denominators = np.sqrt((matrix ** 2).sum(axis=0))
    denominators[denominators == 0] = 1

    normalized = matrix / denominators

    weights = np.array([0.5, 0.5])
    weighted = normalized * weights

    ideal_best = weighted.min(axis=0)
    ideal_worst = weighted.max(axis=0)

    distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    denominator = distance_best + distance_worst
    denominator[denominator == 0] = 1

    df["topsis_score"] = distance_worst / denominator
    df["rank"] = df["topsis_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    df = df.sort_values(
        ["rank", "topsis_score"],
        ascending=[True, False]
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("TOPSIS ranking completed.")
    print(df.to_string(index=False))
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_topsis()
