
import json
import pandas as pd

from config.port_config import PROJECT_ROOT, OUTPUT_DIR

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "maritime"
    / "ports"
    / "port_congestion.csv"
)

OUTPUT_FILE = OUTPUT_DIR / "port_congestion_analysis.csv"
REPORT_FILE = OUTPUT_DIR / "port_congestion_report.json"

REQUIRED_COLUMNS = [
    "port_name",
    "timestamp",
    "vessel_queue",
    "avg_wait_hours",
    "berth_utilization_pct"
]

def classify_congestion(wait_hours, utilization):
    if wait_hours >= 24 or utilization >= 90:
        return "HIGH"

    if wait_hours >= 12 or utilization >= 75:
        return "MODERATE"

    return "LOW"

def run_port_congestion():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Port congestion dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    numeric_columns = [
        "vessel_queue",
        "avg_wait_hours",
        "berth_utilization_pct"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    )

    if data.empty:
        raise ValueError(
            "No valid port congestion observations found."
        )

    if (
        (data["vessel_queue"] < 0).any()
        or (data["avg_wait_hours"] < 0).any()
        or (
            (data["berth_utilization_pct"] < 0)
            | (data["berth_utilization_pct"] > 100)
        ).any()
    ):
        raise ValueError(
            "Invalid congestion values detected."
        )

    summary = (
        data.groupby("port_name")
        .agg(
            observation_count=("timestamp", "count"),
            average_vessel_queue=("vessel_queue", "mean"),
            maximum_vessel_queue=("vessel_queue", "max"),
            average_wait_hours=("avg_wait_hours", "mean"),
            maximum_wait_hours=("avg_wait_hours", "max"),
            average_berth_utilization=(
                "berth_utilization_pct",
                "mean"
            )
        )
        .reset_index()
    )

    summary["congestion_level"] = summary.apply(
        lambda row: classify_congestion(
            row["average_wait_hours"],
            row["average_berth_utilization"]
        ),
        axis=1
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )

    report = {
        "module": "Port Congestion Analysis",
        "total_observations": len(data),
        "ports_analyzed": summary["port_name"].tolist(),
        "congestion_distribution": (
            summary["congestion_level"]
            .value_counts()
            .to_dict()
        ),
        "outputs": {
            "analysis": str(OUTPUT_FILE)
        },
        "limitations": [
            "Congestion classifications use illustrative thresholds.",
            "Input records must come from verified port-operation sources.",
            "Missing observations can affect the results.",
            "Historical averages do not predict future congestion."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_port_congestion()
