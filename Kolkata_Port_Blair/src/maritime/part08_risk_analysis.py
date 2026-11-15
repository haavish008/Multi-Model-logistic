
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "maritime"

WEATHER_FILE = RESULTS_DIR / "weather_waypoint_samples.csv"
OCEAN_FILE = RESULTS_DIR / "ocean_waypoint_samples.csv"

RISK_KEYWORDS = {
    "wind": ["wind_speed", "windspeed", "wind"],
    "rainfall": ["precipitation", "rainfall", "total_precipitation"],
    "wave": ["wave_height", "significant_wave_height", "swh"],
    "current": ["current_speed", "ocean_current_speed"],
}

def identify_risk_category(variable):
    name = variable.lower()

    for category, keywords in RISK_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            return category

    return None

def load_observations(path, source):
    if not path.exists():
        print(f"Missing {source} file:", path)
        return pd.DataFrame()

    dataframe = pd.read_csv(path)

    required = {"waypoint", "variable", "value"}

    if not required.issubset(dataframe.columns):
        raise ValueError(
            f"{source} data does not contain required columns: {required}"
        )

    dataframe["value"] = pd.to_numeric(
        dataframe["value"],
        errors="coerce",
    )

    dataframe["category"] = dataframe["variable"].apply(
        identify_risk_category
    )

    dataframe["source"] = source

    return dataframe

def calculate_relative_indicators(data):
    data = data.dropna(subset=["value"]).copy()
    data = data.loc[data["category"].notna()].copy()

    if data.empty:
        return pd.DataFrame()

    summaries = []

    for (waypoint, category, source), group in data.groupby(
        ["waypoint", "category", "source"]
    ):
        values = group["value"]

        summaries.append({
            "waypoint": waypoint,
            "risk_factor": category,
            "source": source,
            "mean_value": values.mean(),
            "maximum_value": values.max(),
            "units": ", ".join(
                sorted(group["units"].dropna().astype(str).unique())
            ) if "units" in group.columns else "unknown",
        })

    summary = pd.DataFrame(summaries)

    summary["relative_percentile"] = summary.groupby(
        ["risk_factor", "source"]
    )["maximum_value"].rank(pct=True) * 100

    return summary

def generate_risk_report():
    weather = load_observations(WEATHER_FILE, "weather")
    ocean = load_observations(OCEAN_FILE, "ocean")

    available = [
        dataframe for dataframe in [weather, ocean]
        if not dataframe.empty
    ]

    if not available:
        raise FileNotFoundError(
            "No weather or ocean observations available for risk analysis."
        )

    combined = pd.concat(available, ignore_index=True)

    report = calculate_relative_indicators(combined)

    if report.empty:
        raise ValueError(
            "No recognized risk variables found. Inspect the NetCDF "
            "variable names and update RISK_KEYWORDS."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "maritime_risk_indicators.csv"

    report.to_csv(output_path, index=False)

    print("\nMaritime risk indicators:")
    print(report.to_string(index=False))

    print("\nSaved:", output_path)

    return report

if __name__ == "__main__":
    generate_risk_report()
