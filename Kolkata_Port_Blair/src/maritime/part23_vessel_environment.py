
import json
import pandas as pd

from config.port_config import OUTPUT_DIR
from config.vessel_config import VESSEL_CONFIG

WEATHER_FILE = OUTPUT_DIR / "weather_waypoint_samples.csv"
OCEAN_FILE = OUTPUT_DIR / "ocean_waypoint_samples.csv"

OUTPUT_FILE = OUTPUT_DIR / "vessel_environment_assessment.csv"
REPORT_FILE = OUTPUT_DIR / "vessel_environment_report.json"

def find_column(data, candidates):
    for column in candidates:
        if column in data.columns:
            return column
    return None

def load_environment_data():
    frames = []

    for path in [WEATHER_FILE, OCEAN_FILE]:
        if path.exists():
            data = pd.read_csv(path)
            data["dataset_source"] = path.stem
            frames.append(data)

    if not frames:
        raise FileNotFoundError(
            "Weather and ocean sample datasets were not found."
        )

    return pd.concat(frames, ignore_index=True)

def convert_to_ms(value, unit):
    unit = str(unit).strip().lower()

    if unit in ["m/s", "ms-1", "m s-1", "m/s "]:
        return value

    if unit in ["km/h", "kph", "km h-1"]:
        return value / 3.6

    if unit in ["kn", "knot", "knots"]:
        return value * 0.514444

    return None

def convert_to_metres(value, unit):
    unit = str(unit).strip().lower()

    if unit in ["m", "metre", "meter", "metres", "meters"]:
        return value

    if unit in ["cm", "centimeter", "centimetre"]:
        return value / 100

    if unit in ["mm", "millimeter", "millimetre"]:
        return value / 1000

    return None

def run_environment_assessment():
    data = load_environment_data()

    variable_col = find_column(
        data,
        ["variable", "variable_name", "parameter"]
    )

    value_col = find_column(
        data,
        ["value", "sample_value", "measurement"]
    )

    unit_col = find_column(
        data,
        ["units", "unit"]
    )

    waypoint_col = find_column(
        data,
        ["waypoint", "node_id", "location"]
    )

    if variable_col is None or value_col is None:
        raise ValueError(
            "Variable and value columns could not be identified."
        )

    if unit_col is None:
        data["units"] = ""
        unit_col = "units"

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="coerce"
    )

    data = data.dropna(subset=[value_col])

    maximum_wind = VESSEL_CONFIG.get("maximum_wind_speed_ms")
    maximum_wave = VESSEL_CONFIG.get("maximum_wave_height_m")

    results = []

    for _, row in data.iterrows():
        variable = str(row[variable_col]).lower()
        unit = row[unit_col]
        value = float(row[value_col])

        category = None
        converted_value = None
        threshold = None

        if any(term in variable for term in ["wind", "gust"]):
            category = "WIND"
            converted_value = convert_to_ms(value, unit)
            threshold = maximum_wind

        elif any(term in variable for term in ["wave", "swh"]):
            category = "WAVE"
            converted_value = convert_to_metres(value, unit)
            threshold = maximum_wave

        if category is None:
            continue

        if converted_value is None:
            status = "UNKNOWN_UNIT"
        elif threshold is None:
            status = "THRESHOLD_NOT_CONFIGURED"
        elif converted_value > threshold:
            status = "LIMIT_EXCEEDED"
        else:
            status = "WITHIN_CONFIGURED_LIMIT"

        results.append({
            "waypoint": (
                row[waypoint_col]
                if waypoint_col else None
            ),
            "variable": row[variable_col],
            "category": category,
            "original_value": value,
            "original_unit": unit,
            "converted_value": converted_value,
            "threshold": threshold,
            "status": status,
            "dataset_source": row["dataset_source"]
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise ValueError(
            "No identifiable wind or wave variables were found."
        )

    result_df.to_csv(OUTPUT_FILE, index=False)

    report = {
        "total_observations": len(result_df),
        "wind_observations": int(
            (result_df["category"] == "WIND").sum()
        ),
        "wave_observations": int(
            (result_df["category"] == "WAVE").sum()
        ),
        "limit_exceeded": int(
            (result_df["status"] == "LIMIT_EXCEEDED").sum()
        ),
        "unknown_units": int(
            (result_df["status"] == "UNKNOWN_UNIT").sum()
        ),
        "unconfigured_thresholds": int(
            (result_df["status"] == "THRESHOLD_NOT_CONFIGURED").sum()
        ),
        "outputs": {
            "assessment": str(OUTPUT_FILE)
        },
        "limitations": [
            "Only recognized wind and wave variables are assessed.",
            "Unknown units are not converted.",
            "Vessel operating limits must come from authoritative specifications.",
            "This is environmental screening, not a navigational safety determination."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    run_environment_assessment()
