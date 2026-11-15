
import pandas as pd

from config.vessel_config import (
    VESSEL_CONFIG,
    validate_vessel_config
)
from config.port_config import OUTPUT_DIR

BATHYMETRY_FILE = OUTPUT_DIR / "waypoint_bathymetry.csv"
OUTPUT_FILE = OUTPUT_DIR / "vessel_feasibility_report.csv"

def run_vessel_validation():
    config_valid = validate_vessel_config()

    if not BATHYMETRY_FILE.exists():
        print("Bathymetry data is unavailable.")
        return

    bathymetry = pd.read_csv(BATHYMETRY_FILE)

    report = {
        "vessel_configuration_valid": config_valid,
        "bathymetry_available": not bathymetry.empty,
        "draft_available": VESSEL_CONFIG["draft_m"] is not None,
        "clearance_available": (
            VESSEL_CONFIG["minimum_under_keel_clearance_m"] is not None
        )
    }

    if (
        report["vessel_configuration_valid"]
        and report["bathymetry_available"]
    ):
        draft = VESSEL_CONFIG["draft_m"]
        clearance = VESSEL_CONFIG["minimum_under_keel_clearance_m"]

        depth_columns = [
            column for column in bathymetry.columns
            if "depth" in column.lower()
            or "elevation" in column.lower()
            or "bathymetry" in column.lower()
        ]

        if len(depth_columns) == 1:
            depth = pd.to_numeric(
                bathymetry[depth_columns[0]],
                errors="coerce"
            )

            report["bathymetry_values_available"] = depth.notna().any()
            report["required_depth_m"] = draft + clearance
            report["depth_column"] = depth_columns[0]
        else:
            report["bathymetry_values_available"] = False
            report["depth_column_identified"] = False

    pd.DataFrame([report]).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(pd.DataFrame([report]).to_string(index=False))
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_vessel_validation()
