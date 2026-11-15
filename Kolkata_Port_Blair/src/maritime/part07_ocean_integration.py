
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from config.port_config import WAYPOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OCEAN_DIR = PROJECT_ROOT / "data" / "raw" / "ocean"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"

def find_coordinate(dataset, candidates):
    for name in candidates:
        if name in dataset.coords:
            return name
    for name in candidates:
        if name in dataset.variables and dataset[name].ndim == 1:
            return name
    raise ValueError(
        f"Coordinate not found. Available coordinates: {list(dataset.coords)}"
    )

def sample_ocean_file(path):
    records = []

    with xr.open_dataset(path) as dataset:
        latitude_name = find_coordinate(
            dataset, ["latitude", "lat", "LATITUDE"]
        )
        longitude_name = find_coordinate(
            dataset, ["longitude", "lon", "LONGITUDE"]
        )

        print("\nFile:", path.name)
        print("Coordinates:", latitude_name, longitude_name)

        for variable in dataset.data_vars:
            print(
                "Variable:",
                variable,
                "| Units:",
                dataset[variable].attrs.get("units", "unknown"),
                "| Description:",
                dataset[variable].attrs.get("long_name", "not specified"),
            )

        for waypoint in WAYPOINTS:
            point = dataset.sel(
                {
                    latitude_name: waypoint["latitude"],
                    longitude_name: waypoint["longitude"],
                },
                method="nearest",
            )

            for variable in dataset.data_vars:
                data = point[variable]
                values = np.asarray(data.values).squeeze()

                base = {
                    "source_file": path.name,
                    "waypoint": waypoint["name"],
                    "latitude": waypoint["latitude"],
                    "longitude": waypoint["longitude"],
                    "variable": variable,
                    "units": data.attrs.get("units", "unknown"),
                }

                if values.ndim == 0:
                    base["sample_index"] = 0
                    base["value"] = float(values) if np.isfinite(values) else np.nan
                    records.append(base)
                else:
                    for index, value in enumerate(values.reshape(-1)):
                        row = base.copy()
                        row["sample_index"] = index
                        row["value"] = float(value) if np.isfinite(value) else np.nan
                        records.append(row)

    return records

def integrate_ocean():
    files = sorted(OCEAN_DIR.glob("*.nc"))

    if not files:
        raise FileNotFoundError(
            f"No NetCDF ocean files found in {OCEAN_DIR}"
        )

    records = []

    for path in files:
        records.extend(sample_ocean_file(path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "ocean_waypoint_samples.csv"

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_path, index=False)

    print("\nOcean samples saved:", output_path)
    print("Total records:", len(dataframe))

    return dataframe

if __name__ == "__main__":
    integrate_ocean()
