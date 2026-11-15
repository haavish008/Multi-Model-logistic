
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd

from config.port_config import PROJECT_ROOT

PORT_DIR = PROJECT_ROOT / "data" / "raw" / "maritime" / "ports"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"
OUTPUT_FILE = OUTPUT_DIR / "verified_ports.gpkg"
REPORT_FILE = OUTPUT_DIR / "verified_ports_report.json"

PORT_NAMES = {
    "kolkata": ["kolkata", "syama prasad mukherjee"],
    "port_blair": ["port blair", "sri vijaya puram"]
}

LAT_COLUMNS = ["latitude", "lat", "dec_lat", "lat_dec"]
LON_COLUMNS = ["longitude", "lon", "long", "dec_long", "lon_dec"]
NAME_COLUMNS = [
    "port_name",
    "name",
    "port",
    "main_port_name",
    "portname"
]

def normalize_columns(data):
    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]
    return data

def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None

def load_dataset(path):
    suffix = path.suffix.lower()

    if suffix in [".shp", ".gpkg", ".geojson"]:
        return gpd.read_file(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    return None

def run_port_extraction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in PORT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in [
            ".shp", ".gpkg", ".geojson",
            ".csv", ".xlsx", ".xls"
        ]
    ]

    if not files:
        raise FileNotFoundError("No port dataset found.")

    matches = []
    report = []

    for path in files:
        try:
            data = load_dataset(path)

            if data is None or data.empty:
                continue

            data = normalize_columns(data)

            name_column = find_column(
                data.columns,
                NAME_COLUMNS
            )

            if name_column is None:
                report.append({
                    "file": path.name,
                    "status": "Port name column not identified"
                })
                continue

            names = data[name_column].astype(str).str.lower()

            selected = data[
                names.apply(
                    lambda value: any(
                        term in value
                        for terms in PORT_NAMES.values()
                        for term in terms
                    )
                )
            ].copy()

            if selected.empty:
                continue

            if isinstance(selected, gpd.GeoDataFrame):
                if selected.crs is None:
                    raise ValueError("Vector dataset has no CRS.")

                selected = selected.to_crs("EPSG:4326")

                if not selected.geometry.geom_type.isin(
                    ["Point", "MultiPoint"]
                ).all():
                    report.append({
                        "file": path.name,
                        "status": "Non-point geometries need manual review"
                    })
                    continue

                selected = selected[
                    selected.geometry.geom_type == "Point"
                ].copy()

                selected["longitude"] = selected.geometry.x
                selected["latitude"] = selected.geometry.y

            else:
                lat_column = find_column(
                    selected.columns,
                    LAT_COLUMNS
                )

                lon_column = find_column(
                    selected.columns,
                    LON_COLUMNS
                )

                if lat_column is None or lon_column is None:
                    report.append({
                        "file": path.name,
                        "status": "Coordinate columns not identified"
                    })
                    continue

                selected["latitude"] = pd.to_numeric(
                    selected[lat_column],
                    errors="coerce"
                )

                selected["longitude"] = pd.to_numeric(
                    selected[lon_column],
                    errors="coerce"
                )

                selected = selected.dropna(
                    subset=["latitude", "longitude"]
                )

                selected = gpd.GeoDataFrame(
                    selected,
                    geometry=gpd.points_from_xy(
                        selected["longitude"],
                        selected["latitude"]
                    ),
                    crs="EPSG:4326"
                )

            selected["source_file"] = path.name
            matches.append(selected)

            report.append({
                "file": path.name,
                "status": "Matching port records extracted",
                "records": len(selected)
            })

        except Exception as error:
            report.append({
                "file": path.name,
                "status": "Error",
                "error": str(error)
            })

    if not matches:
        with open(REPORT_FILE, "w") as file:
            json.dump(report, file, indent=4)

        raise ValueError(
            "No verified port records extracted. "
            "Inspect the dataset columns and report."
        )

    combined = gpd.GeoDataFrame(
        pd.concat(matches, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326"
    )

    combined = combined[
        combined.geometry.notna()
        & combined.geometry.is_valid
    ]

    combined.to_file(
        OUTPUT_FILE,
        driver="GPKG"
    )

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(combined[
        ["latitude", "longitude", "source_file"]
    ].to_string(index=False))

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")

if __name__ == "__main__":
    run_port_extraction()
