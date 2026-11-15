
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd

from config.port_config import PROJECT_ROOT

PORT_DIR = PROJECT_ROOT / "data" / "raw" / "maritime" / "ports"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"
OUTPUT_FILE = OUTPUT_DIR / "port_dataset_inventory.json"

SUPPORTED_VECTOR = {".shp", ".gpkg", ".geojson"}
SUPPORTED_TABLE = {".csv", ".xlsx"}

def inspect_port_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in PORT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_VECTOR | SUPPORTED_TABLE
    ]

    if not files:
        print("No port datasets found.")
        print(f"Place the downloaded World Port Index data in: {PORT_DIR}")
        return

    inventory = []

    for path in files:
        try:
            if path.suffix.lower() in SUPPORTED_VECTOR:
                data = gpd.read_file(path)

                record = {
                    "filename": path.name,
                    "format": "vector",
                    "rows": len(data),
                    "crs": str(data.crs),
                    "columns": list(data.columns),
                    "geometry_types": (
                        data.geometry.geom_type.value_counts().to_dict()
                        if not data.empty else {}
                    ),
                    "sample": data.drop(
                        columns="geometry"
                    ).head(3).to_dict("records")
                }

            else:
                if path.suffix.lower() == ".csv":
                    data = pd.read_csv(path, nrows=5)
                else:
                    data = pd.read_excel(path, nrows=5)

                record = {
                    "filename": path.name,
                    "format": "table",
                    "columns": list(data.columns),
                    "sample": data.head(3).to_dict("records")
                }

            inventory.append(record)

            print(f"\nFile: {path.name}")
            print(json.dumps(record, indent=2, default=str))

        except Exception as error:
            inventory.append({
                "filename": path.name,
                "error": str(error)
            })

            print(f"Error reading {path.name}: {error}")

    with open(OUTPUT_FILE, "w") as file:
        json.dump(inventory, file, indent=4, default=str)

    print(f"\nPort inventory saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    inspect_port_data()
