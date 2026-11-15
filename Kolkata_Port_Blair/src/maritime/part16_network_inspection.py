
from pathlib import Path
import json
import geopandas as gpd

from config.port_config import PROJECT_ROOT

MARITIME_DIR = PROJECT_ROOT / "data" / "raw" / "maritime"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"
OUTPUT_FILE = OUTPUT_DIR / "network_dataset_inventory.json"

SUPPORTED_FILES = [".shp", ".gpkg", ".geojson"]

def inspect_network_datasets():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in MARITIME_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILES
    ]

    inventory = []

    if not files:
        print("No maritime vector datasets found.")
        print(f"Add maritime network data to: {MARITIME_DIR}")
        return

    for file_path in files:
        try:
            gdf = gpd.read_file(file_path)

            record = {
                "filename": file_path.name,
                "path": str(file_path),
                "feature_count": len(gdf),
                "crs": str(gdf.crs),
                "geometry_types": (
                    gdf.geometry.geom_type.value_counts().to_dict()
                    if not gdf.empty else {}
                ),
                "columns": list(gdf.columns),
                "bounds": (
                    gdf.total_bounds.tolist()
                    if not gdf.empty else None
                ),
                "missing_geometry": int(gdf.geometry.isna().sum()),
                "invalid_geometry": int(
                    (~gdf.geometry.is_valid).sum()
                ) if not gdf.empty else 0
            }

            inventory.append(record)

            print(f"\nDataset: {file_path.name}")
            print(f"Features: {len(gdf)}")
            print(f"CRS: {gdf.crs}")
            print(f"Geometry: {record['geometry_types']}")
            print(f"Columns: {record['columns']}")

        except Exception as error:
            inventory.append({
                "filename": file_path.name,
                "error": str(error)
            })

            print(f"Could not inspect {file_path.name}: {error}")

    with open(OUTPUT_FILE, "w") as file:
        json.dump(inventory, file, indent=4, default=str)

    print("\nMaritime dataset inspection completed.")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    inspect_network_datasets()
