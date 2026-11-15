
from pathlib import Path
import rasterio
import pandas as pd
from rasterio.warp import transform
from config.port_config import WAYPOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATHYMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "bathymetry"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"

def find_bathymetry_file():
    supported = {".tif", ".tiff"}

    files = [
        path for path in BATHYMETRY_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    ]

    if not files:
        raise FileNotFoundError(
            f"No bathymetry raster found in {BATHYMETRY_DIR}"
        )

    if len(files) > 1:
        raise ValueError(
            "Multiple bathymetry rasters found. Keep one selected raster "
            "in the bathymetry folder or specify the intended file."
        )

    return files[0]

def sample_bathymetry():
    raster_path = find_bathymetry_file()
    records = []

    with rasterio.open(raster_path) as dataset:
        if dataset.crs is None:
            raise ValueError("Bathymetry raster has no CRS.")

        print("Raster:", raster_path.name)
        print("CRS:", dataset.crs)
        print("Resolution:", dataset.res)
        print("Bounds:", dataset.bounds)
        print("NoData:", dataset.nodata)

        longitudes = [item["longitude"] for item in WAYPOINTS]
        latitudes = [item["latitude"] for item in WAYPOINTS]

        xs, ys = transform(
            "EPSG:4326",
            dataset.crs,
            longitudes,
            latitudes,
        )

        samples = list(dataset.sample(zip(xs, ys), masked=True))

        for waypoint, sample in zip(WAYPOINTS, samples):
            value = sample[0]

            if value is None or getattr(value, "mask", False):
                depth_value = None
                status = "NoData"
            else:
                depth_value = float(value)
                status = "Sampled"

            records.append({
                "waypoint": waypoint["name"],
                "latitude": waypoint["latitude"],
                "longitude": waypoint["longitude"],
                "raster_value": depth_value,
                "sample_status": status,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "waypoint_bathymetry.csv"

    pd.DataFrame(records).to_csv(
        output_path,
        index=False,
    )

    print("\nBathymetry samples:")
    for record in records:
        print(record)

    print("\nSaved:", output_path)

    return records

if __name__ == "__main__":
    sample_bathymetry()
