
from pathlib import Path
import json
import numpy as np
import rasterio

from config.port_config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bathymetry"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"
OUTPUT_FILE = OUTPUT_DIR / "bathymetry_quality_report.json"

def inspect_bathymetry():
    files = list(DATA_DIR.glob("*.tif")) + list(DATA_DIR.glob("*.tiff"))

    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one GeoTIFF, found {len(files)}"
        )

    raster_path = files[0]

    with rasterio.open(raster_path) as src:
        values = src.read(1, masked=True)
        valid = values.compressed()

        if len(valid) == 0:
            raise ValueError("Raster contains no valid pixels.")

        report = {
            "filename": raster_path.name,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "bounds": list(src.bounds),
            "resolution": list(src.res),
            "nodata": src.nodata,
            "valid_pixels": int(len(valid)),
            "minimum_elevation_m": float(np.min(valid)),
            "maximum_elevation_m": float(np.max(valid)),
            "mean_elevation_m": float(np.mean(valid)),
            "negative_pixel_count": int(np.sum(valid < 0)),
            "positive_pixel_count": int(np.sum(valid > 0)),
            "zero_pixel_count": int(np.sum(valid == 0))
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    inspect_bathymetry()
