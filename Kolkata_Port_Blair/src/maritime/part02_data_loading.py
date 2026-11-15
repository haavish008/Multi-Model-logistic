
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def load_vector(path):
    data = gpd.read_file(path)

    if data.empty:
        raise ValueError(f"Empty vector dataset: {path}")

    if data.crs is None:
        raise ValueError(f"Missing CRS: {path}")

    return data

def load_raster(path):
    with rasterio.open(path) as dataset:
        return {
            "path": str(path),
            "crs": str(dataset.crs),
            "bounds": tuple(dataset.bounds),
            "width": dataset.width,
            "height": dataset.height,
            "resolution": dataset.res,
            "nodata": dataset.nodata,
            "band_count": dataset.count,
            "dtype": dataset.dtypes,
        }

def load_netcdf(path):
    return xr.open_dataset(path)

def load_table(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported table format: {suffix}")

def load_all_datasets():
    datasets = {}

    for path in RAW_DIR.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        key = str(path.relative_to(RAW_DIR))

        try:
            if suffix in {".shp", ".gpkg", ".geojson"}:
                datasets[key] = load_vector(path)

            elif suffix in {".tif", ".tiff"}:
                datasets[key] = load_raster(path)

            elif suffix == ".nc":
                datasets[key] = load_netcdf(path)

            elif suffix in {".csv", ".xlsx", ".xls"}:
                datasets[key] = load_table(path)

            else:
                continue

            print("Loaded:", key)

        except Exception as error:
            print("Could not load:", key, "|", error)

    return datasets

if __name__ == "__main__":
    loaded = load_all_datasets()
    print("\nSuccessfully loaded:", len(loaded), "datasets")
