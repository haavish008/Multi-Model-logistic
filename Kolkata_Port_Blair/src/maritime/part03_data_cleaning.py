
from pathlib import Path
import geopandas as gpd
from shapely import make_valid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "maritime" / "vectors"

VECTOR_EXTENSIONS = {".shp", ".gpkg", ".geojson"}

def clean_vector(path):
    data = gpd.read_file(path)

    if data.empty:
        raise ValueError(f"Dataset is empty: {path}")

    if data.crs is None:
        raise ValueError(f"Dataset has no CRS: {path}")

    data = data.loc[data.geometry.notna()].copy()
    data = data.loc[~data.geometry.is_empty].copy()
    data.geometry = data.geometry.apply(make_valid)
    data = data.loc[~data.geometry.is_empty].copy()
    data = data.drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise ValueError(f"No valid features remain: {path}")

    return data

def process_vector_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed = []

    for path in RAW_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VECTOR_EXTENSIONS:
            continue

        data = clean_vector(path)
        output_path = OUTPUT_DIR / f"{path.stem}_cleaned.gpkg"

        data.to_file(output_path, driver="GPKG")

        processed.append({
            "source": str(path),
            "output": str(output_path),
            "features": len(data),
            "crs": str(data.crs),
        })

        print("Processed:", path.name)
        print("Features:", len(data))
        print("CRS:", data.crs)

    print("\nTotal vector datasets processed:", len(processed))
    return processed

if __name__ == "__main__":
    process_vector_files()
