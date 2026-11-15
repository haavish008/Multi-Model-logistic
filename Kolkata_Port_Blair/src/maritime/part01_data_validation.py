
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SUPPORTED_EXTENSIONS = {
    ".shp",
    ".gpkg",
    ".geojson",
    ".tif",
    ".tiff",
    ".nc",
    ".csv",
    ".xlsx",
    ".json",
}

def inspect_directory(directory):
    records = []

    if not directory.exists():
        return records

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            records.append({
                "name": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            })

    return records

def validate_datasets():
    categories = [
        "maritime",
        "ports",
        "weather",
        "ocean",
        "bathymetry",
    ]

    report = {}

    for category in categories:
        directory = RAW_DIR / category
        files = inspect_directory(directory)

        report[category] = {
            "directory_exists": directory.exists(),
            "file_count": len(files),
            "files": files,
        }

    output_dir = PROJECT_ROOT / "results" / "maritime"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dataset_inventory.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    for category, details in report.items():
        print(f"\n{category.upper()}")
        print("Directory exists:", details["directory_exists"])
        print("Files found:", details["file_count"])

        for item in details["files"]:
            print(" -", item["path"])

    print("\nInventory saved to:", output_file)

    return report

if __name__ == "__main__":
    validate_datasets()
