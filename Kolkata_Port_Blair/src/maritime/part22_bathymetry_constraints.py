
import json
import numpy as np
import geopandas as gpd
import rasterio

from pyproj import Transformer, Geod

from config.port_config import PROJECT_ROOT
from config.vessel_config import VESSEL_CONFIG

OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"
BATHYMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "bathymetry"

EDGE_FILE = OUTPUT_DIR / "constrained_maritime_edges.gpkg"
RASTER_FILE = BATHYMETRY_DIR / "bay_of_bengal_bathymetry.tif"

ALL_EDGES_FILE = OUTPUT_DIR / "bathymetry_assessed_edges.gpkg"
FEASIBLE_EDGES_FILE = OUTPUT_DIR / "bathymetry_feasible_edges.gpkg"
REPORT_FILE = OUTPUT_DIR / "bathymetry_constraint_report.json"

GEOD = Geod(ellps="WGS84")

def sample_edge_coordinates(geometry, spacing_km=2):
    coordinates = list(geometry.coords)
    samples = []

    for start, end in zip(coordinates[:-1], coordinates[1:]):
        lon1, lat1 = start[:2]
        lon2, lat2 = end[:2]

        _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)

        count = max(1, int(np.ceil(distance / (spacing_km * 1000))))

        points = GEOD.npts(
            lon1,
            lat1,
            lon2,
            lat2,
            count - 1
        )

        samples.append((lon1, lat1))
        samples.extend(points)

    if coordinates:
        samples.append(coordinates[-1][:2])

    return samples

def run_bathymetry_constraints():
    if not EDGE_FILE.exists():
        raise FileNotFoundError(
            "Run Part 21 before Part 22."
        )

    if not RASTER_FILE.exists():
        raise FileNotFoundError(
            f"Bathymetry raster not found: {RASTER_FILE}"
        )

    draft = VESSEL_CONFIG.get("draft_m")
    clearance = VESSEL_CONFIG.get(
        "minimum_under_keel_clearance_m"
    )

    if draft is None or clearance is None:
        raise ValueError(
            "Set draft_m and minimum_under_keel_clearance_m "
            "in config/vessel_config.py before running Part 22."
        )

    if draft <= 0 or clearance <= 0:
        raise ValueError("Draft and clearance must be positive.")

    required_depth = float(draft + clearance)

    edges = gpd.read_file(EDGE_FILE)

    if edges.empty:
        raise ValueError("Maritime edge dataset is empty.")

    if edges.crs is None:
        raise ValueError("Maritime edges have no CRS.")

    transformer = None

    with rasterio.open(RASTER_FILE) as raster:
        if raster.crs is None:
            raise ValueError("Bathymetry raster has no CRS.")

        transformer = Transformer.from_crs(
            edges.crs,
            raster.crs,
            always_xy=True
        )

        minimum_depths = []
        valid_sample_counts = []
        total_sample_counts = []
        statuses = []

        for _, row in edges.iterrows():
            geometry = row.geometry

            if geometry is None or geometry.is_empty:
                minimum_depths.append(np.nan)
                valid_sample_counts.append(0)
                total_sample_counts.append(0)
                statuses.append("UNKNOWN")
                continue

            if geometry.geom_type != "LineString":
                minimum_depths.append(np.nan)
                valid_sample_counts.append(0)
                total_sample_counts.append(0)
                statuses.append("UNKNOWN")
                continue

            coordinates = sample_edge_coordinates(geometry)

            raster_coordinates = [
                transformer.transform(x, y)
                for x, y in coordinates
            ]

            sampled = list(
                raster.sample(raster_coordinates, masked=True)
            )

            values = []

            for item in sampled:
                value = item[0]

                if np.ma.is_masked(value):
                    continue

                value = float(value)

                if np.isfinite(value):
                    values.append(value)

            valid_sample_counts.append(len(values))
            total_sample_counts.append(len(coordinates))

            if not values:
                minimum_depths.append(np.nan)
                statuses.append("UNKNOWN")
                continue

            depths = -np.asarray(values, dtype=float)

            minimum_depth = float(np.min(depths))
            minimum_depths.append(minimum_depth)

            if minimum_depth >= required_depth:
                statuses.append("DEPTH_SCREEN_PASSED")
            else:
                statuses.append("DEPTH_SCREEN_FAILED")

    edges["minimum_sampled_depth_m"] = minimum_depths
    edges["valid_bathymetry_samples"] = valid_sample_counts
    edges["total_bathymetry_samples"] = total_sample_counts
    edges["required_depth_m"] = required_depth
    edges["bathymetry_status"] = statuses

    edges.to_file(ALL_EDGES_FILE, driver="GPKG")

    feasible = edges[
        edges["bathymetry_status"] == "DEPTH_SCREEN_PASSED"
    ].copy()

    feasible.to_file(
        FEASIBLE_EDGES_FILE,
        driver="GPKG"
    )

    report = {
        "vessel_draft_m": float(draft),
        "minimum_under_keel_clearance_m": float(clearance),
        "required_depth_m": required_depth,
        "edge_count": len(edges),
        "depth_screen_passed": int(
            (edges["bathymetry_status"] == "DEPTH_SCREEN_PASSED").sum()
        ),
        "depth_screen_failed": int(
            (edges["bathymetry_status"] == "DEPTH_SCREEN_FAILED").sum()
        ),
        "unknown_edges": int(
            (edges["bathymetry_status"] == "UNKNOWN").sum()
        ),
        "sampling_spacing_km": 2,
        "outputs": {
            "assessed_edges": str(ALL_EDGES_FILE),
            "depth_screen_passed_edges": str(FEASIBLE_EDGES_FILE)
        },
        "limitations": [
            "GEBCO grid values are treated as elevation in metres relative to sea level.",
            "Depth is estimated as the negative of elevation.",
            "Raster resolution and vertical datum affect the assessment.",
            "Sampling between raster cells may miss hazards.",
            "This is a preliminary depth screening, not navigational clearance certification."
        ]
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_bathymetry_constraints()
