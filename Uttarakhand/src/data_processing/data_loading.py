import os
import sys
import json
import math
import argparse
import logging
import warnings

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

import geopandas as gpd
import networkx as nx
import rasterio
import xarray as xr

from shapely.geometry import (
    Point,
    LineString,
    MultiLineString
)

from shapely.ops import (
    unary_union,
    linemerge
)

from pyproj import Transformer

warnings.filterwarnings("ignore")

PROJECT_NAME = "AI-Based Uttarakhand Logistics Optimization"

STUDY_AREA = "Dehradun to Joshimath NH-7 Corridor"

ORIGIN_NAME = "Dehradun"

DESTINATION_NAME = "Joshimath"

CASE_STUDY_DATE = "2025-07-20"

SOURCE_CRS = "EPSG:4326"

METRIC_CRS = "EPSG:32644"


@dataclass
class ProjectConfig:

    project_root: Path

    case_date: str = CASE_STUDY_DATE

    vehicle_type: str = "heavy_diesel"

    cargo_tonnes: float = 1.0

    default_speed_kmh: float = 35.0

    max_snap_distance_m: float = 5000.0

    generate_maps: bool = True

    @property
    def raw_dir(self):
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self):
        return self.project_root / "data" / "processed"

    @property
    def results_dir(self):
        return self.project_root / "results"

    @property
    def reports_dir(self):
        return self.project_root / "reports"

    def create_directories(self):

        directories = [
            self.raw_dir,
            self.processed_dir,
            self.results_dir,
            self.reports_dir
        ]

        for directory in directories:

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


def initialize_logging(config):

    config.create_directories()

    log_file = (
        config.reports_dir /
        "uttarakhand_logistics.log"
    )

    logging.basicConfig(

        level=logging.INFO,

        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),

        handlers=[

            logging.StreamHandler(sys.stdout),

            logging.FileHandler(
                log_file,
                encoding="utf-8"
            )

        ],

        force=True
    )

    logging.info("=" * 65)

    logging.info(
        "UTTARAKHAND LOGISTICS OPTIMIZATION STARTED"
    )

    logging.info(
        "Study area: %s",
        STUDY_AREA
    )

    logging.info("=" * 65)


LOCATIONS = {

    "Dehradun": {

        "latitude": 30.3165,

        "longitude": 78.0322,

        "elevation_m": 640,

        "type": "Origin"

    },

    "Jolly Grant Airport": {

        "latitude": 30.1897,

        "longitude": 78.1803,

        "elevation_m": 559,

        "type": "Emergency Airport"

    },

    "Rishikesh": {

        "latitude": 30.0869,

        "longitude": 78.2676,

        "elevation_m": 372,

        "type": "Corridor Location"

    },

    "Devprayag": {

        "latitude": 30.1461,

        "longitude": 78.5987,

        "elevation_m": 830,

        "type": "Corridor Location"

    },

    "Srinagar": {

        "latitude": 30.2225,

        "longitude": 78.7836,

        "elevation_m": 560,

        "type": "Corridor Location"

    },

    "Rudraprayag": {

        "latitude": 30.2844,

        "longitude": 78.9811,

        "elevation_m": 895,

        "type": "Corridor Location"

    },

    "Karnaprayag": {

        "latitude": 30.2584,

        "longitude": 79.2168,

        "elevation_m": 860,

        "type": "Corridor Location"

    },

    "Gauchar Airstrip": {

        "latitude": 30.2908,

        "longitude": 79.2143,

        "elevation_m": 800,

        "type": "Emergency Airstrip"

    },

    "Chamoli": {

        "latitude": 30.4048,

        "longitude": 79.3214,

        "elevation_m": 1300,

        "type": "Corridor Location"

    },

    "Joshimath": {

        "latitude": 30.5552,

        "longitude": 79.5644,

        "elevation_m": 1890,

        "type": "Destination"

    }

}


CORRIDOR_ORDER = [

    "Dehradun",

    "Rishikesh",

    "Devprayag",

    "Srinagar",

    "Rudraprayag",

    "Karnaprayag",

    "Chamoli",

    "Joshimath"

]


EMERGENCY_ROUTE_SEGMENTS = [

    {

        "segment": "Dehradun to Jolly Grant Airport",

        "distance_km": 27,

        "travel_time_minutes": 48

    },

    {

        "segment": "Jolly Grant Airport to Gauchar Airstrip",

        "distance_km": 108,

        "travel_time_minutes": 60

    },

    {

        "segment": "Gauchar Airstrip to Joshimath",

        "distance_km": 87,

        "travel_time_minutes": 165

    }

]


REPORT_ROUTE_COMPARISON = [

    {

        "mode": "Road Only (NH-7)",

        "distance_km": 298,

        "travel_time_hours": 8.5,

        "risk_level": "Road",

        "co2_kg": 45.76

    },

    {

        "mode": "Air + Road",

        "distance_km": 222,

        "travel_time_hours": 5.0,

        "risk_level": "Emergency Air + Road",

        "co2_kg": 46.56

    }

]


FUEL_CONFIG = {

    "heavy_diesel": {

        "fuel_name": "HSD Diesel",

        "price_per_unit": 95.20,

        "mileage": 4.0,

        "unit": "litre"

    },

    "light_petrol": {

        "fuel_name": "Petrol",

        "price_per_unit": 102.12,

        "mileage": 12.0,

        "unit": "litre"

    },

    "cng": {

        "fuel_name": "CNG",

        "price_per_unit": 76.09,

        "mileage": 20.0,

        "unit": "kg"

    }

}


EMISSION_FACTORS = {

    "road": 0.062,

    "air": 0.602

}


TOPSIS_WEIGHTS = {

    "risk": 0.35,

    "time": 0.30,

    "cost": 0.20,

    "distance": 0.15,

    "carbon": 0.10

}


def find_input_file(config, filename):

    search_locations = [

        config.raw_dir / filename,

        config.processed_dir / filename,

        config.project_root / filename,

        config.project_root / "data" / filename

    ]

    for path in search_locations:

        if path.exists():

            return path

    return None


def discover_project_inputs(config):

    filenames = {

        "roads": "uttarakhand_roads.gpkg",

        "boundary": "uttarakhand_boundary.gpkg",

        "dem": "uttarakhand_dem.tif",

        "slope": "slope.tif",

        "aspect": "aspect.tif",

        "tri": "tri.tif",

        "weather": "RF25_ind2025_rfp25.nc",

        "airports": "uttarakhand_airports.gpkg",

        "route_terrain": "rich_route_terrain.gpkg",

        "best_route": "best_route.gpkg"

    }

    inputs = {}

    logging.info("Searching project datasets...")

    for key, filename in filenames.items():

        path = find_input_file(
            config,
            filename
        )

        inputs[key] = path

        if path is not None:

            logging.info(
                "FOUND %-15s : %s",
                key,
                path
            )

        else:

            logging.warning(
                "NOT FOUND %-15s : %s",
                key,
                filename
            )

    return inputs


def export_report_reference_tables(config):

    config.create_directories()

    location_rows = []

    for name in CORRIDOR_ORDER:

        data = LOCATIONS[name]

        location_rows.append({

            "location": name,

            "latitude": data["latitude"],

            "longitude": data["longitude"],

            "elevation_m": data["elevation_m"],

            "type": data["type"]

        })

    pd.DataFrame(
        location_rows
    ).to_csv(

        config.results_dir /
        "report_corridor_locations.csv",

        index=False

    )

    pd.DataFrame(
        EMERGENCY_ROUTE_SEGMENTS
    ).to_csv(

        config.results_dir /
        "report_emergency_route_segments.csv",

        index=False

    )

    pd.DataFrame(
        REPORT_ROUTE_COMPARISON
    ).to_csv(

        config.results_dir /
        "report_route_comparison.csv",

        index=False

    )

    weight_rows = []

    total_weight = sum(
        TOPSIS_WEIGHTS.values()
    )

    for criterion, weight in TOPSIS_WEIGHTS.items():

        weight_rows.append({

            "criterion": criterion,

            "reported_weight": weight,

            "normalized_weight": (
                weight / total_weight
            )

        })

    pd.DataFrame(
        weight_rows
    ).to_csv(

        config.results_dir /
        "report_topsis_weights.csv",

        index=False

    )

    logging.info(
        "Report reference tables exported."
    )


def create_location_geodataframe():

    records = []

    for name, data in LOCATIONS.items():

        records.append({

            "name": name,

            "latitude": data["latitude"],

            "longitude": data["longitude"],

            "elevation_m": data["elevation_m"],

            "type": data["type"],

            "geometry": Point(

                data["longitude"],

                data["latitude"]

            )

        })

    locations_gdf = gpd.GeoDataFrame(

        records,

        geometry="geometry",

        crs=SOURCE_CRS

    )

    return locations_gdf


def save_location_layer(config):

    locations_gdf = create_location_geodataframe()

    output = (
        config.processed_dir /
        "report_locations.gpkg"
    )

    locations_gdf.to_file(

        output,

        layer="report_locations",

        driver="GPKG"

    )

    logging.info(
        "Saved report locations: %s",
        output
    )

    return locations_gdf


def load_vector_dataset(path, dataset_name):

    if path is None:

        logging.warning(
            "%s dataset is unavailable.",
            dataset_name
        )

        return None

    try:

        gdf = gpd.read_file(path)

        if gdf.empty:

            logging.warning(
                "%s dataset is empty.",
                dataset_name
            )

            return None

        if gdf.crs is None:

            raise ValueError(
                f"{dataset_name} does not contain CRS metadata."
            )

        gdf = gdf[
            gdf.geometry.notna()
        ].copy()

        gdf = gdf[
            ~gdf.geometry.is_empty
        ].copy()

        logging.info(

            "%s loaded successfully: %d features | CRS: %s",

            dataset_name,

            len(gdf),

            gdf.crs

        )

        return gdf

    except Exception as error:

        logging.exception(
            "Error loading %s: %s",
            dataset_name,
            error
        )

        return None


def preprocess_road_network(roads, config):

    if roads is None:

        raise FileNotFoundError(
            "Road network is required for A* routing. "
            "Expected data/raw/uttarakhand_roads.gpkg"
        )

    logging.info(
        "Starting road network preprocessing..."
    )

    roads = roads.copy()

    roads = roads[
        roads.geometry.geom_type.isin([

            "LineString",

            "MultiLineString"

        ])
    ].copy()

    roads = roads[
        roads.geometry.is_valid
    ].copy()

    roads = roads[
        ~roads.geometry.is_empty
    ].copy()

    roads = roads.to_crs(
        METRIC_CRS
    )

    roads["geometry"] = roads.geometry.apply(

        lambda geometry:

        linemerge(geometry)

        if geometry.geom_type == "MultiLineString"

        else geometry

    )

    roads = roads[
        roads.geometry.geom_type == "LineString"
    ].copy()

    roads["length_m"] = (
        roads.geometry.length
    )

    roads = roads[
        roads["length_m"] > 1
    ].copy()

    roads = roads.reset_index(
        drop=True
    )

    roads["road_id"] = np.arange(
        len(roads)
    )

    output = (
        config.processed_dir /
        "clean_roads.gpkg"
    )

    roads.to_file(

        output,

        layer="clean_roads",

        driver="GPKG"

    )

    logging.info(
        "Road preprocessing completed."
    )

    logging.info(
        "Clean road segments: %d",
        len(roads)
    )

    logging.info(
        "Road CRS: %s",
        roads.crs
    )

    return roads