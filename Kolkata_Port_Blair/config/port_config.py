
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MARITIME_DIR = RAW_DIR / "maritime"
PORTS_DIR = RAW_DIR / "ports"
WEATHER_DIR = RAW_DIR / "weather"
OCEAN_DIR = RAW_DIR / "ocean"
BATHYMETRY_DIR = RAW_DIR / "bathymetry"

MARITIME_PROCESSED_DIR = PROCESSED_DIR / "maritime"

SOURCE = "Kolkata Port"
DESTINATION = "Port Blair"

WAYPOINTS = [
    {"name": "Kolkata Port", "latitude": 22.57, "longitude": 88.36},
    {"name": "WP-1", "latitude": 20.30, "longitude": 89.30},
    {"name": "WP-2", "latitude": 17.50, "longitude": 90.80},
    {"name": "WP-3", "latitude": 14.50, "longitude": 91.80},
    {"name": "WP-4", "latitude": 12.20, "longitude": 92.30},
    {"name": "Port Blair", "latitude": 11.67, "longitude": 92.75},
]

DATA_FILES = {
    "ports": PORTS_DIR,
    "maritime": MARITIME_DIR,
    "weather": WEATHER_DIR,
    "ocean": OCEAN_DIR,
    "bathymetry": BATHYMETRY_DIR,
}

OUTPUT_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

for directory in [
    MARITIME_PROCESSED_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    LOG_DIR,
    CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
