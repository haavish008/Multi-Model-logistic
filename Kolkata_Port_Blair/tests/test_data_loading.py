
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest

from config.port_config import PROJECT_ROOT, OUTPUT_DIR

def test_project_directory_exists():
    assert PROJECT_ROOT.exists()

def test_raw_data_directory_exists():
    assert (PROJECT_ROOT / "data" / "raw").exists()

def test_processed_directory_exists():
    assert (PROJECT_ROOT / "data" / "processed").exists()

def test_output_directory_exists():
    assert OUTPUT_DIR.exists()

def test_generated_geopackages():
    files = list(OUTPUT_DIR.glob("*.gpkg"))

    for file in files:
        data = gpd.read_file(file)
        assert data.crs is not None
        assert data.geometry.notna().all()
        assert data.geometry.is_valid.all()

def test_generated_csv_files():
    files = list(OUTPUT_DIR.glob("*.csv"))

    for file in files:
        data = pd.read_csv(file)
        assert data is not None
        assert len(data.columns) > 0
