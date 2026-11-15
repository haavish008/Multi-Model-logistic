
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr


def open_weather_dataset(netcdf_path):
    netcdf_path = Path(netcdf_path)

    if not netcdf_path.exists():
        raise FileNotFoundError(netcdf_path)

    return xr.open_dataset(netcdf_path)


def inspect_weather_dataset(dataset):
    print("\nWEATHER DATASET")
    print(dataset)

    print("\nVARIABLES")
    for variable in dataset.data_vars:
        print(
            variable,
            dataset[variable].dims,
            dataset[variable].attrs
        )

    print("\nCOORDINATES")
    for coordinate in dataset.coords:
        print(
            coordinate,
            dataset[coordinate].values[:5]
            if dataset[coordinate].ndim == 1
            else dataset[coordinate].shape
        )


def resolve_coordinate_names(
    dataset,
    latitude_name=None,
    longitude_name=None,
    time_name=None
):
    names = set(dataset.coords) | set(dataset.dims)

    if latitude_name is None:
        latitude_name = next(
            (
                name for name in
                ["LATITUDE", "latitude", "lat", "Latitude"]
                if name in names
            ),
            None
        )

    if longitude_name is None:
        longitude_name = next(
            (
                name for name in
                ["LONGITUDE", "longitude", "lon", "Longitude"]
                if name in names
            ),
            None
        )

    if time_name is None:
        time_name = next(
            (
                name for name in
                ["TIME", "time", "date", "valid_time"]
                if name in names
            ),
            None
        )

    if latitude_name is None or longitude_name is None:
        raise ValueError(
            "Latitude/longitude coordinates not identified. "
            "Pass their exact names explicitly."
        )

    return latitude_name, longitude_name, time_name


def get_segment_midpoints(
    roads,
    source_crs="EPSG:4326"
):
    if roads.crs is None:
        raise ValueError("Road layer CRS is undefined.")

    # Calculate geometric midpoints in a projected CRS.
    projected = roads.to_crs("EPSG:32644")

    midpoints = projected.geometry.interpolate(0.5, normalized=True)

    midpoint_gdf = gpd.GeoDataFrame(
        roads.drop(columns="geometry").copy(),
        geometry=midpoints,
        crs=projected.crs
    )

    return midpoint_gdf.to_crs(source_crs)


def extract_weather_for_roads(
    roads,
    dataset,
    variable_name,
    latitude_name=None,
    longitude_name=None,
    time_name=None,
    start_date=None,
    end_date=None
):
    if variable_name not in dataset.data_vars:
        raise ValueError(
            f"Weather variable '{variable_name}' not found. "
            f"Available: {list(dataset.data_vars)}"
        )

    latitude_name, longitude_name, time_name = (
        resolve_coordinate_names(
            dataset,
            latitude_name,
            longitude_name,
            time_name
        )
    )

    weather = dataset[variable_name]

    # Optional time filtering.
    if time_name and start_date and end_date:
        weather = weather.sel(
            {
                time_name: slice(
                    start_date,
                    end_date
                )
            }
        )

    if time_name and weather.sizes.get(time_name, 0) == 0:
        raise ValueError("No weather records in selected date range.")

    midpoint_gdf = get_segment_midpoints(roads)

    latitude_values = midpoint_gdf.geometry.y.to_numpy()
    longitude_values = midpoint_gdf.geometry.x.to_numpy()

    records = []

    for position, (latitude, longitude) in enumerate(
        zip(latitude_values, longitude_values)
    ):

        # Nearest grid-cell selection.
        point_weather = weather.sel(
            {
                latitude_name: latitude,
                longitude_name: longitude
            },
            method="nearest"
        )

        values = np.asarray(
            point_weather.values,
            dtype=float
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            mean_value = np.nan
            max_value = np.nan
            min_value = np.nan
        else:
            mean_value = float(np.mean(values))
            max_value = float(np.max(values))
            min_value = float(np.min(values))

        records.append({
            "weather_mean": mean_value,
            "weather_max": max_value,
            "weather_min": min_value
        })

    weather_df = pd.DataFrame(records)

    result = roads.copy().reset_index(drop=True)

    weather_df = weather_df.reset_index(drop=True)

    for column in weather_df.columns:
        result[column] = weather_df[column]

    return result


def export_weather_roads(
    roads_with_weather,
    output_path
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    roads_with_weather.to_file(
        output_path,
        layer="roads_weather",
        driver="GPKG"
    )

    print("Saved:", output_path)

    return output_path


if __name__ == "__main__":

    project = Path.home() / "Downloads" / "Uttarakhand"

    netcdf_path = (
        project / "data/raw/weather/RF25_ind2025_rfp25.nc"
    )

    road_path = (
        project / "data/processed/uttarakhand_roads.gpkg"
    )

    output_path = (
        project / "results/rich_route_weather.gpkg"
    )

    dataset = open_weather_dataset(netcdf_path)

    inspect_weather_dataset(dataset)

    roads = gpd.read_file(road_path)

    # Replace "YOUR_RAINFALL_VARIABLE" with the actual
    # rainfall variable printed by inspect_weather_dataset().
    VARIABLE_NAME = "YOUR_RAINFALL_VARIABLE"

    if VARIABLE_NAME == "YOUR_RAINFALL_VARIABLE":
        raise ValueError(
            "Set VARIABLE_NAME to the actual NetCDF rainfall variable."
        )

    result = extract_weather_for_roads(
        roads,
        dataset,
        variable_name=VARIABLE_NAME
    )

    export_weather_roads(
        result,
        output_path
    )

    dataset.close()
