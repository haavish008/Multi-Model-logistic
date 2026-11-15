
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import rowcol


def create_projected_dem(
    dem_path,
    output_path,
    target_crs="EPSG:32644"
):

    with rasterio.open(dem_path) as source:

        transform, width, height = calculate_default_transform(
            source.crs,
            target_crs,
            source.width,
            source.height,
            *source.bounds
        )

        profile = source.profile.copy()

        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as destination:

            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(destination, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

    return output_path


def calculate_slope_aspect(
    dem_path,
    slope_output,
    aspect_output
):

    with rasterio.open(dem_path) as src:

        elevation = src.read(1).astype(float)
        nodata = src.nodata

        valid = np.isfinite(elevation)

        if nodata is not None:
            valid &= elevation != nodata

        elevation[~valid] = np.nan

        x_resolution = abs(src.transform.a)
        y_resolution = abs(src.transform.e)

        # Gradient in elevation units per horizontal metre.
        gradient_y, gradient_x = np.gradient(
            elevation,
            y_resolution,
            x_resolution
        )

        slope = np.degrees(
            np.arctan(
                np.sqrt(
                    gradient_x ** 2
                    + gradient_y ** 2
                )
            )
        )

        # Aspect measured clockwise from north.
        aspect = (
            np.degrees(
                np.arctan2(
                    -gradient_x,
                    gradient_y
                )
            ) + 360
        ) % 360

        aspect[
            (gradient_x == 0) &
            (gradient_y == 0)
        ] = np.nan

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            compress="lzw"
        )

        for output_path, array in [
            (slope_output, slope),
            (aspect_output, aspect)
        ]:

            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst:

                dst.write(
                    array.astype("float32"),
                    1
                )

    return slope_output, aspect_output


def sample_raster_at_points(
    raster_path,
    points
):

    values = []

    with rasterio.open(raster_path) as src:

        if points.crs is None:
            raise ValueError(
                "Point CRS must be defined."
            )

        points = points.to_crs(src.crs)

        for point in points.geometry:

            if point is None or point.is_empty:
                values.append(np.nan)
                continue

            row, col = rowcol(
                src.transform,
                point.x,
                point.y
            )

            if (
                row < 0 or col < 0
                or row >= src.height
                or col >= src.width
            ):
                values.append(np.nan)
                continue

            value = src.read(1)[row, col]

            if src.nodata is not None and value == src.nodata:
                values.append(np.nan)
            else:
                values.append(float(value))

    return np.asarray(values)


def enrich_roads_with_terrain(
    roads,
    dem_path,
    slope_path,
    aspect_path
):

    if roads.crs is None:
        raise ValueError("Road CRS is undefined.")

    projected_roads = roads.to_crs("EPSG:32644")

    midpoints = projected_roads.geometry.interpolate(
        0.5,
        normalized=True
    )

    points = gpd.GeoDataFrame(
        geometry=midpoints,
        crs=projected_roads.crs
    )

    result = roads.copy().reset_index(drop=True)

    result["elevation_m"] = sample_raster_at_points(
        dem_path,
        points
    )

    result["slope_deg"] = sample_raster_at_points(
        slope_path,
        points
    )

    result["aspect_deg"] = sample_raster_at_points(
        aspect_path,
        points
    )

    # Terrain difficulty indicator for analysis.
    # It is not a validated engineering passability score.
    result["terrain_difficulty"] = (
        result["slope_deg"].clip(0, 90) / 90
    )

    return result


def run_terrain_pipeline(
    road_path,
    dem_path,
    output_dir
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    projected_dem = (
        output_dir / "dem_utm44.tif"
    )

    slope_path = (
        output_dir / "slope.tif"
    )

    aspect_path = (
        output_dir / "aspect.tif"
    )

    create_projected_dem(
        dem_path,
        projected_dem
    )

    calculate_slope_aspect(
        projected_dem,
        slope_path,
        aspect_path
    )

    roads = gpd.read_file(road_path)

    result = enrich_roads_with_terrain(
        roads,
        projected_dem,
        slope_path,
        aspect_path
    )

    output_path = (
        output_dir / "rich_route_terrain.gpkg"
    )

    result.to_file(
        output_path,
        layer="roads_terrain",
        driver="GPKG"
    )

    print("Terrain output:", output_path)

    return result


if __name__ == "__main__":

    project = Path.home() / "Downloads" / "Uttarakhand"

    run_terrain_pipeline(
        road_path=(
            project / "data/processed/uttarakhand_roads.gpkg"
        ),
        dem_path=(
            project / "data/raw/elevation/uttarakhand_dem.tif"
        ),
        output_dir=(
            project / "results/terrain"
        )
    )
