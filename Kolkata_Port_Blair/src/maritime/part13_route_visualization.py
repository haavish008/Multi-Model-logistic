
import geopandas as gpd
import folium

from config.port_config import OUTPUT_DIR

NODES_FILE = OUTPUT_DIR / "maritime_nodes.gpkg"
EDGES_FILE = OUTPUT_DIR / "maritime_edges.gpkg"
BEST_ROUTE_FILE = OUTPUT_DIR / "astar_best_route.gpkg"
ALTERNATIVE_FILE = OUTPUT_DIR / "alternative_routes.gpkg"

MAP_FILE = OUTPUT_DIR / "maritime_route_map.html"

def to_wgs84(file_path):
    gdf = gpd.read_file(file_path)

    if gdf.crs is None:
        raise ValueError(f"Missing CRS: {file_path}")

    return gdf.to_crs("EPSG:4326")

def run_visualization():
    nodes = to_wgs84(NODES_FILE)
    edges = to_wgs84(EDGES_FILE)

    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()

    maritime_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    edge_layer = folium.FeatureGroup(name="Conceptual Corridor")

    for _, row in edges.iterrows():
        geometry = row.geometry

        if geometry.geom_type == "LineString":
            coordinates = [
                [lat, lon]
                for lon, lat in geometry.coords
            ]

            folium.PolyLine(
                coordinates,
                color="blue",
                weight=3,
                opacity=0.7
            ).add_to(edge_layer)

    edge_layer.add_to(maritime_map)

    node_layer = folium.FeatureGroup(name="Waypoints")

    for _, row in nodes.iterrows():
        folium.CircleMarker(
            location=[
                row.geometry.y,
                row.geometry.x
            ],
            radius=5,
            color="darkred",
            fill=True,
            fill_opacity=0.9,
            popup=str(row.get("node_id", "Waypoint"))
        ).add_to(node_layer)

    node_layer.add_to(maritime_map)

    if BEST_ROUTE_FILE.exists():
        best_route = to_wgs84(BEST_ROUTE_FILE)

        for _, row in best_route.iterrows():
            if row.geometry.geom_type == "LineString":
                coordinates = [
                    [lat, lon]
                    for lon, lat in row.geometry.coords
                ]

                folium.PolyLine(
                    coordinates,
                    color="green",
                    weight=5,
                    opacity=0.9,
                    tooltip="A* conceptual route"
                ).add_to(maritime_map)

    if ALTERNATIVE_FILE.exists():
        alternatives = to_wgs84(ALTERNATIVE_FILE)

        for _, row in alternatives.iterrows():
            if row.geometry.geom_type == "LineString":
                coordinates = [
                    [lat, lon]
                    for lon, lat in row.geometry.coords
                ]

                folium.PolyLine(
                    coordinates,
                    color="orange",
                    weight=3,
                    opacity=0.8,
                    dash_array="8, 8",
                    tooltip="Alternative conceptual route"
                ).add_to(maritime_map)

    folium.LayerControl().add_to(maritime_map)

    maritime_map.save(str(MAP_FILE))

    print("Maritime map generated successfully.")
    print(f"Saved: {MAP_FILE}")

if __name__ == "__main__":
    run_visualization()
