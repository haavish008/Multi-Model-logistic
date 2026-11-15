
from pathlib import Path

import geopandas as gpd
import networkx as nx

from shapely.geometry import LineString, Point


def export_graph_layers(
    graph,
    output_dir,
    crs="EPSG:32644"
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    node_records = []

    for node, attributes in graph.nodes(data=True):

        if "x" not in attributes or "y" not in attributes:
            continue

        node_records.append({
            "node_id": str(node),
            "geometry": Point(
                attributes["x"],
                attributes["y"]
            )
        })

    edge_records = []

    for u, v, attributes in graph.edges(data=True):

        u_data = graph.nodes[u]
        v_data = graph.nodes[v]

        if not all(
            key in u_data and key in v_data
            for key in ("x", "y")
        ):
            continue

        geometry = LineString([
            (u_data["x"], u_data["y"]),
            (v_data["x"], v_data["y"])
        ])

        edge_records.append({
            "from_node": str(u),
            "to_node": str(v),
            "length_m": attributes.get("length_m"),
            "risk": attributes.get("dynamic_risk"),
            "travel_hours": attributes.get("travel_hours"),
            "is_blocked": attributes.get("is_blocked", False),
            "geometry": geometry
        })

    if node_records:

        nodes_gdf = gpd.GeoDataFrame(
            node_records,
            geometry="geometry",
            crs=crs
        )

        nodes_gdf.to_file(
            output_dir / "graph_nodes.gpkg",
            layer="graph_nodes",
            driver="GPKG"
        )

    if edge_records:

        edges_gdf = gpd.GeoDataFrame(
            edge_records,
            geometry="geometry",
            crs=crs
        )

        edges_gdf.to_file(
            output_dir / "graph_edges.gpkg",
            layer="graph_edges",
            driver="GPKG"
        )


def export_route(
    graph,
    route,
    output_path,
    route_name="Optimized Route",
    crs="EPSG:32644"
):

    if not route or len(route) < 2:
        raise ValueError("Invalid route.")

    coordinates = []

    for node in route:

        attributes = graph.nodes[node]

        if "x" not in attributes or "y" not in attributes:
            raise ValueError(
                f"Node {node} is missing projected coordinates."
            )

        coordinates.append((
            attributes["x"],
            attributes["y"]
        ))

    route_geometry = LineString(coordinates)

    gdf = gpd.GeoDataFrame(
        [{
            "route_name": route_name,
            "nodes": len(route),
            "geometry": route_geometry
        }],
        geometry="geometry",
        crs=crs
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    gdf.to_file(
        output_path,
        layer="optimized_route",
        driver="GPKG"
    )

    return output_path


def export_resources(
    resources_gdf,
    output_path,
    target_crs="EPSG:32644"
):

    if resources_gdf.crs is None:
        raise ValueError(
            "Resource layer CRS is undefined. "
            "Set the correct source CRS before exporting."
        )

    output = resources_gdf.to_crs(target_crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_file(
        output_path,
        layer="emergency_resources",
        driver="GPKG"
    )

    return output_path
