
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium

from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "maritime"

st.set_page_config(
    page_title="Maritime Logistics Optimization",
    page_icon="🚢",
    layout="wide"
)

def load_csv(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()

def load_json(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        with open(path) as file:
            return json.load(file)

    return {}

def load_geodata(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        return gpd.read_file(path)

    return gpd.GeoDataFrame()

st.title("Maritime Logistics Optimization System")
st.caption("Kolkata Port to Port Blair | Research Prototype")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Route Optimization",
        "Environmental Analysis",
        "Fuel and Time",
        "Port Congestion",
        "Performance Evaluation"
    ]
)

comparison = load_csv("final_route_comparison.csv")
ranking = load_csv("advanced_topsis_ranking.csv")
fuel = load_csv("fuel_time_route_estimates.csv")
congestion = load_csv("port_congestion_analysis.csv")
performance = load_csv("performance_evaluation.csv")
risk = load_csv("maritime_risk_indicators.csv")

nodes = load_geodata("real_maritime_nodes.gpkg")
route = load_geodata("genetic_best_route.gpkg")
astar_route = load_geodata("astar_best_route.gpkg")
dstar_route = load_geodata("dstar_updated_route.gpkg")

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Route Records", len(comparison))
    col2.metric("TOPSIS Candidates", len(ranking))
    col3.metric("Environmental Records", len(risk))
    col4.metric("Port Records", len(congestion))

    st.divider()

    st.subheader("Algorithm Comparison")

    if not comparison.empty:
        columns = [
            column for column in [
                "algorithm",
                "distance_km",
                "travel_time_hours",
                "fuel_tonnes",
                "topsis_score"
            ]
            if column in comparison.columns
        ]

        st.dataframe(
            comparison[columns],
            use_container_width=True
        )
    else:
        st.info("Route comparison data is unavailable.")

    st.subheader("Latest Performance Evaluation")

    if not performance.empty:
        st.dataframe(
            performance,
            use_container_width=True
        )
    else:
        st.info("Performance evaluation is unavailable.")

elif page == "Route Optimization":

    st.header("Maritime Route Optimization")

    if not nodes.empty:
        nodes = nodes.to_crs("EPSG:4326")

        center = [
            nodes.geometry.y.mean(),
            nodes.geometry.x.mean()
        ]

        maritime_map = folium.Map(
            location=center,
            zoom_start=5
        )

        for _, row in nodes.iterrows():
            folium.CircleMarker(
                location=[
                    row.geometry.y,
                    row.geometry.x
                ],
                radius=4,
                popup=str(row.get("node_id", "Waypoint")),
                color="blue",
                fill=True
            ).add_to(maritime_map)

        route_layers = [
            (route, "Genetic Algorithm", "green"),
            (astar_route, "A* Route", "blue"),
            (dstar_route, "D* Lite Route", "red")
        ]

        for route_data, label, color in route_layers:
            if route_data.empty:
                continue

            route_data = route_data.to_crs("EPSG:4326")

            for _, row in route_data.iterrows():
                if row.geometry.geom_type == "LineString":
                    coordinates = [
                        [lat, lon]
                        for lon, lat in row.geometry.coords
                    ]

                    folium.PolyLine(
                        coordinates,
                        color=color,
                        weight=4,
                        tooltip=label
                    ).add_to(maritime_map)

        folium.LayerControl().add_to(maritime_map)

        st_folium(
            maritime_map,
            width=None,
            height=600
        )

    else:
        st.warning("Maritime network data is unavailable.")

    st.subheader("TOPSIS Route Ranking")

    if not ranking.empty:
        st.dataframe(ranking, use_container_width=True)
    else:
        st.info("TOPSIS ranking is unavailable.")

elif page == "Environmental Analysis":

    st.header("Environmental Analysis")

    if not risk.empty:
        st.dataframe(risk, use_container_width=True)

        numeric_columns = [
            column for column in risk.select_dtypes(
                include="number"
            ).columns
            if risk[column].notna().any()
        ]

        if numeric_columns:
            selected_metric = st.selectbox(
                "Select environmental metric",
                numeric_columns
            )

            st.plotly_chart(
                px.histogram(
                    risk,
                    x=selected_metric,
                    title=f"Distribution of {selected_metric}"
                ),
                use_container_width=True
            )
    else:
        st.info("Environmental risk data is unavailable.")

    environment = load_csv(
        "vessel_environment_assessment.csv"
    )

    if not environment.empty:
        st.subheader("Vessel Environmental Assessment")
        st.dataframe(environment, use_container_width=True)

elif page == "Fuel and Time":

    st.header("Fuel Consumption and Travel Time")

    if not fuel.empty:
        st.dataframe(fuel, use_container_width=True)

        valid = fuel.dropna(
            subset=["travel_time_hours", "estimated_fuel_tonnes"]
        )

        if not valid.empty:
            st.plotly_chart(
                px.bar(
                    valid,
                    x="route_id",
                    y="travel_time_hours",
                    title="Estimated Travel Time by Route"
                ),
                use_container_width=True
            )

            st.plotly_chart(
                px.bar(
                    valid,
                    x="route_id",
                    y="estimated_fuel_tonnes",
                    title="Estimated Fuel Consumption by Route"
                ),
                use_container_width=True
            )
    else:
        st.info("Fuel and travel-time estimates are unavailable.")

elif page == "Port Congestion":

    st.header("Port Congestion Analysis")

    if not congestion.empty:
        st.dataframe(congestion, use_container_width=True)

        st.plotly_chart(
            px.bar(
                congestion,
                x="port_name",
                y="average_wait_hours",
                title="Average Port Waiting Time"
            ),
            use_container_width=True
        )

        st.plotly_chart(
            px.bar(
                congestion,
                x="port_name",
                y="average_berth_utilization",
                title="Average Berth Utilization"
            ),
            use_container_width=True
        )
    else:
        st.info("Port congestion data is unavailable.")

elif page == "Performance Evaluation":

    st.header("Algorithm Performance Evaluation")

    if not performance.empty:
        st.dataframe(performance, use_container_width=True)

        if "minimum_distance_km" in performance.columns:
            st.plotly_chart(
                px.bar(
                    performance,
                    x="algorithm",
                    y="minimum_distance_km",
                    title="Minimum Route Distance"
                ),
                use_container_width=True
            )

        if "average_travel_time_hours" in performance.columns:
            st.plotly_chart(
                px.bar(
                    performance,
                    x="algorithm",
                    y="average_travel_time_hours",
                    title="Average Travel Time"
                ),
                use_container_width=True
            )
    else:
        st.info("Performance evaluation data is unavailable.")

st.divider()

st.caption(
    "Research prototype only. Displayed routes and depth assessments "
    "are not certified for real-world vessel navigation."
)
