
from pathlib import Path

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

RESULTS_DIR = PROJECT_ROOT / "results"


st.set_page_config(
    page_title="Uttarakhand Logistics Decision Support",
    page_icon="🗺️",
    layout="wide"
)


@st.cache_data
def load_csv(filename):

    path = RESULTS_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_geojson(filename, layer=None):

    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    return gpd.read_file(
        path,
        layer=layer
    )


st.title(
    "AI Military Logistics Decision Support System"
)

st.caption(
    "Uttarakhand | Dehradun–Joshimath Corridor"
)

st.sidebar.header("Mission Planning")

source = st.sidebar.text_input(
    "Source",
    "Dehradun"
)

destination = st.sidebar.text_input(
    "Destination",
    "Joshimath"
)

st.sidebar.info(
    "This dashboard displays previously generated "
    "project results. It does not independently verify "
    "live road or weather conditions."
)


# ------------------------------------------------------------
# ROUTE COMPARISON
# ------------------------------------------------------------

st.header("Algorithm Performance")

performance = load_csv(
    "integrated_route_comparison.csv"
)

if not performance.empty:

    numeric_columns = [
        column for column in [
            "distance_km",
            "travel_hours",
            "fuel_cost_inr",
            "mean_risk",
            "carbon_emissions_kg"
        ]
        if column in performance.columns
    ]

    if numeric_columns:

        selected_metric = st.selectbox(
            "Select performance metric",
            numeric_columns
        )

        if "algorithm" in performance.columns:

            chart = px.bar(
                performance,
                x="algorithm",
                y=selected_metric,
                title=selected_metric.replace(
                    "_",
                    " "
                ).title()
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

    st.dataframe(
        performance,
        use_container_width=True
    )

else:

    st.warning(
        "Route comparison file not found. "
        "Run the master pipeline first."
    )


# ------------------------------------------------------------
# TOPSIS RANKING
# ------------------------------------------------------------

st.header("TOPSIS Route Ranking")

topsis = load_csv(
    "topsis_route_ranking.csv"
)

if not topsis.empty:

    st.dataframe(
        topsis,
        use_container_width=True
    )

else:

    st.info(
        "TOPSIS ranking has not been exported yet."
    )


# ------------------------------------------------------------
# ROUTE MAP
# ------------------------------------------------------------

st.header("Optimized Route Map")

route_file = RESULTS_DIR / "best_route.gpkg"

if route_file.exists():

    try:

        route_gdf = gpd.read_file(
            route_file
        )

        if route_gdf.crs is None:
            st.error(
                "Route CRS is undefined. "
                "The map cannot be displayed reliably."
            )

        else:

            route_wgs84 = route_gdf.to_crs(
                "EPSG:4326"
            )

            st.map(
                route_wgs84
            )

            st.caption(
                "Map coordinates are converted to WGS84."
            )

    except Exception as error:

        st.error(
            f"Unable to load route map: {error}"
        )

else:

    st.warning(
        "best_route.gpkg was not found."
    )


# ------------------------------------------------------------
# EMERGENCY RESOURCES
# ------------------------------------------------------------

st.header("Emergency Resource Accessibility")

resources = load_csv(
    "nearest_resource.csv"
)

if not resources.empty:

    st.dataframe(
        resources,
        use_container_width=True
    )

else:

    st.info(
        "Emergency resource results are not available."
    )


# ------------------------------------------------------------
# BLOCKAGE TEST
# ------------------------------------------------------------

st.header("Road Blockage Scenario")

blockage = load_csv(
    "road_blockage_evaluation.csv"
)

if not blockage.empty:

    st.dataframe(
        blockage,
        use_container_width=True
    )

else:

    st.info(
        "No blockage scenario results are available."
    )


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

st.header("Project Output Status")

output_status = load_csv(
    "final_project_output_check.csv"
)

if not output_status.empty:

    st.dataframe(
        output_status,
        use_container_width=True
    )

st.caption(
    "Research prototype. All outputs depend on the "
    "quality, date and coverage of the underlying datasets."
)
