def launch_streamlit_dashboard():

    import streamlit as st
    import matplotlib.pyplot as plt

    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    RESULTS = ROOT / "results"

    st.set_page_config(
        page_title="Uttarakhand Logistics AI",
        page_icon="🗺️",
        layout="wide"
    )

    st.title(
        "AI-Based Military Logistics Decision Support System"
    )

    st.caption(
        "Dehradun – Joshimath | NH-7 | Uttarakhand"
    )

    st.sidebar.header("Mission Configuration")

    mission_type = st.sidebar.selectbox(
        "Mission Type",
        ["Normal Logistics", "Emergency Logistics"]
    )

    source = st.sidebar.selectbox(
        "Source",
        ["Dehradun"]
    )

    destination = st.sidebar.selectbox(
        "Destination",
        ["Joshimath"]
    )

    st.sidebar.write("Mission:", mission_type)
    st.sidebar.write("Route:", source, "→", destination)

    summary_file = RESULTS / "mission_summary.csv"

    if summary_file.exists():

        summary = pd.read_csv(summary_file)

        if not summary.empty:

            row = summary.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Distance",
                f"{row['distance_km']:.2f} km"
            )

            col2.metric(
                "Travel Time",
                f"{row['travel_hours']:.2f} hours"
            )

            col3.metric(
                "Fuel Cost",
                f"₹{row['fuel_cost_inr']:.2f}"
            )

    else:

        st.warning(
            "Mission summary not found. Run the main pipeline first."
        )

    st.subheader("Route Map")

    map_file = RESULTS / "uttarakhand_logistics_map.html"

    if map_file.exists():

        import streamlit.components.v1 as components

        components.html(
            map_file.read_text(encoding="utf-8"),
            height=600,
            scrolling=True
        )

    else:

        st.info("Route map has not been generated.")

    for filename, heading in [
        ("alternative_routes.csv", "Alternative Routes"),
        ("algorithm_comparison.csv", "Algorithm Comparison"),
        ("topsis_route_ranking.csv", "TOPSIS Ranking"),
        ("landslide_buffer_summary.csv", "Landslide Analysis"),
    ]:

        st.subheader(heading)

        file_path = RESULTS / filename

        if file_path.exists():

            st.dataframe(
                pd.read_csv(file_path),
                use_container_width=True
            )

        else:

            st.info(f"{filename} is not available yet.")
