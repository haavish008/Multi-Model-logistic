
from src.maritime.part01_data_validation import run_validation as validate_datasets
from src.maritime.part02_data_loading import run_data_loading
from src.maritime.part03_data_cleaning import run_data_cleaning
from src.maritime.part04_waypoint_network import run_waypoint_network
from src.maritime.part05_bathymetry_analysis import run_bathymetry_analysis
from src.maritime.part06_weather_integration import run_weather_integration
from src.maritime.part07_ocean_integration import run_ocean_integration
from src.maritime.part08_risk_analysis import run_risk_analysis
from src.maritime.part09_edge_cost import run_edge_cost
from src.maritime.part10_astar_routing import run_astar
from src.maritime.part11_alternative_routes import run_alternative_routes
from src.maritime.part12_topsis import run_topsis
from src.maritime.part13_route_visualization import run_visualization
from src.maritime.part14_validation import run_validation as validate_system

def main():
    steps = [
        ("Dataset Validation", validate_datasets),
        ("Data Loading", run_data_loading),
        ("Data Cleaning", run_data_cleaning),
        ("Waypoint Network", run_waypoint_network),
        ("Bathymetry Analysis", run_bathymetry_analysis),
        ("Weather Integration", run_weather_integration),
        ("Ocean Integration", run_ocean_integration),
        ("Risk Analysis", run_risk_analysis),
        ("Edge Cost Calculation", run_edge_cost),
        ("A* Route Optimization", run_astar),
        ("Alternative Route Generation", run_alternative_routes),
        ("TOPSIS Ranking", run_topsis),
        ("Route Visualization", run_visualization),
        ("System Validation", validate_system)
    ]

    for name, function in steps:
        print(f"\n{'=' * 60}")
        print(name)
        print("=" * 60)
        function()

    print("\nMaritime logistics pipeline execution completed.")

if __name__ == "__main__":
    main()
