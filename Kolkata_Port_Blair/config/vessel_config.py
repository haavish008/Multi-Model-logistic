
VESSEL_CONFIG = {
    "vessel_name": "Research Vessel",
    "vessel_type": "Cargo vessel",
    "draft_m": None,
    "length_m": None,
    "beam_m": None,
    "cruising_speed_knots": None,
    "maximum_wave_height_m": None,
    "maximum_wind_speed_ms": None,
    "minimum_under_keel_clearance_m": None,
    "fuel_consumption_tonnes_per_hour": None
}

REQUIRED_PARAMETERS = [
    "vessel_type",
    "draft_m",
    "cruising_speed_knots",
    "minimum_under_keel_clearance_m"
]

def validate_vessel_config():
    missing = [
        key for key in REQUIRED_PARAMETERS
        if VESSEL_CONFIG.get(key) is None
    ]

    invalid = [
        key for key, value in VESSEL_CONFIG.items()
        if isinstance(value, (int, float)) and value <= 0
    ]

    if missing or invalid:
        print("Vessel configuration is incomplete.")
        print("Missing parameters:", missing)
        print("Invalid parameters:", invalid)
        return False

    print("Vessel configuration validated.")
    return True

if __name__ == "__main__":
    validate_vessel_config()
