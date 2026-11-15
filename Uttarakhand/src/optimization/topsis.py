
import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "risk": 0.30,
    "time": 0.25,
    "cost": 0.20,
    "distance": 0.15,
    "carbon": 0.10
}

DEFAULT_CRITERIA = {
    "risk": "cost",
    "time": "cost",
    "cost": "cost",
    "distance": "cost",
    "carbon": "cost"
}


def calculate_topsis_ranking(
    alternatives,
    weights=None,
    criteria=None
):

    weights = weights or DEFAULT_WEIGHTS
    criteria = criteria or DEFAULT_CRITERIA

    columns = list(weights.keys())

    missing = [
        col for col in columns
        if col not in alternatives.columns
    ]

    if missing:
        raise ValueError(
            f"Missing TOPSIS columns: {missing}"
        )

    data = alternatives[columns].astype(float).copy()

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(
            "TOPSIS input contains missing or non-finite values."
        )

    if (data < 0).any().any():
        raise ValueError(
            "Negative criteria values are not supported."
        )

    weight_values = np.array(
        [weights[col] for col in columns],
        dtype=float
    )

    if (weight_values < 0).any():
        raise ValueError("Weights cannot be negative.")

    if weight_values.sum() == 0:
        raise ValueError("At least one weight must be positive.")

    weight_values = weight_values / weight_values.sum()

    # Vector normalization
    denominator = np.sqrt(
        (data ** 2).sum(axis=0)
    ).replace(0, 1)

    normalized = data / denominator

    weighted = normalized * weight_values

    ideal_best = []
    ideal_worst = []

    for col in columns:

        if criteria[col] == "benefit":
            ideal_best.append(weighted[col].max())
            ideal_worst.append(weighted[col].min())

        elif criteria[col] == "cost":
            ideal_best.append(weighted[col].min())
            ideal_worst.append(weighted[col].max())

        else:
            raise ValueError(
                f"Unknown criterion direction for {col}"
            )

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    distance_best = np.sqrt(
        ((weighted - ideal_best) ** 2).sum(axis=1)
    )

    distance_worst = np.sqrt(
        ((weighted - ideal_worst) ** 2).sum(axis=1)
    )

    denominator = distance_best + distance_worst

    scores = np.divide(
        distance_worst,
        denominator,
        out=np.full(len(data), 0.5),
        where=denominator != 0
    )

    result = alternatives.copy()

    result["topsis_score"] = scores
    result["topsis_rank"] = (
        result["topsis_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return result.sort_values(
        "topsis_rank"
    ).reset_index(drop=True)


def rank_route_alternatives(
    route_performance,
    weights=None
):

    mapping = {
        "mean_risk": "risk",
        "travel_hours": "time",
        "fuel_cost_inr": "cost",
        "distance_km": "distance",
        "carbon_emissions_kg": "carbon"
    }

    missing = [
        source for source in mapping
        if source not in route_performance.columns
    ]

    if missing:
        raise ValueError(
            f"Missing route metrics: {missing}"
        )

    alternatives = route_performance.rename(
        columns=mapping
    ).copy()

    return calculate_topsis_ranking(
        alternatives,
        weights=weights
    )
