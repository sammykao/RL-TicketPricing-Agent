import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from demand_modeling.model_serializer import load_model
from demand_modeling.data_extractor import extract_sales_data
from env.feature_builder import build_features_from_state


def plot_demand_curve_actual(
    db_path: Path,
    model_path: Path,
    target_event_id: int,
    fixed_time_remaining: float = 720.0,
    fixed_quality: float = None,
    is_weekend: bool = None,
    is_playoff: bool = None,
    n_price_points: int = 80
):
    """
    Plot actual observed (raw) demand vs model-predicted demand curve
    for a given event at a fixed time remaining.

    Raw data: empirical sale probability = sold_count / exposure
    Model: predicted P(sale | price, time, quality, context)
    """

    # LOAD MODEL
    model = load_model(model_path)

    # LOAD RAW DATA
    df = extract_sales_data(db_path)
    df_event = df[df["event_id"] == target_event_id].copy()

    if df_event.empty:
        raise ValueError(f"No data found for event_id={target_event_id}")

    # --- FILTER DATA TO FIX TIME BUCKET ---
    # Choose data points near the target time
    df_event["time_diff"] = np.abs(df_event["time_remaining"] - fixed_time_remaining)
    df_subset = df_event[df_event["time_diff"] < 30]   # Within ±30 hours

    if df_subset.empty:
        raise ValueError("No bins near that time_remaining for this event.")

    # --- RAW DATA SCATTER ---
    raw_prices = df_subset["current_price"].values
    raw_probs = (df_subset["sold_count"] / df_subset["exposure"]).values

    # If quality/time/context not fixed, choose defaults from first row
    row0 = df_subset.iloc[0]
    quality = fixed_quality if fixed_quality is not None else row0["quality_score"]
    weekend = is_weekend if is_weekend is not None else bool(row0["is_weekend"])
    playoff = is_playoff if is_playoff is not None else bool(row0["is_playoff"])
    init_price = float(row0["initial_price"])

    event_context = {
        "is_weekend": weekend,
        "is_playoff": playoff,
        "day_of_week": row0["day_of_week"]
    }

    # --- MODEL CURVE ---
    price_grid = np.linspace(raw_prices.min(), raw_prices.max(), n_price_points)
    model_probs = []

    for price in price_grid:
        x = build_features_from_state(
            time_remaining=fixed_time_remaining,
            current_price=price,
            initial_price=init_price,
            quality_score=quality,
            event_context=event_context,
            include_interactions=True
        )
        p = model.predict_proba(x.reshape(1, -1))[0, 1]
        model_probs.append(p)

    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    plt.scatter(raw_prices, raw_probs, alpha=0.5, color="blue", label="Raw empirical demand")
    plt.plot(price_grid, model_probs, color="red", linewidth=3, label="Logistic Regression Model")

    plt.title(f"Demand Curve for Event {target_event_id} (time_remaining={fixed_time_remaining}h)")
    plt.xlabel("Price")
    plt.ylabel("Sale Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    DB = Path("data_generation/db.sqlite")
    MODEL = Path("models/demand_model_v1.pkl")

    plot_demand_curve_actual(
        db_path=DB,
        model_path=MODEL,
        target_event_id=123,      # <-- CHANGE THIS
        fixed_time_remaining=720, # 30 days before event
    )
