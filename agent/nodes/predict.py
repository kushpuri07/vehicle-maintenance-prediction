"""Node 1: Predict.

Loads the trained Decision Tree and produces a maintenance prediction
plus probability for the vehicle.
"""
from functools import lru_cache

import joblib
import pandas as pd

from agent.state import VehicleState

MODEL_PATH = "model_dt.pkl"

# Feature order must match training (see notebook Cell 9)
FEATURE_ORDER = [
    "Brake_Condition",
    "Tire_Condition",
    "Vehicle_Age",
    "Battery_Status",
    "Reported_Issues",
    "Service_History",
    "Odometer_Reading",
    "Insurance_Premium",
    "Accident_History",
    "Maintenance_History",
]


@lru_cache(maxsize=1)
def _load_model():
    """Load the pickled model once per process."""
    return joblib.load(MODEL_PATH)


def predict_node(state: VehicleState) -> VehicleState:
    """Run the ML model and add prediction + probability to state."""
    model = _load_model()
    raw = state["vehicle_input"]

    # Build dataframe in correct feature order
    features = {k: raw[k] for k in FEATURE_ORDER}
    df = pd.DataFrame([features])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])   # P(Need_Maintenance=1)

    return {
        "prediction": prediction,
        "probability": probability,
        "prediction_features": features,
    }
