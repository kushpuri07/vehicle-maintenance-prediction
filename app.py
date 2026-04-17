import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load Model ────────────────────────────────────────────────────
model = joblib.load('model_dt.pkl')

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="")
st.title("Vehicle Maintenance Predictor")
st.write("Fill in the vehicle details below to check if maintenance is needed.")

# ── Input Form ────────────────────────────────────────────────────
# Encodings match Cell 5 of the notebook:
#   Tire_Condition / Brake_Condition: New=0, Good=1, Worn Out=2
#   Battery_Status:                   New=0, Good=1, Weak=2
#   Maintenance_History:              Good=0, Average=1, Poor=2

col1, col2 = st.columns(2)

with col1:
    brake_condition = st.selectbox(
        "Brake Condition", [0, 1, 2],
        format_func=lambda x: {0: "New", 1: "Good", 2: "Worn Out"}[x],
        index=1,
    )
    tire_condition = st.selectbox(
        "Tire Condition", [0, 1, 2],
        format_func=lambda x: {0: "New", 1: "Good", 2: "Worn Out"}[x],
        index=1,
    )
    battery_status = st.selectbox(
        "Battery Status", [0, 1, 2],
        format_func=lambda x: {0: "New", 1: "Good", 2: "Weak"}[x],
        index=1,
    )
    maintenance_history = st.selectbox(
        "Maintenance History", [0, 1, 2],
        format_func=lambda x: {0: "Good", 1: "Average", 2: "Poor"}[x],
        index=1,
    )
    vehicle_age = st.slider("Vehicle Age (years)", 0, 15, 5)

with col2:
    reported_issues = st.selectbox("Reported Issues", [0, 1, 2, 3, 4, 5], index=2)
    service_history = st.slider("Service History (# of past services)", 0, 12, 4)
    accident_history = st.selectbox("Accident History", [0, 1, 2, 3], index=1)
    odometer_reading = st.number_input(
        "Odometer Reading (km)", min_value=0, max_value=400_000, value=60_000, step=1_000,
    )
    insurance_premium = st.number_input(
        "Insurance Premium", min_value=1_500, max_value=100_000, value=15_000, step=500,
    )

# ── Predict ───────────────────────────────────────────────────────
if st.button("Check Maintenance", type="primary"):
    # Column order must match training (useful_features in Cell 9)
    sample = pd.DataFrame([{
        'Brake_Condition':     brake_condition,
        'Tire_Condition':      tire_condition,
        'Vehicle_Age':         vehicle_age,
        'Battery_Status':      battery_status,
        'Reported_Issues':     reported_issues,
        'Service_History':     service_history,
        'Odometer_Reading':    odometer_reading,
        'Insurance_Premium':   insurance_premium,
        'Accident_History':    accident_history,
        'Maintenance_History': maintenance_history,
    }])

    result     = model.predict(sample)[0]
    proba      = model.predict_proba(sample)[0]
    confidence = round(max(proba) * 100, 2)

    # ── Prediction Result ─────────────────────────────────────────
    st.markdown("---")
    if result == 1:
        st.error(f"Maintenance Needed")
    else:
        st.success(f"No Maintenance Needed")

    st.markdown("### Prediction Confidence")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Model Confidence", value=f"{confidence}%")
    with c2:
        st.metric(label="Model Accuracy", value="94.45%")

    # ── Feature Importance Chart ──────────────────────────────────
    st.markdown("### Feature Importance")

    # Must match training order exactly
    feature_names = [
        'Brake_Condition', 'Tire_Condition', 'Vehicle_Age', 'Battery_Status',
        'Reported_Issues', 'Service_History', 'Odometer_Reading',
        'Insurance_Premium', 'Accident_History', 'Maintenance_History',
    ]
    importances = model.named_steps['model'].feature_importances_
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')
    bars = ax.barh(sorted_features, sorted_importances, color='#00C853')
    ax.set_xlabel('Importance Score', color='white')
    ax.set_title('Which features influence the prediction most?', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.xaxis.grid(True, color='#333333', linewidth=0.8)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{w:.3f}', va='center', color='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)