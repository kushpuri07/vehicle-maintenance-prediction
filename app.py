import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load Model
model = joblib.load('model_dt.pkl')

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="")
st.title("Vehicle Maintenance Predictor")
st.write("Fill in the vehicle details below to check if maintenance is needed.")

# ── Input Form ────────────────────────────────────────────────────
reported_issues     = st.selectbox("Reported Issues",    [0, 1, 2, 3, 4, 5])
brake_condition     = st.selectbox("Brake Condition",    [0, 1, 2], format_func=lambda x: {0:"Good", 1:"Worn", 2:"Critical"}[x])
battery_status      = st.selectbox("Battery Status",     [0, 1, 2], format_func=lambda x: {0:"Good", 1:"Weak",  2:"Dead"}[x])
service_history     = st.selectbox("Service History",    [0, 1],    format_func=lambda x: {0:"No",   1:"Yes"}[x])
accident_history    = st.selectbox("Accident History",   [0, 1],    format_func=lambda x: {0:"No",   1:"Yes"}[x])
maintenance_history = st.selectbox("Maintenance History",[0, 1],    format_func=lambda x: {0:"No",   1:"Yes"}[x])

# ── Predict ───────────────────────────────────────────────────────
if st.button("Check Maintenance"):
    sample = pd.DataFrame([{
        'Reported_Issues':     reported_issues,
        'Brake_Condition':     brake_condition,
        'Battery_Status':      battery_status,
        'Service_History':     service_history,
        'Accident_History':    accident_history,
        'Maintenance_History': maintenance_history
    }])

    result     = model.predict(sample)[0]
    proba      = model.predict_proba(sample)[0]
    confidence = round(max(proba) * 100, 2)

    # ── Prediction Result ─────────────────────────────────────────
    st.markdown("---")
    if result == 1:
        st.error(f" Maintenance Needed")
    else:
        st.success(f" No Maintenance Needed")

    # ── Confidence Score ──────────────────────────────────────────
    st.markdown("### Prediction Confidence")
    col1, col2 = st.columns(2)
    # with col1:
    #     st.metric(label="Model Confidence", value=f"{confidence}%")
    with col2:
        st.metric(label="Model Accuracy", value="96.38%")

    # ── Feature Importance Chart ──────────────────────────────────
    st.markdown("### Feature Importance")

    feature_names  = ['Reported_Issues', 'Brake_Condition', 'Battery_Status',
                      'Service_History', 'Accident_History', 'Maintenance_History']
    importances    = model.named_steps['model'].feature_importances_
    indices        = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')
    bars = ax.barh(sorted_features, sorted_importances, color='#00C853')
    ax.set_xlabel('Importance Score', color='white')
    ax.set_title('Which features influence the prediction most?', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.xaxis.grid(True, color='#333333', linewidth=0.8)

    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.3f}', va='center', color='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
