import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── Hide Streamlit UI Elements ────────────────────────
st.markdown("""
<style>
/* Hide Deploy button */
[data-testid="stDeployButton"] { display: none !important; }

/* Hide Rerun button */
[data-testid="stBaseButton-headerNoPadding"] { display: none !important; }

/* Hide Record screencast */
[data-testid="stScreencastButton"] { display: none !important; }

/* Hide Print button */
[data-testid="stPrintButton"] { display: none !important; }

/* Hide footer */
footer { display: none !important; }

/* Hide the bottom toolbar completely */
[data-testid="stToolbar"] { display: none !important; }

/* Hide main menu except settings */
#MainMenu { display: none !important; }

/* Hide top right action buttons except settings */
[data-testid="stActionButtonIcon"] { display: none !important; }

/* Remove blank space left by hidden footer */
.main .block-container { padding-bottom: 20px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["features"]

model, scaler, features = load_model()

@st.cache_data
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)

metrics = load_metrics()

# ── Header ────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.caption("Enter the patient's health details below and click Predict.")
st.divider()

# ── Input Form ────────────────────────────────────────
st.subheader("Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    Age           = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
    BMI           = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.5,
                                     help="Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: 30+")
    Glucose       = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100,
                                     help="Normal: 70–99 | Prediabetes: 100–125 | Diabetes: 126+")

with col2:
    BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=200, value=80,
                                     help="Normal: below 120 | High: 130+")
    Cholesterol   = st.selectbox("Cholesterol Level",
                                  options=[0, 1],
                                  format_func=lambda x: "High" if x else "Normal")
    PhysActivity  = st.selectbox("Physical Activity Level",
                                  options=[0, 1],
                                  format_func=lambda x: "Active (exercises regularly)" if x else "Inactive (little/no exercise)")

st.divider()

# ── Predict Button ────────────────────────────────────
if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):

    # Map user inputs to model features
    # Age: convert actual age to age group (1–13)
    def age_to_group(age):
        brackets = [24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
        for i, b in enumerate(brackets):
            if age <= b:
                return i + 1
        return 13

    # Glucose: derive HighBP and risk signals from raw values
    def glucose_risk(g):
        if g < 100:   return 0   # Normal
        elif g < 126: return 1   # Prediabetes
        else:         return 2   # Diabetic range

    age_group  = age_to_group(Age)
    bp_high    = 1 if BloodPressure >= 130 else 0
    gluc_risk  = glucose_risk(Glucose)
    bmi_risk   = 1 if BMI >= 30 else 0

    # Build full feature vector (21 features the model expects)
    input_data = pd.DataFrame([[
        bp_high,         # HighBP
        Cholesterol,     # HighChol
        1,               # CholCheck (assumed yes if visiting)
        BMI,             # BMI
        0,               # Smoker (not asked)
        0,               # Stroke (not asked)
        0,               # HeartDiseaseorAttack (not asked)
        PhysActivity,    # PhysActivity
        0,               # Fruits (not asked)
        0,               # Veggies (not asked)
        0,               # HvyAlcoholConsump (not asked)
        1,               # AnyHealthcare (assumed yes)
        0,               # NoDocbcCost (not asked)
        3 if gluc_risk == 0 else (4 if gluc_risk == 1 else 5),  # GenHlth mapped from glucose
        0,               # MentHlth
        gluc_risk * 10,  # PhysHlth mapped from glucose risk
        0,               # DiffWalk
        0,               # Sex (not asked)
        age_group,       # Age group
        4,               # Education (assumed average)
        5,               # Income (assumed average)
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    # Adjust risk % using glucose as a strong signal
    base_risk = round((1 - probability[0]) * 100, 1)
    if gluc_risk == 2:
        risk_pct = min(round(base_risk * 1.3, 1), 99.9)
    elif gluc_risk == 1:
        risk_pct = min(round(base_risk * 1.1, 1), 99.9)
    else:
        risk_pct = base_risk

    # Override prediction based on glucose (strong clinical signal)
    if Glucose >= 126:
        final_prediction = 2
    elif Glucose >= 100:
        final_prediction = max(prediction, 1)
    else:
        final_prediction = prediction

    st.divider()
    st.subheader("📊 Result")

    col_result, col_gauge = st.columns([1, 1])

    with col_result:
        if final_prediction == 0:
            st.success("### ✅ Low Risk")
            st.write(f"**Risk Score: {risk_pct}%**")
            st.write("No significant diabetes indicators found.")
            st.write("Maintain a healthy diet and routine check-ups.")
        elif final_prediction == 1:
            st.warning("### ⚠️ Prediabetes Risk")
            st.write(f"**Risk Score: {risk_pct}%**")
            st.write("Some diabetes indicators are present.")
            st.write("Consult a doctor. Lifestyle changes can help prevent progression.")
        else:
            st.error("### 🔴 High Diabetes Risk")
            st.write(f"**Risk Score: {risk_pct}%**")
            st.write("Strong diabetes indicators detected.")
            st.write("Please seek immediate medical attention.")

        st.divider()
        # Input summary
        st.markdown("**📋 Summary of Inputs**")
        st.markdown(f"- **Age:** {Age} years")
        st.markdown(f"- **BMI:** {BMI} ({'Normal' if BMI < 25 else 'Overweight' if BMI < 30 else 'Obese'})")
        st.markdown(f"- **Glucose:** {Glucose} mg/dL ({'Normal' if Glucose < 100 else 'Prediabetic range' if Glucose < 126 else 'Diabetic range'})")
        st.markdown(f"- **Blood Pressure:** {BloodPressure} mm Hg ({'Normal' if BloodPressure < 130 else 'High'})")
        st.markdown(f"- **Cholesterol:** {'High' if Cholesterol else 'Normal'}")
        st.markdown(f"- **Physical Activity:** {'Active' if PhysActivity else 'Inactive'}")
        st.caption("⚠️ For educational purposes only. Not a medical diagnosis.")

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e11d48" if risk_pct > 50 else "#f97316" if risk_pct > 30 else "#16a34a"},
                'steps': [
                    {'range': [0,  30], 'color': "#dcfce7"},
                    {'range': [30, 60], 'color': "#fef9c3"},
                    {'range': [60, 100], 'color': "#fee2e2"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ── Model Info ────────────────────────────────────────
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    | Metric | Score |
    |--------|-------|
    | Accuracy  | {metrics['accuracy']*100:.1f}% |
    | Precision | {metrics['precision']*100:.1f}% |
    | Recall    | {metrics['recall']*100:.1f}% |
    | F1-Score  | {metrics['f1_score']*100:.1f}% |
    | ROC-AUC   | {metrics['roc_auc']} |
    """)
    st.caption("Model: Random Forest | Dataset: CDC BRFSS 2015 | Records: 253,680")

# ── Footer ────────────────────────────────────────────
st.divider()
