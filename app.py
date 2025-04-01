import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("heart_disease_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Detector", layout="centered")
st.title("❤️ Heart Disease Risk Detector")
st.markdown("Upload ECG features and patient info **OR** manually enter the data below.")

# --- Upload Section ---
st.subheader("📤 Upload ECG Features CSV")
ec_data_file = st.file_uploader("Upload ECG features", type=["csv"], key="ecg")

st.subheader("📤 Upload Heart Disease Info CSV")
hd_data_file = st.file_uploader("Upload heart disease info", type=["csv"], key="hd")

# --- Manual Input Section ---
st.subheader("📝 Or Manually Enter Patient Data")
manual_input = st.checkbox("Manually enter patient data")

if manual_input:
    age = st.number_input("🧓 Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("🧬 Sex", ["Male", "Female"])
    chol = st.number_input("🩸 Cholesterol (mg/dl)", min_value=100, max_value=600, step=1)
    ecg_result = st.selectbox("📈 ECG Result", [0, 1, 2])

    if st.button("🔍 Predict from Manual Input"):
        sex_bin = 1 if sex == "Male" else 0
        ecg_input = [11]*15  # Placeholder ECG values

        input_data = pd.DataFrame([[
            *ecg_input, age, sex_bin, 0, 120, chol, 0, 0, 150, 0, 1.0, 1, 0, 1
        ]], columns=[
            "ecg_r_peaks", "ecg_q_peaks", "ecg_s_peaks", "ecg_r_onsets", "ecg_t_peaks", "ecg_p_onsets",
            "ecg_t_onsets", "ecg_r_offsets", "hrv_rmssd", "hrv_meannn", "hrv_sdnn", "hrv_cvnn", "hrv_lf",
            "hrv_hf", "hrv_lfhf", "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        risk_level = "High" if prediction == 1 else "Low"
        color = "red" if prediction == 1 else "green"

        st.markdown("## 🧾 Patient Risk Assessment")
        st.markdown(f"**Heart Disease Risk Level:** <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
        st.markdown(f"**Age:** {age}")
        st.markdown(f"**Sex:** {sex}")
        st.markdown(f"**Cholesterol:** {chol} mg/dl")
        st.markdown(f"**ECG Result:** {ecg_result}")

elif ec_data_file and hd_data_file:
    ecg_df = pd.read_csv(ec_data_file)
    hd_df = pd.read_csv(hd_data_file)
    df = pd.concat([ecg_df.reset_index(drop=True), hd_df.reset_index(drop=True)], axis=1)
    df.fillna(0, inplace=True)

    X = df.drop(columns=["label", "target"], errors="ignore")
    X.columns = X.columns.str.lower()
    expected_columns = scaler.feature_names_in_
    X = X[expected_columns]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    st.subheader("📊 Batch Prediction Results")
    for idx, pred in enumerate(predictions):
        row = df.iloc[idx]
        risk = "High" if pred == 1 else "Low"
        color = "red" if pred == 1 else "green"
        st.markdown(f"### Patient {idx+1}")
        st.markdown(f"**Heart Disease Risk Level:** <span style='color:{color}; font-weight:bold;'>{risk}</span>", unsafe_allow_html=True)
        st.markdown(f"**Age:** {int(row['age'])}")
        st.markdown(f"**Sex:** {'Male' if row['sex'] == 1 else 'Female'}")
        st.markdown(f"**Cholesterol:** {row['chol']} mg/dl")
        st.markdown(f"**ECG Result:** {row['restecg'] if 'restecg' in row else 'N/A'}")
        st.markdown("---")
