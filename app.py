import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("heart_disease_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Risk Predictor")
st.write("Upload ECG features and patient info to predict heart disease risk.")

uploaded_ecg = st.file_uploader("Upload ECG Features CSV", type=["csv"])
uploaded_patient = st.file_uploader("Upload Heart Disease Patient Info CSV", type=["csv"])

if uploaded_ecg and uploaded_patient:
    ecg_df = pd.read_csv(uploaded_ecg)
    patient_df = pd.read_csv(uploaded_patient)

    ecg_df.columns = ecg_df.columns.str.strip().str.lower()
    patient_df.columns = patient_df.columns.str.strip().str.lower()

    # Drop ECG rows with too many NaNs
    ecg_df = ecg_df.dropna(thresh=int(ecg_df.shape[1] * 0.8)).reset_index(drop=True)
    patient_df = patient_df.reset_index(drop=True)

    # Align row counts
    min_len = min(len(ecg_df), len(patient_df))
    ecg_df = ecg_df.iloc[:min_len]
    patient_df = patient_df.iloc[:min_len]

    # Combine and fill NaNs
    combined = pd.concat([ecg_df, patient_df], axis=1)
    combined.fillna(0, inplace=True)

    # Keep original for displaying
    display_data = combined.copy()

    # Drop label/target before scaling
    X = combined.drop(columns=["target", "label"], errors="ignore")
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    prediction_probs = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1

    st.subheader("Prediction Results")
    for i, (row, pred, prob) in enumerate(zip(display_data.iterrows(), predictions, prediction_probs)):
        index, data = row
        risk_level = "High" if pred == 1 else "Low"
        st.markdown("---")
        st.write(f"**Patient {i + 1}**")
        st.write(f"- Heart disease risk level: {risk_level} ({prob * 100:.1f}%)")
        st.write(f"- Age: {data['age']}")
        st.write(f"- Sex: {'Male' if data['sex'] == 1 else 'Female'}")
        st.write(f"- Cholesterol: {data['chol']}")
        st.write(f"- ECG Result: {data.get('restecg', 'N/A')}")
else:
    st.warning("Please upload both ECG and patient info CSV files to proceed.")
