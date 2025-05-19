import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load the trained model and scaler
model = joblib.load("heart_disease_rf_simplified.pkl")
scaler = joblib.load("scaler_simplified.pkl")
feature_names = joblib.load("features_simplified.pkl")
# Set default ECG values as the mean values from training set
default_ecg_values = [
    12.28, 12.28, 12.28, 12.28, 12.28,
    12.28, 12.28, 12.28, 189.63, 1148.07,
    153.09, 0.15, 0.0, 0.04, 0.0  # replaced NaNs with 0.0
]

st.set_page_config(
    page_title="Heart Disease Risk Detector",
    page_icon="❤️", 
    layout="centered"
)

with st.spinner("Waking up the app... Please wait a few seconds."):
    time.sleep(5)  # simulate some load time

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    if age < 29 or age > 77:
        st.warning("Note: This age is outside the range of training data (29–77). Prediction may be inaccurate.")
    sex = st.selectbox("🧬 Sex", ["Male", "Female"])
    chol = st.number_input("🩸 Cholesterol (mg/dl)", min_value=100, max_value=600, step=1)
    ecg_options = {
        0: "0 - Normal",
        1: "1 - ST-T wave abnormality",
        2: "2 - Probable/definite left ventricular hypertrophy"
    }
    ecg_result_label = st.selectbox("📈 ECG Result", list(ecg_options.values()))
    ecg_result = int(ecg_result_label.split(" - ")[0])

    cp_options = {
        0: "0 - Typical angina",
        1: "1 - Atypical angina",
        2: "2 - Non-anginal pain",
        3: "3 - Asymptomatic"
    }
    cp_label = st.selectbox("💥 Chest Pain Type (cp)", list(cp_options.values()))
    cp = int(cp_label.split(" - ")[0])

    fbs_label = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl (fbs)", ["0 - False", "1 - True"])
    fbs = int(fbs_label.split(" - ")[0])

    thalach = st.number_input("🏃 Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, step=1)

    if st.button("🔍 Predict from Manual Input"):
        sex_bin = 1 if sex == "Male" else 0
        input_data = pd.DataFrame([[
            age, sex_bin, cp, chol, fbs, ecg_result, thalach
        ]], columns=feature_names)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        risk_level = "High" if prediction == 1 else "Low"
        color = "red" if prediction == 1 else "green"

        if 0.4 < prob < 0.6:
            st.warning("⚠️ The model is unsure about this prediction. Consider reviewing inputs or using more data.")

        st.markdown(f"**Risk Probability:** {prob:.2%}")
        st.markdown("## 🧾 Patient Risk Assessment")
        st.markdown(f"**Heart Disease Risk Level:** <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
        st.markdown(f"**Age:** {age}")
        st.markdown(f"**Sex:** {sex}")
        st.markdown(f"**Cholesterol:** {chol} mg/dl")
        st.markdown(f"**ECG Result:** {ecg_result}")
        st.markdown(f"**Chest Pain Type:** {cp}")
        st.markdown(f"**Fasting Blood Sugar > 120 mg/dl:** {fbs}")
        st.markdown(f"**Max Heart Rate Achieved:** {thalach}")

elif ec_data_file and hd_data_file:
    ecg_df = pd.read_csv(ec_data_file)
    hd_df = pd.read_csv(hd_data_file)
    df = pd.concat([ecg_df.reset_index(drop=True), hd_df.reset_index(drop=True)], axis=1)
    df.fillna(0, inplace=True)

    X = df.drop(columns=["label", "target"], errors="ignore")
    X.columns = X.columns.str.lower()
    X = X[feature_names]
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
        st.markdown(f"**Chest Pain:** {row['cp']}")
        st.markdown(f"**Fasting Sugar:** {row['fbs']}")
        st.markdown(f"**Max Heart Rate:** {row['thalach']}")
        st.markdown(f"**ECG Result:** {row['restecg'] if 'restecg' in row else 'N/A'}")
        st.markdown("---")
