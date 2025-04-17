import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import zipfile
import os
import matplotlib as plt

# Load ECG and heart disease datasets
ecg_data = pd.read_csv("ecg_features.csv")
heart_data = pd.read_csv("heart_disease_data.csv")

# Clean columns
ecg_data.columns = ecg_data.columns.str.strip().str.lower()
heart_data.columns = heart_data.columns.str.strip().str.lower()

# Drop rows with too many missing values
ecg_data = ecg_data.dropna(thresh=int(ecg_data.shape[1] * 0.8)).reset_index(drop=True)
heart_data = heart_data.reset_index(drop=True)

# Combine ECG and heart disease data
data = pd.concat([ecg_data, heart_data], axis=1)
data.fillna(data.median(), inplace=True)

# Extract PERG "Normal" patient data
zip_path = "a-comprehensive-dataset-of-pattern-electroretinograms-for-ocular-electrophysiology-research-the-perg-ioba-dataset-1.0.0.zip"
extract_path = "perg_ioba"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

csv_folder = os.path.join(extract_path, "a-comprehensive-dataset-of-pattern-electroretinograms-for-ocular-electrophysiology-research-the-perg-ioba-dataset-1.0.0", "csv")
perg_df = pd.read_csv(os.path.join(csv_folder, "participants_info.csv"))

normal_perg = perg_df[perg_df["diagnosis1"].str.strip().str.lower() == "normal"].copy()
normal_perg = normal_perg[["age_years", "sex"]].copy()
normal_perg.columns = ["age", "sex"]
normal_perg["sex"] = normal_perg["sex"].map({"Male": 1, "Female": 0})
normal_perg["target"] = 0
normal_perg.dropna(subset=["age", "sex"], inplace=True)
normal_perg["chol"] = normal_perg["age"] * 5
normal_perg["fbs"] = 0
normal_perg["restecg"] = np.random.choice([0], size=len(normal_perg))
normal_perg["cp"] = np.random.choice([0, 1], size=len(normal_perg))
normal_perg["thalach"] = np.random.normal(170, 10, size=len(normal_perg)).clip(130, 200)

# Only keep training features
selected_features = ["age", "sex", "cp", "chol", "fbs", "restecg", "thalach", "target"]
heart_df = data[selected_features].copy()
combined_df = pd.concat([heart_df, normal_perg], ignore_index=True)
combined_df.fillna(combined_df.median(), inplace=True)


# Optional: oversample high-risk (target=1) patients
high_risk = combined_df[combined_df['target'] == 1]

# You can increase n=50, 100, or 200 depending on how many you have
boosted_high_risk = high_risk.sample(n=100, replace=True, random_state=42)

# Add them to your training data
combined_df = pd.concat([combined_df, boosted_high_risk], ignore_index=True)

print(f"Oversampled high-risk examples: {len(boosted_high_risk)} added.")
print(f"Total training set size: {len(combined_df)} rows.")


# Split features and labels
X = combined_df.drop(columns=["target"])
y = combined_df["target"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train with class balancing
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model artifacts
joblib.dump(model, "heart_disease_rf_simplified.pkl")
joblib.dump(scaler, "scaler_simplified.pkl")
joblib.dump(X.columns.tolist(), "features_simplified.pkl")
print("Model, scaler, and features saved!")