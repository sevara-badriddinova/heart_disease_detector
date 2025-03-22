import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load ECG dataset
ecg_data = pd.read_csv("ecg_features.csv")

# Load Heart Disease dataset
heart_data = pd.read_csv("heart_disease_data.csv")

# Standardize column names
ecg_data.columns = ecg_data.columns.str.strip().str.lower()
heart_data.columns = heart_data.columns.str.strip().str.lower()

# Drop ECG rows with too many NaNs
ecg_data = ecg_data.dropna(thresh=int(ecg_data.shape[1] * 0.8)).reset_index(drop=True)
heart_data = heart_data.reset_index(drop=True)

# Match lengths
ecg_data = ecg_data.iloc[:200]
heart_data = heart_data.iloc[:200]

# Combine and fill missing
data = pd.concat([ecg_data, heart_data], axis=1)
data.fillna(0, inplace=True)

# Prepare features and labels
X = data.drop(columns=["label", "target"], errors="ignore")
y = data["target"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "heart_disease_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved!")
