import numpy as np
import pandas as pd
# save the model
import joblib
# predict heart disease
from sklearn.ensemble import RandomForestClassifier
# splits data into training/testing
from sklearn.model_selection import train_test_split
# normalizing data
from sklearn.preprocessing import StandardScaler
# accuracy 
from sklearn.metrics import classification_report

# load ECG dataset
ecg_data = pd.read_csv("ecg_features.csv")

# load Heart Disease dataset
heart_data = pd.read_csv("heart_disease_data.csv")

# make column names lowercase/remove spaces to make merging easier
ecg_data.columns = ecg_data.columns.str.strip().str.lower()
heart_data.columns = heart_data.columns.str.strip().str.lower()

# drop rows that miss more than 20% of the data
ecg_data = ecg_data.dropna(thresh=int(ecg_data.shape[1] * 0.8)).reset_index(drop=True)
# reset row indexing for heart data
heart_data = heart_data.reset_index(drop=True)

# use the first 200 patients data
# ecg_data = ecg_data.iloc[:200]
# heart_data = heart_data.iloc[:200]

# combine datasets and fill any blanks with 0 
data = pd.concat([ecg_data, heart_data], axis=1)
data.fillna(data.median(), inplace=True)

# all features (age, ecg, chol)
X = data.drop(columns=["label", "target"], errors="ignore")
# prediction - 0 for no heart dis, 1 for yes 
y = data["target"]

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# train random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make predictions on test data and print accuracy
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# save model and scaler
joblib.dump(model, "heart_disease_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
print("Model and scaler saved!")
