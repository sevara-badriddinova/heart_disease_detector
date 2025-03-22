import wfdb
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

dataset_path = "/Users/sevarabadriddinova/Desktop/ecg_code/ECG_Dataset/lobachevsky-university-electrocardiography-database-1.0.1/data"

# ✅ Get only ECG files that have both .dat and .hea
ecg_files = [f[:-4] for f in os.listdir(dataset_path) if f.endswith(".dat") and os.path.exists(os.path.join(dataset_path, f[:-4] + ".hea"))]

# ✅ Function to apply a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff=50, fs=360, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ✅ Initialize a list to store extracted features
data_list = []

# ✅ Process multiple ECG files
for file in ecg_files:
    try:
        # Load ECG signal
        ecg_file = os.path.join(dataset_path, file)
        record = wfdb.rdrecord(ecg_file)
        ecg_signal = record.p_signal[:, 0]

        # Preprocess: Apply filter
        filtered_ecg = butter_lowpass_filter(ecg_signal, fs=360)

        # Normalize ECG between -1 and 1
        normalized_ecg = (filtered_ecg - min(filtered_ecg)) / (max(filtered_ecg) - min(filtered_ecg)) * 2 - 1

        # Extract features using NeuroKit2
        ecg_signals, info = nk.ecg_process(normalized_ecg, sampling_rate=360)

        # ✅ Select a larger set of useful features
        available_features = [
            "ECG_R_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks",
            "ECG_R_Onsets", "ECG_T_Peaks", "ECG_P_Onsets",
            "ECG_T_Onsets", "ECG_R_Offsets"
        ]
        available_features = [f for f in available_features if f in info]

        # Extract HRV features for each sample
        hrv_features = nk.hrv_time(ecg_signals, sampling_rate=360)
        advanced_features = ["HRV_RMSSD", "HRV_MeanNN", "HRV_SDNN", "HRV_CVNN"]
        advanced_features = [f for f in advanced_features if f in hrv_features.columns]

        # Store extracted features
        if available_features:
            feature_values = [len(info[f]) for f in available_features]  # Count peaks
            hrv_values = [hrv_features[f].values[0] for f in advanced_features] if advanced_features else []
            
            # Assign labels randomly for now (Replace this with actual labels if available)
            label = 0 if int(file) % 2 == 0 else 1  
            data_list.append(feature_values + hrv_values + [label])  

    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# ✅ Convert list to DataFrame
if len(data_list) > 1:
    feature_columns = available_features + advanced_features + ["Label"]
    data = pd.DataFrame(data_list, columns=feature_columns)

    # ✅ Check label distribution before balancing
    print("Before SMOTE Balancing:\n", data["Label"].value_counts())

    # ✅ Balance dataset using SMOTE
    X = data.drop(columns=["Label"])
    y = data["Label"]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # ✅ Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # ✅ Train ML Model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,  # More trees for better learning
        max_depth=15,      # Allow deeper trees to capture patterns
        min_samples_split=4,  
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)  # ✅ Train model

    # ✅ Now check feature importance after training
    importances = model.feature_importances_
    feature_names = X.columns

    # Sort features by importance
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:\n", feature_importance)

    print("\n✅ AI Model Trained Successfully!")
    print("\nFinal Dataset Preview:\n", data.head())

    # ✅ Predict on test data
    predictions = model.predict(X_test)

    # ✅ Show evaluation metrics
    print("\nModel Performance:\n", classification_report(y_test, predictions))

else:
    print("❌ Not enough ECG files processed. AI needs at least 2 samples to train!")
