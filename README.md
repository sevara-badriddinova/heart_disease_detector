# Heart Disease Detector

Real-time heart disease risk prediction using **ECG-derived features** and **clinical data**, built with **TensorFlow** and a **Streamlit** UI. The app lets clinicians/students upload or enter patient data, run the model, and see risk predictions instantly with simple visualizations.

> **Highlights**
> - ✅ ~93% accuracy in internal validation (tunable via training config)
> - 🧠 Hybrid approach: CNN for ECG signals + classical ML (e.g., Random Forest) for tabular features
> - ⚡ Real-time UI in Streamlit with input validation and clear outputs
> - 🧩 Modular code: separate data, features, models, and app layers
> - 📦 Easy local setup (venv + `pip install -r requirements.txt`)


## Demo

- **Local**: `streamlit run app.py` (see [Quick Start](#quick-start))
- **Screens**: 
  - Input panel: Age, Sex, Cholesterol, ECG summary, etc.
  - Signal upload: ECG file (e.g., `.csv`/`.npy`) for feature extraction
  - Output: Predicted risk (probability), class, and key feature contributions

> _Tip: Add screenshots/gif once deployed._

---

## Project Structure

```
heart-disease-detector/
├─ app.py                     # Streamlit UI
├─ requirements.txt
├─ src/
│  ├─ data/
│  │  ├─ loaders.py          # Read ECG & tabular data
│  │  └─ preprocess.py       # Cleaning, normalization, splits
│  ├─ features/
│  │  ├─ hrv.py              # HRV + signal features
│  │  └─ tabular.py          # Clinical features (one-hot/scale)
│  ├─ models/
│  │  ├─ cnn.py              # CNN architecture for ECG
│  │  ├─ rf.py               # RandomForest or other classical ML
│  │  └─ ensemble.py         # Late fusion / stacking
│  ├─ training/
│  │  ├─ train_cnn.py        # Train CNN on ECG
│  │  ├─ train_tabular.py    # Train RF/XGB on features
│  │  └─ train_ensemble.py   # Blend/fuse models
│  └─ utils/
│     ├─ metrics.py          # AUC, F1, PR, CM
│     └─ io.py               # Paths, serialization
├─ models/
│  ├─ cnn_best.h5            # (generated) best CNN weights
│  ├─ rf.joblib              # (generated) fitted classical model
│  └─ ensemble.joblib        # (optional) meta model
├─ data/
│  ├─ raw/                   # (gitignored) raw ECG and CSVs
│  └─ processed/             # (gitignored) cleaned features
└─ README.md
```

> File names are suggestions; align with your current repo and adjust imports accordingly.

---

## Quick Start

### 1) Clone & Environment
```bash
git clone https://github.com/sevara-badriddinova/heart-disease-detector.git
cd heart-disease-detector

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Model Weights
Place your trained weights in `models/`:
- `models/cnn_best.h5`
- `models/rf.joblib`
- (optional) `models/ensemble.joblib`

> If you don't have weights yet, see [Training](#training).

### 3) Run the App
```bash
streamlit run app.py
```
Open the local URL printed in the terminal.

---

## Usage (Streamlit)

1. **Enter clinical features** (e.g., age, sex, cholesterol, resting ECG result).
2. **Upload ECG** (CSV or NumPy) *or* paste path to a test sample.
3. Click **Predict**.
4. See:
   - **Probability** and **risk class**
   - **Top contributing features**
   - (Optional) ECG preview and extracted HRV metrics

> The UI is designed so **patient details and prediction** appear **in the app** (not just the console).

---

## Model Overview

- **ECG branch**: 1D CNN on preprocessed ECG beats/segments.
- **Tabular branch**: Classical ML (Random Forest or XGBoost) on engineered features (HRV + clinical).
- **Ensemble**: Late fusion (weighted average or stacking).

### Metrics (example)
- Accuracy: ~0.93
- AUC-ROC: ~0.95
- F1-score: ~0.92

> Results vary by dataset, preprocessing, and train/val split. Log actual numbers from your last run.

---

## Training

> Update paths to your dataset. Ensure raw data is kept out of git if it contains sensitive info.

```bash
# 1) Prepare data
python -m src.data.preprocess --in data/raw --out data/processed --split 0.8

# 2) Train ECG CNN
python -m src.training.train_cnn --data data/processed --out models/cnn_best.h5

# 3) Train tabular model
python -m src.training.train_tabular --data data/processed --out models/rf.joblib

# 4) Train/fit ensemble (optional)
python -m src.training.train_ensemble   --cnn models/cnn_best.h5 --tab models/rf.joblib --out models/ensemble.joblib
```

> For reproducibility, set seeds and log runs (e.g., `wandb`, `mlflow`).

---

## Configuration

Create `.env` (or `config.yaml`) for paths and flags:
```
# .env (example)
DATA_DIR=data
MODEL_DIR=models
ECG_SAMPLE_RATE=360            # set to your dataset
USE_ENSEMBLE=true
TABULAR_MODEL=rf               # or xgb
```

Use `python-dotenv` or `yaml` loader in `src/utils/io.py` to read config.

---

## Requirements

Pin your versions in `requirements.txt`. Example:
```
tensorflow>=2.12
numpy
pandas
scikit-learn
joblib
scipy
matplotlib
streamlit
python-dotenv
```

---

## Deployment

- **Local**: `streamlit run app.py`
- **Cloud**: Streamlit Community Cloud / Hugging Face Spaces / Docker
- **Docker (example)**:
  ```Dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

---

## Testing

- Unit tests under `tests/` for feature extraction and model I/O.
- Add sample ECG and tabular fixtures under `tests/data/`.

```bash
pytest -q
```

---

## Data & Ethics

- Use only datasets you are licensed to use.
- De-identify all records.
- This tool is **educational** and **not a medical device**. Do not use for clinical decision-making without proper validation and approvals.

---

## Contributing

Issues and PRs are welcome. Please open an issue describing the change and include reproduction steps for bugs.

---

## License

Choose a license (e.g., MIT, Apache-2.0) and add `LICENSE` to the repo.

---

## Acknowledgments

- Biomedical signals community for open-source ECG tooling and HRV literature.
- Streamlit/TensorFlow teams for great developer tooling.
