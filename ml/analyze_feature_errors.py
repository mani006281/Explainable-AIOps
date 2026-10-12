"""
Explainable AIOps
Feature-level reconstruction-error analysis.

Loads the existing LSTM Autoencoder and scaler.
Does not retrain the model or modify existing artifacts.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from pathlib import Path


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"

NORMAL_FILE = DATA_DIR / "traffic_features_normal_isolated.csv"
SEQUENCE_FILE = DATA_DIR / "sequence_windows_baseline.npz"

MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"


# ---------------------------------------------------------
# 2. Load existing model artifacts
# ---------------------------------------------------------

if not NORMAL_FILE.exists():
    raise FileNotFoundError(NORMAL_FILE)

if not SEQUENCE_FILE.exists():
    raise FileNotFoundError(SEQUENCE_FILE)

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False,
)

scaler = joblib.load(SCALER_FILE)

with np.load(SEQUENCE_FILE, allow_pickle=False) as data:
    feature_names = data["feature_names"].astype(str).tolist()
    window_size = int(data["window_size"])

threshold_config = pd.read_json(
    THRESHOLD_FILE,
    typ="series",
)

threshold = float(threshold_config["threshold"])


# ---------------------------------------------------------
# 3. Load new normal traffic
# ---------------------------------------------------------

df = pd.read_csv(NORMAL_FILE)

X_raw = df[feature_names].to_numpy(dtype=np.float32)

if len(X_raw) < window_size:
    raise ValueError("Not enough observations to create windows.")

# Create overlapping windows
windows = np.stack([
    X_raw[i:i + window_size]
    for i in range(len(X_raw) - window_size + 1)
])

# Apply the existing scaler
shape = windows.shape

X_scaled = scaler.transform(
    windows.reshape(-1, len(feature_names))
).reshape(shape).astype(np.float32)


# ---------------------------------------------------------
# 4. Calculate feature-level reconstruction errors
# ---------------------------------------------------------

reconstructed = model.predict(
    X_scaled,
    verbose=0,
)

squared_errors = np.square(
    X_scaled - reconstructed
)

# Average over time, leaving one error per feature per window
feature_errors = squared_errors.mean(axis=1)

# Overall error matches the existing detector's calculation
overall_errors = squared_errors.mean(axis=(1, 2))

flagged = overall_errors > threshold


# ---------------------------------------------------------
# 5. Report overall results
# ---------------------------------------------------------

print("=" * 75)
print("FEATURE-LEVEL RECONSTRUCTION-ERROR ANALYSIS")
print("=" * 75)

print("Normal traffic observations:", len(X_raw))
print("Windows evaluated:", len(windows))
print("Existing threshold:", round(threshold, 6))
print("Flagged windows:", int(flagged.sum()), "/", len(flagged))

print("\nOVERALL RECONSTRUCTION ERROR")
print("-" * 75)
print("Mean:", round(float(overall_errors.mean()), 6))
print("Median:", round(float(np.median(overall_errors)), 6))
print("Minimum:", round(float(overall_errors.min()), 6))
print("Maximum:", round(float(overall_errors.max()), 6))


# ---------------------------------------------------------
# 6. Rank features by their contribution to flagged windows
# ---------------------------------------------------------

if flagged.any():
    average_feature_errors = feature_errors[flagged].mean(axis=0)
else:
    average_feature_errors = feature_errors.mean(axis=0)

total_error = average_feature_errors.sum()

contribution_percent = (
    average_feature_errors / total_error * 100
    if total_error > 0
    else np.zeros_like(average_feature_errors)
)

summary = pd.DataFrame({
    "feature": feature_names,
    "mean_squared_error": average_feature_errors,
    "contribution_percent": contribution_percent,
})

summary = summary.sort_values(
    "mean_squared_error",
    ascending=False,
)

print("\nFEATURE CONTRIBUTIONS")
print("-" * 75)
print(summary.round(6).to_string(index=False))


# ---------------------------------------------------------
# 7. Compare flagged and unflagged windows
# ---------------------------------------------------------

window_summary = pd.DataFrame({
    "reconstruction_error": overall_errors,
    "is_anomaly_candidate": flagged,
})

print("\nERROR BY WINDOW CLASSIFICATION")
print("-" * 75)

print(
    window_summary.groupby("is_anomaly_candidate")[
        "reconstruction_error"
    ].agg(["count", "mean", "min", "max"]).to_string()
)

print("\nAnalysis complete. No existing artifacts were modified.")