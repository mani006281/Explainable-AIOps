"""
Explainable AIOps
Inspect feature reconstruction errors for isolated normal traffic.

This script loads the existing model and scaler.
It does not retrain the model or modify saved artifacts.
"""

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"

NORMAL_FILE = DATA_DIR / "traffic_features_normal_isolated.csv"
SEQUENCE_FILE = DATA_DIR / "sequence_windows_baseline.npz"
MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"

# Verify required files.
for file_path in [
    NORMAL_FILE,
    SEQUENCE_FILE,
    MODEL_FILE,
    SCALER_FILE,
    THRESHOLD_FILE,
]:
    if not file_path.exists():
        raise FileNotFoundError(file_path)

# Load existing model, scaler, and threshold.
model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False,
)

scaler = joblib.load(SCALER_FILE)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

threshold = float(threshold_config["threshold"])

# Read feature order and window size from the baseline dataset.
with np.load(SEQUENCE_FILE, allow_pickle=False) as data:
    feature_names = data["feature_names"].astype(str).tolist()
    window_size = int(data["window_size"])

# Load normal traffic features.
df = pd.read_csv(NORMAL_FILE)

X_raw = df[feature_names].to_numpy(dtype=np.float32)

# Create sliding windows.
windows = np.stack([
    X_raw[i:i + window_size]
    for i in range(len(X_raw) - window_size + 1)
])

# Apply the existing scaler.
n_windows, n_steps, n_features = windows.shape

X_scaled = scaler.transform(
    windows.reshape(-1, n_features)
).reshape(windows.shape).astype(np.float32)

# Reconstruct using the existing model.
X_reconstructed = model.predict(
    X_scaled,
    verbose=0,
)

# Calculate squared errors in scaled feature space.
squared_errors = np.square(
    X_scaled - X_reconstructed
)

# Mean error for each feature in each window.
per_window_feature_error = squared_errors.mean(axis=1)

# Overall error for each window.
overall_errors = squared_errors.mean(axis=(1, 2))

flagged = overall_errors > threshold

# Summarize error by feature across all windows.
feature_summary = pd.DataFrame({
    "feature": feature_names,
    "mean_squared_error": per_window_feature_error.mean(axis=0),
    "median_squared_error": np.median(
        per_window_feature_error,
        axis=0,
    ),
    "max_squared_error": per_window_feature_error.max(axis=0),
})

feature_summary = feature_summary.sort_values(
    "mean_squared_error",
    ascending=False,
)

print("=" * 80)
print("RECONSTRUCTION INSPECTION: NORMAL TRAFFIC")
print("=" * 80)
print("Windows evaluated:", n_windows)
print("Window size:", window_size)
print("Existing threshold:", round(threshold, 6))
print("Flagged windows:", int(flagged.sum()), "/", n_windows)

print("\nFEATURE ERRORS ACROSS ALL WINDOWS")
print("-" * 80)
print(feature_summary.round(6).to_string(index=False))

# Identify the windows with the largest reconstruction errors.
window_summary = pd.DataFrame({
    "window_index": np.arange(n_windows),
    "reconstruction_error": overall_errors,
    "flagged": flagged,
})

print("\nFIVE WINDOWS WITH THE HIGHEST ERRORS")
print("-" * 80)
print(
    window_summary.nlargest(
        5,
        "reconstruction_error",
    ).round(6).to_string(index=False)
)

# Show the feature-level errors for the single highest-error window.
highest_error_index = int(np.argmax(overall_errors))

print("\nFEATURE ERRORS IN THE HIGHEST-ERROR WINDOW")
print("-" * 80)

highest_window_features = pd.DataFrame({
    "feature": feature_names,
    "mean_squared_error": per_window_feature_error[
        highest_error_index
    ],
})

highest_window_features = highest_window_features.sort_values(
    "mean_squared_error",
    ascending=False,
)

print("Window index:", highest_error_index)
print("Reconstruction error:", round(
    float(overall_errors[highest_error_index]),
    6,
))

print(
    highest_window_features.round(6).to_string(index=False)
)

print("\nInspection complete.")
print("No model, scaler, threshold, or existing artifact was modified.")