"""
Explainable AIOps
Inspect original and reconstructed values for the
highest-error normal-traffic window.

This script does not retrain the model or modify artifacts.
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

# Verify that all required files exist.
for path in [
    NORMAL_FILE,
    SEQUENCE_FILE,
    MODEL_FILE,
    SCALER_FILE,
    THRESHOLD_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(path)

# Load the existing model and scaler.
model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False,
)
scaler = joblib.load(SCALER_FILE)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    threshold = float(json.load(file)["threshold"])

with np.load(SEQUENCE_FILE, allow_pickle=False) as data:
    feature_names = data["feature_names"].astype(str).tolist()
    window_size = int(data["window_size"])

# Load the normal traffic features.
df = pd.read_csv(NORMAL_FILE)
X_raw = df[feature_names].to_numpy(dtype=np.float32)

# Generate sliding windows.
windows = np.stack([
    X_raw[i:i + window_size]
    for i in range(len(X_raw) - window_size + 1)
])

# Transform each window using the existing scaler.
n_windows, n_steps, n_features = windows.shape

X_scaled = scaler.transform(
    windows.reshape(-1, n_features)
).reshape(windows.shape).astype(np.float32)

# Reconstruct the windows.
reconstructed = model.predict(
    X_scaled,
    verbose=0,
)

# Calculate overall error for each window.
squared_errors = np.square(X_scaled - reconstructed)
overall_errors = squared_errors.mean(axis=(1, 2))

# Find the highest-error window.
window_index = int(np.argmax(overall_errors))

print("=" * 85)
print("HIGHEST-ERROR NORMAL TRAFFIC WINDOW")
print("=" * 85)
print("Window index:", window_index)
print("Window starts at observation:", window_index)
print("Window ends at observation:", window_index + window_size - 1)
print("Reconstruction error:", round(
    float(overall_errors[window_index]),
    6,
))
print("Existing threshold:", round(threshold, 6))
print("Flagged:", bool(overall_errors[window_index] > threshold))

# Compare scaled input and reconstructed values.
input_window = X_scaled[window_index]
reconstructed_window = reconstructed[window_index]

for feature in [
    "response_time_ms",
    "log_response_time",
    "endpoint_health",
    "endpoint_products",
    "endpoint_latency",
]:
    feature_index = feature_names.index(feature)

    comparison = pd.DataFrame({
        "observation_in_window": np.arange(window_size),
        "original_scaled": input_window[:, feature_index],
        "reconstructed_scaled": reconstructed_window[:, feature_index],
        "absolute_error": np.abs(
            input_window[:, feature_index]
            - reconstructed_window[:, feature_index]
        ),
    })

    print(f"\nFEATURE: {feature}")
    print("-" * 85)
    print(comparison.round(4).to_string(index=False))

print("\nInspection complete.")
print("No existing model, scaler, threshold, or artifact was modified.")