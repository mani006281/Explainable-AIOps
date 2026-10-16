"""
Explainable AIOps
Feature-level reconstruction-error analysis on normal test traffic.

Uses the existing normal_extended_v1 model, scaler, and threshold.
Does not retrain or modify existing artifacts.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DEMO_DIR = PROJECT_DIR / "ecommerce-demo"

ARTIFACT_DIR = ML_DIR / "artifacts_normal_extended_v1"
DATA_DIR = DEMO_DIR / "data"

MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"

TEST_WINDOWS_FILE = DATA_DIR / "sequence_windows_normal_extended_test.csv"

OUTPUT_DIR = ML_DIR / "normal_test_feature_analysis_v1"


# ---------------------------------------------------------
# 2. Validate input files
# ---------------------------------------------------------

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    THRESHOLD_FILE,
    TEST_WINDOWS_FILE,
]

for file_path in required_files:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}"
        )

# Never overwrite previous analysis results.
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Inspect or rename it before running this script again."
    )


# ---------------------------------------------------------
# 3. Load model configuration and artifacts
# ---------------------------------------------------------

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    config = json.load(file)

feature_names = config["feature_names"]
threshold = float(config["threshold"])
window_size = int(config["window_size"])

scaler = joblib.load(SCALER_FILE)

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False,
)

if len(feature_names) != scaler.n_features_in_:
    raise ValueError("Feature count does not match saved scaler.")

if model.input_shape[1] != window_size:
    raise ValueError("Window size does not match the model.")

if model.input_shape[2] != len(feature_names):
    raise ValueError("Feature count does not match the model.")

if model.output_shape != model.input_shape:
    raise ValueError("Autoencoder input and output shapes differ.")


# ---------------------------------------------------------
# 4. Load flattened test windows and restore their shape
# ---------------------------------------------------------

df = pd.read_csv(TEST_WINDOWS_FILE)

if "window_id" not in df.columns:
    raise ValueError("Test CSV does not contain window_id.")

# Validate the expected flattened column order.
expected_columns = ["window_id"] + [
    f"t{step:02d}_{feature}"
    for step in range(1, window_size + 1)
    for feature in feature_names
]

if df.columns.tolist() != expected_columns:
    raise ValueError(
        "Unexpected test CSV column names or order. "
        "No analysis was performed."
    )

window_ids = df["window_id"].astype(str).to_numpy()

if len(set(window_ids)) != len(window_ids):
    raise ValueError("Duplicate window IDs were found.")

raw_flat = df[expected_columns[1:]].apply(
    pd.to_numeric,
    errors="raise",
).to_numpy(dtype=np.float64)

if not np.isfinite(raw_flat).all():
    raise ValueError("Test data contains NaN or infinite values.")

raw_windows = raw_flat.reshape(
    len(df),
    window_size,
    len(feature_names),
)

if raw_windows.shape != (81, window_size, len(feature_names)):
    raise ValueError(
        f"Expected 81 test windows, got {raw_windows.shape}."
    )


# ---------------------------------------------------------
# 5. Apply the existing scaler
# ---------------------------------------------------------

scaled_windows = scaler.transform(
    raw_windows.reshape(-1, len(feature_names))
).reshape(raw_windows.shape).astype(np.float32)


# ---------------------------------------------------------
# 6. Reconstruct test windows
# ---------------------------------------------------------

reconstructed = model.predict(
    scaled_windows,
    verbose=0,
)

if reconstructed.shape != scaled_windows.shape:
    raise ValueError("Unexpected reconstruction shape.")

if not np.isfinite(reconstructed).all():
    raise ValueError("Model output contains invalid values.")

squared_errors = np.square(
    scaled_windows - reconstructed
)

# Mean squared error for each window and each feature.
feature_errors = squared_errors.mean(axis=1)

# Overall window error: mean across time steps and features.
overall_errors = squared_errors.mean(axis=(1, 2))

predicted_anomaly = overall_errors > threshold


# ---------------------------------------------------------
# 7. Calculate feature contributions
# ---------------------------------------------------------

def make_feature_summary(indices):
    """Summarize feature errors for a selected set of windows."""

    if len(indices) == 0:
        return pd.DataFrame({
            "feature": feature_names,
            "mean_squared_error": np.zeros(len(feature_names)),
            "contribution_percent": np.zeros(len(feature_names)),
        })

    means = feature_errors[indices].mean(axis=0)
    total = means.sum()

    percentages = (
        means / total * 100
        if total > 0
        else np.zeros_like(means)
    )

    return (
        pd.DataFrame({
            "feature": feature_names,
            "mean_squared_error": means,
            "contribution_percent": percentages,
        })
        .sort_values("mean_squared_error", ascending=False)
        .reset_index(drop=True)
    )


all_indices = np.arange(len(df))
flagged_indices = np.flatnonzero(predicted_anomaly)
normal_indices = np.flatnonzero(~predicted_anomaly)

all_summary = make_feature_summary(all_indices)
flagged_summary = make_feature_summary(flagged_indices)
normal_summary = make_feature_summary(normal_indices)


# ---------------------------------------------------------
# 8. Create output tables
# ---------------------------------------------------------

window_results = pd.DataFrame({
    "window_id": window_ids,
    "reconstruction_error": overall_errors,
    "threshold": threshold,
    "predicted_anomaly": predicted_anomaly,
})

all_feature_errors = pd.DataFrame(
    feature_errors,
    columns=feature_names,
)

all_feature_errors.insert(0, "window_id", window_ids)
all_feature_errors.insert(1, "reconstruction_error", overall_errors)
all_feature_errors.insert(2, "predicted_anomaly", predicted_anomaly)


# ---------------------------------------------------------
# 9. Save to a new directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

window_results.to_csv(
    OUTPUT_DIR / "normal_test_window_results.csv",
    index=False,
)

all_feature_errors.to_csv(
    OUTPUT_DIR / "normal_test_feature_errors.csv",
    index=False,
)

all_summary.to_csv(
    OUTPUT_DIR / "feature_contributions_all_windows.csv",
    index=False,
)

flagged_summary.to_csv(
    OUTPUT_DIR / "feature_contributions_flagged_windows.csv",
    index=False,
)

normal_summary.to_csv(
    OUTPUT_DIR / "feature_contributions_unflagged_windows.csv",
    index=False,
)

summary = {
    "experiment": config["experiment"],
    "dataset": "normal_extended_test",
    "test_windows": int(len(df)),
    "window_size": window_size,
    "feature_count": len(feature_names),
    "threshold": threshold,
    "mean_reconstruction_error": float(overall_errors.mean()),
    "median_reconstruction_error": float(np.median(overall_errors)),
    "minimum_reconstruction_error": float(overall_errors.min()),
    "maximum_reconstruction_error": float(overall_errors.max()),
    "flagged_windows": int(predicted_anomaly.sum()),
    "unflagged_windows": int((~predicted_anomaly).sum()),
    "feature_error_method": (
        "Mean squared scaled reconstruction error across the 10 time steps "
        "for each feature."
    ),
    "contribution_method": (
        "Each feature's mean squared error divided by the sum of mean "
        "feature errors, expressed as a percentage."
    ),
    "warning": (
        "Feature contributions are reconstruction-error attributions, "
        "not causal explanations. Overlapping windows are not independent."
    ),
}

with open(
    OUTPUT_DIR / "analysis_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2)

print("=" * 70)
print("NORMAL TEST FEATURE-LEVEL ANALYSIS COMPLETED")
print("=" * 70)

print(f"Test windows: {len(df)}")
print(f"Window shape: {scaled_windows.shape}")
print(f"Threshold: {threshold:.12f}")
print(f"Mean reconstruction error: {overall_errors.mean():.8f}")
print(f"Flagged windows: {predicted_anomaly.sum()}/{len(df)}")

print("\nFEATURE CONTRIBUTIONS — ALL NORMAL TEST WINDOWS")
print(all_summary.round(6).to_string(index=False))

print("\nFEATURE CONTRIBUTIONS — FLAGGED WINDOWS")
if len(flagged_indices):
    print(flagged_summary.round(6).to_string(index=False))
else:
    print("No normal test windows were flagged.")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("Existing model, scaler, and threshold were not modified.")