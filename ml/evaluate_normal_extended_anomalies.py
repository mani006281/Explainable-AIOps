"""
Evaluate the normal_extended_v1 LSTM autoencoder on isolated slow traffic.

This script:
- Loads the existing trained model, scaler, and threshold.
- Preserves the original training artifacts.
- Uses the exact feature order stored in anomaly_threshold.json.
- Generates overlapping windows from isolated slow traffic.
- Calculates reconstruction errors and anomaly candidates.
- Saves results in a separate evaluation directory.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEMO_DIR = PROJECT_ROOT / "ecommerce-demo"

ARTIFACT_DIR = SCRIPT_DIR / "artifacts_normal_extended_v1"

INPUT_FILE = DEMO_DIR / "data" / "traffic_features_slow_isolated.csv"

OUTPUT_DIR = SCRIPT_DIR / "evaluation_normal_extended_v1"


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"

required_files = [
    INPUT_FILE,
    MODEL_FILE,
    SCALER_FILE,
    THRESHOLD_FILE,
]

for file_path in required_files:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}"
        )


# ============================================================
# 3. LOAD EXISTING ARTIFACTS
# ============================================================

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

feature_names = threshold_config["feature_names"]
threshold = float(threshold_config["threshold"])
window_size = int(threshold_config["window_size"])

scaler = joblib.load(SCALER_FILE)

model = tf.keras.models.load_model(MODEL_FILE)

print("=" * 65)
print("NORMAL EXTENDED V1 - SLOW TRAFFIC EVALUATION")
print("=" * 65)

print(f"Experiment: {threshold_config['experiment']}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Scaler type: {type(scaler).__name__}")
print(f"Features expected: {scaler.n_features_in_}")
print(f"Window size: {window_size}")
print(f"Existing threshold: {threshold:.12f}")


# ============================================================
# 4. VALIDATE MODEL AND SCALER CONFIGURATION
# ============================================================

if len(feature_names) != scaler.n_features_in_:
    raise ValueError(
        "Feature count in threshold configuration does not "
        "match the saved scaler."
    )

if model.input_shape[1] != window_size:
    raise ValueError(
        "Model window size does not match threshold configuration."
    )

if model.input_shape[2] != len(feature_names):
    raise ValueError(
        "Model feature count does not match threshold configuration."
    )

if model.output_shape != model.input_shape:
    raise ValueError(
        "Autoencoder input and output shapes do not match."
    )


# ============================================================
# 5. LOAD AND VALIDATE SLOW TRAFFIC DATA
# ============================================================

data = pd.read_csv(INPUT_FILE)

required_columns = ["timestamp_utc"] + feature_names

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Parse timestamps as UTC and sort observations chronologically.
data["timestamp_utc"] = pd.to_datetime(
    data["timestamp_utc"],
    utc=True,
    errors="raise",
)

data = data.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

# Convert the 12 model features to numeric values.
features_df = data[feature_names].apply(
    pd.to_numeric,
    errors="raise",
)

raw_features = features_df.to_numpy(dtype=np.float64)

if not np.isfinite(raw_features).all():
    raise ValueError(
        "Feature data contains NaN or infinite values."
    )

observation_count = len(data)

if observation_count < window_size:
    raise ValueError(
        f"At least {window_size} observations are required, "
        f"but only {observation_count} were found."
    )

print("\nInput data validation:")
print(f"Input file: {INPUT_FILE}")
print(f"Observations: {observation_count}")
print(f"Features: {len(feature_names)}")
print(f"First timestamp: {data['timestamp_utc'].iloc[0]}")
print(f"Last timestamp: {data['timestamp_utc'].iloc[-1]}")


# ============================================================
# 6. APPLY THE SAVED SCALER
# ============================================================

# IMPORTANT:
# Do not fit a new scaler on slow traffic.
# Use the scaler fitted on the original normal training data.

scaled_features = scaler.transform(
    raw_features
).astype(np.float32)


# ============================================================
# 7. GENERATE OVERLAPPING SEQUENCES
# ============================================================

sequences = np.stack(
    [
        scaled_features[index:index + window_size]
        for index in range(observation_count - window_size + 1)
    ],
    axis=0,
).astype(np.float32)

expected_windows = observation_count - window_size + 1

if sequences.shape != (
    expected_windows,
    window_size,
    len(feature_names),
):
    raise ValueError(
        f"Unexpected sequence shape: {sequences.shape}"
    )

print("\nSequence preparation:")
print(f"Sequence array shape: {sequences.shape}")
print(f"Expected windows: {expected_windows}")


# ============================================================
# 8. RECONSTRUCT WINDOWS AND CALCULATE ERRORS
# ============================================================

reconstructed = model.predict(
    sequences,
    verbose=0,
)

if reconstructed.shape != sequences.shape:
    raise ValueError(
        "Reconstructed sequences have an unexpected shape."
    )

# Match the training experiment's mean squared reconstruction
# error calculation across both time steps and features.
reconstruction_errors = np.mean(
    np.square(sequences - reconstructed),
    axis=(1, 2),
)

if not np.isfinite(reconstruction_errors).all():
    raise ValueError(
        "Reconstruction errors contain invalid values."
    )

# Keep the threshold unchanged.
is_anomaly = reconstruction_errors > threshold


# ============================================================
# 9. PREPARE WINDOW-LEVEL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "model": "normal_extended_v1",
        "dataset": "slow_isolated",
        "window_number": np.arange(
            1,
            expected_windows + 1,
        ),
        "start_timestamp": [
            data["timestamp_utc"].iloc[index].isoformat()
            for index in range(expected_windows)
        ],
        "end_timestamp": [
            data["timestamp_utc"].iloc[
                index + window_size - 1
            ].isoformat()
            for index in range(expected_windows)
        ],
        "mean_response_time_ms": [
            float(
                data["response_time_ms"].iloc[
                    index:index + window_size
                ].mean()
            )
            for index in range(expected_windows)
        ],
        "max_response_time_ms": [
            float(
                data["response_time_ms"].iloc[
                    index:index + window_size
                ].max()
            )
            for index in range(expected_windows)
        ],
        "reconstruction_error": reconstruction_errors,
        "threshold": threshold,
        "is_anomaly_candidate": is_anomaly,
    }
)


# ============================================================
# 10. CREATE SUMMARY STATISTICS
# ============================================================

flagged_count = int(np.sum(is_anomaly))
total_windows = int(len(reconstruction_errors))

flagged_percentage = (
    100.0 * flagged_count / total_windows
    if total_windows
    else 0.0
)

summary = {
    "experiment": "normal_extended_v1",
    "dataset": "slow_isolated",
    "input_file": str(INPUT_FILE),
    "model_file": str(MODEL_FILE),
    "scaler_file": str(SCALER_FILE),
    "observations": int(observation_count),
    "features": int(len(feature_names)),
    "window_size": int(window_size),
    "windows_evaluated": total_windows,
    "threshold": threshold,
    "threshold_source": (
        "Existing 95th percentile of normal validation errors"
    ),
    "mean_reconstruction_error": float(
        np.mean(reconstruction_errors)
    ),
    "median_reconstruction_error": float(
        np.median(reconstruction_errors)
    ),
    "minimum_reconstruction_error": float(
        np.min(reconstruction_errors)
    ),
    "maximum_reconstruction_error": float(
        np.max(reconstruction_errors)
    ),
    "flagged_windows": flagged_count,
    "unflagged_windows": total_windows - flagged_count,
    "flagged_percentage": flagged_percentage,
    "evaluation_note": (
        "Results describe this isolated slow-traffic dataset only. "
        "They do not establish real-world detection performance, "
        "and overlapping windows are not independent observations."
    ),
}


# ============================================================
# 11. SAVE RESULTS WITHOUT MODIFYING MODEL ARTIFACTS
# ============================================================

# Do not silently overwrite a previous evaluation.
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory already contains files: {OUTPUT_DIR}\n"
        "Rename or move that directory before running a new evaluation."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_file = OUTPUT_DIR / "slow_traffic_results.csv"
summary_file = OUTPUT_DIR / "evaluation_summary.json"

results.to_csv(
    results_file,
    index=False,
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# 12. PRINT THE FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print(f"Observations: {observation_count}")
print(f"Windows evaluated: {total_windows}")
print(f"Threshold: {threshold:.12f}")

print(
    f"Mean reconstruction error: "
    f"{summary['mean_reconstruction_error']:.8f}"
)

print(
    f"Median reconstruction error: "
    f"{summary['median_reconstruction_error']:.8f}"
)

print(
    f"Minimum reconstruction error: "
    f"{summary['minimum_reconstruction_error']:.8f}"
)

print(
    f"Maximum reconstruction error: "
    f"{summary['maximum_reconstruction_error']:.8f}"
)

print(
    f"Flagged windows: {flagged_count}/{total_windows}"
)

print(
    f"Flagged percentage: {flagged_percentage:.2f}%"
)

print(f"\nResults CSV: {results_file}")
print(f"Summary JSON: {summary_file}")

print("\nExisting model and scaler were not modified.")