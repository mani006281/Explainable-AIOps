
"""
Generate feature-level explanations for isolated slow traffic.

Uses the existing normal_extended_v1 model, scaler, and threshold.
Does not retrain the model, refit the scaler, or modify existing results.

Feature contributions describe reconstruction-error attribution.
They do not establish causal root causes.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEMO_DIR = PROJECT_ROOT / "ecommerce-demo"

ARTIFACT_DIR = SCRIPT_DIR / "artifacts_normal_extended_v1"

INPUT_FILE = (
    DEMO_DIR / "data" / "traffic_features_slow_isolated.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "slow_traffic_explanations_v3"


# ============================================================
# 2. REQUIRED FILES
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
# 3. LOAD EXISTING CONFIGURATION AND ARTIFACTS
# ============================================================

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

feature_names = threshold_config["feature_names"]
threshold = float(threshold_config["threshold"])
window_size = int(threshold_config["window_size"])

scaler = joblib.load(SCALER_FILE)
model = tf.keras.models.load_model(MODEL_FILE)

if len(feature_names) != scaler.n_features_in_:
    raise ValueError(
        "Feature count does not match the saved scaler."
    )

if model.input_shape[1] != window_size:
    raise ValueError(
        "Model window size does not match the saved configuration."
    )

if model.input_shape[2] != len(feature_names):
    raise ValueError(
        "Model feature count does not match the configuration."
    )

if model.output_shape != model.input_shape:
    raise ValueError(
        "Model input and output shapes do not match."
    )


# ============================================================
# 4. LOAD AND VALIDATE INPUT DATA
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
        f"Input data is missing columns: {missing_columns}"
    )

data["timestamp_utc"] = pd.to_datetime(
    data["timestamp_utc"],
    utc=True,
    errors="raise",
)

data = data.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

raw_features = data[feature_names].apply(
    pd.to_numeric,
    errors="raise",
).to_numpy(dtype=np.float64)

if not np.isfinite(raw_features).all():
    raise ValueError(
        "Input features contain NaN or infinite values."
    )

observation_count = len(data)

if observation_count < window_size:
    raise ValueError(
        "There are fewer observations than the required window size."
    )


# ============================================================
# 5. SCALE FEATURES AND GENERATE WINDOWS
# ============================================================

# Apply the existing scaler only. Never refit it on slow traffic.
scaled_features = scaler.transform(
    raw_features
).astype(np.float32)

sequences = np.stack(
    [
        scaled_features[index:index + window_size]
        for index in range(
            observation_count - window_size + 1
        )
    ],
    axis=0,
).astype(np.float32)

expected_shape = (
    observation_count - window_size + 1,
    window_size,
    len(feature_names),
)

if sequences.shape != expected_shape:
    raise ValueError(
        f"Unexpected sequence shape: {sequences.shape}"
    )


# ============================================================
# 6. RECONSTRUCT WINDOWS
# ============================================================

reconstructed = model.predict(
    sequences,
    verbose=0,
)

if reconstructed.shape != sequences.shape:
    raise ValueError(
        "Reconstructed data shape does not match input sequences."
    )

# Squared errors are calculated in the same scaled feature space
# used for the overall anomaly score.
squared_errors = np.square(
    sequences - reconstructed
)

if not np.isfinite(squared_errors).all():
    raise ValueError(
        "Squared reconstruction errors contain invalid values."
    )

# Shape: (number_of_windows, number_of_features)
feature_errors = np.mean(
    squared_errors,
    axis=1,
)

# Mean over both time and feature dimensions.
overall_errors = np.mean(
    squared_errors,
    axis=(1, 2),
)

is_anomaly = overall_errors > threshold

window_count = len(sequences)


# ============================================================
# 7. CREATE WINDOW-LEVEL RESULTS
# ============================================================

window_rows = []
feature_error_rows = []
contribution_rows = []
explanation_rows = []

for index in range(window_count):
    window_number = index + 1

    start_time = data["timestamp_utc"].iloc[index]
    end_time = data["timestamp_utc"].iloc[
        index + window_size - 1
    ]

    window_data = data.iloc[
        index:index + window_size
    ]

    overall_error = float(overall_errors[index])
    flagged = bool(is_anomaly[index])

    feature_error_values = feature_errors[index]

    total_feature_error = float(
        np.sum(feature_error_values)
    )

    if total_feature_error > 0:
        contributions = (
            feature_error_values / total_feature_error
        ) * 100.0
    else:
        contributions = np.zeros(
            len(feature_names),
            dtype=np.float64,
        )

    window_rows.append(
        {
            "window_number": window_number,
            "start_timestamp": start_time.isoformat(),
            "end_timestamp": end_time.isoformat(),
            "mean_response_time_ms": float(
                window_data["response_time_ms"].mean()
            ),
            "max_response_time_ms": float(
                window_data["response_time_ms"].max()
            ),
            "reconstruction_error": overall_error,
            "threshold": threshold,
            "threshold_exceedance": (
                overall_error - threshold
            ),
            "is_anomaly_candidate": flagged,
        }
    )

    for feature_index, feature_name in enumerate(feature_names):
        error_value = float(
            feature_error_values[feature_index]
        )

        contribution_value = float(
            contributions[feature_index]
        )

        feature_error_rows.append(
            {
                "window_number": window_number,
                "feature": feature_name,
                "mean_squared_scaled_reconstruction_error": (
                    error_value
                ),
                "window_is_anomaly": flagged,
            }
        )

        contribution_rows.append(
            {
                "window_number": window_number,
                "feature": feature_name,
                "mean_squared_scaled_reconstruction_error": (
                    error_value
                ),
                "contribution_percentage": contribution_value,
                "window_is_anomaly": flagged,
            }
        )

    # Rank all features by their contribution to this window.
    ranked_indices = np.argsort(
        feature_errors[index]
    )[::-1]

    top_indices = ranked_indices[:3]

    top_features = [
        (
            f"{feature_names[feature_index]} "
            f"({contributions[feature_index]:.2f}%)"
        )
        for feature_index in top_indices
    ]

    top_features_text = "; ".join(top_features)

    if flagged:
        explanation = (
            f"Window {window_number} was flagged because its "
            f"reconstruction error ({overall_error:.6f}) exceeded "
            f"the existing threshold ({threshold:.6f}). "
            f"The largest feature-level reconstruction-error "
            f"contributions were: {top_features_text}. "
            f"Mean response time was "
            f"{window_data['response_time_ms'].mean():.2f} ms, "
            f"and maximum response time was "
            f"{window_data['response_time_ms'].max():.2f} ms. "
            f"These feature contributions describe model "
            f"reconstruction error and do not prove causal root causes."
        )
    else:
        explanation = (
            f"Window {window_number} was not flagged because its "
            f"reconstruction error ({overall_error:.6f}) did not "
            f"exceed the existing threshold ({threshold:.6f}). "
            f"The largest feature-level reconstruction-error "
            f"contributions were: {top_features_text}. "
            f"These contributions describe reconstruction error, "
            f"not causal root causes."
        )

    explanation_rows.append(
        {
            "window_number": window_number,
            "start_timestamp": start_time.isoformat(),
            "end_timestamp": end_time.isoformat(),
            "is_anomaly_candidate": flagged,
            "reconstruction_error": overall_error,
            "threshold": threshold,
            "threshold_exceedance": (
                overall_error - threshold
            ),
            "top_contributing_features": top_features_text,
            "explanation": explanation,
        }
    )


# ============================================================
# 8. BUILD OUTPUT DATAFRAMES
# ============================================================

window_results_df = pd.DataFrame(window_rows)
feature_errors_df = pd.DataFrame(feature_error_rows)
contributions_df = pd.DataFrame(contribution_rows)
explanations_df = pd.DataFrame(explanation_rows)

flagged_count = int(np.sum(is_anomaly))
unflagged_count = int(window_count - flagged_count)

summary = {
    "experiment": threshold_config.get(
        "experiment",
        "normal_extended_v1",
    ),
    "report_version": 3,
    "source_report_version": 2,
    "feature_flag_semantics": (
        "window_is_anomaly indicates whether the complete sequence "
        "was classified as anomalous using the saved overall "
        "reconstruction-error threshold. It is repeated across "
        "feature rows for context and is not an independent "
        "anomaly decision for each feature."
    ),
    "model_inference_rerun": True,
    "model_artifacts_modified": False,
    "dataset": "slow_isolated",
    "input_file": str(INPUT_FILE),
    "model_file": str(MODEL_FILE),
    "scaler_file": str(SCALER_FILE),
    "threshold_file": str(THRESHOLD_FILE),
    "observations": int(observation_count),
    "feature_count": int(len(feature_names)),
    "feature_names": feature_names,
    "window_size": int(window_size),
    "windows_evaluated": int(window_count),
    "threshold": threshold,
    "flagged_windows": flagged_count,
    "unflagged_windows": unflagged_count,
    "flagged_percentage": (
        100.0 * flagged_count / window_count
        if window_count
        else 0.0
    ),
    "mean_reconstruction_error": float(
        np.mean(overall_errors)
    ),
    "median_reconstruction_error": float(
        np.median(overall_errors)
    ),
    "minimum_reconstruction_error": float(
        np.min(overall_errors)
    ),
    "maximum_reconstruction_error": float(
        np.max(overall_errors)
    ),
    "feature_error_method": (
        "Mean squared scaled reconstruction error over the "
        "window's time steps, calculated separately for each feature."
    ),
    "contribution_method": (
        "A feature's mean squared scaled reconstruction error "
        "divided by the sum of feature errors in the same window, "
        "expressed as a percentage."
    ),
    "warning": (
        "Feature contributions are reconstruction-error attributions, "
        "not causal explanations. Windows overlap and are not "
        "independent observations. Results describe this dataset only "
        "and do not establish real-world detection performance."
    ),
}


# ============================================================
# 9. SAVE TO A NEW DIRECTORY WITHOUT OVERWRITING
# ============================================================

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory already contains files: {OUTPUT_DIR}\n"
        "Move or rename that directory before running this script again."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

window_results_df.to_csv(
    OUTPUT_DIR / "slow_window_results.csv",
    index=False,
)

feature_errors_df.to_csv(
    OUTPUT_DIR / "slow_window_feature_errors.csv",
    index=False,
)

contributions_df.to_csv(
    OUTPUT_DIR / "slow_window_feature_contributions.csv",
    index=False,
)

explanations_df.to_csv(
    OUTPUT_DIR / "slow_window_explanations.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "slow_traffic_explanation_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# 10. PRINT VERIFICATION SUMMARY
# ============================================================

print("=" * 70)
print("SLOW-TRAFFIC FEATURE EXPLANATION COMPLETED")
print("=" * 70)

print(f"Observations: {observation_count}")
print(f"Features: {len(feature_names)}")
print(f"Window size: {window_size}")
print(f"Windows evaluated: {window_count}")
print(f"Existing threshold: {threshold:.12f}")

print(
    f"Mean reconstruction error: "
    f"{summary['mean_reconstruction_error']:.8f}"
)

print(
    f"Flagged windows: {flagged_count}/{window_count}"
)

print(
    f"Unflagged windows: {unflagged_count}/{window_count}"
)

print("\nGenerated files:")

for output_file in sorted(OUTPUT_DIR.iterdir()):
    print(f"- {output_file.name}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nExisting model, scaler, and threshold were not modified.")

