
"""
Explainable AIOps
Additional Traffic Anomaly Detection

Uses the existing trained LSTM Autoencoder, feature scaler,
and anomaly threshold to evaluate additional traffic.

This script does not retrain the model or modify baseline results.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# 1. Define project paths and model configuration
# --------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"

FEATURES_FILE = DATA_DIR / "traffic_features_additional.csv"
METADATA_FILE = DATA_DIR / "traffic_feature_metadata_additional.csv"

MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"

OUTPUT_FILE = ARTIFACT_DIR / "additional_traffic_anomaly_results.csv"

FEATURE_NAMES = [
    "response_time_ms",
    "log_response_time",
    "status_200",
    "status_400",
    "status_404",
    "status_other",
    "endpoint_health",
    "endpoint_products",
    "endpoint_latency",
    "endpoint_other",
    "is_connection_error",
    "is_http_error",
]


# --------------------------------------------------
# 2. Load and validate input data
# --------------------------------------------------

def load_input_data():
    """Load additional features and corresponding metadata."""

    required_files = [
        FEATURES_FILE,
        METADATA_FILE,
        MODEL_FILE,
        SCALER_FILE,
        THRESHOLD_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    features_df = pd.read_csv(FEATURES_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)

    if len(features_df) != len(metadata_df):
        raise ValueError(
            "Feature and metadata row counts do not match."
        )

    if "timestamp_utc" not in features_df.columns:
        raise ValueError(
            "Feature dataset is missing timestamp_utc."
        )

    if "timestamp_utc" not in metadata_df.columns:
        raise ValueError(
            "Metadata dataset is missing timestamp_utc."
        )

    feature_timestamps = pd.to_datetime(
        features_df["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    metadata_timestamps = pd.to_datetime(
        metadata_df["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    if not feature_timestamps.equals(metadata_timestamps):
        raise ValueError(
            "Feature and metadata timestamps are not aligned."
        )

    missing_features = [
        name for name in FEATURE_NAMES
        if name not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required model features: {missing_features}"
        )

    if len(features_df) < 10:
        raise ValueError(
            "At least 10 observations are required."
        )

    X = features_df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature data contains NaN or infinite values."
        )

    metadata_df = metadata_df.copy()
    metadata_df["timestamp_utc"] = metadata_timestamps

    print(f"Additional traffic observations: {len(X)}")
    print(f"Number of model features: {X.shape[1]}")
    print("Feature and metadata timestamps aligned: True")

    return X, metadata_df


# --------------------------------------------------
# 3. Create chronological sliding windows
# --------------------------------------------------

def create_sliding_windows(X, metadata, window_size):
    """Build overlapping windows without changing time order."""

    if len(X) < window_size:
        raise ValueError(
            f"Need at least {window_size} observations."
        )

    windows = np.stack([
        X[start:start + window_size]
        for start in range(len(X) - window_size + 1)
    ])

    window_metadata = []

    for start in range(len(X) - window_size + 1):
        end = start + window_size
        block = metadata.iloc[start:end]

        scenarios = (
            block["scenario"]
            .fillna("unknown")
            .astype(str)
            .unique()
            .tolist()
            if "scenario" in block.columns
            else ["unknown"]
        )

        statuses = (
            pd.to_numeric(
                block["status_code"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
            if "status_code" in block.columns
            else []
        )

        response_times = (
            pd.to_numeric(
                block["measured_response_time_ms"],
                errors="coerce",
            )
            if "measured_response_time_ms" in block.columns
            else pd.Series(dtype=float)
        )

        expected_delays = (
            pd.to_numeric(
                block["expected_delay_ms"],
                errors="coerce",
            )
            if "expected_delay_ms" in block.columns
            else pd.Series(dtype=float)
        )

        contains_slow_latency = (
            bool(block["scenario"].eq("slow_latency").any())
            if "scenario" in block.columns
            else False
        )

        contains_http_error = (
            bool(
                (
                    pd.to_numeric(
                        block["status_code"],
                        errors="coerce",
                    ) >= 400
                ).any()
            )
            if "status_code" in block.columns
            else False
        )

        window_metadata.append({
            "window_index": start,
            "start_timestamp_utc": block["timestamp_utc"].iloc[0],
            "end_timestamp_utc": block["timestamp_utc"].iloc[-1],
            "scenarios_in_window": "|".join(scenarios),
            "status_codes_in_window": "|".join(
                str(status) for status in statuses
            ),
            "contains_slow_latency": contains_slow_latency,
            "contains_http_error": contains_http_error,
            "max_response_time_ms": (
                float(response_times.max())
                if response_times.notna().any()
                else np.nan
            ),
            "max_expected_delay_ms": (
                float(expected_delays.max())
                if expected_delays.notna().any()
                else np.nan
            ),
        })

    return windows, pd.DataFrame(window_metadata)


# --------------------------------------------------
# 4. Load existing trained artifacts
# --------------------------------------------------

def load_artifacts():
    """Load the original model, scaler, and threshold."""

    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_data = json.load(file)

    threshold = float(threshold_data["threshold"])
    window_size = int(threshold_data["window_size"])
    saved_feature_names = threshold_data["feature_names"]

    if saved_feature_names != FEATURE_NAMES:
        raise ValueError(
            "Feature order does not match the saved configuration."
        )

    if scaler.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(
            "Scaler feature count does not match."
        )

    if model.input_shape[-1] != len(FEATURE_NAMES):
        raise ValueError(
            "Model input feature count does not match."
        )

    if model.input_shape[1] not in (None, window_size):
        raise ValueError(
            "Model window size does not match the threshold configuration."
        )

    print("Existing LSTM Autoencoder loaded.")
    print("Existing StandardScaler loaded.")
    print(f"Saved anomaly threshold: {threshold:.6f}")
    print(f"Configured window size: {window_size}")

    return model, scaler, threshold, window_size


# --------------------------------------------------
# 5. Scale and calculate reconstruction errors
# --------------------------------------------------

def calculate_reconstruction_errors(model, scaler, windows):
    """Scale with the original scaler and score each window."""

    number_of_windows, window_size, number_of_features = windows.shape

    flattened = windows.reshape(
        -1,
        number_of_features,
    )

    scaled = scaler.transform(flattened)

    scaled_windows = scaled.reshape(
        number_of_windows,
        window_size,
        number_of_features,
    ).astype(np.float32)

    reconstructed = model.predict(
        scaled_windows,
        verbose=0,
    )

    errors = np.mean(
        np.square(scaled_windows - reconstructed),
        axis=(1, 2),
    )

    return errors


# --------------------------------------------------
# 6. Save results and summarize detections
# --------------------------------------------------

def save_and_summarize(errors, window_metadata, threshold):
    """Save reconstruction scores and detection statistics."""

    results = window_metadata.copy()

    results["reconstruction_error"] = errors
    results["threshold"] = threshold
    results["is_anomaly_candidate"] = errors > threshold

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    anomaly_count = int(
        results["is_anomaly_candidate"].sum()
    )

    print("\n" + "=" * 70)
    print("ADDITIONAL TRAFFIC ANOMALY DETECTION SUMMARY")
    print("=" * 70)

    print(f"Scored windows: {len(results)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Mean reconstruction error: {errors.mean():.6f}")
    print(f"Median reconstruction error: {np.median(errors):.6f}")
    print(f"Maximum reconstruction error: {errors.max():.6f}")
    print(f"Minimum reconstruction error: {errors.min():.6f}")
    print(f"Anomaly candidate windows: {anomaly_count}")

    print("\nDetection by window contents:")

    for column, label in [
        (
            "contains_slow_latency",
            "Windows containing slow-latency traffic",
        ),
        (
            "contains_http_error",
            "Windows containing HTTP errors",
        ),
    ]:
        subset = results[results[column]]

        if len(subset) > 0:
            detected = int(
                subset["is_anomaly_candidate"].sum()
            )

            percentage = detected / len(subset) * 100

            print(
                f"{label}: {detected}/{len(subset)} "
                f"flagged ({percentage:.1f}%)"
            )

    print("\nResults grouped by window contents:")

    summary = results.groupby(
        [
            "contains_slow_latency",
            "contains_http_error",
        ],
        dropna=False,
    )["is_anomaly_candidate"].agg(
        window_count="count",
        flagged_count="sum",
        flagged_fraction="mean",
    )

    print(summary.to_string())

    print(f"\nResults saved to:\n{OUTPUT_FILE}")

    print(
        "\nReminder: These are anomaly candidates, not confirmed "
        "incidents. Overlapping windows can contain both normal "
        "and abnormal traffic. The threshold is preliminary and "
        "requires evaluation using representative labeled traffic."
    )


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

def main():
    print("=" * 70)
    print("EXPLAINABLE AIOPS: ADDITIONAL TRAFFIC EVALUATION")
    print("=" * 70)

    X, metadata = load_input_data()

    model, scaler, threshold, window_size = load_artifacts()

    windows, window_metadata = create_sliding_windows(
        X,
        metadata,
        window_size,
    )

    print(f"\nSliding-window tensor shape: {windows.shape}")
    print("Applying the existing scaler and model...")

    errors = calculate_reconstruction_errors(
        model,
        scaler,
        windows,
    )

    save_and_summarize(
        errors,
        window_metadata,
        threshold,
    )

    print("\nAdditional traffic evaluation completed.")


if __name__ == "__main__":
    main()

