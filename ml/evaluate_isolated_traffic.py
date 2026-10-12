
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "ecommerce-demo" / "data"
ARTIFACT_DIR = ROOT_DIR / "ml" / "artifacts"

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

DATASETS = {
    "normal": {
        "features": "traffic_features_normal_isolated.csv",
        "metadata": "traffic_feature_metadata_normal_isolated.csv",
    },
    "slow": {
        "features": "traffic_features_slow_isolated.csv",
        "metadata": "traffic_feature_metadata_slow_isolated.csv",
    },
}


# --------------------------------------------------
# Load existing artifacts
# --------------------------------------------------

with open(
    ARTIFACT_DIR / "anomaly_threshold.json",
    "r",
    encoding="utf-8",
) as file:
    threshold_config = json.load(file)

threshold = float(threshold_config["threshold"])
window_size = int(threshold_config["window_size"])

scaler = joblib.load(
    ARTIFACT_DIR / "feature_scaler.joblib"
)

model = tf.keras.models.load_model(
    ARTIFACT_DIR / "lstm_autoencoder.keras"
)

print("=" * 72)
print("EXPLAINABLE AIOPS: ISOLATED TRAFFIC EVALUATION")
print("=" * 72)
print(f"Existing threshold: {threshold:.6f}")
print(f"Window size: {window_size}")
print("Existing model and scaler loaded.")
print("No retraining will be performed.")


# --------------------------------------------------
# Evaluate one dataset
# --------------------------------------------------

def evaluate_dataset(dataset_name, filenames):
    feature_path = DATA_DIR / filenames["features"]
    metadata_path = DATA_DIR / filenames["metadata"]

    features_df = pd.read_csv(feature_path)
    metadata_df = pd.read_csv(metadata_path)

    missing_features = [
        name for name in FEATURE_NAMES
        if name not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in {dataset_name}: "
            f"{missing_features}"
        )

    if len(features_df) != len(metadata_df):
        raise ValueError(
            f"Feature/metadata row counts differ for "
            f"{dataset_name}."
        )

    if not features_df["timestamp_utc"].equals(
        metadata_df["timestamp_utc"]
    ):
        raise ValueError(
            f"Feature/metadata timestamps differ for "
            f"{dataset_name}."
        )

    X = features_df[FEATURE_NAMES].to_numpy(
        dtype=np.float32
    )

    if len(X) < window_size:
        raise ValueError(
            f"{dataset_name} has fewer observations than "
            f"the window size."
        )

    # Create chronological, overlapping windows.
    windows = np.stack([
        X[i:i + window_size]
        for i in range(len(X) - window_size + 1)
    ])

    n_windows, size, n_features = windows.shape

    scaled = scaler.transform(
        windows.reshape(-1, n_features)
    ).reshape(
        n_windows, size, n_features
    ).astype(np.float32)

    reconstructed = model.predict(
        scaled,
        verbose=0,
    )

    # One mean squared reconstruction error per window.
    errors = np.mean(
        np.square(scaled - reconstructed),
        axis=(1, 2),
    )

    flags = errors > threshold

    # Build a report for every window.
    rows = []

    for i in range(n_windows):
        start = i
        end = i + window_size

        window_metadata = metadata_df.iloc[start:end]

        rows.append({
            "dataset": dataset_name,
            "window_number": i + 1,
            "start_timestamp": window_metadata[
                "timestamp_utc"
            ].iloc[0],
            "end_timestamp": window_metadata[
                "timestamp_utc"
            ].iloc[-1],
            "mean_response_time_ms": float(
                features_df["response_time_ms"]
                .iloc[start:end]
                .mean()
            ),
            "max_response_time_ms": float(
                features_df["response_time_ms"]
                .iloc[start:end]
                .max()
            ),
            "reconstruction_error": float(errors[i]),
            "threshold": threshold,
            "is_anomaly_candidate": bool(flags[i]),
        })

    result_df = pd.DataFrame(rows)

    output_path = (
        ARTIFACT_DIR
        / f"isolated_{dataset_name}_evaluation.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("\n" + "-" * 72)
    print(f"{dataset_name.upper()} TRAFFIC")
    print("-" * 72)
    print(f"Observations: {len(X)}")
    print(f"Windows evaluated: {n_windows}")
    print(f"Mean response time: {X[:, 0].mean():.2f} ms")
    print(f"Maximum response time: {X[:, 0].max():.2f} ms")
    print(f"Mean reconstruction error: {errors.mean():.6f}")
    print(f"Median reconstruction error: "
          f"{np.median(errors):.6f}")
    print(f"Minimum reconstruction error: {errors.min():.6f}")
    print(f"Maximum reconstruction error: {errors.max():.6f}")
    print(f"Flagged windows: {int(flags.sum())}/{n_windows}")
    print(f"Flagged fraction: {flags.mean():.2%}")
    print(f"Results saved to: {output_path}")

    return result_df


# --------------------------------------------------
# Run both independent evaluations
# --------------------------------------------------

normal_results = evaluate_dataset(
    "normal",
    DATASETS["normal"],
)

slow_results = evaluate_dataset(
    "slow",
    DATASETS["slow"],
)

print("\n" + "=" * 72)
print("COMPARISON SUMMARY")
print("=" * 72)

for name, results in [
    ("Normal", normal_results),
    ("Slow", slow_results),
]:
    print(
        f"{name:8} | "
        f"Mean error: "
        f"{results['reconstruction_error'].mean():10.6f} | "
        f"Flagged: "
        f"{int(results['is_anomaly_candidate'].sum()):2d}/"
        f"{len(results):2d}"
    )

print("\nEvaluation completed.")
print("The trained model, scaler, and threshold were not modified.")

