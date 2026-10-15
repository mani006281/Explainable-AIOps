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
ML_DIR = ROOT_DIR / "ml"

MODELS = {
    "Original": ML_DIR / "artifacts",
    "Baseline_V2": ML_DIR / "artifacts_baseline_v2",
}

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

OUTPUT_DIR = ML_DIR / "comparison_baseline_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load and validate datasets
# --------------------------------------------------

def load_dataset(dataset_name, filenames):
    feature_path = DATA_DIR / filenames["features"]
    metadata_path = DATA_DIR / filenames["metadata"]

    features_df = pd.read_csv(feature_path)
    metadata_df = pd.read_csv(metadata_path)

    missing = [
        name for name in FEATURE_NAMES
        if name not in features_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features in {dataset_name}: {missing}"
        )

    if len(features_df) != len(metadata_df):
        raise ValueError(
            f"Feature and metadata row counts differ: "
            f"{dataset_name}"
        )

    if not features_df["timestamp_utc"].equals(
        metadata_df["timestamp_utc"]
    ):
        raise ValueError(
            f"Timestamp mismatch in {dataset_name}"
        )

    X = features_df[FEATURE_NAMES].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite feature values in {dataset_name}"
        )

    return X, features_df, metadata_df


def create_windows(X, window_size):
    if len(X) < window_size:
        raise ValueError(
            f"Only {len(X)} observations for "
            f"window size {window_size}"
        )

    return np.stack([
        X[i:i + window_size]
        for i in range(len(X) - window_size + 1)
    ])


# --------------------------------------------------
# Evaluate one model on one dataset
# --------------------------------------------------

def evaluate_model(model_name, artifact_dir, dataset_name, data):
    X, features_df, metadata_df = data

    with open(
        artifact_dir / "anomaly_threshold.json",
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    threshold = float(config["threshold"])
    window_size = int(config["window_size"])

    if config.get("feature_names") != FEATURE_NAMES:
        raise ValueError(
            f"Feature order mismatch for {model_name}"
        )

    scaler = joblib.load(
        artifact_dir / "feature_scaler.joblib"
    )

    model = tf.keras.models.load_model(
        artifact_dir / "lstm_autoencoder.keras"
    )

    windows = create_windows(X, window_size)

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

    errors = np.mean(
        np.square(scaled - reconstructed),
        axis=(1, 2),
    )

    flags = errors > threshold

    rows = []

    for i in range(n_windows):
        start = i
        end = i + window_size

        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "window_number": i + 1,
            "start_timestamp": metadata_df[
                "timestamp_utc"
            ].iloc[start],
            "end_timestamp": metadata_df[
                "timestamp_utc"
            ].iloc[end - 1],
            "mean_response_time_ms": float(
                features_df["response_time_ms"]
                .iloc[start:end].mean()
            ),
            "max_response_time_ms": float(
                features_df["response_time_ms"]
                .iloc[start:end].max()
            ),
            "reconstruction_error": float(errors[i]),
            "threshold": threshold,
            "is_anomaly_candidate": bool(flags[i]),
        })

    result = pd.DataFrame(rows)

    result_path = OUTPUT_DIR / (
        f"{model_name.lower()}_{dataset_name}_results.csv"
    )

    result.to_csv(result_path, index=False)

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "observations": len(X),
        "windows": n_windows,
        "threshold": threshold,
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "minimum_error": float(errors.min()),
        "maximum_error": float(errors.max()),
        "flagged_windows": int(flags.sum()),
        "flagged_percentage": float(flags.mean() * 100),
        "results_file": str(result_path),
    }

    print(
        f"{model_name:12} | "
        f"{dataset_name:6} | "
        f"Threshold: {threshold:.6f} | "
        f"Mean error: {errors.mean():.6f} | "
        f"Flagged: {int(flags.sum())}/{n_windows} "
        f"({flags.mean():.2%})"
    )

    return summary, result


# --------------------------------------------------
# Run the comparison
# --------------------------------------------------

def main():
    print("=" * 90)
    print("EXPLAINABLE AIOPS: ORIGINAL VS BASELINE V2")
    print("=" * 90)

    datasets = {
        name: load_dataset(name, filenames)
        for name, filenames in DATASETS.items()
    }

    summaries = []
    detailed_results = {}

    for model_name, artifact_dir in MODELS.items():
        print(f"\nLoading and evaluating: {model_name}")

        for dataset_name, data in datasets.items():
            summary, result = evaluate_model(
                model_name,
                artifact_dir,
                dataset_name,
                data,
            )

            summaries.append(summary)
            detailed_results[
                (model_name, dataset_name)
            ] = result

    summary_df = pd.DataFrame(summaries)

    summary_path = OUTPUT_DIR / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 90)

    columns = [
        "model",
        "dataset",
        "threshold",
        "mean_error",
        "flagged_windows",
        "windows",
        "flagged_percentage",
    ]

    print(
        summary_df[columns].to_string(
            index=False,
            formatters={
                "threshold": "{:.6f}".format,
                "mean_error": "{:.6f}".format,
                "flagged_percentage": "{:.2f}%".format,
            },
        )
    )

    # Compare each model's reconstruction error distributions.
    print("\nSLOW-TRAFFIC VS NORMAL-TRAFFIC ERROR SEPARATION")

    for model_name in MODELS:
        normal = detailed_results[
            (model_name, "normal")
        ]["reconstruction_error"].mean()

        slow = detailed_results[
            (model_name, "slow")
        ]["reconstruction_error"].mean()

        ratio = slow / normal if normal > 0 else float("inf")

        print(
            f"{model_name:12} | "
            f"Normal mean error: {normal:.6f} | "
            f"Slow mean error: {slow:.6f} | "
            f"Slow/normal ratio: {ratio:.2f}x"
        )

    print(f"\nComparison summary saved to: {summary_path}")
    print(f"All comparison reports saved in: {OUTPUT_DIR}")
    print("\nEvaluation completed. Original model artifacts were not modified.")


if __name__ == "__main__":
    main()