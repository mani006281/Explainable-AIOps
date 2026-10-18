import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "ml"
DATA_DIR = ROOT_DIR / "ecommerce-demo" / "data"

ARTIFACTS = {
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
    "normal": "traffic_features_normal_isolated.csv",
    "slow": "traffic_features_slow_isolated.csv",
}


def load_model_and_scaler(name, artifact_dir):
    print(f"\nLoading {name}...")

    model_path = artifact_dir / "lstm_autoencoder.keras"
    scaler_path = artifact_dir / "feature_scaler.joblib"

    with open(
        artifact_dir / "anomaly_threshold.json",
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    scaler = joblib.load(scaler_path)

    print("Model path:", model_path)
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    print("Threshold:", config["threshold"])
    print("Scaler feature count:", scaler.n_features_in_)

    return model, scaler, config


def create_windows(features, window_size):
    return np.stack([
        features[i:i + window_size]
        for i in range(len(features) - window_size + 1)
    ])


def prepare_data(dataset_name, filename, scaler, window_size):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    windows = create_windows(features, window_size)

    n_windows, sequence_length, n_features = windows.shape

    scaled = scaler.transform(
        windows.reshape(-1, n_features)
    ).reshape(
        n_windows,
        sequence_length,
        n_features,
    ).astype(np.float32)

    print(
        f"{dataset_name}: observations={len(df)}, "
        f"windows={n_windows}, shape={scaled.shape}"
    )

    return scaled


def main():
    loaded = {}

    for name, artifact_dir in ARTIFACTS.items():
        loaded[name] = load_model_and_scaler(
            name,
            artifact_dir,
        )

    original_model, original_scaler, original_config = loaded["Original"]
    baseline_model, baseline_scaler, baseline_config = loaded["Baseline_V2"]

    print("\n" + "=" * 70)
    print("MODEL AND SCALER COMPARISON")
    print("=" * 70)

    print(
        "Model architectures identical:",
        original_model.to_json() == baseline_model.to_json(),
    )

    original_weights = original_model.get_weights()
    baseline_weights = baseline_model.get_weights()

    print("Original weight arrays:", len(original_weights))
    print("Baseline_V2 weight arrays:", len(baseline_weights))

    if len(original_weights) == len(baseline_weights):
        weight_differences = []

        for index, (a, b) in enumerate(
            zip(original_weights, baseline_weights)
        ):
            same_shape = a.shape == b.shape

            if same_shape:
                max_difference = float(
                    np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                )
                exactly_equal = np.array_equal(a, b)
            else:
                max_difference = float("nan")
                exactly_equal = False

            weight_differences.append(exactly_equal)

            print(
                f"Weights {index}: "
                f"shape_original={a.shape}, "
                f"shape_baseline={b.shape}, "
                f"exactly_equal={exactly_equal}, "
                f"max_abs_difference={max_difference}"
            )

        print(
            "All model weights exactly equal:",
            all(weight_differences),
        )

    original_scaler_params = [
        "mean_", "scale_", "var_"
    ]

    print("\nScaler parameter comparison:")

    for attribute in original_scaler_params:
        if hasattr(original_scaler, attribute) and hasattr(
            baseline_scaler, attribute
        ):
            a = getattr(original_scaler, attribute)
            b = getattr(baseline_scaler, attribute)

            print(
                f"{attribute}:",
                np.array_equal(a, b),
            )

    if original_config["window_size"] != baseline_config["window_size"]:
        raise ValueError("Window sizes differ.")

    window_size = int(original_config["window_size"])

    if original_config["feature_names"] != baseline_config["feature_names"]:
        raise ValueError("Feature orders differ.")

    print("\n" + "=" * 70)
    print("INDEPENDENT PREDICTION COMPARISON")
    print("=" * 70)

    for dataset_name, filename in DATASETS.items():
        print(f"\n--- {dataset_name.upper()} DATASET ---")

        # Transform separately using each model's own scaler.
        original_windows = prepare_data(
            dataset_name,
            filename,
            original_scaler,
            window_size,
        )

        baseline_windows = prepare_data(
            dataset_name,
            filename,
            baseline_scaler,
            window_size,
        )

        print(
            "Scaled input arrays exactly equal:",
            np.array_equal(original_windows, baseline_windows),
        )

        original_predictions = original_model.predict(
            original_windows,
            verbose=0,
        )

        baseline_predictions = baseline_model.predict(
            baseline_windows,
            verbose=0,
        )

        if original_predictions.shape != baseline_predictions.shape:
            print("Prediction shapes differ:")
            print("Original:", original_predictions.shape)
            print("Baseline_V2:", baseline_predictions.shape)
            continue

        prediction_difference = np.abs(
            original_predictions.astype(np.float64)
            - baseline_predictions.astype(np.float64)
        )

        print("Prediction shape:", original_predictions.shape)
        print(
            "Predictions exactly equal:",
            np.array_equal(
                original_predictions,
                baseline_predictions,
            ),
        )
        print(
            "Maximum absolute prediction difference:",
            float(prediction_difference.max()),
        )
        print(
            "Mean absolute prediction difference:",
            float(prediction_difference.mean()),
        )

        original_errors = np.mean(
            np.square(original_windows - original_predictions),
            axis=(1, 2),
        )

        baseline_errors = np.mean(
            np.square(baseline_windows - baseline_predictions),
            axis=(1, 2),
        )

        print(
            "Reconstruction errors exactly equal:",
            np.array_equal(original_errors, baseline_errors),
        )
        print(
            "Maximum absolute reconstruction-error difference:",
            float(np.max(np.abs(original_errors - baseline_errors))),
        )

        original_flags = (
            original_errors > float(original_config["threshold"])
        )
        baseline_flags = (
            baseline_errors > float(baseline_config["threshold"])
        )

        print(
            "Different anomaly flags:",
            int(np.sum(original_flags != baseline_flags)),
        )

    print("\nDiagnostic completed. No model artifacts or reports were saved.")


if __name__ == "__main__":
    main()