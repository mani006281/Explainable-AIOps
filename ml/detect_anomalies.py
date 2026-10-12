"""
Explainable AIOps
LSTM Autoencoder Anomaly Detection

Loads a trained model, scores traffic sequences,
and saves anomaly candidates to a CSV file.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"

MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"

INPUT_FILE = DATA_DIR / "sequence_windows_baseline.npz"

OUTPUT_FILE = ARTIFACT_DIR / "anomaly_detection_results.csv"


# --------------------------------------------------
# 2. Load saved artifacts
# --------------------------------------------------

def load_artifacts():
    """Load the trained model, scaler, and threshold."""

    required_files = [
        MODEL_FILE,
        SCALER_FILE,
        THRESHOLD_FILE,
        INPUT_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    print("Model loaded successfully.")
    print("Feature scaler loaded successfully.")
    print("Threshold configuration loaded successfully.")

    return model, scaler, threshold_data


# --------------------------------------------------
# 3. Load test sequences
# --------------------------------------------------

def load_test_data():
    """Load test sequences and feature names."""

    with np.load(INPUT_FILE, allow_pickle=False) as data:
        X_test = data["X_test"].astype(np.float32)
        feature_names = data["feature_names"].astype(str)
        window_size = int(data["window_size"])

    if X_test.ndim != 3:
        raise ValueError(
            "Expected a 3D array: "
            "(number_of_windows, window_size, number_of_features)"
        )

    if not np.isfinite(X_test).all():
        raise ValueError("Input data contains NaN or infinite values.")

    print(f"Test sequences shape: {X_test.shape}")
    print(f"Window size: {window_size}")
    print(f"Number of features: {len(feature_names)}")

    return X_test, feature_names


# --------------------------------------------------
# 4. Scale the data
# --------------------------------------------------

def scale_sequences(X, scaler):
    """Scale sequences using the scaler fitted during training."""

    number_of_windows, window_size, number_of_features = X.shape

    flattened = X.reshape(-1, number_of_features)
    scaled = scaler.transform(flattened)

    return scaled.reshape(
        number_of_windows,
        window_size,
        number_of_features,
    ).astype(np.float32)


# --------------------------------------------------
# 5. Calculate reconstruction errors
# --------------------------------------------------

def calculate_errors(model, X_scaled):
    """Calculate mean squared reconstruction error per window."""

    reconstructed = model.predict(
        X_scaled,
        verbose=0,
    )

    errors = np.mean(
        np.square(X_scaled - reconstructed),
        axis=(1, 2),
    )

    return errors


# --------------------------------------------------
# 6. Save anomaly detection results
# --------------------------------------------------

def save_results(errors, threshold_data):
    """Classify sequences using the saved threshold."""

    threshold = float(
        threshold_data["threshold"]
    )

    is_anomaly = errors > threshold

    results = pd.DataFrame({
        "window_index": np.arange(len(errors)),
        "reconstruction_error": errors,
        "threshold": threshold,
        "is_anomaly_candidate": is_anomaly,
    })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return results, threshold


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

def main():
    print("=" * 65)
    print("EXPLAINABLE AIOPS: ANOMALY DETECTION")
    print("=" * 65)

    model, scaler, threshold_data = load_artifacts()

    X_test, feature_names = load_test_data()

    expected_features = model.input_shape[-1]

    if X_test.shape[-1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but input contains {X_test.shape[-1]}."
        )

    if len(feature_names) != expected_features:
        raise ValueError(
            "Feature-name count does not match model input."
        )

    X_scaled = scale_sequences(X_test, scaler)

    print("\nCalculating reconstruction errors...")

    errors = calculate_errors(model, X_scaled)

    results, threshold = save_results(
        errors,
        threshold_data,
    )

    anomaly_count = int(
        results["is_anomaly_candidate"].sum()
    )

    print("\n" + "=" * 65)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 65)

    print(f"Total sequences: {len(results)}")
    print(f"Anomaly threshold: {threshold:.6f}")
    print(f"Mean reconstruction error: {errors.mean():.6f}")
    print(f"Maximum reconstruction error: {errors.max():.6f}")
    print(f"Minimum reconstruction error: {errors.min():.6f}")
    print(f"Anomaly candidates: {anomaly_count}")

    print(f"\nResults saved to:\n{OUTPUT_FILE}")

    print("\nNote:")
    print(
        "These are anomaly candidates, not confirmed incidents. "
        "Evaluate the model using labeled abnormal traffic."
    )

    print("\nDetection completed successfully.")


if __name__ == "__main__":
    main()