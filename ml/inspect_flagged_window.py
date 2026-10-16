
"""
Explainable AIOps
Inspect actual versus reconstructed values for one flagged window.

Uses the existing trained model, scaler, and threshold.
Does not retrain or modify existing artifacts.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------
# 1. Paths and target window
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DEMO_DIR = PROJECT_DIR / "ecommerce-demo"

ARTIFACT_DIR = ML_DIR / "artifacts_normal_extended_v1"
DATA_DIR = DEMO_DIR / "data"

MODEL_FILE = ARTIFACT_DIR / "lstm_autoencoder.keras"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.joblib"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"
TEST_FILE = DATA_DIR / "sequence_windows_normal_extended_test.csv"

OUTPUT_DIR = ML_DIR / "flagged_window_diagnostic_v1"
WINDOW_ID = "test_0036"


# ---------------------------------------------------------
# 2. Validate files and output directory
# ---------------------------------------------------------

for path in [
    MODEL_FILE,
    SCALER_FILE,
    THRESHOLD_FILE,
    TEST_FILE,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Inspect the existing results before creating new output."
    )


# ---------------------------------------------------------
# 3. Load existing configuration and model
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
    raise ValueError("Feature count does not match the saved scaler.")

if model.input_shape[1] != window_size:
    raise ValueError("Window size does not match the saved model.")

if model.input_shape[2] != len(feature_names):
    raise ValueError("Feature count does not match the saved model.")


# ---------------------------------------------------------
# 4. Load and validate the selected window
# ---------------------------------------------------------

df = pd.read_csv(TEST_FILE)

if "window_id" not in df.columns:
    raise ValueError("Input file has no window_id column.")

matches = df[df["window_id"].astype(str) == WINDOW_ID]

if len(matches) != 1:
    raise ValueError(
        f"Expected exactly one row for {WINDOW_ID}; "
        f"found {len(matches)}."
    )

expected_columns = ["window_id"] + [
    f"t{step:02d}_{feature}"
    for step in range(1, window_size + 1)
    for feature in feature_names
]

if df.columns.tolist() != expected_columns:
    raise ValueError("Input feature columns or their order are unexpected.")

raw_flat = matches[expected_columns[1:]].apply(
    pd.to_numeric,
    errors="raise",
).to_numpy(dtype=np.float64)

if not np.isfinite(raw_flat).all():
    raise ValueError("Input window contains invalid numerical values.")

actual = raw_flat.reshape(
    1,
    window_size,
    len(feature_names),
)


# ---------------------------------------------------------
# 5. Scale using the existing scaler and reconstruct
# ---------------------------------------------------------

scaled_actual = scaler.transform(
    actual.reshape(-1, len(feature_names))
).reshape(actual.shape).astype(np.float32)

scaled_reconstructed = model.predict(
    scaled_actual,
    verbose=0,
)

if scaled_reconstructed.shape != scaled_actual.shape:
    raise ValueError("Unexpected reconstruction shape.")

if not np.isfinite(scaled_reconstructed).all():
    raise ValueError("Reconstructed values contain invalid numbers.")

squared_errors = np.square(
    scaled_actual - scaled_reconstructed
)

window_error = float(squared_errors.mean())
predicted_anomaly = window_error > threshold
exceedance = window_error - threshold


# ---------------------------------------------------------
# 6. Build actual-versus-reconstructed comparison
# ---------------------------------------------------------

rows = []

for step in range(window_size):
    for feature_index, feature in enumerate(feature_names):
        actual_raw = float(actual[0, step, feature_index])
        actual_scaled = float(scaled_actual[0, step, feature_index])
        reconstructed_scaled = float(
            scaled_reconstructed[0, step, feature_index]
        )

        squared_error = float(
            squared_errors[0, step, feature_index]
        )

        # Convert reconstructed values back to the original
        # feature scale for easier interpretation.
        reconstructed_raw = float(
            scaler.inverse_transform(
                scaled_reconstructed[0, step, :].reshape(1, -1)
            )[0, feature_index]
        )

        rows.append({
            "window_id": WINDOW_ID,
            "time_step": step + 1,
            "feature": feature,
            "actual_value": actual_raw,
            "reconstructed_value": reconstructed_raw,
            "actual_scaled": actual_scaled,
            "reconstructed_scaled": reconstructed_scaled,
            "absolute_scaled_error": abs(
                actual_scaled - reconstructed_scaled
            ),
            "squared_scaled_error": squared_error,
        })

comparison = pd.DataFrame(rows)


# ---------------------------------------------------------
# 7. Summarize errors by feature
# ---------------------------------------------------------

feature_summary = (
    comparison.groupby("feature", as_index=False)
    .agg(
        mean_squared_scaled_error=("squared_scaled_error", "mean"),
        mean_absolute_scaled_error=("absolute_scaled_error", "mean"),
    )
    .sort_values(
        "mean_squared_scaled_error",
        ascending=False,
    )
    .reset_index(drop=True)
)

total_feature_error = feature_summary[
    "mean_squared_scaled_error"
].sum()

if total_feature_error > 0:
    feature_summary["contribution_percent"] = (
        feature_summary["mean_squared_scaled_error"]
        / total_feature_error
        * 100
    )
else:
    feature_summary["contribution_percent"] = 0.0


# ---------------------------------------------------------
# 8. Save results to a new directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

comparison.to_csv(
    OUTPUT_DIR / "actual_vs_reconstructed_test_0036.csv",
    index=False,
)

feature_summary.to_csv(
    OUTPUT_DIR / "feature_error_summary_test_0036.csv",
    index=False,
)

summary = {
    "window_id": WINDOW_ID,
    "experiment": config.get("experiment"),
    "threshold": threshold,
    "reconstruction_error": window_error,
    "threshold_exceedance": exceedance,
    "exceedance_percent_of_threshold": (
        exceedance / threshold * 100 if threshold > 0 else None
    ),
    "predicted_anomaly": bool(predicted_anomaly),
    "window_size": window_size,
    "feature_count": len(feature_names),
    "model_retrained": False,
    "existing_artifacts_modified": False,
    "interpretation_warning": (
        "Feature errors describe reconstruction discrepancies, "
        "not causal explanations. Reconstructed raw values are inverse-"
        "scaled model outputs and may not be valid categorical indicators."
    ),
}

with open(
    OUTPUT_DIR / "diagnostic_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2)

print("=" * 70)
print("FLAGGED WINDOW DIAGNOSTIC COMPLETED")
print("=" * 70)
print(f"Window ID: {WINDOW_ID}")
print(f"Reconstruction error: {window_error:.9f}")
print(f"Threshold: {threshold:.9f}")
print(f"Threshold exceedance: {exceedance:.9f}")
print(f"Predicted anomaly: {predicted_anomaly}")

print("\nTOP FEATURE ERRORS")
print(feature_summary.round(6).head(6).to_string(index=False))

print("\nRESULT FILES")
print(f"Output directory: {OUTPUT_DIR}")
print(" - actual_vs_reconstructed_test_0036.csv")
print(" - feature_error_summary_test_0036.csv")
print(" - diagnostic_summary.json")

print("\nExisting model, scaler, and threshold were not modified.")

