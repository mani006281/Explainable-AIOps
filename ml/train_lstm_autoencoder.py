"""
Explainable AIOps
LSTM Autoencoder for baseline traffic anomaly detection.

Input:
    ecommerce-demo/data/sequence_windows_baseline.npz

Outputs:
    ml/artifacts/lstm_autoencoder.keras
    ml/artifacts/feature_scaler.joblib
    ml/artifacts/anomaly_threshold.json
    ml/artifacts/training_history.csv
    ml/artifacts/test_reconstruction_errors.csv
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"

INPUT_FILE = DATA_DIR / "sequence_windows_baseline.npz"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------
# 2. Load and validate prepared sequences
# ---------------------------------------------------------

print("=" * 72)
print("EXPLAINABLE AIOPS: LSTM AUTOENCODER TRAINING")
print("=" * 72)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Prepared sequence file was not found: {INPUT_FILE}\n"
        "Run scripts/prepare_baseline_sequences.py first."
    )

with np.load(INPUT_FILE, allow_pickle=False) as data:
    X_train_raw = data["X_train"].astype(np.float32)
    X_val_raw = data["X_val"].astype(np.float32)
    X_test_raw = data["X_test"].astype(np.float32)
    feature_names = data["feature_names"].astype(str)
    window_size = int(data["window_size"])

print(f"\nInput file: {INPUT_FILE}")
print(f"Window size: {window_size}")
print(f"Features: {len(feature_names)}")
print(f"Training sequences: {X_train_raw.shape}")
print(f"Validation sequences: {X_val_raw.shape}")
print(f"Test sequences: {X_test_raw.shape}")

if X_train_raw.ndim != 3:
    raise ValueError("Expected three-dimensional training sequences.")

if X_train_raw.shape[1] != window_size:
    raise ValueError("Training sequence length does not match window_size.")

if X_train_raw.shape[2] != len(feature_names):
    raise ValueError("Feature count does not match feature_names.")

if not all(
    np.isfinite(array).all()
    for array in (X_train_raw, X_val_raw, X_test_raw)
):
    raise ValueError("Input sequences contain NaN or infinite values.")

if min(len(X_train_raw), len(X_val_raw), len(X_test_raw)) == 0:
    raise ValueError("Training, validation, and test splits must be non-empty.")


# ---------------------------------------------------------
# 3. Scale features without fitting on validation/test data
# ---------------------------------------------------------

# Fit the scaler using training observations only.
# Overlapping windows repeat observations, so flattening the
# training sequences weights interior observations more heavily.
# This is acceptable for this initial prototype, but can be
# improved later by fitting on the original 210 train observations.

number_of_features = X_train_raw.shape[2]

scaler = StandardScaler()

scaler.fit(
    X_train_raw.reshape(-1, number_of_features)
)


def transform_sequences(sequences):
    """Scale each feature while preserving sequence dimensions."""
    original_shape = sequences.shape

    scaled = scaler.transform(
        sequences.reshape(-1, number_of_features)
    )

    return scaled.reshape(original_shape).astype(np.float32)


X_train = transform_sequences(X_train_raw)
X_val = transform_sequences(X_val_raw)
X_test = transform_sequences(X_test_raw)

print("\nFeature scaling completed.")
print("Scaler fitted using training sequences only.")


# ---------------------------------------------------------
# 4. Build the LSTM Autoencoder
# ---------------------------------------------------------

sequence_input = layers.Input(
    shape=(window_size, number_of_features),
    name="traffic_sequence",
)

# Encoder: compress the sequence into a compact representation.
encoded = layers.LSTM(
    32,
    activation="tanh",
    name="encoder_lstm",
)(sequence_input)

encoded = layers.Dropout(
    0.1,
    name="encoder_dropout",
)(encoded)

# Decoder: reconstruct the original sequence.
decoded = layers.RepeatVector(
    window_size,
    name="repeat_encoded_vector",
)(encoded)

decoded = layers.LSTM(
    32,
    activation="tanh",
    return_sequences=True,
    name="decoder_lstm",
)(decoded)

reconstructed = layers.TimeDistributed(
    layers.Dense(number_of_features),
    name="reconstructed_features",
)(decoded)

model = Model(
    inputs=sequence_input,
    outputs=reconstructed,
    name="lstm_autoencoder",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse",
)

model.summary()


# ---------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\nStarting model training...")

history = model.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=1,
)

history_df = pd.DataFrame(history.history)
history_path = ARTIFACT_DIR / "training_history.csv"
history_df.to_csv(history_path, index=False)

print(f"\nTraining history saved: {history_path}")


# ---------------------------------------------------------
# 6. Calculate reconstruction errors
# ---------------------------------------------------------

def reconstruction_errors(model, sequences):
    """
    Calculate one reconstruction error per sequence.

    Error = mean squared difference across all time steps
            and all features in a sequence.
    """
    reconstructed_sequences = model.predict(
        sequences,
        verbose=0,
    )

    errors = np.mean(
        np.square(sequences - reconstructed_sequences),
        axis=(1, 2),
    )

    return errors


train_errors = reconstruction_errors(model, X_train)
val_errors = reconstruction_errors(model, X_val)
test_errors = reconstruction_errors(model, X_test)


# ---------------------------------------------------------
# 7. Determine a threshold from validation errors
# ---------------------------------------------------------

# This percentile is an initial operating choice, not a
# statistically validated optimum. A small validation set
# makes the threshold uncertain.
THRESHOLD_PERCENTILE = 95

threshold = float(
    np.percentile(val_errors, THRESHOLD_PERCENTILE)
)

print("\nReconstruction-error summary:")
print(f"Training mean error: {train_errors.mean():.6f}")
print(f"Validation mean error: {val_errors.mean():.6f}")
print(f"Test mean error: {test_errors.mean():.6f}")
print(
    f"Threshold (validation {THRESHOLD_PERCENTILE}th percentile): "
    f"{threshold:.6f}"
)


# ---------------------------------------------------------
# 8. Flag test sequences
# ---------------------------------------------------------

test_predictions = test_errors > threshold

test_results = pd.DataFrame(
    {
        "test_sequence_index": np.arange(1, len(test_errors) + 1),
        "reconstruction_error": test_errors,
        "threshold": threshold,
        "predicted_anomaly": test_predictions,
    }
)

test_results_path = (
    ARTIFACT_DIR / "test_reconstruction_errors.csv"
)
test_results.to_csv(test_results_path, index=False)

print(f"\nTest error report saved: {test_results_path}")
print(f"Test sequences flagged: {int(test_predictions.sum())}")
print(f"Total test sequences: {len(test_predictions)}")


# ---------------------------------------------------------
# 9. Save model, scaler, and threshold configuration
# ---------------------------------------------------------

model_path = ARTIFACT_DIR / "lstm_autoencoder.keras"
scaler_path = ARTIFACT_DIR / "feature_scaler.joblib"
threshold_path = ARTIFACT_DIR / "anomaly_threshold.json"

model.save(model_path)
joblib.dump(scaler, scaler_path)

threshold_config = {
    "method": "validation_reconstruction_error_percentile",
    "percentile": THRESHOLD_PERCENTILE,
    "threshold": threshold,
    "window_size": window_size,
    "feature_names": feature_names.tolist(),
    "training_sequences": int(len(X_train)),
    "validation_sequences": int(len(X_val)),
    "test_sequences": int(len(X_test)),
    "seed": SEED,
    "warning": (
        "Prototype threshold. Evaluate on representative labeled "
        "normal and anomalous traffic before production use."
    ),
}

with open(threshold_path, "w", encoding="utf-8") as file:
    json.dump(threshold_config, file, indent=2)

print("\nSaved artifacts:")
print(f"  Model: {model_path}")
print(f"  Scaler: {scaler_path}")
print(f"  Threshold: {threshold_path}")
print(f"  Training history: {history_path}")
print(f"  Test errors: {test_results_path}")

print("\n" + "=" * 72)
print("LSTM AUTOENCODER TRAINING COMPLETED")
print("=" * 72)
print(
    "Important: test flags are anomaly candidates, not verified "
    "anomalies. Labeled anomaly data is needed to evaluate detection quality."
)