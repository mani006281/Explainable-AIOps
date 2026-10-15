
"""
Train an experimental LSTM Autoencoder using the extended normal dataset.

Safety:
- Does not overwrite the original model.
- Reads the new normal-extended sequence dataset.
- Fits StandardScaler on unique training observations only.
- Saves all artifacts in a separate experiment directory.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
import joblib



# 1. PATHS AND CONFIGURATION

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"

INPUT_FILE = DATA_DIR / "sequence_windows_normal_extended.npz"
FEATURE_FILE = DATA_DIR / "traffic_normal_extended_features.csv"

ARTIFACT_DIR = ML_DIR / "artifacts_normal_extended_v1"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WINDOW_SIZE_EXPECTED = 10

# Chronological observation split used during sequence preparation.
TRAIN_RATIO = 0.70


# 2. REPRODUCIBILITY


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



# 3. SAFETY CHECKS


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Sequence dataset not found: {INPUT_FILE}"
    )

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {FEATURE_FILE}"
    )

# Never overwrite existing experiment artifacts.
if ARTIFACT_DIR.exists() and any(ARTIFACT_DIR.iterdir()):
    raise FileExistsError(
        f"Experiment directory already contains files:\n"
        f"{ARTIFACT_DIR}\n"
        "This script will not overwrite existing experiment artifacts. "
        "Use a new versioned directory for another experiment."
    )

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("EXPERIMENTAL LSTM AUTOENCODER TRAINING")
print("=" * 65)
print(f"Input sequences: {INPUT_FILE}")
print(f"Raw features:    {FEATURE_FILE}")
print(f"Output artifacts:{ARTIFACT_DIR}")


# 4. LOAD SEQUENCE DATA

with np.load(INPUT_FILE, allow_pickle=False) as data:
    X_train_raw = data["X_train"].astype(np.float32)
    X_val_raw = data["X_val"].astype(np.float32)
    X_test_raw = data["X_test"].astype(np.float32)

    feature_names = data["feature_names"].astype(str).tolist()
    window_size = int(data["window_size"])

print("\nLoaded sequence data:")
print("X_train:", X_train_raw.shape)
print("X_val:  ", X_val_raw.shape)
print("X_test: ", X_test_raw.shape)
print("Features:", len(feature_names))
print("Window size:", window_size)

if window_size != WINDOW_SIZE_EXPECTED:
    raise ValueError(
        f"Expected window size {WINDOW_SIZE_EXPECTED}, "
        f"but found {window_size}."
    )

if not (
    X_train_raw.ndim == 3
    and X_val_raw.ndim == 3
    and X_test_raw.ndim == 3
):
    raise ValueError("All sequence arrays must be three-dimensional.")

if not (
    X_train_raw.shape[2]
    == X_val_raw.shape[2]
    == X_test_raw.shape[2]
    == len(feature_names)
):
    raise ValueError("Feature count does not match the sequence arrays.")

if not (
    X_train_raw.shape[1]
    == X_val_raw.shape[1]
    == X_test_raw.shape[1]
    == window_size
):
    raise ValueError("Sequence window dimensions do not match.")

if min(len(X_train_raw), len(X_val_raw), len(X_test_raw)) == 0:
    raise ValueError("Training, validation, and test data must be nonempty.")

for name, array in [
    ("X_train", X_train_raw),
    ("X_val", X_val_raw),
    ("X_test", X_test_raw),
]:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


# 5. LOAD RAW OBSERVATIONS FOR SCALER FITTING
#
# Fit the scaler on unique chronological training observations.
# Do not fit it on flattened overlapping windows because that
# repeats many observations and gives them unequal weights.
#
# The sequence preparation split the 600 observations chronologically:
# 420 training, 90 validation, 90 testing.

features_df = pd.read_csv(FEATURE_FILE)

required_columns = ["timestamp_utc"] + feature_names
missing_columns = [
    column for column in required_columns
    if column not in features_df.columns
]

if missing_columns:
    raise ValueError(
        f"Feature CSV is missing required columns: {missing_columns}"
    )

features_df["timestamp_utc"] = pd.to_datetime(
    features_df["timestamp_utc"],
    utc=True,
    errors="raise",
)

features_df = features_df.sort_values(
    "timestamp_utc",
    kind="stable",
).reset_index(drop=True)

if features_df["timestamp_utc"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps found in the feature CSV. "
        "Resolve them before training."
    )

raw_features = features_df[feature_names].apply(
    pd.to_numeric,
    errors="raise",
).to_numpy(dtype=np.float32)

if not np.isfinite(raw_features).all():
    raise ValueError("Raw feature observations contain NaN or infinity.")

observation_count = len(raw_features)

if observation_count != 600:
    raise ValueError(
        f"Expected 600 raw observations, found {observation_count}. "
        "Check the source dataset and split assumptions."
    )

train_observation_count = int(observation_count * TRAIN_RATIO)

if train_observation_count != 420:
    raise ValueError(
        f"Expected 420 training observations, found "
        f"{train_observation_count}."
    )

raw_train_observations = raw_features[:train_observation_count]

print("\nRaw chronological observations:")
print("All observations:", observation_count)
print("Training observations used for scaler:", len(raw_train_observations))
print("Validation observations excluded from scaler: 90")
print("Test observations excluded from scaler: 90")


# 6. FIT SCALER ON UNIQUE TRAINING OBSERVATIONS

scaler = StandardScaler()
scaler.fit(raw_train_observations)

def scale_sequences(sequences, fitted_scaler):
    original_shape = sequences.shape

    flattened = sequences.reshape(-1, original_shape[-1])
    transformed = fitted_scaler.transform(flattened)

    return transformed.reshape(original_shape).astype(np.float32)


X_train = scale_sequences(X_train_raw, scaler)
X_val = scale_sequences(X_val_raw, scaler)
X_test = scale_sequences(X_test_raw, scaler)

print("\nScaled sequence shapes:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# 7. BUILD LSTM AUTOENCODER

n_features = X_train.shape[2]

inputs = Input(shape=(window_size, n_features))

encoded = LSTM(
    32,
    activation="tanh",
    name="encoder_lstm",
)(inputs)

encoded = Dropout(0.1, name="encoder_dropout")(encoded)

repeated = RepeatVector(window_size)(encoded)

decoded = LSTM(
    32,
    activation="tanh",
    return_sequences=True,
    name="decoder_lstm",
)(repeated)

outputs = TimeDistributed(
    Dense(n_features),
    name="reconstruction",
)(decoded)

model = Model(inputs, outputs, name="normal_extended_lstm_autoencoder")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse",
)

model.summary()


# 8. TRAIN

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
history_df.insert(
    0,
    "epoch",
    np.arange(1, len(history_df) + 1),
)

history_path = ARTIFACT_DIR / "training_history.csv"
history_df.to_csv(history_path, index=False)


# 9. CALCULATE RECONSTRUCTION ERRORS

def reconstruction_errors(input_sequences, trained_model):
    reconstructed = trained_model.predict(
        input_sequences,
        verbose=0,
    )

    errors = np.mean(
        np.square(input_sequences - reconstructed),
        axis=(1, 2),
    )

    return errors, reconstructed


train_errors, _ = reconstruction_errors(X_train, model)
val_errors, _ = reconstruction_errors(X_val, model)
test_errors, test_reconstructed = reconstruction_errors(X_test, model)

# Initial threshold: 95th percentile of validation-normal errors.
# This is a prototype threshold, not a guaranteed production threshold.
threshold = float(np.percentile(val_errors, 95))

test_predictions = test_errors > threshold

test_results = pd.DataFrame({
    "window_index": np.arange(len(test_errors)),
    "reconstruction_error": test_errors,
    "threshold": threshold,
    "predicted_anomaly": test_predictions,
})

test_results.to_csv(
    ARTIFACT_DIR / "test_reconstruction_errors.csv",
    index=False,
)

validation_results = pd.DataFrame({
    "window_index": np.arange(len(val_errors)),
    "reconstruction_error": val_errors,
    "threshold": threshold,
    "predicted_anomaly": val_errors > threshold,
})

validation_results.to_csv(
    ARTIFACT_DIR / "validation_reconstruction_errors.csv",
    index=False,
)


# 10. SAVE MODEL AND SCALER

model.save(ARTIFACT_DIR / "lstm_autoencoder.keras")
joblib.dump(scaler, ARTIFACT_DIR / "feature_scaler.joblib")

threshold_config = {
    "experiment": "normal_extended_v1",
    "method": "validation_reconstruction_error_percentile",
    "percentile": 95,
    "threshold": threshold,
    "window_size": window_size,
    "feature_names": feature_names,
    "training_sequences": int(len(X_train)),
    "validation_sequences": int(len(X_val)),
    "test_sequences": int(len(X_test)),
    "raw_training_observations_for_scaler": int(len(raw_train_observations)),
    "seed": SEED,
    "scaler_fit_method": "unique_chronological_training_observations",
    "warning": (
        "Prototype threshold. Evaluate on independent labeled normal "
        "and anomalous traffic before drawing performance conclusions."
    ),
}

with open(
    ARTIFACT_DIR / "anomaly_threshold.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(threshold_config, file, indent=2)


# 11. SAVE EXPERIMENT SUMMARY

summary = {
    "experiment": "normal_extended_v1",
    "raw_observations": int(observation_count),
    "raw_training_observations": int(len(raw_train_observations)),
    "training_sequences": int(len(X_train)),
    "validation_sequences": int(len(X_val)),
    "test_sequences": int(len(X_test)),
    "features": int(n_features),
    "window_size": int(window_size),
    "epochs_completed": int(len(history.history["loss"])),
    "best_validation_loss": float(min(history.history["val_loss"])),
    "final_training_loss": float(history.history["loss"][-1]),
    "final_validation_loss": float(history.history["val_loss"][-1]),
    "threshold": threshold,
    "validation_windows_flagged": int(np.sum(val_errors > threshold)),
    "validation_windows_total": int(len(val_errors)),
    "test_windows_flagged": int(np.sum(test_predictions)),
    "test_windows_total": int(len(test_errors)),
    "mean_test_reconstruction_error": float(np.mean(test_errors)),
    "median_test_reconstruction_error": float(np.median(test_errors)),
    "max_test_reconstruction_error": float(np.max(test_errors)),
    "warning": (
        "Test windows are normal-only, but the threshold was selected "
        "using validation errors. Test results estimate normal false "
        "alarms on this dataset only; they do not measure anomaly recall."
    ),
}

with open(
    ARTIFACT_DIR / "experiment_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2)

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED")
print("=" * 65)
print(f"Artifacts saved to: {ARTIFACT_DIR}")
print(f"Epochs completed: {summary['epochs_completed']}")
print(f"Best validation loss: {summary['best_validation_loss']:.6f}")
print(f"Threshold: {threshold:.6f}")
print(
    "Validation windows flagged: "
    f"{summary['validation_windows_flagged']}/"
    f"{summary['validation_windows_total']}"
)
print(
    "Normal test windows flagged: "
    f"{summary['test_windows_flagged']}/"
    f"{summary['test_windows_total']}"
)
print(f"Summary: {ARTIFACT_DIR / 'experiment_summary.json'}")
print("\nOriginal model artifacts were not modified.")
