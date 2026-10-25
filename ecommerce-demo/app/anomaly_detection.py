"""
app/anomaly_detection.py
========================
Standalone LSTM Autoencoder anomaly detection module for the
Explainable AIOps e-commerce monitoring system.

Responsibilities
----------------
* Load the trained LSTM Autoencoder, feature scaler, and anomaly
  threshold from ml/artifacts_normal_extended_v1/ at import time.
* Accept one HTTP observation at a time via record_observation().
* Maintain a thread-safe rolling window of the last 10 observations.
* When the window is full, run LSTM inference and compute the mean
  squared error (MSE) reconstruction error.
* Compare the reconstruction error against the saved threshold and
  return an AnomalyResult with full diagnostic details.
* Catch all internal exceptions so that inference errors NEVER
  interrupt normal HTTP request processing.

Design constraints
------------------
* Do NOT retrain, overwrite, or modify any model artifact.
* Do NOT modify any existing file in the project.
* Feature order must exactly match the training-time feature order
  documented in ml/artifacts_normal_extended_v1/anomaly_threshold.json.
* Rolling window uses collections.deque with maxlen=10 and is
  protected by a threading.Lock for concurrent request safety.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
# Resolved relative to this file's location:
#   ecommerce-demo/app/anomaly_detection.py
#   → ../../ml/artifacts_normal_extended_v1/

_ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "artifacts_normal_extended_v1"
)

_MODEL_PATH     = _ARTIFACTS_DIR / "lstm_autoencoder.keras"
_SCALER_PATH    = _ARTIFACTS_DIR / "feature_scaler.joblib"
_THRESHOLD_PATH = _ARTIFACTS_DIR / "anomaly_threshold.json"


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# This order is fixed and must match the order used during training.
# Source of truth: anomaly_threshold.json → "feature_names"

FEATURE_NAMES: list[str] = [
    "response_time_ms",     # 0  raw HTTP response time in milliseconds
    "log_response_time",    # 1  math.log1p(response_time_ms)
    "status_200",           # 2  1 if HTTP 200, else 0
    "status_400",           # 3  1 if HTTP 400, else 0
    "status_404",           # 4  1 if HTTP 404, else 0
    "status_other",         # 5  1 if any other status code, else 0
    "endpoint_health",      # 6  1 if path == "/health"
    "endpoint_products",    # 7  1 if path == "/products/"
    "endpoint_latency",     # 8  1 if path == "/debug/test-latency"
    "endpoint_other",       # 9  1 if path does not match any above
    "is_connection_error",  # 10 1 if no status code (connection failed)
    "is_http_error",        # 11 1 if status_code >= 400
]

WINDOW_SIZE: int = 10
N_FEATURES: int = len(FEATURE_NAMES)  # must be 12


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """
    Returned by record_observation() after every call.

    Fields
    ------
    window_ready : bool
        True once 10 or more observations have been collected.
        False during the warm-up period (first 9 requests).

    is_anomaly : bool
        True when window_ready is True and the reconstruction error
        exceeds the saved threshold.  Always False during warm-up.

    reconstruction_error : Optional[float]
        Mean squared error between the model's input and its
        reconstruction.  None during warm-up.

    threshold : float
        The saved anomaly threshold loaded from anomaly_threshold.json.

    window_size : int
        Number of observations currently in the rolling window
        (0–10).

    feature_vector : list[dict]
        The 12 feature values for the most recent observation.
        Always present, even during warm-up, for diagnostic logging.

    per_feature_errors : Optional[list[dict]]
        Mean squared error per feature across all 10 timesteps.
        Useful for explainability: the feature with the highest error
        contributed most to the anomaly.  None during warm-up.

    error_message : Optional[str]
        Non-None only when an internal exception was caught.
        The HTTP request always succeeds regardless of this field.
    """

    window_ready: bool = False
    is_anomaly: bool = False
    reconstruction_error: Optional[float] = None
    threshold: float = 0.0
    window_size: int = 0
    feature_vector: list[dict] = field(default_factory=list)
    per_feature_errors: Optional[list[dict]] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Module-level state  (loaded once, shared across all requests)
# ---------------------------------------------------------------------------

# These are set during _load_artifacts() and never modified afterwards.
_model = None          # keras.Model
_scaler = None         # sklearn.preprocessing.MinMaxScaler
_threshold: float = 0.0
_artifacts_loaded: bool = False
_load_error: Optional[str] = None

# Rolling window — holds up to WINDOW_SIZE feature dicts.
# deque automatically discards the oldest entry when full.
_window: deque = deque(maxlen=WINDOW_SIZE)
_window_lock = threading.Lock()

# Guard against parallel import-time loading.
_load_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_artifacts() -> None:
    """
    Load model, scaler, and threshold from disk.

    Called automatically the first time record_observation() is used.
    Subsequent calls are no-ops (idempotent guard with _artifacts_loaded).

    This function intentionally imports TensorFlow lazily so that the
    module can be imported without TensorFlow installed — the import
    error only surfaces when record_observation() is actually called.
    """
    global _model, _scaler, _threshold, _artifacts_loaded, _load_error

    with _load_lock:
        # Double-checked locking: another thread may have loaded between
        # the outer check and acquiring the lock.
        if _artifacts_loaded:
            return

        try:
            # --- Verify all artifact files exist before loading -------
            for path, label in [
                (_MODEL_PATH,     "LSTM model"),
                (_SCALER_PATH,    "Feature scaler"),
                (_THRESHOLD_PATH, "Anomaly threshold"),
            ]:
                if not path.exists():
                    raise FileNotFoundError(
                        f"{label} not found at: {path}"
                    )

            # --- Load anomaly threshold (plain JSON, no ML libs) -------
            with _THRESHOLD_PATH.open("r", encoding="utf-8") as fh:
                threshold_data = json.load(fh)

            _threshold = float(threshold_data["threshold"])
            saved_window = int(threshold_data["window_size"])
            saved_features = threshold_data["feature_names"]

            # Sanity-check: confirm saved config matches our constants.
            if saved_window != WINDOW_SIZE:
                raise ValueError(
                    f"Window size mismatch: expected {WINDOW_SIZE}, "
                    f"threshold.json has {saved_window}"
                )
            if saved_features != FEATURE_NAMES:
                raise ValueError(
                    f"Feature name mismatch.\n"
                    f"  Expected : {FEATURE_NAMES}\n"
                    f"  Saved    : {saved_features}"
                )

            # --- Load feature scaler (joblib, requires scikit-learn) ---
            _scaler = joblib.load(_SCALER_PATH)

            # --- Load LSTM model (TensorFlow / Keras) ------------------
            # Import here so TensorFlow is not imported at module level.
            import tensorflow as tf  # noqa: F401 (needed for keras)
            from tensorflow import keras  # noqa: F401

            _model = keras.models.load_model(_MODEL_PATH)

            _artifacts_loaded = True
            _load_error = None

        except Exception as exc:
            _load_error = f"Artifact loading failed: {type(exc).__name__}: {exc}"
            # _artifacts_loaded stays False so we retry next time (or
            # you can choose to set it True and cache the error — here
            # we retry to handle transient I/O issues on first request).


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _extract_features(
    path: str,
    status_code: Optional[int],
    response_time_ms: float,
) -> dict:
    """
    Convert one HTTP observation into the 12-element feature dict.

    This replicates the logic from:
        ecommerce-demo/scripts/engineer_features.py

    Parameters
    ----------
    path : str
        URL path of the request, e.g. "/health" or "/products/".
        Should NOT include the query string.

    status_code : Optional[int]
        HTTP response status code, e.g. 200, 404.
        Pass None (or omit) for connection errors where no response
        was received.

    response_time_ms : float
        Measured round-trip time in milliseconds (≥ 0.0).

    Returns
    -------
    dict
        Keys are exactly FEATURE_NAMES in the correct order.
    """
    # Clamp negative latency to zero and round to 2 decimal places.
    resp_ms = round(max(0.0, float(response_time_ms)), 2)

    # Feature 1: log1p transformation of response time rounded to 4 decimal places.
    # log1p(x) = log(1 + x), matching original training feature engineering.
    log_resp = round(math.log1p(resp_ms), 4)

    # ---- Status code one-hot encoding ---------------------------------
    # is_connection_error is 1 when status_code is None, meaning the
    # server never responded (network timeout, refused connection, etc.)
    if status_code is None:
        is_connection_error = 1
        is_http_error       = 0  # not an HTTP error — no response at all
        status_200          = 0
        status_400          = 0
        status_404          = 0
        status_other        = 1  # treated as "other" for the status group
    else:
        is_connection_error = 0
        sc = int(status_code)
        if sc == 200:
            status_200    = 1
            status_400    = 0
            status_404    = 0
            status_other  = 0
            is_http_error = 0
        elif sc == 400:
            status_200    = 0
            status_400    = 1
            status_404    = 0
            status_other  = 0
            is_http_error = 1
        elif sc == 404:
            status_200    = 0
            status_400    = 0
            status_404    = 1
            status_other  = 0
            is_http_error = 1
        else:
            status_200    = 0
            status_400    = 0
            status_404    = 0
            status_other  = 1
            is_http_error = 1 if sc >= 400 else 0

    # ---- Endpoint one-hot encoding ------------------------------------
    # Strip the query string before matching (path already should be
    # clean, but guard against accidental inclusion).
    clean_path = path.split("?")[0].strip()

    if clean_path == "/health":
        endpoint_health   = 1
        endpoint_products = 0
        endpoint_latency  = 0
        endpoint_other    = 0
    elif clean_path == "/products/":
        endpoint_health   = 0
        endpoint_products = 1
        endpoint_latency  = 0
        endpoint_other    = 0
    elif clean_path == "/debug/test-latency":
        endpoint_health   = 0
        endpoint_products = 0
        endpoint_latency  = 1
        endpoint_other    = 0
    else:
        endpoint_health   = 0
        endpoint_products = 0
        endpoint_latency  = 0
        endpoint_other    = 1

    # ---- Assemble in the exact training order -------------------------
    return {
        "response_time_ms":    resp_ms,
        "log_response_time":   log_resp,
        "status_200":          status_200,
        "status_400":          status_400,
        "status_404":          status_404,
        "status_other":        status_other,
        "endpoint_health":     endpoint_health,
        "endpoint_products":   endpoint_products,
        "endpoint_latency":    endpoint_latency,
        "endpoint_other":      endpoint_other,
        "is_connection_error": is_connection_error,
        "is_http_error":       is_http_error,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(window_snapshot: list[dict]) -> tuple[float, list[dict]]:
    """
    Scale, reshape, run LSTM inference, and compute reconstruction MSE.

    Parameters
    ----------
    window_snapshot : list[dict]
        Exactly WINDOW_SIZE feature dicts in chronological order.
        Each dict has keys matching FEATURE_NAMES.

    Returns
    -------
    reconstruction_error : float
        Mean squared error (MSE) averaged over all timesteps and
        features.  Higher values indicate more anomalous behaviour.

    per_feature_errors : list[dict]
        One entry per feature.  Each entry contains:
            "feature"  : feature name (str)
            "mse"      : mean squared error for that feature across
                         all 10 timesteps (float)
        Sorted descending by "mse" — the first entry is the feature
        that contributed most to the anomaly score.
    """
    # --- Build raw numpy array (10, 12) --------------------------------
    raw = np.array(
        [[obs[feat] for feat in FEATURE_NAMES] for obs in window_snapshot],
        dtype=np.float32,
    )  # shape: (WINDOW_SIZE, N_FEATURES) = (10, 12)

    # --- Scale using the saved MinMaxScaler ----------------------------
    # The scaler was fitted on (n_observations, 12).
    # We pass (10, 12), which is perfectly valid.
    scaled = _scaler.transform(raw)  # shape: (10, 12)

    # --- Reshape for LSTM: (batch, timesteps, features) ---------------
    model_input = scaled.reshape(1, WINDOW_SIZE, N_FEATURES)
    # shape: (1, 10, 12)

    # --- Run forward pass through the autoencoder ---------------------
    reconstruction = _model.predict(model_input, verbose=0)
    # reconstruction shape: (1, 10, 12)

    # --- Squeeze batch dimension --------------------------------------
    reconstruction = reconstruction[0]  # shape: (10, 12)

    # --- Compute overall MSE ------------------------------------------
    # MSE averaged across all 10 × 12 = 120 values.
    diff_sq = (scaled - reconstruction) ** 2
    overall_mse = float(np.mean(diff_sq))

    # --- Per-feature MSE ----------------------------------------------
    # Average squared error per feature across all 10 timesteps.
    per_feature_mse = np.mean(diff_sq, axis=0)  # shape: (12,)

    per_feature_errors = sorted(
        [
            {"feature": feat, "mse": float(per_feature_mse[i])}
            for i, feat in enumerate(FEATURE_NAMES)
        ],
        key=lambda x: x["mse"],
        reverse=True,  # highest MSE first
    )

    return overall_mse, per_feature_errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_observation(
    path: str,
    status_code: Optional[int],
    response_time_ms: float,
) -> AnomalyResult:
    """
    Record one HTTP observation and (when the window is full) run
    anomaly detection.

    This is the only function that the FastAPI middleware needs to call.
    It is safe to call from any thread — the rolling window is
    protected by a threading.Lock.

    Parameters
    ----------
    path : str
        URL path of the request (e.g. "/health", "/products/").
        Do NOT include the query string — strip it beforehand.

    status_code : Optional[int]
        HTTP response status code.  Pass None for connection errors.

    response_time_ms : float
        Measured round-trip response time in milliseconds.

    Returns
    -------
    AnomalyResult
        Always returned, even if an internal error occurs.
        Check result.error_message for non-None to detect failures.

    Examples
    --------
    >>> result = record_observation("/health", 200, 12.5)
    >>> result.window_ready
    False   # during warm-up (fewer than 10 observations)

    >>> # After 10 or more calls:
    >>> result.window_ready
    True
    >>> result.reconstruction_error
    0.0043...
    >>> result.is_anomaly
    False
    """
    result = AnomalyResult(threshold=_threshold)

    try:
        # --- Ensure artifacts are loaded (idempotent) -----------------
        if not _artifacts_loaded:
            _load_artifacts()

        if not _artifacts_loaded:
            # Loading failed — return a safe result with the error.
            result.error_message = _load_error or "Artifacts not loaded"
            return result

        result.threshold = _threshold

        # --- Extract 12 features from this observation ----------------
        features = _extract_features(path, status_code, response_time_ms)
        result.feature_vector = [
            {"feature": k, "value": v} for k, v in features.items()
        ]

        # --- Append to the rolling window (thread-safe) ---------------
        with _window_lock:
            _window.append(features)
            current_size = len(_window)
            # Take a snapshot so we don't hold the lock during inference.
            window_snapshot = list(_window) if current_size == WINDOW_SIZE else []

        result.window_size = current_size

        # --- Run inference only when window is full -------------------
        if current_size < WINDOW_SIZE:
            # Warm-up period: not enough observations yet.
            result.window_ready = False
            result.is_anomaly   = False
            return result

        result.window_ready = True

        # --- Run inference --------------------------------------------
        mse, per_feature_errors = _run_inference(window_snapshot)

        result.reconstruction_error = mse
        result.per_feature_errors   = per_feature_errors
        result.is_anomaly           = mse > _threshold

    except Exception as exc:
        # Catch-all: inference errors must never propagate to the
        # HTTP response layer.
        result.error_message = f"{type(exc).__name__}: {exc}"

    return result


def reset_window() -> None:
    """
    Clear the rolling window.

    Useful for testing or for resetting state when the server
    restarts / receives a signal to flush its internal buffer.
    This function is NOT called by the middleware automatically.
    """
    with _window_lock:
        _window.clear()


def get_detector_status() -> dict:
    """
    Return a status dictionary for diagnostics and health endpoints.

    Returns
    -------
    dict
        {
            "artifacts_loaded"  : bool,
            "load_error"        : str or None,
            "threshold"         : float,
            "window_size"       : int,
            "window_capacity"   : int,
            "window_ready"      : bool,
            "model_path"        : str,
            "scaler_path"       : str,
            "threshold_path"    : str,
        }
    """
    with _window_lock:
        current_size = len(_window)

    return {
        "artifacts_loaded": _artifacts_loaded,
        "load_error":       _load_error,
        "threshold":        _threshold,
        "window_size":      current_size,
        "window_capacity":  WINDOW_SIZE,
        "window_ready":     current_size >= WINDOW_SIZE,
        "model_path":       str(_MODEL_PATH),
        "scaler_path":      str(_SCALER_PATH),
        "threshold_path":   str(_THRESHOLD_PATH),
    }
