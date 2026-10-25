"""
ecommerce-demo/app/incident_store.py
====================================
In-memory incident history and explainability store for the
Explainable AIOps e-commerce monitoring system.

Responsibilities
----------------
* Store a bounded history of the most recent 100 detected anomaly incidents.
* Provide thread-safe append and retrieval operations.
* Extract and rank the top three contributing features based on
  per-feature reconstruction error (MSE).
* Explicitly label feature contributions as reconstruction-error
  contributions rather than proven causal root causes.
* Ensure NO request bodies, credentials, headers, PII, or database
  connection strings are stored.
* Return incidents in newest-first chronological order.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional


MAX_INCIDENTS: int = 100

# Thread-safe bounded deque (automatically evicts oldest when full)
_incident_history: deque[dict] = deque(maxlen=MAX_INCIDENTS)
_incident_lock = threading.Lock()


def format_top_contributing_features(
    per_feature_errors: Optional[list[dict]],
    top_n: int = 3,
) -> list[dict]:
    """
    Extract and format the top N contributing features from per-feature errors.

    Parameters
    ----------
    per_feature_errors : list[dict] or None
        List of dicts containing 'feature' and 'mse' (or 'reconstruction_error').
    top_n : int
        Number of top features to return (default: 3).

    Returns
    -------
    list[dict]
        Top N features ordered descending by reconstruction error,
        explicitly labeled as reconstruction-error contributions.
    """
    if not per_feature_errors:
        return []

    # Ensure strictly sorted descending by error
    sorted_features = sorted(
        per_feature_errors,
        key=lambda item: float(item.get("mse", item.get("reconstruction_error", 0.0))),
        reverse=True,
    )

    top_features = []
    for rank, item in enumerate(sorted_features[:top_n], start=1):
        feature_name = str(item.get("feature", "unknown"))
        error_val = float(item.get("mse", item.get("reconstruction_error", 0.0)))

        top_features.append({
            "rank": rank,
            "feature": feature_name,
            "reconstruction_error": round(error_val, 6),
            "metric_type": "reconstruction_error_contribution",
            "note": (
                f"Feature '{feature_name}' had reconstruction error {round(error_val, 6)}. "
                "This indicates feature-level reconstruction error contribution, "
                "not a proven causal root cause."
            ),
        })

    return top_features


def create_incident_record(
    path: str,
    status_code: Optional[int],
    response_time_ms: float,
    reconstruction_error: float,
    threshold: float,
    per_feature_errors: Optional[list[dict]] = None,
) -> dict:
    """
    Build a sanitized, explainable incident record.

    Excludes request bodies, headers, credentials, and internal paths.
    """
    clean_path = str(path).split("?")[0].strip()
    sc = int(status_code) if status_code is not None else 0
    resp_ms = round(max(0.0, float(response_time_ms)), 2)
    recon_err = round(float(reconstruction_error), 6)
    thresh = round(float(threshold), 6)

    top_three = format_top_contributing_features(per_feature_errors, top_n=3)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "path": clean_path,
        "status_code": sc,
        "response_time_ms": resp_ms,
        "reconstruction_error": recon_err,
        "threshold": thresh,
        "top_contributing_features": top_three,
        "explanation": (
            "Top contributing features ordered by per-feature reconstruction error. "
            "These indicate which observation dimensions deviated most from the "
            "learned normal baseline. They are statistical reconstruction-error "
            "contributions, not proven causal root causes."
        ),
    }


def record_incident(
    path: str,
    status_code: Optional[int],
    response_time_ms: float,
    reconstruction_error: float,
    threshold: float,
    per_feature_errors: Optional[list[dict]] = None,
) -> dict:
    """
    Create and record an anomaly incident into bounded in-memory history.

    Thread-safe and bounded to MAX_INCIDENTS (100).
    """
    incident = create_incident_record(
        path=path,
        status_code=status_code,
        response_time_ms=response_time_ms,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
        per_feature_errors=per_feature_errors,
    )

    with _incident_lock:
        _incident_history.append(incident)

    return incident


def get_incident_history(limit: int = MAX_INCIDENTS) -> list[dict]:
    """
    Retrieve incident history, newest first.

    Parameters
    ----------
    limit : int
        Maximum number of records to return (up to 100).

    Returns
    -------
    list[dict]
        Incidents ordered from newest to oldest.
    """
    clamped_limit = max(0, min(limit, MAX_INCIDENTS))
    with _incident_lock:
        # reversed(_incident_history) yields newest first
        records = list(reversed(_incident_history))
        return records[:clamped_limit]


def get_incident_count() -> int:
    """Return the current number of stored incidents (0 to 100)."""
    with _incident_lock:
        return len(_incident_history)


def clear_incident_history() -> None:
    """
    Clear all stored incidents.
    Primarily used for test isolation and application reset.
    """
    with _incident_lock:
        _incident_history.clear()
