"""
Phase 3.4 — prepare_sequences.py
=================================
Reads the feature matrix (traffic_features_expanded.csv or fallback to
traffic_features.csv), assembles chronological sliding-window sequences,
performs a time-based train / validation / test split, and writes
deterministic output CSV files.

Design decisions
----------------
- Window length is configurable (default: 10 observations).
- Windows are formed from a SINGLE chronologically sorted batch;
  overlapping windows that cross a split boundary are dropped, not
  carried over, to prevent leakage.
- Timestamps and row-level metadata are kept in a companion metadata
  file, NOT inside the feature tensor.
- Scenario labels and expected_delay_ms are NEVER passed as numerical
  model inputs.
- HTTP 400 and 404 outcomes are preserved in metadata for post-detection
  evaluation; they are not automatically labelled as anomalies here.

Split fractions (time-based, chronological)
-------------------------------------------
  train      : first 70 % of windows
  validation : next  15 % of windows
  test       : last  15 % of windows
"""

import csv
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_LENGTH = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC = 0.15

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Prefer expanded features; fall back to Phase-3.3 features
EXPANDED_FEATURES = DATA_DIR / "traffic_features_expanded.csv"
FALLBACK_FEATURES = DATA_DIR / "traffic_features.csv"

EXPANDED_METADATA = DATA_DIR / "traffic_feature_metadata_expanded.csv"
FALLBACK_METADATA = DATA_DIR / "traffic_feature_metadata.csv"

OUTPUT_WINDOWS = DATA_DIR / "sequence_windows.csv"
OUTPUT_WINDOW_META = DATA_DIR / "sequence_window_metadata.csv"
OUTPUT_TRAIN = DATA_DIR / "sequence_windows_train.csv"
OUTPUT_VAL = DATA_DIR / "sequence_windows_val.csv"
OUTPUT_TEST = DATA_DIR / "sequence_windows_test.csv"

# Numerical feature columns used as model input (no timestamps / metadata)
NUMERICAL_FEATURES = [
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if any(v.strip() for v in row.values()):
                rows.append(row)
    return rows


def _is_finite(value: str) -> bool:
    try:
        v = float(value)
        return math.isfinite(v)
    except (ValueError, TypeError):
        return False


def _split_windows(windows: list, train_f: float, val_f: float):
    n = len(windows)
    train_end = int(n * train_f)
    val_end = train_end + int(n * val_f)
    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


def _write_window_csv(path: Path, windows: list, feature_cols: list):
    """Writes flattened sequence windows to CSV.

    Each row represents one complete window of W observations.
    Columns are named  feat_step0, feat_step1, ... (step × feature).
    """
    if not windows:
        return
    w_len = len(windows[0]["features"])
    fieldnames = (
        ["window_id", "split", "first_timestamp", "last_timestamp"]
        + [f"{col}_step{i}" for i in range(w_len) for col in feature_cols]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for win in windows:
            row = {
                "window_id": win["window_id"],
                "split": win["split"],
                "first_timestamp": win["first_timestamp"],
                "last_timestamp": win["last_timestamp"],
            }
            for i, feat_row in enumerate(win["features"]):
                for col in feature_cols:
                    row[f"{col}_step{i}"] = feat_row.get(col, "")
            writer.writerow(row)


def _write_meta_csv(path: Path, windows: list):
    fieldnames = [
        "window_id",
        "split",
        "first_timestamp",
        "last_timestamp",
        "window_scenarios",
        "window_status_codes",
        "window_paths",
        "contains_slow_latency",
        "contains_http_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for win in windows:
            writer.writerow({
                "window_id": win["window_id"],
                "split": win["split"],
                "first_timestamp": win["first_timestamp"],
                "last_timestamp": win["last_timestamp"],
                "window_scenarios": "|".join(win["scenarios"]),
                "window_status_codes": "|".join(win["status_codes"]),
                "window_paths": "|".join(win["paths"]),
                "contains_slow_latency": int(win["contains_slow_latency"]),
                "contains_http_error": int(win["contains_http_error"]),
            })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_sequences(
    window_length: int = DEFAULT_WINDOW_LENGTH,
    features_path: Path = None,
    metadata_path: Path = None,
) -> int:
    print("=" * 72)
    print(" PHASE 3.4 — SEQUENCE PREPARATION PIPELINE")
    print("=" * 72)

    # Choose input files (prefer expanded, fall back to Phase-3.3)
    if features_path is None:
        features_path = (
            EXPANDED_FEATURES if EXPANDED_FEATURES.exists() else FALLBACK_FEATURES
        )
    if metadata_path is None:
        metadata_path = (
            EXPANDED_METADATA if EXPANDED_METADATA.exists() else FALLBACK_METADATA
        )

    print(f"Feature input      : {features_path.resolve()}")
    print(f"Metadata input     : {metadata_path.resolve()}")
    print(f"Window length      : {window_length}")
    print()

    if not features_path.exists():
        print(f"[CRITICAL ERROR] Feature file not found: {features_path}")
        return 1
    if not metadata_path.exists():
        print(f"[CRITICAL ERROR] Metadata file not found: {metadata_path}")
        return 1

    features_rows = _read_csv(features_path)
    metadata_rows = _read_csv(metadata_path)

    total_obs = len(features_rows)
    print(f"  Observations loaded  : {total_obs}")

    if total_obs != len(metadata_rows):
        print(
            f"[CRITICAL ERROR] Row count mismatch: "
            f"features={total_obs}, metadata={len(metadata_rows)}"
        )
        return 1

    # Sort chronologically by timestamp (both files identically)
    features_rows.sort(key=lambda r: r.get("timestamp_utc", ""))
    metadata_rows.sort(key=lambda r: r.get("timestamp_utc", ""))

    # Verify timestamps align 1-to-1
    misaligned = sum(
        1
        for f, m in zip(features_rows, metadata_rows)
        if f.get("timestamp_utc") != m.get("timestamp_utc")
    )
    if misaligned:
        print(f"[CRITICAL ERROR] {misaligned} timestamp misalignments between feature and metadata rows.")
        return 1
    print(f"  Timestamp alignment  : OK (0 misalignments)")

    # Validate all numerical feature values are finite
    non_finite = 0
    for row in features_rows:
        for col in NUMERICAL_FEATURES:
            if not _is_finite(row.get(col, "")):
                non_finite += 1
    print(f"  Non-finite values    : {non_finite}")
    if non_finite:
        print("[WARN] Non-finite feature values detected. They will be preserved but may affect model training.")

    # Assess dataset adequacy for sequence windowing
    max_windows = max(0, total_obs - window_length + 1)
    print(f"\n  Total observations   : {total_obs}")
    print(f"  Window length        : {window_length}")
    print(f"  Max possible windows : {max_windows}")

    # Minimum viable threshold for a meaningful train/val/test split
    min_viable = 30
    if max_windows < min_viable:
        print(
            f"\n[!] DATASET ADEQUACY WARNING:"
            f"\n    Only {max_windows} sequence window(s) can be formed from {total_obs} observations"
            f"\n    with window_length={window_length}."
            f"\n    This is BELOW the recommended minimum of {min_viable} windows for a meaningful"
            f"\n    LSTM Autoencoder experiment."
            f"\n    The pipeline will complete, but results will NOT be suitable for model training."
            f"\n    Expand the dataset to at least {window_length + min_viable - 1} observations first."
        )

    if max_windows == 0:
        print(
            f"\n[CRITICAL ERROR] Cannot form even one window of length {window_length}"
            f" from {total_obs} observations. Reduce window_length or expand the dataset."
        )
        return 1

    # Build sliding windows
    all_windows = []
    for i in range(max_windows):
        feat_slice = features_rows[i: i + window_length]
        meta_slice = metadata_rows[i: i + window_length]

        scenarios = [r.get("scenario", "") for r in meta_slice]
        status_codes = [str(r.get("status_code", "")) for r in meta_slice]
        paths = [r.get("path", "") for r in meta_slice]

        contains_slow = any("slow" in sc for sc in scenarios)
        contains_err = any(
            int(s) >= 400 for s in status_codes if s.isdigit()
        )

        all_windows.append({
            "window_id": i,
            "split": "",  # assigned below
            "first_timestamp": feat_slice[0].get("timestamp_utc", ""),
            "last_timestamp": feat_slice[-1].get("timestamp_utc", ""),
            "features": [{col: r.get(col, "") for col in NUMERICAL_FEATURES} for r in feat_slice],
            "scenarios": scenarios,
            "status_codes": status_codes,
            "paths": paths,
            "contains_slow_latency": contains_slow,
            "contains_http_error": contains_err,
        })

    # Time-based split (chronological — NO random shuffle)
    train_wins, val_wins, test_wins = _split_windows(
        all_windows, TRAIN_FRAC, VAL_FRAC
    )
    for w in train_wins:
        w["split"] = "train"
    for w in val_wins:
        w["split"] = "val"
    for w in test_wins:
        w["split"] = "test"

    print(f"\n  Windows produced     : {len(all_windows)}")
    print(f"  Train windows        : {len(train_wins)}")
    print(f"  Validation windows   : {len(val_wins)}")
    print(f"  Test windows         : {len(test_wins)}")

    # Write combined windows + per-split files
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _write_window_csv(OUTPUT_WINDOWS, all_windows, NUMERICAL_FEATURES)
    _write_meta_csv(OUTPUT_WINDOW_META, all_windows)

    _write_window_csv(OUTPUT_TRAIN, train_wins, NUMERICAL_FEATURES)
    _write_window_csv(OUTPUT_VAL, val_wins, NUMERICAL_FEATURES)
    _write_window_csv(OUTPUT_TEST, test_wins, NUMERICAL_FEATURES)

    print(f"\n[✓] sequence_windows.csv          : {OUTPUT_WINDOWS}")
    print(f"[✓] sequence_window_metadata.csv  : {OUTPUT_WINDOW_META}")
    print(f"[✓] sequence_windows_train.csv    : {OUTPUT_TRAIN}")
    print(f"[✓] sequence_windows_val.csv      : {OUTPUT_VAL}")
    print(f"[✓] sequence_windows_test.csv     : {OUTPUT_TEST}")

    print("\n" + "=" * 72)
    print(" SEQUENCE PREPARATION: COMPLETED")
    print("=" * 72)
    return 0


def main():
    wl = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WINDOW_LENGTH
    fp = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    mp = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    sys.exit(prepare_sequences(wl, fp, mp))


if __name__ == "__main__":
    main()
