"""
Prepare leakage-safe time-series sequences from normal baseline traffic.

This script:
1. Reads baseline features and their aligned metadata.
2. Validates timestamps and numerical feature values.
3. Splits chronological observations into train, validation and test.
4. Generates independent windows within each split.
5. Saves CSV and NumPy outputs for later model training.

No model is trained by this script.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

FEATURES_FILE = DATA_DIR / "traffic_baseline_features.csv"
METADATA_FILE = DATA_DIR / "traffic_baseline_feature_metadata.csv"

WINDOW_SIZE = 10
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

FEATURE_COLUMNS = [
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


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def fail(message):
    """Print an error message and exit with a failure code."""
    print(f"\nERROR: {message}")
    raise SystemExit(1)


def main():
    print("=" * 72)
    print("EXPLAINABLE-AIOPS: BASELINE SEQUENCE PREPARATION")
    print("=" * 72)

    # 1. Confirm that the input files exist.
    if not FEATURES_FILE.exists():
        fail(f"Feature file not found: {FEATURES_FILE}")

    if not METADATA_FILE.exists():
        fail(f"Metadata file not found: {METADATA_FILE}")

    print(f"Feature input : {FEATURES_FILE}")
    print(f"Metadata input: {METADATA_FILE}")
    print(f"Window size   : {WINDOW_SIZE}")
    print()

    # 2. Read the datasets.
    features = pd.read_csv(FEATURES_FILE)
    metadata = pd.read_csv(METADATA_FILE)

    required_feature_columns = ["timestamp_utc"] + FEATURE_COLUMNS
    required_metadata_columns = [
        "timestamp_utc",
        "scenario",
        "status_code",
    ]

    missing_features = [
        column
        for column in required_feature_columns
        if column not in features.columns
    ]

    missing_metadata = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_features:
        fail(f"Missing feature columns: {missing_features}")

    if missing_metadata:
        fail(f"Missing metadata columns: {missing_metadata}")

    if len(features) != len(metadata):
        fail(
            "Feature and metadata record counts differ: "
            f"{len(features)} versus {len(metadata)}."
        )

    # 3. Parse timestamps and reject invalid timestamps.
    features["_parsed_timestamp"] = pd.to_datetime(
        features["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    metadata["_parsed_timestamp"] = pd.to_datetime(
        metadata["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    if features["_parsed_timestamp"].isna().any():
        fail("The feature dataset contains invalid timestamps.")

    if metadata["_parsed_timestamp"].isna().any():
        fail("The metadata dataset contains invalid timestamps.")

    # Sort the two datasets chronologically.
    features = (
        features.sort_values("_parsed_timestamp")
        .reset_index(drop=True)
    )

    metadata = (
        metadata.sort_values("_parsed_timestamp")
        .reset_index(drop=True)
    )

    # Duplicate timestamps can cause ambiguous alignment.
    if features["_parsed_timestamp"].duplicated().any():
        fail("Duplicate timestamps found in the feature dataset.")

    if metadata["_parsed_timestamp"].duplicated().any():
        fail("Duplicate timestamps found in the metadata dataset.")

    # Ensure each feature row matches the correct metadata row.
    if not features["_parsed_timestamp"].equals(
        metadata["_parsed_timestamp"]
    ):
        fail(
            "Feature and metadata timestamps do not match after "
            "chronological sorting."
        )

    # 4. Convert the model features to numeric values.
    numeric_features = features[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    feature_values = numeric_features.to_numpy(dtype=np.float64)

    if not np.isfinite(feature_values).all():
        fail(
            "The numerical feature matrix contains missing or "
            "non-finite values."
        )

    if len(feature_values) < WINDOW_SIZE:
        fail(
            f"At least {WINDOW_SIZE} observations are required "
            "to create a sequence."
        )

    total_observations = len(feature_values)

    print(f"Validated observations: {total_observations}")
    print(f"Number of features    : {len(FEATURE_COLUMNS)}")
    print("Timestamp alignment   : PASS")
    print("Numerical validation  : PASS")
    print()

    # 5. Split observations chronologically BEFORE windowing.
    train_end = int(total_observations * TRAIN_FRACTION)

    validation_count = int(
        total_observations * VALIDATION_FRACTION
    )

    validation_end = train_end + validation_count

    split_ranges = [
        ("train", 0, train_end),
        ("val", train_end, validation_end),
        ("test", validation_end, total_observations),
    ]

    print("Chronological observation split:")
    for split_name, start, stop in split_ranges:
        print(
            f"  {split_name:5s}: "
            f"{stop - start:3d} observations"
        )

    # 6. Create windows independently inside each split.
    all_window_rows = []
    all_metadata_rows = []
    split_arrays = {}
    split_window_rows = {}

    for split_name, start, stop in split_ranges:
        split_observations = feature_values[start:stop]

        number_of_windows = (
            len(split_observations) - WINDOW_SIZE + 1
        )

        if number_of_windows <= 0:
            fail(
                f"The {split_name} split has too few observations "
                f"for a window size of {WINDOW_SIZE}."
            )

        windows = []
        window_rows = []

        for local_start in range(number_of_windows):
            global_start = start + local_start
            global_stop = global_start + WINDOW_SIZE

            window = feature_values[global_start:global_stop]
            windows.append(window.astype(np.float32))

            window_id = (
                f"{split_name}_{local_start + 1:04d}"
            )

            # Flatten one 10 x 12 window for CSV inspection.
            row = {
                "window_id": window_id,
                "split": split_name,
            }

            for time_index in range(WINDOW_SIZE):
                for feature_index, feature_name in enumerate(
                    FEATURE_COLUMNS
                ):
                    column_name = (
                        f"t{time_index + 1:02d}_{feature_name}"
                    )

                    row[column_name] = float(
                        window[time_index, feature_index]
                    )

            window_rows.append(row)
            all_window_rows.append(row)

            # Metadata is not supplied as an additional model feature.
            # It is retained for audit and interpretation.
            scenarios = (
                metadata["scenario"]
                .iloc[global_start:global_stop]
                .astype(str)
                .tolist()
            )

            status_codes = pd.to_numeric(
                metadata["status_code"]
                .iloc[global_start:global_stop],
                errors="coerce",
            ).to_numpy()

            all_http_200 = bool(
                np.all(status_codes == 200)
            )

            normal_scenario_only = all(
                scenario.startswith("normal_")
                for scenario in scenarios
            )

            all_metadata_rows.append(
                {
                    "window_id": window_id,
                    "split": split_name,
                    "start_timestamp_utc": features[
                        "timestamp_utc"
                    ].iloc[global_start],
                    "end_timestamp_utc": features[
                        "timestamp_utc"
                    ].iloc[global_stop - 1],
                    "source_row_start": global_start + 1,
                    "source_row_end": global_stop,
                    "scenario_sequence": " | ".join(scenarios),
                    "all_http_200": all_http_200,
                    "normal_scenario_only": normal_scenario_only,
                    "is_normal_baseline_window": (
                        all_http_200 and normal_scenario_only
                    ),
                }
            )

        split_arrays[split_name] = np.stack(windows)
        split_window_rows[split_name] = window_rows

    # 7. Save all windows in one CSV, including the split label.
    all_windows_df = pd.DataFrame(all_window_rows)

    all_windows_csv = DATA_DIR / "sequence_windows_baseline.csv"
    all_windows_df.to_csv(all_windows_csv, index=False)

    # Also save separate train, validation and test CSVs.
    for split_name, rows in split_window_rows.items():
        split_df = pd.DataFrame(rows).drop(columns=["split"])

        output_path = (
            DATA_DIR / f"sequence_windows_baseline_{split_name}.csv"
        )

        split_df.to_csv(output_path, index=False)

    # 8. Save NumPy arrays in LSTM-compatible three-dimensional form.
    # Each array has shape:
    # (number_of_windows, window_size, number_of_features).
    npz_path = DATA_DIR / "sequence_windows_baseline.npz"

    np.savez_compressed(
        npz_path,
        X_train=split_arrays["train"],
        X_val=split_arrays["val"],
        X_test=split_arrays["test"],
        feature_names=np.array(FEATURE_COLUMNS, dtype=str),
        window_size=np.int64(WINDOW_SIZE),
    )

    # 9. Save auditable metadata for each window.
    window_metadata_df = pd.DataFrame(all_metadata_rows)

    metadata_output = (
        DATA_DIR / "sequence_window_metadata_baseline.csv"
    )

    window_metadata_df.to_csv(metadata_output, index=False)

    # 10. Save a machine-readable summary.
    window_counts = {
        split_name: int(array.shape[0])
        for split_name, array in split_arrays.items()
    }

    summary = {
        "input_observations": int(total_observations),
        "window_size": WINDOW_SIZE,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "observation_split": {
            split_name: int(stop - start)
            for split_name, start, stop in split_ranges
        },
        "windows_per_split": window_counts,
        "total_windows": int(sum(window_counts.values())),
        "normal_baseline_windows": int(
            window_metadata_df[
                "is_normal_baseline_window"
            ].sum()
        ),
        "split_method": (
            "Chronological observation split before window generation"
        ),
        "model_trained": False,
    }

    summary_path = DATA_DIR / "sequence_baseline_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # 11. Display a concise report.
    print()
    print("=" * 72)
    print("BASELINE SEQUENCE PREPARATION COMPLETED")
    print("=" * 72)

    print(f"Training windows   : {split_arrays['train'].shape[0]}")
    print(f"Validation windows : {split_arrays['val'].shape[0]}")
    print(f"Test windows       : {split_arrays['test'].shape[0]}")
    print(f"Total windows      : {sum(window_counts.values())}")

    print()
    print("NumPy array shapes:")
    for split_name in ["train", "val", "test"]:
        print(
            f"  X_{split_name}: "
            f"{split_arrays[split_name].shape}"
        )

    print()
    print(
        "Windows marked as normal baseline: "
        f"{summary['normal_baseline_windows']}"
    )

    print()
    print("Saved outputs:")
    print(f"  {all_windows_csv}")
    print(f"  {metadata_output}")
    print(f"  {npz_path}")
    print(f"  {summary_path}")

    for split_name in ["train", "val", "test"]:
        print(
            "  "
            + str(
                DATA_DIR
                / f"sequence_windows_baseline_{split_name}.csv"
            )
        )

    print()
    print("No model was trained.")
    print("Review the counts and shapes before proceeding.")


if __name__ == "__main__":
    main()