import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

FEATURES_FILE = DATA_DIR / "traffic_normal_extended_features.csv"
METADATA_FILE = DATA_DIR / "traffic_normal_extended_feature_metadata.csv"

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


def main():
    print("=" * 72)
    print("NORMAL EXTENDED TRAFFIC: SEQUENCE PREPARATION")
    print("=" * 72)

    # Protect existing outputs from accidental overwriting.
    output_files = [
        DATA_DIR / "sequence_windows_normal_extended.csv",
        DATA_DIR / "sequence_window_metadata_normal_extended.csv",
        DATA_DIR / "sequence_windows_normal_extended.npz",
        DATA_DIR / "sequence_normal_extended_summary.json",
        DATA_DIR / "sequence_windows_normal_extended_train.csv",
        DATA_DIR / "sequence_windows_normal_extended_val.csv",
        DATA_DIR / "sequence_windows_normal_extended_test.csv",
    ]

    existing_files = [path for path in output_files if path.exists()]

    if existing_files:
        print("ERROR: The following output files already exist:")
        for path in existing_files:
            print(f"  {path}")
        print("No files were changed. Inspect the existing files first.")
        raise SystemExit(1)

    # Verify input files.
    for path in [FEATURES_FILE, METADATA_FILE]:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            raise SystemExit(1)

    features = pd.read_csv(FEATURES_FILE)
    metadata = pd.read_csv(METADATA_FILE)

    required_features = ["timestamp_utc"] + FEATURE_COLUMNS
    required_metadata = [
        "timestamp_utc",
        "scenario",
        "status_code",
    ]

    for column in required_features:
        if column not in features.columns:
            raise ValueError(f"Missing feature column: {column}")

    for column in required_metadata:
        if column not in metadata.columns:
            raise ValueError(f"Missing metadata column: {column}")

    if len(features) != len(metadata):
        raise ValueError("Feature and metadata record counts differ.")

    # Parse and validate timestamps.
    features["_timestamp"] = pd.to_datetime(
        features["timestamp_utc"], utc=True, errors="coerce"
    )
    metadata["_timestamp"] = pd.to_datetime(
        metadata["timestamp_utc"], utc=True, errors="coerce"
    )

    if features["_timestamp"].isna().any():
        raise ValueError("Invalid timestamps found in feature data.")

    if metadata["_timestamp"].isna().any():
        raise ValueError("Invalid timestamps found in metadata.")

    # Sort both datasets chronologically.
    features = features.sort_values("_timestamp").reset_index(drop=True)
    metadata = metadata.sort_values("_timestamp").reset_index(drop=True)

    # Verify alignment and timestamp uniqueness.
    if features["_timestamp"].duplicated().any():
        raise ValueError("Duplicate feature timestamps found.")

    if metadata["_timestamp"].duplicated().any():
        raise ValueError("Duplicate metadata timestamps found.")

    if not features["_timestamp"].equals(metadata["_timestamp"]):
        raise ValueError("Feature and metadata timestamps do not align.")

    # Validate the numeric feature matrix.
    numeric_features = features[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    values = numeric_features.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite feature values found.")

    total = len(values)

    if total < WINDOW_SIZE:
        raise ValueError("Not enough observations to create windows.")

    # Split chronologically before window generation.
    train_end = int(total * TRAIN_FRACTION)
    val_count = int(total * VALIDATION_FRACTION)
    val_end = train_end + val_count

    split_ranges = [
        ("train", 0, train_end),
        ("val", train_end, val_end),
        ("test", val_end, total),
    ]

    all_windows = []
    all_window_metadata = []
    arrays = {}
    split_rows = {}

    for split_name, start, stop in split_ranges:
        split_values = values[start:stop]
        number_of_windows = len(split_values) - WINDOW_SIZE + 1

        if number_of_windows <= 0:
            raise ValueError(
                f"Not enough observations in {split_name} split."
            )

        windows = []
        rows = []

        for local_start in range(number_of_windows):
            global_start = start + local_start
            global_stop = global_start + WINDOW_SIZE

            window = values[global_start:global_stop].astype(np.float32)
            windows.append(window)

            window_id = f"{split_name}_{local_start + 1:04d}"

            row = {
                "window_id": window_id,
                "split": split_name,
            }

            for timestep in range(WINDOW_SIZE):
                for feature_index, feature_name in enumerate(
                    FEATURE_COLUMNS
                ):
                    row[f"t{timestep + 1:02d}_{feature_name}"] = float(
                        window[timestep, feature_index]
                    )

            rows.append(row)
            all_windows.append(row)

            window_metadata = metadata.iloc[global_start:global_stop]

            scenarios = window_metadata["scenario"].astype(str).tolist()
            statuses = pd.to_numeric(
                window_metadata["status_code"], errors="coerce"
            ).to_numpy()

            all_window_metadata.append(
                {
                    "window_id": window_id,
                    "split": split_name,
                    "start_timestamp_utc": features[
                        "timestamp_utc"
                    ].iloc[global_start],
                    "end_timestamp_utc": features[
                        "timestamp_utc"
                    ].iloc[global_stop - 1],
                    "scenario_sequence": " | ".join(scenarios),
                    "all_http_200": bool(np.all(statuses == 200)),
                    "normal_scenario_only": all(
                        scenario.startswith("normal_")
                        for scenario in scenarios
                    ),
                }
            )

        arrays[split_name] = np.stack(windows)
        split_rows[split_name] = rows

    # Save outputs exclusively under new filenames.
    all_windows_path = DATA_DIR / "sequence_windows_normal_extended.csv"
    metadata_path = (
        DATA_DIR / "sequence_window_metadata_normal_extended.csv"
    )
    npz_path = DATA_DIR / "sequence_windows_normal_extended.npz"
    summary_path = DATA_DIR / "sequence_normal_extended_summary.json"

    pd.DataFrame(all_windows).to_csv(all_windows_path, index=False)
    pd.DataFrame(all_window_metadata).to_csv(metadata_path, index=False)

    for split_name, rows in split_rows.items():
        pd.DataFrame(rows).drop(columns=["split"]).to_csv(
            DATA_DIR
            / f"sequence_windows_normal_extended_{split_name}.csv",
            index=False,
        )

    np.savez_compressed(
        npz_path,
        X_train=arrays["train"],
        X_val=arrays["val"],
        X_test=arrays["test"],
        feature_names=np.array(FEATURE_COLUMNS, dtype=str),
        window_size=np.int64(WINDOW_SIZE),
    )

    summary = {
        "input_observations": total,
        "window_size": WINDOW_SIZE,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "observation_split": {
            name: stop - start for name, start, stop in split_ranges
        },
        "windows_per_split": {
            name: int(array.shape[0]) for name, array in arrays.items()
        },
        "total_windows": int(sum(array.shape[0] for array in arrays.values())),
        "model_trained": False,
        "split_method": (
            "Chronological observation split before window generation"
        ),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Validated observations: {total}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Window size: {WINDOW_SIZE}")
    print("\nObservation splits:")

    for name, start, stop in split_ranges:
        print(f"  {name}: {stop - start}")

    print("\nSequence shapes:")
    for name, array in arrays.items():
        print(f"  X_{name}: {array.shape}")

    print(f"\nTotal sequence windows: {summary['total_windows']}")
    print("\nSaved files:")
    for path in output_files:
        print(f"  {path}")

    print("\nNo model was trained.")
    print("Sequence preparation completed successfully.")


if __name__ == "__main__":
    main()