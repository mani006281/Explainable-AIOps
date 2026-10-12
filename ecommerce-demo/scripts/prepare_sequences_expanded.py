import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FEATURES_FILE = DATA_DIR / "traffic_features_expanded.csv"
METADATA_FILE = DATA_DIR / "traffic_feature_metadata_expanded.csv"

WINDOW_LENGTH = 10

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Remaining 15% is used for testing.

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

NORMAL_SCENARIOS = {
    "normal_health",
    "normal_products",
    "normal_latency",
}

# All outputs use new filenames to preserve previous results.

OUTPUT_WINDOWS = DATA_DIR / "sequence_windows_expanded.csv"
OUTPUT_METADATA = DATA_DIR / "sequence_window_metadata_expanded.csv"

OUTPUT_TRAIN = DATA_DIR / "sequence_windows_expanded_train.csv"
OUTPUT_VALIDATION = DATA_DIR / "sequence_windows_expanded_val.csv"
OUTPUT_TEST = DATA_DIR / "sequence_windows_expanded_test.csv"

OUTPUT_NORMAL_TRAIN = (
    DATA_DIR / "sequence_windows_expanded_train_normal_only.csv"
)


# =========================================================
# CSV HELPERS
# =========================================================

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        return [
            row for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]


def parse_timestamp(value: str) -> datetime:
    if not value or not value.strip():
        raise ValueError("Empty timestamp")

    parsed = datetime.fromisoformat(value.strip())

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def validate_numeric_features(row: dict, row_number: int) -> dict:
    converted = {}

    for column in NUMERICAL_FEATURES:
        value = row.get(column, "")

        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid numerical feature '{column}' "
                f"at row {row_number}: {value!r}"
            )

        if not math.isfinite(number):
            raise ValueError(
                f"Non-finite feature '{column}' at row {row_number}"
            )

        converted[column] = number

    return converted


# =========================================================
# WINDOW CREATION
# =========================================================

def create_windows(
    paired_rows: list[tuple[dict, dict]],
    split_name: str,
    window_length: int,
    first_window_id: int,
) -> list[dict]:

    windows = []

    # Windows are formed only inside this split.
    # They cannot overlap with observations in another split.

    max_start = len(paired_rows) - window_length

    for start in range(max_start + 1):
        current_rows = paired_rows[start:start + window_length]

        feature_rows = [pair[0] for pair in current_rows]
        metadata_rows = [pair[1] for pair in current_rows]

        scenarios = [
            row.get("scenario", "").strip()
            for row in metadata_rows
        ]

        statuses = [
            str(row.get("status_code", "")).strip()
            for row in metadata_rows
        ]

        paths = [
            row.get("path", "").strip()
            for row in metadata_rows
        ]

        # A normal candidate window is a preliminary selection
        # based on experiment metadata. It is not proof of
        # independently verified normal ground truth.

        normal_candidate = all(
            scenario in NORMAL_SCENARIOS
            and status == "200"
            for scenario, status in zip(scenarios, statuses)
        )

        contains_slow_latency = any(
            scenario == "slow_latency"
            for scenario in scenarios
        )

        contains_http_error = any(
            status.isdigit() and int(status) >= 400
            for status in statuses
        )

        windows.append({
            "window_id": first_window_id + len(windows),
            "split": split_name,
            "first_timestamp": feature_rows[0]["timestamp_utc"],
            "last_timestamp": feature_rows[-1]["timestamp_utc"],
            "features": [
                validate_numeric_features(row, start + step + 1)
                for step, row in enumerate(feature_rows)
            ],
            "scenarios": scenarios,
            "status_codes": statuses,
            "paths": paths,
            "normal_candidate": normal_candidate,
            "contains_slow_latency": contains_slow_latency,
            "contains_http_error": contains_http_error,
        })

    return windows


# =========================================================
# OUTPUT HELPERS
# =========================================================

def feature_fieldnames() -> list[str]:
    fields = [
        "window_id",
        "split",
        "first_timestamp",
        "last_timestamp",
    ]

    for step in range(WINDOW_LENGTH):
        for feature in NUMERICAL_FEATURES:
            fields.append(f"{feature}_step{step}")

    return fields


def write_feature_windows(path: Path, windows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = feature_fieldnames()

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for window in windows:
            output_row = {
                "window_id": window["window_id"],
                "split": window["split"],
                "first_timestamp": window["first_timestamp"],
                "last_timestamp": window["last_timestamp"],
            }

            for step, feature_row in enumerate(window["features"]):
                for feature in NUMERICAL_FEATURES:
                    output_row[f"{feature}_step{step}"] = (
                        feature_row[feature]
                    )

            writer.writerow(output_row)


def write_window_metadata(path: Path, windows: list[dict]) -> None:
    fields = [
        "window_id",
        "split",
        "first_timestamp",
        "last_timestamp",
        "window_scenarios",
        "window_status_codes",
        "window_paths",
        "normal_candidate_window",
        "contains_slow_latency",
        "contains_http_error",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for window in windows:
            writer.writerow({
                "window_id": window["window_id"],
                "split": window["split"],
                "first_timestamp": window["first_timestamp"],
                "last_timestamp": window["last_timestamp"],
                "window_scenarios": "|".join(window["scenarios"]),
                "window_status_codes": "|".join(window["status_codes"]),
                "window_paths": "|".join(window["paths"]),
                "normal_candidate_window": int(
                    window["normal_candidate"]
                ),
                "contains_slow_latency": int(
                    window["contains_slow_latency"]
                ),
                "contains_http_error": int(
                    window["contains_http_error"]
                ),
            })


# =========================================================
# MAIN PIPELINE
# =========================================================

def main() -> int:
    print("=" * 72)
    print("PHASE 3.4 — LEAKAGE-SAFE SEQUENCE PREPARATION")
    print("=" * 72)

    print(f"Feature input : {FEATURES_FILE}")
    print(f"Metadata input: {METADATA_FILE}")
    print(f"Window length : {WINDOW_LENGTH}")

    try:
        feature_rows = read_csv(FEATURES_FILE)
        metadata_rows = read_csv(METADATA_FILE)

        if len(feature_rows) != len(metadata_rows):
            raise ValueError(
                "Feature/metadata row count mismatch: "
                f"{len(feature_rows)} vs {len(metadata_rows)}"
            )

        if not feature_rows:
            raise ValueError("Feature dataset is empty")

        required_feature_columns = {
            "timestamp_utc",
            *NUMERICAL_FEATURES,
        }

        required_metadata_columns = {
            "timestamp_utc",
            "scenario",
            "path",
            "status_code",
            "expected_delay_ms",
        }

        missing_features = (
            required_feature_columns
            - set(feature_rows[0].keys())
        )

        missing_metadata = (
            required_metadata_columns
            - set(metadata_rows[0].keys())
        )

        if missing_features:
            raise ValueError(
                f"Missing feature columns: {sorted(missing_features)}"
            )

        if missing_metadata:
            raise ValueError(
                f"Missing metadata columns: {sorted(missing_metadata)}"
            )

        # Index both CSVs by timestamp so features and metadata
        # are paired correctly even if their original ordering differs.

        feature_by_timestamp = {}

        for row in feature_rows:
            timestamp = row["timestamp_utc"]

            if timestamp in feature_by_timestamp:
                raise ValueError(
                    f"Duplicate feature timestamp found: {timestamp}"
                )

            feature_by_timestamp[timestamp] = row

        metadata_by_timestamp = {}

        for row in metadata_rows:
            timestamp = row["timestamp_utc"]

            if timestamp in metadata_by_timestamp:
                raise ValueError(
                    f"Duplicate metadata timestamp found: {timestamp}"
                )

            metadata_by_timestamp[timestamp] = row

        if (
            feature_by_timestamp.keys()
            != metadata_by_timestamp.keys()
        ):
            raise ValueError(
                "Feature and metadata timestamp sets do not match"
            )

        timestamps = sorted(
            feature_by_timestamp.keys(),
            key=parse_timestamp,
        )

        paired_rows = [
            (
                feature_by_timestamp[timestamp],
                metadata_by_timestamp[timestamp],
            )
            for timestamp in timestamps
        ]

        # Validate every numerical value before making windows.
        for index, (feature_row, _) in enumerate(paired_rows, start=1):
            validate_numeric_features(feature_row, index)

        observation_count = len(paired_rows)

        if observation_count < WINDOW_LENGTH:
            raise ValueError(
                f"Only {observation_count} observations are available; "
                f"at least {WINDOW_LENGTH} are required"
            )

        # IMPORTANT:
        # Split chronological observations BEFORE constructing windows.
        # This prevents overlapping windows from crossing split boundaries.

        train_end = int(observation_count * TRAIN_FRACTION)

        validation_count = int(
            observation_count * VALIDATION_FRACTION
        )

        validation_end = train_end + validation_count

        train_rows = paired_rows[:train_end]
        validation_rows = paired_rows[train_end:validation_end]
        test_rows = paired_rows[validation_end:]

        print(f"\nTotal observations : {observation_count}")
        print(f"Training observations: {len(train_rows)}")
        print(f"Validation observations: {len(validation_rows)}")
        print(f"Test observations: {len(test_rows)}")

        # Windows are generated independently inside each split.
        train_windows = create_windows(
            train_rows,
            "train",
            WINDOW_LENGTH,
            0,
        )

        validation_windows = create_windows(
            validation_rows,
            "val",
            WINDOW_LENGTH,
            len(train_windows),
        )

        test_windows = create_windows(
            test_rows,
            "test",
            WINDOW_LENGTH,
            len(train_windows) + len(validation_windows),
        )

        all_windows = (
            train_windows
            + validation_windows
            + test_windows
        )

        normal_training_windows = [
            window
            for window in train_windows
            if window["normal_candidate"]
        ]

        # Write new expanded filenames only.
        write_feature_windows(OUTPUT_WINDOWS, all_windows)
        write_window_metadata(OUTPUT_METADATA, all_windows)

        write_feature_windows(OUTPUT_TRAIN, train_windows)
        write_feature_windows(OUTPUT_VALIDATION, validation_windows)
        write_feature_windows(OUTPUT_TEST, test_windows)

        # This separate file is intended for future normal-only
        # autoencoder training. It may contain zero rows if the
        # experiment does not provide sufficiently long normal runs.
        write_feature_windows(
            OUTPUT_NORMAL_TRAIN,
            normal_training_windows,
        )

        print("\n" + "-" * 72)
        print("SEQUENCE PREPARATION SUMMARY")
        print("-" * 72)

        print(f"Window length: {WINDOW_LENGTH}")
        print(f"Train windows: {len(train_windows)}")
        print(f"Validation windows: {len(validation_windows)}")
        print(f"Test windows: {len(test_windows)}")
        print(f"Total windows: {len(all_windows)}")
        print(
            "Normal-only candidate training windows: "
            f"{len(normal_training_windows)}"
        )

        print("\nGenerated files:")

        for path in [
            OUTPUT_WINDOWS,
            OUTPUT_METADATA,
            OUTPUT_TRAIN,
            OUTPUT_VALIDATION,
            OUTPUT_TEST,
            OUTPUT_NORMAL_TRAIN,
        ]:
            print(f"  {path}")

        if not normal_training_windows:
            print(
                "\nWARNING: No normal-only training windows were found."
            )
            print(
                "The current traffic experiment interleaves normal and "
                "controlled abnormal scenarios."
            )
            print(
                "Collect a dedicated, chronological normal-baseline run "
                "before training the LSTM Autoencoder."
            )

        print("\nSequence preparation completed.")
        print("No model has been trained.")

        return 0

    except Exception as exc:
        print(f"\n[ERROR] Sequence preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())