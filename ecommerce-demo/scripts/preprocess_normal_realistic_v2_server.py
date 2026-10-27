"""
Preprocess realistic server-side normal traffic for Explainable AIOps.

Converts the server-side dataset into the column format expected
by the existing feature engineering pipeline.

Does not overwrite existing files.
"""

import csv
import math
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

INPUT_FILE = DATA_DIR / "traffic_normal_realistic_v2_server.csv"

OUTPUT_FILE = (
    DATA_DIR / "traffic_normal_realistic_v2_server_preprocessed.csv"
)

EXPECTED_ROWS = 600

OUTPUT_COLUMNS = [
    "timestamp_utc",
    "scenario",
    "expected_delay_ms",
    "path",
    "status_code",
    "measured_response_time_ms",
    "error",
]


def main():

    print("=" * 65)
    print("PREPROCESS REALISTIC NORMAL TRAFFIC - SERVER DATASET")
    print("=" * 65)

    # Safety checks
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "timestamp_utc",
            "endpoint",
            "method",
            "status_code",
            "response_time_ms",
            "error_message",
            "expected_delay_ms",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        source_rows = list(reader)

    if len(source_rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(source_rows)}"
        )

    processed_rows = []

    previous_timestamp = None

    for index, row in enumerate(source_rows, start=1):

        timestamp = row["timestamp_utc"].strip()
        endpoint = row["endpoint"].strip()
        method = row["method"].strip()
        status = row["status_code"].strip()

        # Validate timestamp
        parsed_timestamp = datetime.fromisoformat(timestamp)

        if parsed_timestamp.tzinfo is None:
            raise ValueError(
                f"Row {index}: Timestamp has no timezone."
            )

        if (
            previous_timestamp is not None
            and parsed_timestamp < previous_timestamp
        ):
            raise ValueError(
                f"Row {index}: Timestamps are not chronological."
            )

        previous_timestamp = parsed_timestamp

        # Validate request
        if method != "GET":
            raise ValueError(
                f"Row {index}: Unexpected method {method}"
            )

        if endpoint not in {"/", "/products/", "/users/"}:
            raise ValueError(
                f"Row {index}: Unexpected endpoint {endpoint}"
            )

        if status != "200":
            raise ValueError(
                f"Row {index}: Unexpected HTTP status {status}"
            )

        # Validate response time
        response_time = float(row["response_time_ms"])

        if not math.isfinite(response_time) or response_time < 0:
            raise ValueError(
                f"Row {index}: Invalid response time."
            )

        expected_delay = float(row["expected_delay_ms"])

        if not math.isfinite(expected_delay) or expected_delay < 0:
            raise ValueError(
                f"Row {index}: Invalid expected delay."
            )

        error_message = row["error_message"].strip()

        if error_message:
            raise ValueError(
                f"Row {index}: Unexpected error message."
            )

        # Convert to the existing pipeline schema
        processed_rows.append({
            "timestamp_utc": timestamp,
            "scenario": "normal_realistic_v2",
            "expected_delay_ms": row["expected_delay_ms"],
            "path": endpoint,
            "status_code": status,
            "measured_response_time_ms": row["response_time_ms"],
            "error": error_message,
        })

    # Exclusive creation prevents overwriting existing datasets
    with OUTPUT_FILE.open(
        "x",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS
        )

        writer.writeheader()
        writer.writerows(processed_rows)

    print("\nInput records:", len(source_rows))
    print("Processed records:", len(processed_rows))

    print("\nOutput columns:")
    print(OUTPUT_COLUMNS)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()