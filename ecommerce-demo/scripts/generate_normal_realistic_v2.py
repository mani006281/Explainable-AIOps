"""
Generate varied, read-only normal e-commerce traffic.

Experiment: normal_realistic_v2

Safety:
- Uses GET requests only.
- Does not create orders or payments.
- Does not overwrite an existing dataset.
- Does not modify existing ML artifacts.
"""

import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_URL = "http://127.0.0.1:8001"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_FILE = DATA_DIR / "traffic_normal_realistic_v2.csv"

TOTAL_REQUESTS = 600
RANDOM_SEED = 42

MIN_INTERVAL_SECONDS = 0.1
MAX_INTERVAL_SECONDS = 0.5

REQUEST_TIMEOUT_SECONDS = 10

ENDPOINTS = [
    "/products/",
    "/users/",
    "/",
]


# --------------------------------------------------
# 2. SAFETY CHECKS
# --------------------------------------------------

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Dataset already exists: {OUTPUT_FILE}\n"
        "The existing dataset will not be overwritten."
    )

DATA_DIR.mkdir(parents=True, exist_ok=True)

random_generator = random.Random(RANDOM_SEED)


# --------------------------------------------------
# 3. GENERATE VARIED REQUEST SEQUENCE
# --------------------------------------------------

def generate_request_sequence():
    sequence = []

    while len(sequence) < TOTAL_REQUESTS:

        endpoint = random_generator.choice(ENDPOINTS)

        # Occasionally repeat the same endpoint.
        repeat_count = random_generator.choice(
            [1, 1, 1, 2, 2, 3, 4, 5]
        )

        sequence.extend([endpoint] * repeat_count)

    return sequence[:TOTAL_REQUESTS]


# --------------------------------------------------
# 4. SEND REQUEST
# --------------------------------------------------

def send_request(endpoint):
    url = BASE_URL + endpoint

    timestamp = datetime.now(timezone.utc).isoformat()

    start_time = time.perf_counter()

    status_code = None
    error_message = ""

    try:
        with urlopen(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:

            status_code = response.status
            response.read()

    except HTTPError as error:
        status_code = error.code
        error_message = str(error)

    except URLError as error:
        status_code = 0
        error_message = str(error)

    except TimeoutError as error:
        status_code = 0
        error_message = str(error)

    response_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "timestamp_utc": timestamp,
        "endpoint": endpoint,
        "method": "GET",
        "status_code": status_code,
        "response_time_ms": round(response_time_ms, 2),
        "error_message": error_message,
        "expected_delay_ms": 0,
    }


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("REALISTIC NORMAL TRAFFIC GENERATOR - V2")
    print("=" * 60)

    print(f"Target API: {BASE_URL}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Output file: {OUTPUT_FILE}")

    sequence = generate_request_sequence()

    records = []

    print("\nGenerating traffic...\n")

    for index, endpoint in enumerate(sequence, start=1):

        record = send_request(endpoint)

        records.append(record)

        if index % 50 == 0:
            print(
                f"Completed {index}/{TOTAL_REQUESTS} requests"
            )

        interval = random_generator.uniform(
            MIN_INTERVAL_SECONDS,
            MAX_INTERVAL_SECONDS
        )

        time.sleep(interval)

    # Save only after all requests have completed.
    # Exclusive creation prevents overwriting an existing file.
    with OUTPUT_FILE.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp_utc",
                "endpoint",
                "method",
                "status_code",
                "response_time_ms",
                "error_message",
                "expected_delay_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 60)
    print("TRAFFIC GENERATION COMPLETED")
    print("=" * 60)

    print(f"Requests generated: {len(records)}")
    print(f"Dataset saved: {OUTPUT_FILE}")

    print("\nEndpoint distribution:")

    for endpoint in ENDPOINTS:

        count = sum(
            record["endpoint"] == endpoint
            for record in records
        )

        print(f"{endpoint}: {count}")

    print("\nHTTP status distribution:")

    status_codes = sorted(
        set(record["status_code"] for record in records)
    )

    for status_code in status_codes:

        count = sum(
            record["status_code"] == status_code
            for record in records
        )

        print(f"HTTP {status_code}: {count}")


if __name__ == "__main__":
    main()