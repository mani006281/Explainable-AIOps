import csv
import time
import sys

from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


# ==================================================
# CONFIGURATION
# ==================================================

BASE_URL = "http://127.0.0.1:8001"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_CSV = DATA_DIR / "traffic_baseline.csv"

# 100 iterations × 3 scenarios = 300 requests
ITERATIONS = 100

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# NORMAL BASELINE SCENARIOS
# ==================================================

SCENARIOS = [
    {
        "name": "normal_health",
        "path": "/health",
        "expected_delay_ms": 0,
    },
    {
        "name": "normal_products",
        "path": "/products/",
        "expected_delay_ms": 0,
    },
    {
        "name": "normal_latency",
        "path": "/debug/test-latency?delay_ms=100",
        "expected_delay_ms": 100,
    },
]


FIELDNAMES = [
    "timestamp_utc",
    "scenario",
    "path",
    "expected_delay_ms",
    "status_code",
    "measured_response_time_ms",
    "error",
]


# ==================================================
# CHECK SERVER
# ==================================================

def check_server():
    try:
        with urlopen(
            BASE_URL + "/health",
            timeout=5,
        ) as response:
            return response.status == 200

    except (HTTPError, URLError, TimeoutError, OSError):
        return False


# ==================================================
# SEND REQUEST
# ==================================================

def send_request(scenario):
    url = BASE_URL + scenario["path"]

    started = time.perf_counter()

    status_code = None
    error_message = ""

    try:
        with urlopen(url, timeout=10) as response:
            status_code = response.status
            response.read()

    except HTTPError as exc:
        status_code = exc.code
        error_message = f"HTTPError: {exc.code}"

    except (URLError, TimeoutError, OSError) as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    response_time_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    return {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "scenario": scenario["name"],
        "path": scenario["path"].split("?")[0],
        "expected_delay_ms": scenario["expected_delay_ms"],
        "status_code": status_code,
        "measured_response_time_ms": response_time_ms,
        "error": error_message,
    }


# ==================================================
# GENERATE BASELINE DATA
# ==================================================

def main():
    print("=" * 60)
    print("NORMAL BASELINE TRAFFIC GENERATOR")
    print("=" * 60)

    print(f"Server: {BASE_URL}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Scenarios per iteration: {len(SCENARIOS)}")

    if not check_server():
        print("\nERROR: The e-commerce server is not responding.")
        print("Start FastAPI on port 8001 and run this script again.")
        return 1

    print("\nServer health check passed.")

    results = []
    total_requests = ITERATIONS * len(SCENARIOS)

    for iteration in range(ITERATIONS):

        for scenario in SCENARIOS:
            result = send_request(scenario)
            results.append(result)

        completed = (iteration + 1) * len(SCENARIOS)

        if (iteration + 1) % 10 == 0:
            print(
                f"Progress: {completed}/{total_requests} requests"
            )

    # Write a fresh baseline file.
    # Other datasets are not modified.
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(results)

    status_counts = {}

    for row in results:
        status = (
            str(row["status_code"])
            if row["status_code"] is not None
            else "connection_error"
        )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    success_count = sum(
        1 for row in results
        if row["status_code"] == 200
    )

    connection_errors = sum(
        1 for row in results
        if row["status_code"] is None
    )

    print("\n" + "=" * 60)
    print("BASELINE COLLECTION COMPLETED")
    print("=" * 60)

    print(f"Requests attempted: {len(results)}")
    print(f"HTTP 200 responses: {success_count}")
    print(f"Status distribution: {status_counts}")
    print(f"Connection errors: {connection_errors}")
    print(f"Dataset saved to: {OUTPUT_CSV}")

    print(
        "\nNote: These are controlled local baseline observations, "
        "not production traffic."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())