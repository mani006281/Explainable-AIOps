import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_URL = "http://127.0.0.1:8001"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
CSV_FILE = DATA_DIR / "traffic_runs.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Test scenarios
# --------------------------------------------------

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
    {
        "name": "slow_latency",
        "path": "/debug/test-latency?delay_ms=1500",
        "expected_delay_ms": 1500,
    },
    {
        "name": "slow_latency",
        "path": "/debug/test-latency?delay_ms=3000",
        "expected_delay_ms": 3000,
    },
    {
        "name": "invalid_delay",
        "path": "/debug/test-latency?delay_ms=6000",
        "expected_delay_ms": 0,
    },
    {
        "name": "unknown_endpoint",
        "path": "/does-not-exist",
        "expected_delay_ms": 0,
    },
]


# --------------------------------------------------
# Send one request and measure response time
# --------------------------------------------------

def send_request(scenario):
    url = BASE_URL + scenario["path"]

    start_time = time.perf_counter()
    status_code = None
    error_message = ""

    try:
        with urlopen(url, timeout=10) as response:
            status_code = response.status
            response.read()

    except HTTPError as error:
        # HTTP 400 and 404 responses are recorded here.
        status_code = error.code
        error_message = str(error)

    except (URLError, TimeoutError, OSError) as error:
        # Record connection and timeout problems.
        error_message = str(error)

    elapsed_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario["name"],
        "path": scenario["path"].split("?")[0],
        "expected_delay_ms": scenario["expected_delay_ms"],
        "status_code": status_code,
        "measured_response_time_ms": elapsed_ms,
        "error": error_message,
    }

    print(
        f"{result['scenario']:<16} "
        f"status={str(status_code):<4} "
        f"time={elapsed_ms:>8.2f} ms"
    )

    return result


# --------------------------------------------------
# Generate traffic and save results
# --------------------------------------------------

def main():
    results = []

    print("\nStarting controlled traffic generation...\n")

    for iteration in range(3):
        print(f"Iteration {iteration + 1}/3")

        for scenario in SCENARIOS:
            results.append(send_request(scenario))

    # CSV columns
    fieldnames = [
        "timestamp_utc",
        "scenario",
        "path",
        "expected_delay_ms",
        "status_code",
        "measured_response_time_ms",
        "error",
    ]

    file_exists = CSV_FILE.exists()

    # Append results to preserve previous experiments.
    with CSV_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(results)

    # Count successful requests.
    successful_requests = sum(
        1
        for result in results
        if result["status_code"] == 200
    )

    # Count HTTP status codes and connection errors.
    status_counts = {}

    for result in results:
        if result["status_code"] is None:
            status = "connection_error"
        else:
            status = str(result["status_code"])

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    # Final summary
    print("\nTraffic generation completed.")
    print(f"Requests sent: {len(results)}")
    print(f"HTTP 200 responses: {successful_requests}")
    print(f"HTTP status counts: {status_counts}")
    print(f"Results saved to: {CSV_FILE}")


if __name__ == "__main__":
    main()