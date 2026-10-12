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

# Save additional traffic separately.
# Do not modify the original traffic_runs.csv.
CSV_FILE = DATA_DIR / "traffic_runs_additional.csv"

# Number of times to execute all scenarios.
ITERATIONS = 10

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
        # Record HTTP errors such as 400 and 404.
        status_code = error.code
        error_message = str(error)

    except (URLError, TimeoutError, OSError) as error:
        # Record connection failures and timeouts.
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
        f"status={str(status_code):<18} "
        f"time={elapsed_ms:>8.2f} ms"
    )

    return result


# --------------------------------------------------
# Generate additional traffic and save results
# --------------------------------------------------

def main():
    results = []

    total_requests = ITERATIONS * len(SCENARIOS)

    print("\nStarting additional traffic generation...")
    print(f"Base URL: {BASE_URL}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Scenarios per iteration: {len(SCENARIOS)}")
    print(f"Expected total requests: {total_requests}\n")

    # Execute each scenario repeatedly.
    for iteration in range(ITERATIONS):
        print(
            f"\nIteration {iteration + 1}/{ITERATIONS}"
        )

        for scenario in SCENARIOS:
            result = send_request(scenario)
            results.append(result)

    # CSV columns.
    fieldnames = [
        "timestamp_utc",
        "scenario",
        "path",
        "expected_delay_ms",
        "status_code",
        "measured_response_time_ms",
        "error",
    ]

    # Write a fresh additional dataset.
    # This avoids duplicate records when the script is rerun.
    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------
    # Summarize HTTP status codes
    # --------------------------------------------------

    status_counts = {}

    for result in results:
        if result["status_code"] is None:
            status = "connection_error"
        else:
            status = str(result["status_code"])

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    successful_requests = sum(
        1
        for result in results
        if result["status_code"] == 200
    )

    failed_http_requests = sum(
        1
        for result in results
        if result["status_code"] is not None
        and result["status_code"] >= 400
    )

    connection_errors = sum(
        1
        for result in results
        if result["status_code"] is None
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("ADDITIONAL TRAFFIC GENERATION COMPLETED")
    print("=" * 55)

    print(f"Requests attempted: {len(results)}")
    print(f"HTTP 200 responses: {successful_requests}")
    print(f"HTTP error responses: {failed_http_requests}")
    print(f"Connection errors: {connection_errors}")
    print(f"HTTP status distribution: {status_counts}")

    print(f"\nResults saved to:\n{CSV_FILE}")

    print("\nExpected status distribution if all requests behave as designed:")
    print("HTTP 200: 50")
    print("HTTP 400: 10")
    print("HTTP 404: 10")
    print("Total:    70")

    print("\nUse the actual status distribution above to verify the experiment.")
    print("=" * 55)


if __name__ == "__main__":
    main()