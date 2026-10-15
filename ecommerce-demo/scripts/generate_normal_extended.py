
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

OUTPUT_CSV = DATA_DIR / "traffic_normal_extended.csv"

# 200 iterations x 3 scenarios = 600 requests
ITERATIONS = 200
REQUEST_INTERVAL_SECONDS = 0.1

DATA_DIR.mkdir(parents=True, exist_ok=True)

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


def check_server():
    try:
        with urlopen(BASE_URL + "/health", timeout=5) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


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

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario["name"],
        "path": scenario["path"].split("?")[0],
        "expected_delay_ms": scenario["expected_delay_ms"],
        "status_code": status_code,
        "measured_response_time_ms": elapsed_ms,
        "error": error_message,
    }


def main():
    print("=" * 60)
    print("EXTENDED NORMAL-TRAFFIC COLLECTION")
    print("=" * 60)
    print(f"Server: {BASE_URL}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Total planned requests: {ITERATIONS * len(SCENARIOS)}")
    print(f"Output: {OUTPUT_CSV}")

    # Avoid accidentally overwriting an earlier collection.
    if OUTPUT_CSV.exists():
        print("\nERROR: The output file already exists.")
        print("No data was collected. Choose a new filename first.")
        return 1

    if not check_server():
        print("\nERROR: The server health check failed.")
        print("Start FastAPI on port 8001 and try again.")
        return 1

    results = []
    total = ITERATIONS * len(SCENARIOS)

    for iteration in range(ITERATIONS):
        for scenario in SCENARIOS:
            result = send_request(scenario)
            results.append(result)

            if len(results) % 30 == 0:
                print(f"Progress: {len(results)}/{total}")

            time.sleep(REQUEST_INTERVAL_SECONDS)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    successful = sum(row["status_code"] == 200 for row in results)
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETED")
    print("=" * 60)
    print(f"Requests attempted: {len(results)}")
    print(f"HTTP 200 responses: {successful}")
    print(f"Other responses or connection errors: {failed}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nScenario counts:")
    for scenario in SCENARIOS:
        count = sum(
            row["scenario"] == scenario["name"]
            for row in results
        )
        print(f"{scenario['name']}: {count}")

    if failed:
        print("\nWARNING: Review unsuccessful requests before training.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())