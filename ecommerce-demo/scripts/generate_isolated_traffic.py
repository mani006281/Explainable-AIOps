
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

DATA_DIR.mkdir(parents=True, exist_ok=True)

ITERATIONS = 20

FIELDNAMES = [
    "timestamp_utc",
    "scenario",
    "path",
    "expected_delay_ms",
    "status_code",
    "measured_response_time_ms",
    "error",
]

# --------------------------------------------------
# Isolated traffic scenarios
# --------------------------------------------------

SCENARIOS = {
    "normal": [
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
    ],
    "slow": [
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
    ],
}


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
        status_code = error.code
        error_message = str(error)

    except (URLError, TimeoutError, OSError) as error:
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
        f"{result['scenario']:<18} "
        f"status={str(status_code):<8} "
        f"time={elapsed_ms:>8.2f} ms"
    )

    return result


def main():
    print("\nISOLATED TRAFFIC GENERATION")
    print("=" * 55)
    print("1. Normal traffic")
    print("2. Slow-latency traffic")
    print("3. Both datasets separately")
    print("=" * 55)

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        selected_modes = ["normal"]
    elif choice == "2":
        selected_modes = ["slow"]
    elif choice == "3":
        selected_modes = ["normal", "slow"]
    else:
        print("Invalid choice. Enter 1, 2, or 3.")
        return

    for mode in selected_modes:
        output_file = DATA_DIR / f"traffic_runs_{mode}_isolated.csv"
        results = []

        scenarios = SCENARIOS[mode]
        total_requests = ITERATIONS * len(scenarios)

        print(f"\nGenerating {mode.upper()} traffic")
        print(f"Requests planned: {total_requests}")
        print(f"Output: {output_file}\n")

        for iteration in range(ITERATIONS):
            print(
                f"\nIteration {iteration + 1}/{ITERATIONS}"
            )

            for scenario in scenarios:
                results.append(send_request(scenario))

        with output_file.open(
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

        successful = sum(
            row["status_code"] == 200
            for row in results
        )

        errors = sum(
            row["status_code"] is None
            or row["status_code"] >= 400
            for row in results
        )

        print("\n" + "-" * 55)
        print(f"{mode.upper()} TRAFFIC SUMMARY")
        print(f"Requests: {len(results)}")
        print(f"HTTP 200 responses: {successful}")
        print(f"HTTP errors or connection errors: {errors}")
        print(f"Saved to: {output_file}")
        print("-" * 55)


if __name__ == "__main__":
    main()

