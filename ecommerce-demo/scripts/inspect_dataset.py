import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Default paths and schema configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "traffic_runs.csv"
)

EXPECTED_COLUMNS = [
    "timestamp_utc",
    "scenario",
    "path",
    "expected_delay_ms",
    "status_code",
    "measured_response_time_ms",
    "error",
]


def inspect_dataset(csv_path: Path) -> int:
    """Inspects and validates the traffic CSV dataset.

    Returns:
        int: Exit status code (0 for success, 1 for critical failure).
    """
    print("=" * 72)
    print(" EXPLAINABLE-AIOPS: DATASET INSPECTION & VALIDATION REPORT")
    print("=" * 72)
    print(f"Target CSV File : {csv_path.resolve()}\n")

    # 1. Check file existence
    if not csv_path.exists():
        print(f"[CRITICAL ERROR] File does not exist: {csv_path}")
        return 1

    if not csv_path.is_file():
        print(f"[CRITICAL ERROR] Path is not a file: {csv_path}")
        return 1

    file_size_bytes = csv_path.stat().st_size
    if file_size_bytes == 0:
        print("[CRITICAL ERROR] File is completely empty (0 bytes).")
        return 1

    # 2. Read and parse CSV
    records = []
    headers = []
    critical_errors = []
    warnings = []

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                print("[CRITICAL ERROR] CSV file has no rows.")
                return 1

            for row_idx, row in enumerate(reader, start=2):
                if not row:
                    # Skip trailing empty lines
                    continue
                if len(row) != len(headers):
                    critical_errors.append(
                        f"Line {row_idx}: Column count mismatch (expected {len(headers)}, found {len(row)})."
                    )
                records.append(dict(zip(headers, row)))
    except Exception as exc:
        print(f"[CRITICAL ERROR] Failed to read CSV: {exc}")
        return 1

    total_records = len(records)
    print(f"Total Records   : {total_records}")
    print(f"Total Columns   : {len(headers)}")
    print(f"Columns Found   : {', '.join(headers)}\n")

    # 3. Column schema validation
    missing_expected_cols = [col for col in EXPECTED_COLUMNS if col not in headers]
    unexpected_cols = [col for col in headers if col not in EXPECTED_COLUMNS]

    if missing_expected_cols:
        critical_errors.append(
            f"Missing required columns: {missing_expected_cols}"
        )
    if unexpected_cols:
        warnings.append(f"Unexpected extra columns: {unexpected_cols}")

    if total_records == 0:
        print("[CRITICAL ERROR] CSV contains header row but zero data records.")
        return 1

    # 4. Field validation, missing values, duplicates, and type parsing
    missing_per_col = {col: 0 for col in headers}
    duplicates_exact = 0
    duplicate_timestamps = 0
    seen_rows = set()
    seen_timestamps = set()

    parsed_response_times = []
    parsed_expected_delays = []
    parsed_status_codes = []
    scenarios = []
    connection_error_records = []
    scenario_inconsistencies = []

    scenario_metrics = {}

    for idx, rec in enumerate(records, start=1):
        # Exact row duplicate check
        row_tuple = tuple(rec.get(h, "") for h in headers)
        if row_tuple in seen_rows:
            duplicates_exact += 1
        else:
            seen_rows.add(row_tuple)

        # Missing values check
        for col in headers:
            val = rec.get(col, "")
            # 'error' is legitimately empty for successful requests
            if val is None or val == "":
                if col != "error":
                    missing_per_col[col] += 1
            elif val.strip() == "":
                if col != "error":
                    missing_per_col[col] += 1

        # Timestamp check
        ts_str = rec.get("timestamp_utc", "")
        if ts_str:
            if ts_str in seen_timestamps:
                duplicate_timestamps += 1
            else:
                seen_timestamps.add(ts_str)
            try:
                datetime.fromisoformat(ts_str)
            except ValueError:
                critical_errors.append(
                    f"Record {idx}: Invalid ISO-8601 timestamp '{ts_str}'."
                )

        # Expected delay check
        delay_val = rec.get("expected_delay_ms", "")
        parsed_delay = None
        try:
            parsed_delay = int(delay_val)
            if parsed_delay < 0:
                critical_errors.append(
                    f"Record {idx}: Negative expected_delay_ms '{delay_val}'."
                )
            parsed_expected_delays.append(parsed_delay)
        except ValueError:
            critical_errors.append(
                f"Record {idx}: Non-integer expected_delay_ms '{delay_val}'."
            )

        # Status code check
        status_val = rec.get("status_code", "")
        parsed_status = None
        if not status_val or status_val.lower() == "none":
            connection_error_records.append(idx)
        else:
            try:
                parsed_status = int(status_val)
                parsed_status_codes.append(parsed_status)
            except ValueError:
                critical_errors.append(
                    f"Record {idx}: Non-integer status_code '{status_val}'."
                )

        # Response time check
        resp_val = rec.get("measured_response_time_ms", "")
        parsed_resp = None
        try:
            parsed_resp = float(resp_val)
            if parsed_resp < 0:
                critical_errors.append(
                    f"Record {idx}: Negative measured_response_time_ms '{resp_val}'."
                )
            else:
                parsed_response_times.append(parsed_resp)
        except ValueError:
            critical_errors.append(
                f"Record {idx}: Non-float measured_response_time_ms '{resp_val}'."
            )

        # Scenario consistency check
        sc_name = rec.get("scenario", "")
        path_val = rec.get("path", "")
        error_val = rec.get("error", "")
        scenarios.append(sc_name)

        if parsed_resp is not None:
            if sc_name not in scenario_metrics:
                scenario_metrics[sc_name] = []
            scenario_metrics[sc_name].append(parsed_resp)

        # Validate scenario expectations
        if sc_name == "normal_health":
            if path_val != "/health" or parsed_status != 200:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): expected /health & 200, got {path_val} & {parsed_status}"
                )
        elif sc_name == "normal_products":
            if path_val != "/products/" or parsed_status != 200:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): expected /products/ & 200, got {path_val} & {parsed_status}"
                )
        elif sc_name in ("normal_latency", "slow_latency"):
            if path_val != "/debug/test-latency" or parsed_status != 200:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): expected /debug/test-latency & 200, got {path_val} & {parsed_status}"
                )
            if (
                parsed_delay is not None
                and parsed_resp is not None
                and parsed_resp < (parsed_delay - 50)
            ):
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): measured time ({parsed_resp}ms) significantly lower than delay ({parsed_delay}ms)"
                )
        elif sc_name == "invalid_delay":
            if path_val != "/debug/test-latency" or parsed_status != 400:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): expected /debug/test-latency & 400, got {path_val} & {parsed_status}"
                )
            if not error_val:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): HTTP 400 has missing error message."
                )
        elif sc_name == "unknown_endpoint":
            if path_val != "/does-not-exist" or parsed_status != 404:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): expected /does-not-exist & 404, got {path_val} & {parsed_status}"
                )
            if not error_val:
                scenario_inconsistencies.append(
                    f"Record {idx} ({sc_name}): HTTP 404 has missing error message."
                )

    # 5. Output missing values table
    print("-" * 72)
    print(" 1. MISSING OR BLANK VALUES PER COLUMN")
    print("-" * 72)
    for col in headers:
        count = missing_per_col[col]
        pct = (count / total_records * 100) if total_records else 0
        note = " (error message is empty for 200 OK)" if col == "error" else ""
        print(f"  - {col:<26}: {count:>3} missing ({pct:>5.1f}%){note}")

    blank_errors = sum(1 for r in records if not r.get("error", "").strip())
    populated_errors = total_records - blank_errors
    print(f"  * Note on 'error' column   : {populated_errors} populated (for errors), {blank_errors} blank (for HTTP 200)")

    # 6. Duplicates
    print("\n" + "-" * 72)
    print(" 2. DUPLICATE ANALYSIS")
    print("-" * 72)
    print(f"  - Exact Duplicate Rows     : {duplicates_exact}")
    print(f"  - Duplicate Timestamps     : {duplicate_timestamps}")

    # 7. HTTP status code distribution
    print("\n" + "-" * 72)
    print(" 3. HTTP STATUS-CODE DISTRIBUTION")
    print("-" * 72)
    status_counter = Counter(parsed_status_codes)
    for code, count in sorted(status_counter.items()):
        pct = (count / total_records * 100) if total_records else 0
        print(f"  - HTTP {code:<4} : {count:>3} records ({pct:>5.1f}%)")
    if connection_error_records:
        conn_count = len(connection_error_records)
        pct = (conn_count / total_records * 100) if total_records else 0
        print(f"  - Connection Errors (None) : {conn_count:>3} records ({pct:>5.1f}%)")
    else:
        print("  - Connection Errors (None) :   0 records (0.0%)")

    # 8. Scenario distribution
    print("\n" + "-" * 72)
    print(" 4. SCENARIO DISTRIBUTION")
    print("-" * 72)
    scenario_counter = Counter(scenarios)
    for sc, count in sorted(scenario_counter.items()):
        pct = (count / total_records * 100) if total_records else 0
        print(f"  - {sc:<22} : {count:>3} records ({pct:>5.1f}%)")

    # 9. Response-time statistics
    print("\n" + "-" * 72)
    print(" 5. RESPONSE-TIME MEASUREMENTS (ms)")
    print("-" * 72)
    if parsed_response_times:
        overall_min = min(parsed_response_times)
        overall_avg = mean(parsed_response_times)
        overall_med = median(parsed_response_times)
        overall_max = max(parsed_response_times)
        print(f"  - Overall Minimum : {overall_min:>8.2f} ms")
        print(f"  - Overall Average : {overall_avg:>8.2f} ms")
        print(f"  - Overall Median  : {overall_med:>8.2f} ms")
        print(f"  - Overall Maximum : {overall_max:>8.2f} ms")

        print("\n  Per-Scenario Response Times:")
        print(f"  {'Scenario':<20} {'Count':>6} {'Min (ms)':>10} {'Mean (ms)':>10} {'Median (ms)':>12} {'Max (ms)':>10}")
        print("  " + "-" * 68)
        for sc, times in sorted(scenario_metrics.items()):
            sc_min = min(times)
            sc_mean = mean(times)
            sc_med = median(times)
            sc_max = max(times)
            print(
                f"  {sc:<20} {len(times):>6} {sc_min:>10.2f} {sc_mean:>10.2f} {sc_med:>12.2f} {sc_max:>10.2f}"
            )
    else:
        print("  [WARN] No valid numerical response times found.")

    # 10. Consistency & Error Summary
    print("\n" + "-" * 72)
    print(" 6. DATA CONSISTENCY & QUALITY ISSUES")
    print("-" * 72)
    if scenario_inconsistencies:
        print(f"  [!] Found {len(scenario_inconsistencies)} scenario inconsistencies:")
        for inc in scenario_inconsistencies:
            print(f"      - {inc}")
    else:
        print("  [✓] All scenarios match expected HTTP status codes, paths, and delay bounds.")

    if critical_errors:
        print(f"\n  [!] {len(critical_errors)} CRITICAL ERRORS detected:")
        for err in critical_errors[:10]:
            print(f"      - {err}")
        if len(critical_errors) > 10:
            print(f"      ... and {len(critical_errors) - 10} more.")
    else:
        print("  [✓] Zero critical formatting, missing column, or numeric type errors.")

    # Warning on dataset size for ML
    if total_records < 500:
        print(f"\n  [!] NOTICE FOR PHASE 4 (LSTM MODELING):")
        print(f"      Current dataset size ({total_records} rows) is adequate for ingestion validation,")
        print("      but insufficient for LSTM sequence modeling. Recommended minimum is 1,000–3,000 events.")

    print("\n" + "=" * 72)
    if critical_errors:
        print(" VALIDATION STATUS: FAILED (Critical errors present)")
        print("=" * 72)
        return 1
    else:
        print(" VALIDATION STATUS: PASSED (Dataset is structurally valid)")
        print("=" * 72)
        return 0


def main():
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    exit_code = inspect_dataset(csv_file)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
