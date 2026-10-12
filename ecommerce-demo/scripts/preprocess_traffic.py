import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DEFAULT_INPUT_CSV = DATA_DIR / "traffic_runs.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "traffic_runs_preprocessed.csv"
DEFAULT_REJECTED_CSV = DATA_DIR / "traffic_runs_rejected.csv"

EXPECTED_COLUMNS = [
    "timestamp_utc",
    "scenario",
    "path",
    "expected_delay_ms",
    "status_code",
    "measured_response_time_ms",
    "error",
]


def parse_and_normalize_timestamp(ts_str: str) -> str:
    """Parses a timestamp string and normalizes it to standard ISO-8601 UTC."""
    if not ts_str or not ts_str.strip():
        raise ValueError("Timestamp is empty")

    clean_str = ts_str.strip()
    dt = datetime.fromisoformat(clean_str)

    # Ensure UTC timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.isoformat()


def preprocess_traffic_data(
    input_path: Path = DEFAULT_INPUT_CSV,
    output_path: Path = DEFAULT_OUTPUT_CSV,
    rejected_path: Path = DEFAULT_REJECTED_CSV,
) -> int:
    """Reads, validates, cleans, and standardizes traffic runs CSV.

    Preserves original dataset, handles experimental metadata,
    and isolates any rejected rows.

    Returns:
        int: 0 on success, 1 on critical failure.
    """
    print("=" * 72)
    print(" EXPLAINABLE-AIOPS: PHASE 3.2 LOG DATA PREPROCESSING PIPELINE")
    print("=" * 72)
    print(f"Source Input CSV   : {input_path.resolve()}")
    print(f"Cleaned Output CSV : {output_path.resolve()}")

    # 1. Verify input file existence and safety
    if not input_path.exists():
        print(f"[CRITICAL ERROR] Source CSV does not exist: {input_path}")
        return 1

    if input_path.resolve() == output_path.resolve():
        print("[CRITICAL ERROR] Output path cannot be the same as input path (protection rule).")
        return 1

    # 2. Read source CSV
    source_records = []
    headers = []

    try:
        with input_path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.reader(infile)
            try:
                headers = [h.strip() for h in next(reader)]
            except StopIteration:
                print("[CRITICAL ERROR] Source CSV is empty.")
                return 1

            for line_no, row in enumerate(reader, start=2):
                if not row or all(c.strip() == "" for c in row):
                    # Skip empty blank rows at EOF
                    continue
                if len(row) != len(headers):
                    print(
                        f"[WARN] Line {line_no} column count mismatch "
                        f"({len(row)} vs {len(headers)} expected)."
                    )
                source_records.append(dict(zip(headers, [c.strip() for c in row])))
    except Exception as exc:
        print(f"[CRITICAL ERROR] Failed to read source CSV: {exc}")
        return 1

    total_source = len(source_records)
    print(f"Source Records Read: {total_source}")
    print(f"Headers Found      : {', '.join(headers)}\n")

    # 3. Check expected columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in headers]
    if missing_cols:
        print(f"[CRITICAL ERROR] Missing expected columns: {missing_cols}")
        return 1

    # 4. Process and clean records
    valid_records = []
    rejected_records = []

    seen_exact_rows = set()
    seen_timestamps = set()
    exact_duplicates_count = 0
    timestamp_duplicates_count = 0

    missing_values_per_col = {col: 0 for col in headers}

    for idx, raw_record in enumerate(source_records, start=1):
        rejection_reasons = []

        # Duplicate checking
        row_signature = tuple(raw_record.get(c, "") for c in EXPECTED_COLUMNS)
        if row_signature in seen_exact_rows:
            exact_duplicates_count += 1
        else:
            seen_exact_rows.add(row_signature)

        # Check required fields (error can be empty for 200 OK)
        for col in EXPECTED_COLUMNS:
            val = raw_record.get(col, "")
            if val == "" and col != "error":
                missing_values_per_col[col] += 1
                rejection_reasons.append(f"Missing required field '{col}'")
            elif col == "error" and val == "":
                missing_values_per_col[col] += 1

        # A. Timestamp normalization
        normalized_timestamp = None
        ts_val = raw_record.get("timestamp_utc", "")
        if ts_val:
            try:
                normalized_timestamp = parse_and_normalize_timestamp(ts_val)
                if normalized_timestamp in seen_timestamps:
                    timestamp_duplicates_count += 1
                else:
                    seen_timestamps.add(normalized_timestamp)
            except Exception as e:
                rejection_reasons.append(f"Invalid timestamp format '{ts_val}': {e}")

        # B. Scenario & Path validation
        scenario_val = raw_record.get("scenario", "").strip()
        path_val = raw_record.get("path", "").strip()

        if not scenario_val:
            rejection_reasons.append("Empty scenario name")
        if not path_val:
            rejection_reasons.append("Empty path")

        # C. Numeric fields validation
        parsed_delay = None
        delay_val = raw_record.get("expected_delay_ms", "")
        try:
            parsed_delay = int(delay_val)
            if parsed_delay < 0:
                rejection_reasons.append(f"Negative expected_delay_ms ({parsed_delay})")
        except (ValueError, TypeError):
            rejection_reasons.append(f"Non-integer expected_delay_ms '{delay_val}'")

        parsed_status = None
        status_val = raw_record.get("status_code", "")
        try:
            parsed_status = int(status_val)
            if not (100 <= parsed_status <= 599):
                rejection_reasons.append(f"Invalid HTTP status code range ({parsed_status})")
        except (ValueError, TypeError):
            rejection_reasons.append(f"Non-integer status_code '{status_val}'")

        parsed_response_time = None
        resp_val = raw_record.get("measured_response_time_ms", "")
        try:
            parsed_response_time = float(resp_val)
            if parsed_response_time < 0.0:
                rejection_reasons.append(f"Negative response time ({parsed_response_time} ms)")
        except (ValueError, TypeError):
            rejection_reasons.append(f"Non-float measured_response_time_ms '{resp_val}'")

        error_val = raw_record.get("error", "").strip()

        # D. Decide acceptance or rejection
        if rejection_reasons:
            rejected_entry = dict(raw_record)
            rejected_entry["source_row_index"] = idx
            rejected_entry["rejection_reasons"] = "; ".join(rejection_reasons)
            rejected_records.append(rejected_entry)
        else:
            # Clean standardized record
            clean_record = {
                "timestamp_utc": normalized_timestamp,
                "scenario": scenario_val,
                "path": path_val,
                "expected_delay_ms": parsed_delay,
                "status_code": parsed_status,
                "measured_response_time_ms": round(parsed_response_time, 2),
                "error": error_val,
            }
            valid_records.append(clean_record)

    # 5. Sort valid records chronologically
    valid_records.sort(key=lambda r: r["timestamp_utc"])

    # 6. Write preprocessed CSV (Deterministic fresh write)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(valid_records)

    print(f"[✓] Preprocessed dataset written successfully to: {output_path}")

    # 7. Write rejected CSV if any rejected records exist
    if rejected_records:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_headers = EXPECTED_COLUMNS + ["source_row_index", "rejection_reasons"]
        with rejected_path.open("w", encoding="utf-8", newline="") as rej_file:
            writer = csv.DictWriter(rej_file, fieldnames=rejected_headers)
            writer.writeheader()
            writer.writerows(rejected_records)
        print(f"[!] {len(rejected_records)} rejected records written to: {rejected_path}")
    else:
        # If no rejected records, ensure no stale rejected file lingers
        if rejected_path.exists():
            rejected_path.unlink()
        print("[✓] Zero rejected records. No rejected CSV needed.")

    # 8. Compute and display statistics
    print("\n" + "-" * 72)
    print(" PREPROCESSING PIPELINE SUMMARY STATISTICS")
    print("-" * 72)
    print(f"  - Total Source Records    : {total_source}")
    print(f"  - Valid Processed Records : {len(valid_records)}")
    print(f"  - Rejected Records        : {len(rejected_records)}")
    print(f"  - Exact Duplicate Rows    : {exact_duplicates_count}")
    print(f"  - Duplicate Timestamps    : {timestamp_duplicates_count}")

    print("\n  Missing / Blank Field Counts:")
    for col, count in missing_values_per_col.items():
        note = " (normal for successful requests)" if col == "error" else ""
        print(f"    * {col:<26} : {count:>3} blank{note}")

    status_counts = Counter(r["status_code"] for r in valid_records)
    print("\n  HTTP Status Code Distribution:")
    for code, cnt in sorted(status_counts.items()):
        pct = (cnt / len(valid_records) * 100) if valid_records else 0
        print(f"    * HTTP {code:<4} : {cnt:>3} records ({pct:>5.1f}%)")

    scenario_counts = Counter(r["scenario"] for r in valid_records)
    print("\n  Scenario Distribution:")
    for sc, cnt in sorted(scenario_counts.items()):
        pct = (cnt / len(valid_records) * 100) if valid_records else 0
        print(f"    * {sc:<22} : {cnt:>3} records ({pct:>5.1f}%)")

    response_times = [r["measured_response_time_ms"] for r in valid_records]
    if response_times:
        print("\n  Response Time Measurements (ms):")
        print(f"    * Minimum : {min(response_times):>8.2f} ms")
        print(f"    * Average : {mean(response_times):>8.2f} ms")
        print(f"    * Median  : {median(response_times):>8.2f} ms")
        print(f"    * Maximum : {max(response_times):>8.2f} ms")

    print("\n" + "=" * 72)
    print(" PREPROCESSING STATUS: COMPLETED SUCCESSFULLY")
    print("=" * 72)
    return 0


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_CSV
    rej = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_REJECTED_CSV
    code = preprocess_traffic_data(inp, out, rej)
    sys.exit(code)


if __name__ == "__main__":
    main()
