import csv
import math
from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

INPUT_PREPROCESSED_CSV = DATA_DIR / "traffic_runs_preprocessed.csv"
OUTPUT_FEATURES_CSV = DATA_DIR / "traffic_features.csv"
OUTPUT_METADATA_CSV = DATA_DIR / "traffic_feature_metadata.csv"

# Feature columns for model input (strictly numerical, no experiment metadata)
FEATURE_COLUMNS = [
    "timestamp_utc",
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

# Aligned metadata columns for evaluation and post-detection analysis
METADATA_COLUMNS = [
    "timestamp_utc",
    "scenario",
    "expected_delay_ms",
    "path",
    "status_code",
    "measured_response_time_ms",
    "error",
]


def extract_features_and_metadata(
    input_path: Path = INPUT_PREPROCESSED_CSV,
    features_path: Path = OUTPUT_FEATURES_CSV,
    metadata_path: Path = OUTPUT_METADATA_CSV,
) -> int:
    """Reads preprocessed traffic runs, extracts numerical model features,

    and creates aligned metadata while preventing data leakage.

    Returns:
        int: 0 on success, 1 on critical failure.
    """
    print("=" * 72)
    print(" EXPLAINABLE-AIOPS: PHASE 3.3 FEATURE ENGINEERING PIPELINE")
    print("=" * 72)
    print(f"Input Preprocessed CSV : {input_path.resolve()}")
    print(f"Output Features CSV    : {features_path.resolve()}")
    print(f"Output Metadata CSV    : {metadata_path.resolve()}\n")

    # 1. Verify input file existence
    if not input_path.exists():
        print(f"[CRITICAL ERROR] Input preprocessed CSV does not exist: {input_path}")
        return 1

    source_records = []
    try:
        with input_path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row:
                    source_records.append(row)
    except Exception as exc:
        print(f"[CRITICAL ERROR] Failed to read preprocessed CSV: {exc}")
        return 1

    total_source = len(source_records)
    print(f"Source Records Read: {total_source}")

    if total_source == 0:
        print("[CRITICAL ERROR] Source dataset is empty.")
        return 1

    feature_rows = []
    metadata_rows = []
    errors = []

    for idx, row in enumerate(source_records, start=1):
        # A. Timestamp validation
        ts = row.get("timestamp_utc", "").strip()
        if not ts:
            errors.append(f"Row {idx}: Missing timestamp_utc.")

        # B. Numerical response time & log transformation
        resp_str = row.get("measured_response_time_ms", "")
        try:
            resp_ms = float(resp_str)
            if resp_ms < 0.0:
                errors.append(f"Row {idx}: Negative response time ({resp_ms}).")
            log_resp = math.log1p(resp_ms)
        except (ValueError, TypeError):
            errors.append(
                f"Row {idx}: Non-numeric measured_response_time_ms '{resp_str}'."
            )
            resp_ms = 0.0
            log_resp = 0.0

        # C. HTTP Status One-Hot Encoding
        status_str = row.get("status_code", "")
        status_200 = 0
        status_400 = 0
        status_404 = 0
        status_other = 0
        is_http_error = 0
        is_connection_error = 0

        if not status_str or status_str.lower() == "none":
            is_connection_error = 1
            status_other = 1
        else:
            try:
                status_code = int(status_str)
                if status_code == 200:
                    status_200 = 1
                elif status_code == 400:
                    status_400 = 1
                    is_http_error = 1
                elif status_code == 404:
                    status_404 = 1
                    is_http_error = 1
                else:
                    status_other = 1
                    if status_code >= 400:
                        is_http_error = 1
            except ValueError:
                errors.append(f"Row {idx}: Invalid status code '{status_str}'.")
                status_other = 1

        # D. Endpoint One-Hot Encoding
        path = row.get("path", "").strip()
        endpoint_health = 0
        endpoint_products = 0
        endpoint_latency = 0
        endpoint_other = 0

        if path == "/health":
            endpoint_health = 1
        elif path == "/products/":
            endpoint_products = 1
        elif path == "/debug/test-latency":
            endpoint_latency = 1
        else:
            endpoint_other = 1

        # E. Verify all feature values are finite numbers
        numerical_values = [
            resp_ms,
            log_resp,
            status_200,
            status_400,
            status_404,
            status_other,
            endpoint_health,
            endpoint_products,
            endpoint_latency,
            endpoint_other,
            is_connection_error,
            is_http_error,
        ]
        if not all(math.isfinite(v) for v in numerical_values):
            errors.append(f"Row {idx}: Non-finite numerical value detected.")

        # F. Build Feature Row (Excludes scenario and expected_delay_ms)
        feature_row = {
            "timestamp_utc": ts,
            "response_time_ms": round(resp_ms, 2),
            "log_response_time": round(log_resp, 4),
            "status_200": status_200,
            "status_400": status_400,
            "status_404": status_404,
            "status_other": status_other,
            "endpoint_health": endpoint_health,
            "endpoint_products": endpoint_products,
            "endpoint_latency": endpoint_latency,
            "endpoint_other": endpoint_other,
            "is_connection_error": is_connection_error,
            "is_http_error": is_http_error,
        }
        feature_rows.append(feature_row)

        # G. Build Metadata Row (Preserves experimental context for evaluation)
        metadata_row = {
            "timestamp_utc": ts,
            "scenario": row.get("scenario", "").strip(),
            "expected_delay_ms": row.get("expected_delay_ms", "").strip(),
            "path": path,
            "status_code": status_str,
            "measured_response_time_ms": round(resp_ms, 2),
            "error": row.get("error", "").strip(),
        }
        metadata_rows.append(metadata_row)

    if errors:
        print(f"[CRITICAL ERROR] Validation failed with {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  - {err}")
        return 1

    # Ensure consistent row order (chronological by timestamp)
    feature_rows.sort(key=lambda r: r["timestamp_utc"])
    metadata_rows.sort(key=lambda r: r["timestamp_utc"])

    # 4. Write Feature CSV
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with features_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(feature_rows)

    # 5. Write Metadata CSV
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as m_out:
        writer = csv.DictWriter(m_out, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"[✓] Feature dataset written successfully  : {features_path}")
    print(f"[✓] Metadata dataset written successfully : {metadata_path}")

    # 6. Verification and Reporting
    print("\n" + "-" * 72)
    print(" VERIFICATION & FEATURE MATRIX SUMMARY")
    print("-" * 72)
    print(f"  - Source Records Read      : {total_source}")
    print(f"  - Feature Records Written  : {len(feature_rows)}")
    print(f"  - Metadata Records Written : {len(metadata_rows)}")
    print(f"  - Row Alignment Match      : {len(feature_rows) == len(metadata_rows)}")

    # Check 1-to-1 timestamp alignment
    misaligned = sum(
        1
        for f, m in zip(feature_rows, metadata_rows)
        if f["timestamp_utc"] != m["timestamp_utc"]
    )
    print(f"  - Timestamp Misalignments  : {misaligned}")

    print("\n  Feature Matrix Columns (12 model features + 1 timestamp):")
    for col in FEATURE_COLUMNS:
        print(f"    * {col}")

    print("\n  Feature Distributions:")
    print(f"    * status_200 active     : {sum(r['status_200'] for r in feature_rows)}")
    print(f"    * status_400 active     : {sum(r['status_400'] for r in feature_rows)}")
    print(f"    * status_404 active     : {sum(r['status_404'] for r in feature_rows)}")
    print(f"    * is_http_error active  : {sum(r['is_http_error'] for r in feature_rows)}")
    print(f"    * endpoint_health       : {sum(r['endpoint_health'] for r in feature_rows)}")
    print(f"    * endpoint_products     : {sum(r['endpoint_products'] for r in feature_rows)}")
    print(f"    * endpoint_latency      : {sum(r['endpoint_latency'] for r in feature_rows)}")
    print(f"    * endpoint_other        : {sum(r['endpoint_other'] for r in feature_rows)}")

    log_times = [r["log_response_time"] for r in feature_rows]
    print("\n  Transformed Latency Statistics (log1p):")
    print(f"    * Min log(1+ms) : {min(log_times):.4f}")
    print(f"    * Avg log(1+ms) : {sum(log_times)/len(log_times):.4f}")
    print(f"    * Max log(1+ms) : {max(log_times):.4f}")

    print("\n" + "=" * 72)
    print(" FEATURE ENGINEERING COMPLETED: SUCCESS")
    print("=" * 72)
    return 0


def main():
    code = extract_features_and_metadata()
    sys.exit(code)


if __name__ == "__main__":
    main()
