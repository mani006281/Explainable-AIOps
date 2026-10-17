import csv
import sys
from pathlib import Path


ML_DIR = Path(__file__).resolve().parent

NORMAL_RESULTS = (
    ML_DIR / "normal_test_feature_analysis_v1"
    / "normal_test_window_results.csv"
)

NORMAL_FEATURE_ERRORS = (
    ML_DIR / "normal_test_feature_analysis_v1"
    / "normal_test_feature_errors.csv"
)

SLOW_RESULTS = (
    ML_DIR / "slow_traffic_explanations_v1"
    / "slow_window_results.csv"
)

SLOW_EXPLANATIONS = (
    ML_DIR / "slow_traffic_explanations_v1"
    / "slow_window_explanations.csv"
)

SLOW_FEATURE_CONTRIBUTIONS = (
    ML_DIR / "slow_traffic_explanations_v1"
    / "slow_window_feature_contributions.csv"
)

OUTPUT_DIR = ML_DIR / "consolidated_anomaly_report_v1"
OUTPUT_FILE = OUTPUT_DIR / "consolidated_anomaly_report.csv"

FEATURE_COLUMNS = [
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

OUTPUT_COLUMNS = [
    "traffic_type",
    "window_id",
    "start_timestamp",
    "end_timestamp",
    "mean_response_time_ms",
    "max_response_time_ms",
    "reconstruction_error",
    "threshold",
    "threshold_exceedance",
    "is_anomaly",
    "top_contributing_features",
    "explanation",
]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def as_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def as_bool(value):
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Unrecognized Boolean value: {value!r}")


def format_features(items):
    return "; ".join(
        f"{feature} ({percentage:.2f}%)"
        for feature, percentage in items
    )


def normal_feature_ranking(row):
    values = []

    for feature in FEATURE_COLUMNS:
        error = as_float(row.get(feature))

        if error is not None:
            values.append((feature, max(error, 0.0)))

    total = sum(value for _, value in values)

    if total <= 0:
        return []

    percentages = [
        (feature, value / total * 100.0)
        for feature, value in values
    ]

    return sorted(percentages, key=lambda item: item[1], reverse=True)


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite it:\n{OUTPUT_FILE}\n"
            "Review or rename the existing report before running again."
        )

    normal_results = read_csv(NORMAL_RESULTS)
    normal_errors = read_csv(NORMAL_FEATURE_ERRORS)
    slow_results = read_csv(SLOW_RESULTS)
    slow_explanations = read_csv(SLOW_EXPLANATIONS)
    slow_contributions = read_csv(SLOW_FEATURE_CONTRIBUTIONS)

    if len(normal_results) != 81:
        raise ValueError(
            f"Expected 81 normal test windows; found {len(normal_results)}."
        )

    if len(normal_errors) != 81:
        raise ValueError(
            f"Expected 81 normal feature-error rows; found {len(normal_errors)}."
        )

    if len(slow_results) != 31:
        raise ValueError(
            f"Expected 31 slow traffic windows; found {len(slow_results)}."
        )

    if len(slow_explanations) != 31:
        raise ValueError(
            f"Expected 31 slow explanations; found {len(slow_explanations)}."
        )

    normal_result_by_id = {
        row["window_id"]: row for row in normal_results
    }

    normal_error_by_id = {
        row["window_id"]: row for row in normal_errors
    }

    slow_explanation_by_window = {
        row["window_number"]: row for row in slow_explanations
    }

    # Build the slow-traffic feature rankings directly from the
    # existing per-window contribution percentages.
    slow_rankings = {}

    for row in slow_contributions:
        window_id = str(row["window_number"])
        feature = row["feature"]
        percentage = as_float(row["contribution_percentage"])

        if percentage is not None:
            slow_rankings.setdefault(window_id, []).append(
                (feature, percentage)
            )

    for window_id in slow_rankings:
        slow_rankings[window_id].sort(
            key=lambda item: item[1],
            reverse=True
        )

    consolidated = []

    # Normal test windows
    for result in normal_results:
        window_id = result["window_id"]

        if window_id not in normal_error_by_id:
            raise ValueError(
                f"Missing feature-error record for normal window {window_id}"
            )

        feature_row = normal_error_by_id[window_id]
        error = float(result["reconstruction_error"])
        threshold = float(result["threshold"])
        flagged = as_bool(result["predicted_anomaly"])

        ranking = normal_feature_ranking(feature_row)
        top_three = ranking[:3]
        feature_text = format_features(top_three)

        if flagged:
            explanation = (
                f"Normal test window {window_id} was flagged because its "
                f"reconstruction error ({error:.8f}) exceeded the existing "
                f"threshold ({threshold:.8f}). "
                f"Leading feature-level reconstruction-error contributions: "
                f"{feature_text}. These contributions describe model "
                f"reconstruction error and do not prove causal root causes."
            )
        else:
            explanation = (
                f"Normal test window {window_id} was not flagged because its "
                f"reconstruction error ({error:.8f}) did not exceed the "
                f"existing threshold ({threshold:.8f}). "
                f"Largest feature-level reconstruction-error contributions: "
                f"{feature_text}. These contributions describe model "
                f"reconstruction error and do not prove causal root causes."
            )

        consolidated.append({
            "traffic_type": "normal_test",
            "window_id": window_id,
            "start_timestamp": "",
            "end_timestamp": "",
            "mean_response_time_ms": "",
            "max_response_time_ms": "",
            "reconstruction_error": error,
            "threshold": threshold,
            "threshold_exceedance": error - threshold,
            "is_anomaly": flagged,
            "top_contributing_features": feature_text,
            "explanation": explanation,
        })

    # Slow traffic windows
    for result in slow_results:
        window_number = str(result["window_number"])

        if window_number not in slow_explanation_by_window:
            raise ValueError(
                f"Missing explanation for slow window {window_number}"
            )

        explanation_row = slow_explanation_by_window[window_number]
        error = float(result["reconstruction_error"])
        threshold = float(result["threshold"])
        flagged = as_bool(result["is_anomaly_candidate"])

        top_three = slow_rankings.get(window_number, [])[:3]
        feature_text = format_features(top_three)

        # Prefer the existing explanation text and contribution summary.
        existing_feature_text = explanation_row.get(
            "top_contributing_features", ""
        )
        existing_explanation = explanation_row.get("explanation", "")

        if existing_feature_text.strip():
            feature_text = existing_feature_text

        if not existing_explanation.strip():
            existing_explanation = (
                f"Slow traffic window {window_number} has reconstruction "
                f"error {error:.8f} and threshold {threshold:.8f}. "
                f"Leading feature contributions: {feature_text}. "
                f"Feature contributions describe reconstruction error "
                f"and do not establish causal root causes."
            )

        consolidated.append({
            "traffic_type": "slow_traffic",
            "window_id": f"slow_{int(window_number):04d}",
            "start_timestamp": result.get("start_timestamp", ""),
            "end_timestamp": result.get("end_timestamp", ""),
            "mean_response_time_ms": result.get(
                "mean_response_time_ms", ""
            ),
            "max_response_time_ms": result.get(
                "max_response_time_ms", ""
            ),
            "reconstruction_error": error,
            "threshold": threshold,
            "threshold_exceedance": (
                as_float(result.get("threshold_exceedance"))
                if result.get("threshold_exceedance", "").strip()
                else error - threshold
            ),
            "is_anomaly": flagged,
            "top_contributing_features": feature_text,
            "explanation": existing_explanation,
        })

    if len(consolidated) != 112:
        raise ValueError(
            f"Expected 112 consolidated rows; got {len(consolidated)}."
        )

    window_ids = [
        (row["traffic_type"], row["window_id"])
        for row in consolidated
    ]

    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Duplicate window identifiers detected.")

    normal_rows = [
        row for row in consolidated
        if row["traffic_type"] == "normal_test"
    ]

    slow_rows = [
        row for row in consolidated
        if row["traffic_type"] == "slow_traffic"
    ]

    normal_flagged = sum(row["is_anomaly"] for row in normal_rows)
    slow_flagged = sum(row["is_anomaly"] for row in slow_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recheck before creating the output to avoid overwriting an existing file.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output appeared during processing; refusing to overwrite: "
            f"{OUTPUT_FILE}"
        )

    with OUTPUT_FILE.open(
        "x", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(consolidated)

    print("\nCONSOLIDATED ANOMALY REPORT CREATED")
    print("=" * 55)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total windows: {len(consolidated)}")
    print(f"Normal test windows: {len(normal_rows)}")
    print(f"Normal test anomalies: {normal_flagged}")
    print(f"Slow traffic windows: {len(slow_rows)}")
    print(f"Slow traffic anomalies: {slow_flagged}")
    print(f"Total flagged windows: {normal_flagged + slow_flagged}")
    print(f"Total not flagged: {len(consolidated) - normal_flagged - slow_flagged}")
    print(f"Output columns: {len(OUTPUT_COLUMNS)}")
    print("\nExisting model artifacts and input reports were not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
