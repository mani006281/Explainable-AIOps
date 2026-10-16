
"""
Explainable AIOps - Version 2

Generates explanations for every flagged window using existing CSV files.
Does not load or retrain the model.
Does not modify existing analysis or diagnostic outputs.
"""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Configure paths
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent

INPUT_DIR = ML_DIR / "normal_test_feature_analysis_v1"
OUTPUT_DIR = ML_DIR / "anomaly_explanations_v2"

WINDOW_RESULTS_FILE = INPUT_DIR / "normal_test_window_results.csv"
FEATURE_ERRORS_FILE = INPUT_DIR / "normal_test_feature_errors.csv"
SUMMARY_FILE = INPUT_DIR / "analysis_summary.json"


# ---------------------------------------------------------
# 2. Validate input files
# ---------------------------------------------------------

for path in [
    WINDOW_RESULTS_FILE,
    FEATURE_ERRORS_FILE,
    SUMMARY_FILE,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

# Never overwrite previous output.
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists:\n{OUTPUT_DIR}\n"
        "Inspect existing results before running again."
    )

windows = pd.read_csv(WINDOW_RESULTS_FILE)
feature_errors = pd.read_csv(FEATURE_ERRORS_FILE)

with open(SUMMARY_FILE, "r", encoding="utf-8") as file:
    experiment_summary = json.load(file)


# ---------------------------------------------------------
# 3. Validate columns and experiment settings
# ---------------------------------------------------------

required_window_columns = {
    "window_id",
    "reconstruction_error",
    "threshold",
    "predicted_anomaly",
}

required_feature_columns = {
    "window_id",
    "reconstruction_error",
    "predicted_anomaly",
}

if not required_window_columns.issubset(windows.columns):
    raise ValueError("Window results CSV is missing required columns.")

if not required_feature_columns.issubset(feature_errors.columns):
    raise ValueError("Feature errors CSV is missing required columns.")

if windows["window_id"].duplicated().any():
    raise ValueError("Duplicate window IDs found in window results.")

if feature_errors["window_id"].duplicated().any():
    raise ValueError("Duplicate window IDs found in feature errors.")

if len(windows) != experiment_summary.get("test_windows"):
    raise ValueError(
        "Window count does not match the experiment summary."
    )

feature_columns = [
    column
    for column in feature_errors.columns
    if column not in {
        "window_id",
        "reconstruction_error",
        "predicted_anomaly",
    }
]

if len(feature_columns) != experiment_summary.get("feature_count"):
    raise ValueError(
        "Feature count does not match the experiment summary."
    )

# Normalize boolean values safely.
def to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1"}:
        return True

    if normalized in {"false", "0"}:
        return False

    raise ValueError(f"Unexpected prediction value: {value!r}")


windows["predicted_anomaly"] = windows["predicted_anomaly"].map(to_bool)
feature_errors["predicted_anomaly"] = (
    feature_errors["predicted_anomaly"].map(to_bool)
)

windows["reconstruction_error"] = pd.to_numeric(
    windows["reconstruction_error"],
    errors="raise",
)

windows["threshold"] = pd.to_numeric(
    windows["threshold"],
    errors="raise",
)

for column in feature_columns:
    feature_errors[column] = pd.to_numeric(
        feature_errors[column],
        errors="raise",
    )

if not windows["reconstruction_error"].map(
    lambda value: pd.notna(value) and value >= 0
).all():
    raise ValueError("Invalid reconstruction errors found.")

if not windows["threshold"].map(
    lambda value: pd.notna(value) and value > 0
).all():
    raise ValueError("Invalid thresholds found.")

if not feature_errors[feature_columns].notna().all().all():
    raise ValueError("Missing feature-error values found.")

if (feature_errors[feature_columns] < 0).any().any():
    raise ValueError("Feature reconstruction errors cannot be negative.")

# Verify predictions are consistent with score and threshold.
expected_predictions = (
    windows["reconstruction_error"] > windows["threshold"]
)

if not expected_predictions.equals(windows["predicted_anomaly"]):
    raise ValueError(
        "Prediction values do not match reconstruction errors and thresholds."
    )


# ---------------------------------------------------------
# 4. Identify flagged windows
# ---------------------------------------------------------

flagged_windows = windows.loc[
    windows["predicted_anomaly"]
].copy()

if flagged_windows.empty:
    print("No anomalous windows were flagged.")
    print("No explanation reports were generated.")
    raise SystemExit(0)

feature_index = feature_errors.set_index("window_id", drop=False)

missing_ids = set(flagged_windows["window_id"]) - set(feature_index.index)

if missing_ids:
    raise ValueError(
        f"Feature errors missing for flagged windows: {sorted(missing_ids)}"
    )


# ---------------------------------------------------------
# 5. Prepare output directory and summary
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

summary_rows = []

endpoint_features = {
    "endpoint_health",
    "endpoint_products",
    "endpoint_latency",
    "endpoint_other",
}

response_features = {
    "response_time_ms",
    "log_response_time",
}

status_features = {
    "status_200",
    "status_400",
    "status_404",
    "status_other",
}


# ---------------------------------------------------------
# 6. Generate one explanation per flagged window
# ---------------------------------------------------------

for _, window in flagged_windows.iterrows():

    window_id = str(window["window_id"])
    score = float(window["reconstruction_error"])
    threshold = float(window["threshold"])
    exceedance = score - threshold
    exceedance_percent = exceedance / threshold * 100

    row = feature_index.loc[window_id]

    # Each feature's error contribution as a percentage of total
    # feature reconstruction error for this window.
    error_values = row[feature_columns].astype(float)
    total_feature_error = float(error_values.sum())

    if total_feature_error <= 0:
        raise ValueError(
            f"Feature errors sum to zero for {window_id}."
        )

    ranked_features = pd.DataFrame({
        "feature": feature_columns,
        "mean_squared_scaled_error": error_values.values,
    })

    ranked_features["contribution_percent"] = (
        ranked_features["mean_squared_scaled_error"]
        / total_feature_error
        * 100
    )

    ranked_features = ranked_features.sort_values(
        "mean_squared_scaled_error",
        ascending=False,
    ).reset_index(drop=True)

    top_five = ranked_features.head(5).copy()

    leading_names = set(top_five["feature"])

    if leading_names & endpoint_features and leading_names & response_features:
        interpretation = (
            "The leading reconstruction-error contributions involve "
            "both endpoint indicators and response-time features."
        )
    elif leading_names & endpoint_features:
        interpretation = (
            "The leading reconstruction-error contributions involve "
            "endpoint indicators."
        )
    elif leading_names & response_features:
        interpretation = (
            "The leading reconstruction-error contributions involve "
            "response-time features."
        )
    elif leading_names & status_features:
        interpretation = (
            "The leading reconstruction-error contributions involve "
            "HTTP status indicator features."
        )
    else:
        interpretation = (
            "Review the leading feature contributions to understand "
            "which inputs the model reconstructed least accurately."
        )

    exceedance_relative = (
        f"{exceedance_percent:.2f}% above the threshold."
    )

    lines = [
        "EXPLAINABLE AIOPS — AUTOMATIC ANOMALY EXPLANATION",
        "=" * 64,
        "",
        f"Window ID: {window_id}",
        "Classification: Anomaly detected",
        "",
        "ANOMALY SCORE",
        "-" * 64,
        f"Reconstruction error: {score:.9f}",
        f"Threshold:            {threshold:.9f}",
        f"Threshold exceedance: {exceedance:.9f}",
        f"Relative exceedance:  {exceedance_relative}",
        "",
        "TOP CONTRIBUTING FEATURES",
        "-" * 64,
    ]

    for rank, feature_row in enumerate(
        top_five.itertuples(index=False),
        start=1,
    ):
        lines.append(
            f"{rank}. {feature_row.feature}: "
            f"{feature_row.contribution_percent:.2f}% contribution"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 64,
        interpretation,
        "",
        "IMPORTANT LIMITATIONS",
        "-" * 64,
        "- Contributions explain reconstruction error, not causality.",
        "- An anomaly prediction does not prove a production incident.",
        "- Validate alerts against independent operational evidence.",
        "- This script does not determine whether a window is a true or false positive.",
        "- Overlapping windows are not statistically independent.",
        "",
        "Source: existing normal_test_window_results.csv,",
        "normal_test_feature_errors.csv, and analysis_summary.json.",
        "No model retraining was performed.",
    ])

    report_text = "\n".join(lines) + "\n"

    # Save a separate text and JSON report for each flagged window.
    (OUTPUT_DIR / f"explanation_{window_id}.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    explanation_json = {
        "experiment": experiment_summary.get("experiment"),
        "dataset": experiment_summary.get("dataset"),
        "window_id": window_id,
        "classification": "anomaly_detected",
        "reconstruction_error": score,
        "threshold": threshold,
        "threshold_exceedance": exceedance,
        "threshold_exceedance_percent": exceedance_percent,
        "top_features": top_five.to_dict(orient="records"),
        "interpretation": interpretation,
        "limitations": [
            "Feature contributions are reconstruction-error attributions, not causal explanations.",
            "An anomaly prediction does not independently establish an operational incident.",
            "The script does not determine ground-truth incident status.",
            "Overlapping windows are not statistically independent.",
        ],
    }

    with open(
        OUTPUT_DIR / f"explanation_{window_id}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(explanation_json, file, indent=2)

    # Save all feature contributions for this window.
    ranked_features.to_csv(
        OUTPUT_DIR / f"feature_contributions_{window_id}.csv",
        index=False,
    )

    summary_rows.append({
        "window_id": window_id,
        "reconstruction_error": score,
        "threshold": threshold,
        "threshold_exceedance": exceedance,
        "threshold_exceedance_percent": exceedance_percent,
        "top_feature_1": top_five.iloc[0]["feature"],
        "top_feature_1_contribution_percent": (
            top_five.iloc[0]["contribution_percent"]
        ),
        "top_feature_2": top_five.iloc[1]["feature"],
        "top_feature_2_contribution_percent": (
            top_five.iloc[1]["contribution_percent"]
        ),
        "top_feature_3": top_five.iloc[2]["feature"],
        "top_feature_3_contribution_percent": (
            top_five.iloc[2]["contribution_percent"]
        ),
    })

    print(f"Generated explanation for {window_id}")
    print(f"  Reconstruction error: {score:.9f}")
    print(f"  Threshold:            {threshold:.9f}")
    print("  Top features:")

    for feature_row in top_five.itertuples(index=False):
        print(
            f"    {feature_row.feature}: "
            f"{feature_row.contribution_percent:.2f}%"
        )

    print()


# ---------------------------------------------------------
# 7. Save experiment-wide explanation summary
# ---------------------------------------------------------

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "flagged_windows_explanation_summary.csv",
    index=False,
)

run_summary = {
    "experiment": experiment_summary.get("experiment"),
    "dataset": experiment_summary.get("dataset"),
    "total_test_windows": int(len(windows)),
    "flagged_windows": int(len(flagged_windows)),
    "explanations_generated": int(len(summary_rows)),
    "output_directory": str(OUTPUT_DIR),
    "model_retrained": False,
}

with open(
    OUTPUT_DIR / "generation_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(run_summary, file, indent=2)

print("=" * 64)
print("ANOMALY EXPLANATION GENERATION COMPLETED")
print("=" * 64)
print(f"Total test windows:       {len(windows)}")
print(f"Flagged windows:          {len(flagged_windows)}")
print(f"Explanations generated:   {len(summary_rows)}")
print(f"Output directory:         {OUTPUT_DIR}")
print("Model retrained:          No")

