
"""
Explainable AIOps
Generate a human-readable explanation for a flagged window.

Uses existing diagnostic CSV and JSON files.
Does not load, retrain, or modify the model.
"""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent

INPUT_DIR = ML_DIR / "flagged_window_diagnostic_v1"
OUTPUT_DIR = ML_DIR / "anomaly_explanation_v1"

SUMMARY_FILE = INPUT_DIR / "diagnostic_summary.json"
FEATURE_FILE = INPUT_DIR / "feature_error_summary_test_0036.csv"
DETAIL_FILE = INPUT_DIR / "actual_vs_reconstructed_test_0036.csv"

WINDOW_ID = "test_0036"


# ---------------------------------------------------------
# 2. Validate inputs
# ---------------------------------------------------------

for path in [SUMMARY_FILE, FEATURE_FILE, DETAIL_FILE]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Inspect the existing results before generating another report."
    )

with open(SUMMARY_FILE, "r", encoding="utf-8") as file:
    summary = json.load(file)

features = pd.read_csv(FEATURE_FILE)
details = pd.read_csv(DETAIL_FILE)

if summary.get("window_id") != WINDOW_ID:
    raise ValueError("Unexpected window ID in diagnostic summary.")

required_feature_columns = {
    "feature",
    "mean_squared_scaled_error",
    "contribution_percent",
}

if not required_feature_columns.issubset(features.columns):
    raise ValueError("Feature summary is missing required columns.")

required_detail_columns = {
    "window_id",
    "time_step",
    "feature",
    "actual_value",
    "reconstructed_value",
    "absolute_scaled_error",
}

if not required_detail_columns.issubset(details.columns):
    raise ValueError("Detailed comparison file is missing required columns.")

if not (details["window_id"].astype(str) == WINDOW_ID).all():
    raise ValueError("Unexpected window IDs in detailed comparison.")

if features.empty or details.empty:
    raise ValueError("Diagnostic inputs contain no records.")


# ---------------------------------------------------------
# 3. Calculate score and threshold information
# ---------------------------------------------------------

score = float(summary["reconstruction_error"])
threshold = float(summary["threshold"])
exceedance = score - threshold

is_anomaly = bool(summary["predicted_anomaly"])

exceedance_percent = (
    exceedance / threshold * 100
    if threshold > 0
    else None
)


# ---------------------------------------------------------
# 4. Rank contributing features
# ---------------------------------------------------------

features = features.sort_values(
    "mean_squared_scaled_error",
    ascending=False,
).reset_index(drop=True)

top_features = features.head(5).copy()

top_features["contribution_percent"] = pd.to_numeric(
    top_features["contribution_percent"],
    errors="raise",
)

top_feature_names = top_features["feature"].tolist()


# ---------------------------------------------------------
# 5. Find largest individual reconstruction discrepancies
# ---------------------------------------------------------

details["absolute_scaled_error"] = pd.to_numeric(
    details["absolute_scaled_error"],
    errors="raise",
)

details["time_step"] = pd.to_numeric(
    details["time_step"],
    errors="raise",
)

largest_discrepancies = (
    details.sort_values(
        "absolute_scaled_error",
        ascending=False,
    )
    .head(10)
    .copy()
)

# ---------------------------------------------------------
# 6. Describe the evidence cautiously
# ---------------------------------------------------------

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

top_set = set(top_feature_names)

has_endpoint_contribution = bool(top_set & endpoint_features)
has_response_contribution = bool(top_set & response_features)

if has_endpoint_contribution and has_response_contribution:
    interpretation = (
        "The leading reconstruction-error contributions involve both "
        "endpoint indicators and response-time features."
    )
elif has_endpoint_contribution:
    interpretation = (
        "The leading reconstruction-error contributions involve "
        "endpoint indicators."
    )
elif has_response_contribution:
    interpretation = (
        "The leading reconstruction-error contributions involve "
        "response-time features."
    )
else:
    interpretation = (
        "The leading feature contributions should be reviewed "
        "to identify the dominant reconstruction mismatches."
    )

if is_anomaly:
    classification = "Anomaly detected"
else:
    classification = "No anomaly detected"

assessment = (
    "Candidate false positive requiring validation against the known "
    "normal test data and relevant operational context."
    if is_anomaly
    else
    "The reconstruction error did not exceed the configured threshold."
)


# ---------------------------------------------------------
# 7. Generate report text
# ---------------------------------------------------------

lines = [
    "EXPLAINABLE AIOPS — ANOMALY EXPLANATION",
    "=" * 60,
    "",
    f"Window ID: {WINDOW_ID}",
    f"Classification: {classification}",
    "",
    "ANOMALY SCORE",
    "-" * 60,
    f"Reconstruction error: {score:.9f}",
    f"Threshold:            {threshold:.9f}",
    f"Threshold exceedance: {exceedance:.9f}",
]

if exceedance_percent is not None:
    lines.append(
        f"Exceedance relative to threshold: {exceedance_percent:.2f}%"
    )

lines.extend([
    "",
    "TOP CONTRIBUTING FEATURES",
    "-" * 60,
])

for rank, row in enumerate(top_features.itertuples(index=False), start=1):
    lines.append(
        f"{rank}. {row.feature}: "
        f"{row.contribution_percent:.2f}% contribution"
    )

lines.extend([
    "",
    "INTERPRETATION",
    "-" * 60,
    interpretation,
    "",
    "LARGEST INDIVIDUAL RECONSTRUCTION DISCREPANCIES",
    "-" * 60,
    "Values below are inverse-scaled feature values. The error column",
    "is measured in the scaled feature space.",
    "",
    f"{'Step':<6}{'Feature':<25}{'Actual':>15}"
    f"{'Reconstructed':>18}{'Scaled abs. error':>20}",
    "-" * 84,
])

for row in largest_discrepancies.itertuples(index=False):
    lines.append(
        f"{int(row.time_step):<6}"
        f"{row.feature:<25}"
        f"{float(row.actual_value):>15.6f}"
        f"{float(row.reconstructed_value):>18.6f}"
        f"{float(row.absolute_scaled_error):>20.6f}"
    )

lines.extend([
    "",
    "ASSESSMENT",
    "-" * 60,
    assessment,
    "",
    "LIMITATIONS",
    "-" * 60,
    "- Feature contributions describe reconstruction error, not causality.",
    "- Reconstructed categorical indicators can be fractional or negative.",
    "- A model alert does not independently establish an operational incident.",
    "- Confirm the data labels and operational context before declaring a false positive.",
    "",
    "Provenance: generated from existing diagnostic CSV/JSON files.",
    "The trained model, scaler, and threshold were not modified.",
])

report_text = "\n".join(lines) + "\n"


# ---------------------------------------------------------
# 8. Save report and supporting tables
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

report_path = OUTPUT_DIR / "explanation_test_0036.txt"
report_path.write_text(report_text, encoding="utf-8")

top_features.to_csv(
    OUTPUT_DIR / "top_features_test_0036.csv",
    index=False,
)

largest_discrepancies.to_csv(
    OUTPUT_DIR / "largest_discrepancies_test_0036.csv",
    index=False,
)

report_json = {
    "window_id": WINDOW_ID,
    "classification": classification,
    "reconstruction_error": score,
    "threshold": threshold,
    "threshold_exceedance": exceedance,
    "exceedance_percent_of_threshold": exceedance_percent,
    "top_features": top_features[
        ["feature", "mean_squared_scaled_error", "contribution_percent"]
    ].to_dict(orient="records"),
    "interpretation": interpretation,
    "assessment": assessment,
    "limitations": [
        "Feature contributions are reconstruction-error attributions, not causal explanations.",
        "Reconstructed categorical indicators may be fractional or negative.",
        "Operational incident status requires independent validation.",
    ],
}

with open(
    OUTPUT_DIR / "explanation_test_0036.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(report_json, file, indent=2)

print(report_text)
print(f"\nReport saved to: {report_path}")
print(f"Supporting files saved to: {OUTPUT_DIR}")

