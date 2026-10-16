import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ML_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ML_DIR / "artifacts_normal_extended_v1"
EVALUATION_DIR = ML_DIR / "evaluation_normal_extended_v1"

NORMAL_FILE = ARTIFACT_DIR / "test_reconstruction_errors.csv"
SLOW_FILE = EVALUATION_DIR / "slow_traffic_results.csv"
THRESHOLD_FILE = ARTIFACT_DIR / "anomaly_threshold.json"
EXPERIMENT_FILE = ARTIFACT_DIR / "experiment_summary.json"

OUTPUT_DIR = EVALUATION_DIR / "comparison_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "normal_vs_slow_comparison.csv"
JSON_OUTPUT = OUTPUT_DIR / "normal_vs_slow_comparison.json"
PDF_OUTPUT = OUTPUT_DIR / "normal_vs_slow_comparison.pdf"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def to_float(value, column_name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid numeric value in column '{column_name}': {value!r}"
        )

    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite numeric value in column '{column_name}': {value!r}"
        )

    return number


def summarize_dataset(name, rows, error_column, flag_column, threshold):
    if not rows:
        raise ValueError(f"No rows found for dataset: {name}")

    required = {error_column, flag_column}
    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"{name} results are missing columns: {sorted(missing)}"
        )

    errors = [
        to_float(row[error_column], error_column)
        for row in rows
    ]

    flags = [
        str(row[flag_column]).strip().lower() == "true"
        for row in rows
    ]

    flagged = sum(flags)
    total = len(rows)

    return {
        "dataset": name,
        "windows_evaluated": total,
        "windows_flagged": flagged,
        "windows_not_flagged": total - flagged,
        "flagged_percentage": round(100.0 * flagged / total, 4),
        "mean_reconstruction_error": statistics.mean(errors),
        "median_reconstruction_error": statistics.median(errors),
        "minimum_reconstruction_error": min(errors),
        "maximum_reconstruction_error": max(errors),
        "threshold": threshold,
    }


def fmt(number, digits=6):
    return f"{number:.{digits}f}"


def create_pdf(report):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=23,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
    ))

    document = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []
    story.append(Paragraph(
        "Explainable-AIOps: Normal vs Slow Traffic Evaluation",
        styles["ReportTitle"],
    ))
    story.append(Paragraph(
        "Model: normal_extended_v1 — LSTM Autoencoder",
        styles["Heading2"],
    ))
    story.append(Paragraph(
        "Purpose: Compare reconstruction-error-based anomaly flags "
        "for the saved held-out normal test dataset and the isolated "
        "slow-traffic dataset.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Experiment Configuration", styles["Heading2"]))
    config = [
        ["Parameter", "Value"],
        ["Model experiment", report["experiment"]],
        ["Window size", str(report["window_size"])],
        ["Features per time step", str(report["features"])],
        ["Anomaly threshold", fmt(report["threshold"], 12)],
        ["Threshold method", report["threshold_method"]],
        ["Model retrained for this report", "No"],
    ]
    table = Table(config, colWidths=[2.25 * inch, 4.6 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26364A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F4F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("2. Dataset Comparison", styles["Heading2"]))
    comparison = [[
        "Metric",
        "Normal test",
        "Slow traffic",
    ]]

    metric_rows = [
        ("Windows evaluated", "windows_evaluated", "integer"),
        ("Windows flagged", "windows_flagged", "integer"),
        ("Windows not flagged", "windows_not_flagged", "integer"),
        ("Flagged percentage", "flagged_percentage", "percent"),
        ("Mean reconstruction error", "mean_reconstruction_error", "decimal"),
        ("Median reconstruction error", "median_reconstruction_error", "decimal"),
        ("Minimum reconstruction error", "minimum_reconstruction_error", "decimal"),
        ("Maximum reconstruction error", "maximum_reconstruction_error", "decimal"),
    ]

    normal = report["datasets"][0]
    slow = report["datasets"][1]

    for label, key, kind in metric_rows:
        a = normal[key]
        b = slow[key]

        if kind == "integer":
            a, b = str(a), str(b)
        elif kind == "percent":
            a, b = f"{a:.2f}%", f"{b:.2f}%"
        else:
            a, b = fmt(a, 8), fmt(b, 8)

        comparison.append([label, a, b])

    table = Table(
        comparison,
        colWidths=[2.7 * inch, 2.05 * inch, 2.05 * inch],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26364A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F4F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("3. Interpretation", styles["Heading2"]))
    story.append(Paragraph(
        f"The saved normal test results contain {normal['windows_evaluated']} "
        f"windows, of which {normal['windows_flagged']} exceeded the threshold "
        f"({normal['flagged_percentage']:.2f}%). The isolated slow-traffic "
        f"results contain {slow['windows_evaluated']} windows, of which "
        f"{slow['windows_flagged']} exceeded the threshold "
        f"({slow['flagged_percentage']:.2f}%).",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "These results show how this model behaved on these particular "
        "datasets. They do not establish general production accuracy or "
        "prove that every flagged window corresponds to a real incident.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Limitations", styles["Heading2"]))
    limitations = [
        "The normal and slow-traffic datasets are different samples and "
        "have different numbers of windows.",
        "The slow-traffic dataset contains only 40 observations and produces "
        "31 overlapping windows.",
        "Overlapping windows share observations and are not statistically "
        "independent.",
        "The threshold was selected from normal validation reconstruction "
        "errors; it was not adjusted for this comparison.",
        "A normal-only test set estimates false alarms on that sample but "
        "does not measure anomaly recall. Independent, labeled datasets are "
        "needed to assess broader performance.",
        "Reconstruction error is an anomaly indicator, not an explanation "
        "of the operational root cause.",
    ]
    for item in limitations:
        story.append(Paragraph(
            "&bull; " + item,
            styles["SmallNote"],
        ))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Source files: test_reconstruction_errors.csv, "
        "slow_traffic_results.csv, anomaly_threshold.json, and "
        "experiment_summary.json. This report was generated from existing "
        "saved results; it does not retrain or modify the model.",
        styles["SmallNote"],
    ))

    document.build(story)


def main():
    normal_rows = read_csv(NORMAL_FILE)
    slow_rows = read_csv(SLOW_FILE)
    threshold_config = read_json(THRESHOLD_FILE)
    experiment_summary = read_json(EXPERIMENT_FILE)

    threshold = float(threshold_config["threshold"])

    if not math.isfinite(threshold):
        raise ValueError("Saved threshold is not finite.")

    normal = summarize_dataset(
        "normal_held_out_test",
        normal_rows,
        "reconstruction_error",
        "predicted_anomaly",
        threshold,
    )
    slow = summarize_dataset(
        "isolated_slow_traffic",
        slow_rows,
        "reconstruction_error",
        "is_anomaly_candidate",
        threshold,
    )

    # Ensure that every saved row uses the expected threshold.
    for dataset_name, rows in [
        ("normal", normal_rows),
        ("slow", slow_rows),
    ]:
        for row in rows:
            if "threshold" not in row:
                raise ValueError(
                    f"The {dataset_name} CSV does not contain a threshold column."
                )
            row_threshold = to_float(row["threshold"], "threshold")
            if not math.isclose(
                row_threshold, threshold, rel_tol=1e-8, abs_tol=1e-10
            ):
                raise ValueError(
                    f"Threshold mismatch in {dataset_name} results: "
                    f"{row_threshold} != {threshold}"
                )

    report = {
        "report_title": "Explainable-AIOps Normal vs Slow Traffic Comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": threshold_config["experiment"],
        "model": "LSTM Autoencoder",
        "window_size": int(threshold_config["window_size"]),
        "features": len(threshold_config["feature_names"]),
        "threshold": threshold,
        "threshold_method": threshold_config["method"],
        "threshold_percentile": threshold_config.get("percentile"),
        "datasets": [normal, slow],
        "source_files": {
            "normal_test": str(NORMAL_FILE),
            "slow_traffic": str(SLOW_FILE),
            "threshold_config": str(THRESHOLD_FILE),
            "experiment_summary": str(EXPERIMENT_FILE),
        },
        "original_experiment_summary": experiment_summary,
        "limitations": [
            "Results apply only to the datasets evaluated.",
            "Overlapping windows are not independent observations.",
            "Normal-only test data does not measure anomaly recall.",
            "The threshold is a prototype threshold derived from normal validation errors.",
            "Reconstruction-error flags do not establish root cause.",
        ],
        "model_retrained": False,
        "artifacts_modified": False,
    }

    csv_columns = [
        "dataset",
        "windows_evaluated",
        "windows_flagged",
        "windows_not_flagged",
        "flagged_percentage",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "minimum_reconstruction_error",
        "maximum_reconstruction_error",
        "threshold",
    ]

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for dataset in report["datasets"]:
            writer.writerow({
                key: dataset[key]
                for key in csv_columns
            })

    with JSON_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    create_pdf(report)

    print("\nREPORT GENERATION COMPLETED")
    print("Model retrained: No")
    print("Original model/scaler/threshold modified: No")
    print(f"Normal windows: {normal['windows_evaluated']}")
    print(
        f"Normal flagged: {normal['windows_flagged']} "
        f"({normal['flagged_percentage']:.2f}%)"
    )
    print(f"Slow windows: {slow['windows_evaluated']}")
    print(
        f"Slow flagged: {slow['windows_flagged']} "
        f"({slow['flagged_percentage']:.2f}%)"
    )
    print(f"Threshold: {threshold:.12f}")
    print(f"\nCSV:  {CSV_OUTPUT}")
    print(f"JSON: {JSON_OUTPUT}")
    print(f"PDF:  {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
