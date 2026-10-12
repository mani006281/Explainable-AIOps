import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ORIGINAL_CSV = DATA_DIR / "traffic_runs.csv"
ADDITIONAL_CSV = DATA_DIR / "traffic_runs_additional.csv"
EXPANDED_CSV = DATA_DIR / "traffic_runs_expanded.csv"

FIELDNAMES = [
    "timestamp_utc",
    "scenario",
    "path",
    "expected_delay_ms",
    "status_code",
    "measured_response_time_ms",
    "error",
]


def read_csv(path: Path) -> list:
    """Reads a CSV and returns a list of dicts. Returns [] if file is missing."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(v.strip() for v in row.values()):
                rows.append(row)
    return rows


def combine_datasets():
    """Combines original + additional records into traffic_runs_expanded.csv.

    Does NOT overwrite the original CSV.
    This script must be run AFTER additional records have been collected.
    """
    print("=" * 72)
    print(" PHASE 3.4 — DATASET EXPANSION: COMBINE SCRIPT")
    print("=" * 72)

    original_rows = read_csv(ORIGINAL_CSV)
    additional_rows = read_csv(ADDITIONAL_CSV)

    if not original_rows:
        print(f"[CRITICAL ERROR] Original dataset is missing: {ORIGINAL_CSV}")
        sys.exit(1)

    print(f"  Original records : {len(original_rows)}")
    print(f"  Additional records : {len(additional_rows)}")

    combined = original_rows + additional_rows
    combined.sort(key=lambda r: r.get("timestamp_utc", ""))

    print(f"  Combined total   : {len(combined)}")

    with EXPANDED_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(combined)

    print(f"\n[✓] Expanded dataset written to: {EXPANDED_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    combine_datasets()
