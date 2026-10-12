"""
Explainable AIOps
Compare latency-related features between baseline training
data and isolated normal traffic.

This script only reads existing files.
It does not retrain the model or modify artifacts.
"""

import numpy as np
import pandas as pd

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
DATA_DIR = PROJECT_DIR / "ecommerce-demo" / "data"

BASELINE_FILE = DATA_DIR / "sequence_windows_baseline.npz"
NORMAL_FILE = DATA_DIR / "traffic_features_normal_isolated.csv"

if not BASELINE_FILE.exists():
    raise FileNotFoundError(BASELINE_FILE)

if not NORMAL_FILE.exists():
    raise FileNotFoundError(NORMAL_FILE)

# Load original baseline training windows.
with np.load(BASELINE_FILE, allow_pickle=False) as data:
    X_train = data["X_train"]
    feature_names = data["feature_names"].astype(str).tolist()

# Load newly generated normal traffic.
normal_df = pd.read_csv(NORMAL_FILE)

# Extract feature arrays.
# X_train has overlapping windows, so flattening it repeats observations.
baseline_flat = X_train.reshape(-1, X_train.shape[-1])

baseline_df = pd.DataFrame(
    baseline_flat,
    columns=feature_names
)

normal_df = normal_df[feature_names]

features_to_compare = [
    "response_time_ms",
    "log_response_time",
]

print("=" * 75)
print("LATENCY FEATURE DISTRIBUTION COMPARISON")
print("=" * 75)

print("Baseline training windows:", len(X_train))
print("Baseline flattened feature rows:", len(baseline_df))
print("New normal traffic observations:", len(normal_df))

for feature in features_to_compare:
    baseline_values = baseline_df[feature]
    normal_values = normal_df[feature]

    print(f"\nFEATURE: {feature}")
    print("-" * 75)

    print(
        f"{'Metric':<15}"
        f"{'Baseline':>18}"
        f"{'New normal':>18}"
    )

    metrics = {
        "Mean": (baseline_values.mean(), normal_values.mean()),
        "Median": (baseline_values.median(), normal_values.median()),
        "Std deviation": (
            baseline_values.std(),
            normal_values.std(),
        ),
        "Minimum": (baseline_values.min(), normal_values.min()),
        "Maximum": (baseline_values.max(), normal_values.max()),
        "25th percentile": (
            baseline_values.quantile(0.25),
            normal_values.quantile(0.25),
        ),
        "75th percentile": (
            baseline_values.quantile(0.75),
            normal_values.quantile(0.75),
        ),
        "95th percentile": (
            baseline_values.quantile(0.95),
            normal_values.quantile(0.95),
    ),
    }

    for metric, (baseline_value, normal_value) in metrics.items():
        print(
            f"{metric:<15}"
            f"{baseline_value:>18.4f}"
            f"{normal_value:>18.4f}"
        )

print("\nComparison complete.")
print("No model, scaler, threshold, or existing artifact was modified.")