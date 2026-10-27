"""
Prepare server-side normal traffic dataset for Explainable AIOps.

Safety:
- Reads existing client CSV and backend JSON logs.
- Validates request counts, endpoint sequence, and timestamps.
- Uses server-side response times for model training.
- Never overwrites an existing output file.
- Does not modify ML artifacts or PostgreSQL.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"

CLIENT_FILE = DATA_DIR / "traffic_normal_realistic_v2.csv"
LOG_FILE = LOG_DIR / "ecommerce.log"

OUTPUT_FILE = DATA_DIR / "traffic_normal_realistic_v2_server.csv"

ENDPOINTS = {"/", "/products/", "/users/"}

EXPECTED_REQUESTS = 600


def main():

    print("=" * 65)
    print("PREPARE REALISTIC NORMAL TRAFFIC - SERVER DATASET")
    print("=" * 65)

    # --------------------------------------------------
    # 1. SAFETY CHECKS
    # --------------------------------------------------

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Existing datasets will not be overwritten."
        )

    if not CLIENT_FILE.exists():
        raise FileNotFoundError(CLIENT_FILE)

    if not LOG_FILE.exists():
        raise FileNotFoundError(LOG_FILE)

    # --------------------------------------------------
    # 2. LOAD CLIENT DATA
    # --------------------------------------------------

    client = pd.read_csv(CLIENT_FILE)

    if len(client) != EXPECTED_REQUESTS:
        raise ValueError(
            f"Expected {EXPECTED_REQUESTS} client requests, "
            f"found {len(client)}."
        )

    client["timestamp_utc"] = pd.to_datetime(
        client["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    if not client["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("Client timestamps are not chronological.")

    # --------------------------------------------------
    # 3. LOAD BACKEND LOGS
    # --------------------------------------------------

    records = []

    with LOG_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("event") == "http_request":
                records.append(record)

    logs = pd.DataFrame(records)

    if logs.empty:
        raise ValueError("No HTTP request logs found.")

    logs["timestamp"] = pd.to_datetime(
        logs["timestamp"],
        utc=True,
        errors="raise",
    )

    start = client["timestamp_utc"].iloc[0]
    end = client["timestamp_utc"].iloc[-1] + pd.Timedelta(
        seconds=15
    )

    logs = logs[
        (logs["timestamp"] >= start)
        & (logs["timestamp"] <= end)
        & (logs["path"].isin(ENDPOINTS))
        & (logs["method"] == "GET")
    ].copy()

    logs = logs.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 4. VALIDATE RECORD ALIGNMENT
    # --------------------------------------------------

    print("\nClient records:", len(client))
    print("Server records:", len(logs))

    if len(logs) != len(client):
        raise ValueError(
            "Client/server record counts differ. "
            "Cannot safely align observations."
        )

    if not np.array_equal(
        client["endpoint"].to_numpy(),
        logs["path"].to_numpy(),
    ):
        raise ValueError(
            "Client/server endpoint sequences do not match."
        )

    if not np.array_equal(
        client["status_code"].to_numpy(),
        logs["status_code"].to_numpy(),
    ):
        raise ValueError(
            "Client/server HTTP status codes do not match."
        )

    # Server logs are written after the request starts.
    # Allow up to 15 seconds for the request to complete.
    time_difference = (
        logs["timestamp"] - client["timestamp_utc"]
    ).dt.total_seconds()

    if (time_difference < 0).any():
        raise ValueError(
            "Some server timestamps precede client timestamps."
        )

    if (time_difference > 15).any():
        raise ValueError(
            "Some server timestamps are more than 15 seconds "
            "after their corresponding client timestamps."
        )

    print("Endpoint sequence: MATCH")
    print("HTTP status codes: MATCH")
    print("Timestamp alignment: PASS")

    # --------------------------------------------------
    # 5. PREPARE SERVER-SIDE DATASET
    # --------------------------------------------------

    server = client.copy()

    server["response_time_ms"] = pd.to_numeric(
        logs["response_time_ms"],
        errors="raise",
    ).to_numpy()

    if server["response_time_ms"].isna().any():
        raise ValueError("Missing server response times.")

    if (server["response_time_ms"] < 0).any():
        raise ValueError("Negative server response times found.")

    # Use the server timestamp because it is associated
    # with the server-side response-time measurement.
    server["timestamp_utc"] = logs[
        "timestamp"
    ].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z").to_numpy()

    # --------------------------------------------------
    # 6. SAVE WITHOUT OVERWRITING
    # --------------------------------------------------

    server.to_csv(
        OUTPUT_FILE,
        index=False,
        mode="x",
    )

    print("\n" + "=" * 65)
    print("SERVER-SIDE DATASET CREATED")
    print("=" * 65)

    print("\nDataset shape:", server.shape)

    print("\nEndpoint distribution:")
    print(server["endpoint"].value_counts())

    print("\nHTTP status distribution:")
    print(server["status_code"].value_counts())

    print("\nServer response-time statistics:")
    print(server["response_time_ms"].describe().round(2))

    print("\nSaved dataset:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()