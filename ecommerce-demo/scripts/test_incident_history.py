"""
ecommerce-demo/scripts/test_incident_history.py
================================================
Focused test suite for Phase 8: Explainable Anomaly Results & Incident History.

What this test suite verifies
-----------------------------
1. Incident creation & required fields:
   - timestamp_utc (ISO 8601 UTC string)
   - path, status_code, response_time_ms
   - reconstruction_error, threshold
   - top_contributing_features (exactly 3 items)
   - causal explanation disclaimer
2. Correct top-three feature ordering:
   - Strictly ordered descending by per-feature reconstruction error.
   - Explicitly labeled as reconstruction-error contributions, not root causes.
3. Newest-first chronological ordering:
   - Most recently recorded incident appears first (index 0).
4. Maximum history size bound (100 records):
   - Inserting 150 records results in exactly 100 stored.
   - Oldest 50 records are automatically evicted in FIFO order.
5. Thread safety:
   - 20 concurrent threads inserting incidents with zero data corruption.
6. Exclusion of monitoring endpoints:
   - Neither /aiops/status nor /aiops/incidents increments observation window.
7. Live anomaly incident creation via FastAPI:
   - Triggering an anomalous HTTP request creates an incident in GET /aiops/incidents.
8. Error isolation:
   - Simulated incident storage errors never cause HTTP requests to fail.
9. Preservation of existing endpoints:
   - /health, /aiops/status, and / still function perfectly.
"""

import asyncio
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure ecommerce-demo root is on sys.path
ECOMMERCE_ROOT = Path(__file__).resolve().parents[1]
if str(ECOMMERCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ECOMMERCE_ROOT))

from app.main import app
from app.anomaly_detection import reset_window, get_detector_status, record_observation
from app.incident_store import (
    clear_incident_history,
    create_incident_record,
    format_top_contributing_features,
    get_incident_count,
    get_incident_history,
    record_incident,
    MAX_INCIDENTS,
)
import app.main as main_module

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

_pass_count = 0
_fail_count = 0


def _ok(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def _fail(msg: str) -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {RED}[FAIL]{RESET} {msg}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")


def _check(condition: bool, pass_msg: str, fail_msg: str) -> None:
    if condition:
        _ok(pass_msg)
    else:
        _fail(fail_msg)


async def make_request(method: str = "GET", path: str = "/health"):
    """ASGI request helper for in-memory FastAPI testing."""
    status_code = None
    response_headers = []
    response_body = bytearray()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8001),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code, response_headers, response_body
        if message["type"] == "http.response.start":
            status_code = message["status"]
            response_headers = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    await app(scope, receive, send)

    parsed_json = None
    try:
        parsed_json = json.loads(response_body.decode("utf-8"))
    except Exception:
        pass

    return status_code, parsed_json


# ==================================================================
# TEST 1: Incident Creation and Required Fields
# ==================================================================
def test_incident_record_creation():
    _section("TEST 1: Incident Creation & Required Fields Verification")

    clear_incident_history()

    sample_errors = [
        {"feature": "response_time_ms", "mse": 0.085},
        {"feature": "log_response_time", "mse": 0.062},
        {"feature": "endpoint_latency", "mse": 0.031},
        {"feature": "status_200", "mse": 0.005},
    ]

    record = create_incident_record(
        path="/debug/test-latency",
        status_code=200,
        response_time_ms=1500.5,
        reconstruction_error=0.045678,
        threshold=0.025899,
        per_feature_errors=sample_errors,
    )

    required_fields = [
        "timestamp_utc",
        "path",
        "status_code",
        "response_time_ms",
        "reconstruction_error",
        "threshold",
        "top_contributing_features",
        "explanation",
    ]
    for field in required_fields:
        _check(field in record, f"Field '{field}' present in incident record", f"Missing field '{field}'")

    _check(record["path"] == "/debug/test-latency", "path matches input", f"path: {record['path']}")
    _check(record["status_code"] == 200, "status_code matches input", f"status: {record['status_code']}")
    _check(record["response_time_ms"] == 1500.5, "response_time_ms matches input", f"latency: {record['response_time_ms']}")
    _check(record["reconstruction_error"] == 0.045678, "reconstruction_error matches input", f"error: {record['reconstruction_error']}")
    _check(record["threshold"] == 0.025899, "threshold matches input", f"thresh: {record['threshold']}")
    _check(len(record["top_contributing_features"]) == 3, "top_contributing_features contains exactly 3 features", f"Count: {len(record['top_contributing_features'])}")

    # Security check: no bodies, no credentials
    raw_str = json.dumps(record)
    _check("password" not in raw_str and "secret" not in raw_str and "authorization" not in raw_str.lower(),
           "No credentials, auth headers, or secrets in incident record", "Security leak in incident record")


# ==================================================================
# TEST 2: Correct Top-Three Feature Ordering & Causal Disclaimers
# ==================================================================
def test_top_three_feature_ordering():
    _section("TEST 2: Top-Three Feature Ordering & Causal Disclaimers")

    # Deliberately unordered feature error list
    unordered_errors = [
        {"feature": "status_404", "mse": 0.012},
        {"feature": "endpoint_latency", "mse": 0.089},
        {"feature": "status_200", "mse": 0.003},
        {"feature": "response_time_ms", "mse": 0.145},
        {"feature": "log_response_time", "mse": 0.075},
        {"feature": "is_http_error", "mse": 0.001},
    ]

    top_three = format_top_contributing_features(unordered_errors, top_n=3)

    _check(len(top_three) == 3, f"Returned exactly 3 features (got {len(top_three)})", "Wrong feature count")
    _check(top_three[0]["feature"] == "response_time_ms", f"Rank 1 is 'response_time_ms' (mse=0.145, got {top_three[0]['feature']})", "Rank 1 incorrect")
    _check(top_three[1]["feature"] == "endpoint_latency", f"Rank 2 is 'endpoint_latency' (mse=0.089, got {top_three[1]['feature']})", "Rank 2 incorrect")
    _check(top_three[2]["feature"] == "log_response_time", f"Rank 3 is 'log_response_time' (mse=0.075, got {top_three[2]['feature']})", "Rank 3 incorrect")

    # Check ordering descending
    _check(
        top_three[0]["reconstruction_error"] >= top_three[1]["reconstruction_error"] >= top_three[2]["reconstruction_error"],
        "Features are strictly ordered descending by reconstruction error",
        "Feature ordering is not descending",
    )

    # Check causal disclaimers
    for item in top_three:
        _check(
            item.get("metric_type") == "reconstruction_error_contribution",
            f"Metric type for '{item['feature']}' is 'reconstruction_error_contribution'",
            f"Metric type missing or wrong: {item.get('metric_type')}",
        )
        _check(
            "not a proven causal root cause" in item.get("note", "").lower() or "not a proven root cause" in item.get("note", "").lower(),
            f"Explicit causal disclaimer present for '{item['feature']}'",
            f"Missing causal disclaimer in note: {item.get('note')}",
        )


# ==================================================================
# TEST 3: Newest-First Chronological Ordering
# ==================================================================
def test_newest_first_ordering():
    _section("TEST 3: Newest-First Chronological Incident Ordering")

    clear_incident_history()

    # Record 5 incidents with distinct paths
    for i in range(1, 6):
        record_incident(
            path=f"/test-endpoint-{i}",
            status_code=200,
            response_time_ms=100.0 * i,
            reconstruction_error=0.030 + (i * 0.005),
            threshold=0.025899,
            per_feature_errors=[{"feature": "response_time_ms", "mse": 0.05}],
        )
        time.sleep(0.002)  # Tiny pause to ensure monotonically increasing timestamps

    history = get_incident_history()

    _check(len(history) == 5, f"Retrieved 5 incidents (got {len(history)})", f"Expected 5, got {len(history)}")
    _check(history[0]["path"] == "/test-endpoint-5", f"Newest incident is at index 0 ({history[0]['path']})", f"Index 0 wrong: {history[0]['path']}")
    _check(history[-1]["path"] == "/test-endpoint-1", f"Oldest incident is at last index ({history[-1]['path']})", f"Last index wrong: {history[-1]['path']}")

    # Verify timestamps are descending
    timestamps = [h["timestamp_utc"] for h in history]
    is_descending = all(timestamps[i] >= timestamps[i+1] for i in range(len(timestamps)-1))
    _check(is_descending, "Incident timestamps are monotonically descending (newest first)", "Timestamps not newest-first")


# ==================================================================
# TEST 4: Maximum History Size Bound (100)
# ==================================================================
def test_maximum_history_size():
    _section("TEST 4: Maximum Bounded History Size (100 Records)")

    clear_incident_history()

    # Insert 150 incidents
    for i in range(1, 151):
        record_incident(
            path=f"/path-{i}",
            status_code=200,
            response_time_ms=10.0,
            reconstruction_error=0.040,
            threshold=0.025899,
        )

    count = get_incident_count()
    history = get_incident_history()

    _check(count == MAX_INCIDENTS, f"Incident count is strictly capped at {MAX_INCIDENTS} (got {count})", f"Count exceeded: {count}")
    _check(len(history) == MAX_INCIDENTS, f"History list length is exactly {MAX_INCIDENTS} (got {len(history)})", f"Length wrong: {len(history)}")

    # Verify FIFO eviction: items 1..50 should be evicted, item 150 newest, item 51 oldest
    _check(history[0]["path"] == "/path-150", f"Newest item is /path-150 (got {history[0]['path']})", f"Newest item wrong: {history[0]['path']}")
    _check(history[-1]["path"] == "/path-51", f"Oldest retained item is /path-51 (got {history[-1]['path']})", f"Oldest item wrong: {history[-1]['path']}")

    # Verify /path-1 through /path-50 are not present
    evicted_paths = {h["path"] for h in history}
    _check("/path-1" not in evicted_paths and "/path-50" not in evicted_paths,
           "Oldest 50 records were safely evicted in FIFO order", "Evicted items still found in history")


# ==================================================================
# TEST 5: Thread Safety
# ==================================================================
def test_thread_safety():
    _section("TEST 5: Thread Safety Under Concurrent Load")

    clear_incident_history()
    errors = []

    def worker(thread_idx: int):
        try:
            for j in range(10):
                record_incident(
                    path=f"/thread-{thread_idx}/call-{j}",
                    status_code=200,
                    response_time_ms=50.0,
                    reconstruction_error=0.035,
                    threshold=0.025899,
                )
        except Exception as exc:
            errors.append(f"Thread {thread_idx}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _check(len(errors) == 0, f"20 concurrent threads executed with 0 errors", f"Thread errors: {errors}")
    count = get_incident_count()
    _check(count == MAX_INCIDENTS, f"Bounded queue maintained exact {MAX_INCIDENTS} cap under concurrency (got {count})", f"Queue corrupted: {count}")


# ==================================================================
# TEST 6: Monitoring Endpoints Exclusion & JSON Structure
# ==================================================================
async def test_monitoring_endpoint_exclusion_and_structure():
    _section("TEST 6: Monitoring Endpoint Exclusion & Response Structure")

    reset_window()
    clear_incident_history()

    # Pre-check
    _check(get_incident_count() == 0, "Incident store starts empty", f"Store has {get_incident_count()} items")
    _check(get_detector_status()["window_size"] == 0, "Window starts empty", "Window not empty")

    # Call /aiops/status and /aiops/incidents multiple times
    for _ in range(3):
        await make_request("GET", "/aiops/status")
        await make_request("GET", "/aiops/incidents")

    # Verify neither endpoint was recorded in the rolling window
    _check(
        get_detector_status()["window_size"] == 0,
        "Neither /aiops/status nor /aiops/incidents incremented the rolling window (remains 0)",
        f"Window polluted: size={get_detector_status()['window_size']}",
    )
    _check(
        get_incident_count() == 0,
        "Querying monitoring endpoints did not create any incident records",
        f"Incidents created: {get_incident_count()}",
    )

    # Test GET /aiops/incidents JSON response structure
    status, data = await make_request("GET", "/aiops/incidents")
    _check(status == 200, f"GET /aiops/incidents returned HTTP 200 (got {status})", f"Status: {status}")
    _check(isinstance(data, dict), "Response is a valid JSON dictionary", f"Response type: {type(data)}")

    expected_keys = ["service", "incident_count", "max_capacity", "explanation_disclaimer", "incidents"]
    for k in expected_keys:
        _check(k in data, f"Key '{k}' present in /aiops/incidents response", f"Missing key '{k}'")

    _check(data.get("service") == "ecommerce-aiops", "service == 'ecommerce-aiops'", f"service: {data.get('service')}")
    _check(data.get("max_capacity") == 100, "max_capacity == 100", f"max_capacity: {data.get('max_capacity')}")
    _check(isinstance(data.get("incidents"), list), "'incidents' is a JSON list", "Not a list")
    _check(
        "not proven causal root cause" in data.get("explanation_disclaimer", "").lower() or "not proven root cause" in data.get("explanation_disclaimer", "").lower(),
        "Top-level explanation disclaimer properly states contributions are not proven root causes",
        f"Disclaimer missing: {data.get('explanation_disclaimer')}",
    )


# ==================================================================
# TEST 7: Live Anomaly Incident Creation via FastAPI Middleware
# ==================================================================
async def test_live_anomaly_incident_creation():
    _section("TEST 7: Live Anomaly Incident Creation via Middleware")

    reset_window()
    clear_incident_history()

    # Step 7a: Warm up the rolling window to 10 observations
    for i in range(10):
        await make_request("GET", "/health")

    _check(get_detector_status()["window_ready"] is True, "Detector window is ready (size=10)", "Window not ready")
    initial_incidents = get_incident_count()

    # Step 7b: Trigger an anomalous request burst
    # Sending requests to an unmapped endpoint (HTTP 404 + unmapped category) simulates out-of-distribution anomaly
    for _ in range(10):
        await make_request("GET", "/does-not-exist")

    incidents_after = get_incident_count()
    print(f"  Incidents recorded after anomalous traffic: {incidents_after}")
    _check(incidents_after > 0, f"At least 1 anomaly incident recorded during anomalous traffic (got {incidents_after})", "No incidents recorded")

    # Query GET /aiops/incidents
    status, payload = await make_request("GET", "/aiops/incidents")
    _check(status == 200, "GET /aiops/incidents returned HTTP 200", f"Failed with {status}")
    _check(payload["incident_count"] == incidents_after, f"incident_count ({payload['incident_count']}) matches stored count ({incidents_after})", "Count mismatch")

    if payload["incidents"]:
        newest = payload["incidents"][0]
        _check(newest["reconstruction_error"] > newest["threshold"],
               f"Incident reconstruction_error ({newest['reconstruction_error']}) > threshold ({newest['threshold']})",
               "Error not exceeding threshold")
        _check(len(newest["top_contributing_features"]) <= 3,
               f"Incident has {len(newest['top_contributing_features'])} top contributing features (<= 3)",
               "Wrong top features count")


# ==================================================================
# TEST 8: Error Isolation
# ==================================================================
async def test_incident_error_isolation():
    _section("TEST 8: Error Isolation During Incident Recording")

    import app.main as m
    original_record_incident = m.record_incident

    # Simulate exception inside record_incident
    def broken_record_incident(*args, **kwargs):
        raise RuntimeError("Simulated crash in incident recording storage")

    m.record_incident = broken_record_incident

    try:
        # Request should still return HTTP 200 even if incident storage fails
        status, data = await make_request("GET", "/health")
        _check(
            status == 200,
            "HTTP request succeeded with HTTP 200 despite simulated incident recording exception",
            f"HTTP request failed with status {status}",
        )
        _check(
            data.get("status") == "healthy",
            "Response body remained healthy and intact",
            f"Response corrupted: {data}",
        )
    finally:
        m.record_incident = original_record_incident

    status_restored, _ = await make_request("GET", "/health")
    _check(status_restored == 200, "Normal operation resumed after fault test", "Failed to resume")


# ==================================================================
# TEST 9: Preservation of Existing Endpoints
# ==================================================================
async def test_preservation_of_existing_endpoints():
    _section("TEST 9: Preservation of Existing API Endpoints")

    status_root, data_root = await make_request("GET", "/")
    _check(status_root == 200, "GET / returned HTTP 200", f"Root failed: {status_root}")
    _check(data_root.get("status") == "healthy", "GET / returns healthy", f"Root payload: {data_root}")

    status_health, data_health = await make_request("GET", "/health")
    _check(status_health == 200, "GET /health returned HTTP 200", f"Health failed: {status_health}")
    _check(data_health.get("status") == "healthy", "GET /health returns healthy", f"Health payload: {data_health}")

    status_aiops, data_aiops = await make_request("GET", "/aiops/status")
    _check(status_aiops == 200, "GET /aiops/status returned HTTP 200", f"AIOps status failed: {status_aiops}")
    _check("detector_ready" in data_aiops, "/aiops/status contains 'detector_ready'", "Missing key")


# ==================================================================
# Main Runner
# ==================================================================
async def run_all_tests():
    print(f"\n{BOLD}{'='*65}")
    print("  PHASE 8: EXPLAINABLE INCIDENT HISTORY TEST SUITE")
    print(f"{'='*65}{RESET}")
    print("  Application : ecommerce-demo/app/main.py")
    print("  Store       : ecommerce-demo/app/incident_store.py")
    print("  Model       : ml/artifacts_normal_extended_v1/lstm_autoencoder.keras")
    print()

    test_incident_record_creation()
    test_top_three_feature_ordering()
    test_newest_first_ordering()
    test_maximum_history_size()
    test_thread_safety()
    await test_monitoring_endpoint_exclusion_and_structure()
    await test_live_anomaly_incident_creation()
    await test_incident_error_isolation()
    await test_preservation_of_existing_endpoints()

    print(f"\n{BOLD}{'='*65}{RESET}")
    total = _pass_count + _fail_count
    if _fail_count == 0:
        print(f"{GREEN}{BOLD}  ALL {total} PHASE 8 INCIDENT TESTS PASSED{RESET}")
    else:
        print(f"{RED}{BOLD}  {_fail_count}/{total} PHASE 8 INCIDENT TESTS FAILED{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
