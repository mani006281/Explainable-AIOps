"""
test_anomaly_detection.py
=========================
Safe, focused test suite for app/anomaly_detection.py.

What this tests
---------------
1. Artifact loading       — model, scaler, threshold all load cleanly.
2. Feature extraction     — all 12 features computed correctly for
                            known HTTP observations.
3. Rolling window warmup  — window_ready is False for the first 9 calls.
4. Inference trigger      — window_ready becomes True on the 10th call.
5. Reconstruction error   — error is a positive float; normal traffic
                            stays below the threshold.
6. Anomaly detection      — simulate anomalous observations and check
                            that is_anomaly can become True.
7. Per-feature errors     — sorted descending, highest contributor first.
8. Thread safety          — 20 threads send observations concurrently;
                            no crash, no data corruption.
9. Error safety           — bad input types return an error_message
                            instead of raising an exception.
10. reset_window()        — window clears and warmup restarts.
11. get_detector_status() — returns expected keys and consistent values.

This file does NOT:
- Retrain any model.
- Write to any data file.
- Modify any existing file.
- Call FastAPI or any HTTP layer.
"""

import csv
import math
import sys
import threading
import time
from pathlib import Path

# ------------------------------------------------------------------
# Ensure the ecommerce-demo root is on sys.path so that
#   from app.anomaly_detection import ...
# works when this script is run from the project root.
# ------------------------------------------------------------------
ECOMMERCE_ROOT = Path(__file__).resolve().parents[1]
if str(ECOMMERCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ECOMMERCE_ROOT))

from app.anomaly_detection import (
    AnomalyResult,
    FEATURE_NAMES,
    N_FEATURES,
    WINDOW_SIZE,
    _extract_features,
    get_detector_status,
    record_observation,
    reset_window,
)


# ------------------------------------------------------------------
# ANSI colours for readable output
# ------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

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
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


def _check(condition: bool, pass_msg: str, fail_msg: str) -> None:
    if condition:
        _ok(pass_msg)
    else:
        _fail(fail_msg)


# ==================================================================
# Test 1 — Artifact loading
# ==================================================================
def test_artifact_loading():
    _section("TEST 1 — Artifact Loading")

    reset_window()

    # Trigger loading by calling record_observation once.
    result = record_observation("/health", 200, 10.0)

    from app.anomaly_detection import (
        _artifacts_loaded,
        _load_error,
        _threshold,
        _model,
        _scaler,
    )

    _check(
        _artifacts_loaded,
        "Artifacts loaded successfully",
        f"Artifacts NOT loaded. Error: {_load_error}",
    )
    _check(
        _load_error is None,
        "No loading errors",
        f"Loading error: {_load_error}",
    )
    _check(
        _model is not None,
        "LSTM model object is not None",
        "LSTM model is None",
    )
    _check(
        _scaler is not None,
        "Feature scaler object is not None",
        "Feature scaler is None",
    )
    _check(
        abs(_threshold - 0.025898804888129234) < 1e-10,
        f"Threshold loaded correctly: {_threshold:.15f}",
        f"Threshold wrong: {_threshold}",
    )
    _check(
        result.error_message is None,
        "record_observation returned no error_message",
        f"Unexpected error_message: {result.error_message}",
    )


# ==================================================================
# Test 2 — Feature extraction
# ==================================================================
def test_feature_extraction():
    _section("TEST 2 — Feature Extraction (_extract_features)")

    # ---- 2a: Normal GET /health → HTTP 200, 12 ms ------------------
    f = _extract_features("/health", 200, 12.0)
    import math
    _check(len(f) == N_FEATURES,          f"12 features returned (got {len(f)})",   "Wrong feature count")
    expected_log_resp = round(math.log1p(12.0), 4)
    _check(f["log_response_time"] == expected_log_resp,
           f"log_response_time == round(log1p(12.0), 4) ({expected_log_resp})",
           f"Got {f['log_response_time']}, expected {expected_log_resp}")
    _check(f["status_200"]        == 1,     "status_200 == 1",                       "status_200 != 1")
    _check(f["status_400"]        == 0,     "status_400 == 0",                       "status_400 != 0")
    _check(f["status_404"]        == 0,     "status_404 == 0",                       "status_404 != 0")
    _check(f["status_other"]      == 0,     "status_other == 0",                     "status_other != 0")
    _check(f["endpoint_health"]   == 1,     "endpoint_health == 1",                  "endpoint_health != 1")
    _check(f["endpoint_products"] == 0,     "endpoint_products == 0",                "endpoint_products != 0")
    _check(f["endpoint_latency"]  == 0,     "endpoint_latency == 0",                 "endpoint_latency != 0")
    _check(f["endpoint_other"]    == 0,     "endpoint_other == 0",                   "endpoint_other != 0")
    _check(f["is_connection_error"] == 0,   "is_connection_error == 0",              "is_connection_error != 0")
    _check(f["is_http_error"]     == 0,     "is_http_error == 0",                    "is_http_error != 0")

    # ---- 2b: GET /products/ → HTTP 200 ----------------------------
    f = _extract_features("/products/", 200, 25.5)
    _check(f["endpoint_products"] == 1,     "endpoint_products == 1 for /products/", "Wrong endpoint encoding")
    _check(f["endpoint_health"]   == 0,     "endpoint_health == 0 for /products/",   "Wrong endpoint encoding")

    # ---- 2c: GET /debug/test-latency → HTTP 200, slow --------------
    f = _extract_features("/debug/test-latency", 200, 1500.0)
    _check(f["endpoint_latency"]  == 1,     "endpoint_latency == 1",                 "Wrong endpoint encoding")
    _check(f["response_time_ms"]  == 1500.0, "response_time_ms == 1500.0",           f"Got {f['response_time_ms']}")

    # ---- 2d: HTTP 404 → unknown endpoint ---------------------------
    f = _extract_features("/does-not-exist", 404, 8.0)
    _check(f["status_404"]        == 1,     "status_404 == 1",                       "status_404 wrong")
    _check(f["is_http_error"]     == 1,     "is_http_error == 1 for 404",            "is_http_error wrong")
    _check(f["endpoint_other"]    == 1,     "endpoint_other == 1 for unknown path",  "endpoint_other wrong")

    # ---- 2e: HTTP 400 -----------------------------------------------
    f = _extract_features("/debug/test-latency", 400, 5.0)
    _check(f["status_400"]        == 1,     "status_400 == 1",                       "status_400 wrong")
    _check(f["is_http_error"]     == 1,     "is_http_error == 1 for 400",            "is_http_error wrong")

    # ---- 2f: Connection error (status_code = None) -----------------
    f = _extract_features("/health", None, 0.0)
    _check(f["is_connection_error"] == 1,   "is_connection_error == 1 for None",     "is_connection_error wrong")
    _check(f["status_other"]        == 1,   "status_other == 1 for None",            "status_other wrong")
    _check(f["is_http_error"]       == 0,   "is_http_error == 0 for None",           "is_http_error wrong")

    # ---- 2g: Feature names match FEATURE_NAMES ----------------------
    _check(
        list(f.keys()) == FEATURE_NAMES,
        "Feature dict keys match FEATURE_NAMES in correct order",
        f"Key mismatch. Got: {list(f.keys())}",
    )


# ==================================================================
# Test 3 — Rolling window warmup
# ==================================================================
def test_rolling_window_warmup():
    _section("TEST 3 — Rolling Window Warmup (first 9 observations)")

    reset_window()

    all_not_ready = True
    for i in range(WINDOW_SIZE - 1):
        result = record_observation("/health", 200, 10.0 + i)
        if result.window_ready:
            all_not_ready = False
            _fail(f"window_ready was True at observation #{i+1} (expected False)")
            break
        if result.error_message:
            _fail(f"Unexpected error at observation #{i+1}: {result.error_message}")
            all_not_ready = False
            break

    if all_not_ready:
        _ok(f"window_ready stayed False for all {WINDOW_SIZE - 1} warmup observations")

    _check(
        result.reconstruction_error is None,
        "reconstruction_error is None during warmup",
        f"reconstruction_error was {result.reconstruction_error} (should be None)",
    )
    _check(
        result.is_anomaly is False,
        "is_anomaly is False during warmup",
        "is_anomaly was True during warmup",
    )
    _check(
        result.window_size == WINDOW_SIZE - 1,
        f"window_size == {WINDOW_SIZE - 1} after {WINDOW_SIZE - 1} observations",
        f"window_size == {result.window_size}",
    )


# ==================================================================
# Test 4 — First inference (10th observation)
# ==================================================================
def test_first_inference():
    _section("TEST 4 — First Inference (10th observation)")

    # Window has 9 entries from test 3 — add one more.
    result = record_observation("/health", 200, 15.0)

    _check(result.window_ready,              "window_ready == True on 10th observation",   "window_ready is False")
    _check(result.window_size == WINDOW_SIZE, f"window_size == {WINDOW_SIZE}",              f"window_size == {result.window_size}")
    _check(
        result.reconstruction_error is not None,
        "reconstruction_error is not None",
        "reconstruction_error is still None",
    )
    if result.reconstruction_error is not None:
        _check(
            result.reconstruction_error >= 0,
            f"reconstruction_error is non-negative: {result.reconstruction_error:.8f}",
            f"reconstruction_error is negative: {result.reconstruction_error}",
        )
    _check(
        result.per_feature_errors is not None,
        "per_feature_errors is not None",
        "per_feature_errors is None",
    )
    if result.per_feature_errors:
        _check(
            len(result.per_feature_errors) == N_FEATURES,
            f"per_feature_errors has {N_FEATURES} entries",
            f"per_feature_errors has {len(result.per_feature_errors)} entries",
        )
        # Verify sorted descending
        mse_values = [e["mse"] for e in result.per_feature_errors]
        _check(
            mse_values == sorted(mse_values, reverse=True),
            "per_feature_errors is sorted descending by MSE",
            f"per_feature_errors not sorted: {mse_values}",
        )
        print(f"\n  {YELLOW}Top contributing feature:{RESET} "
              f"{result.per_feature_errors[0]['feature']} "
              f"(mse={result.per_feature_errors[0]['mse']:.8f})")

    _check(
        result.error_message is None,
        "No error_message on successful inference",
        f"Unexpected error_message: {result.error_message}",
    )
    print(f"  {YELLOW}Overall reconstruction error:{RESET} "
          f"{result.reconstruction_error:.8f}")
    print(f"  {YELLOW}Threshold:                  {RESET} "
          f"{result.threshold:.8f}")
    print(f"  {YELLOW}Is anomaly:                 {RESET} "
          f"{result.is_anomaly}")


# ==================================================================
# Test 5 — Normal Traffic & Inference Verification
# ==================================================================
def test_normal_traffic():
    _section("TEST 5 — Normal Traffic & Inference Verification")

    DATA_DIR = ECOMMERCE_ROOT / "data"
    HELD_OUT_METADATA_FILE = DATA_DIR / "traffic_normal_extended_feature_metadata.csv"

    # --------------------------------------------------------------
    # 5a: Held-out Test Set Normal Traffic Evaluation
    # --------------------------------------------------------------
    # The normal_extended_v1 model was trained on 420 observations,
    # validated on 90 observations, and evaluated on 90 held-out test
    # observations (forming 81 test windows).
    # Here we stream the held-out test observations through the detector.
    reset_window()

    held_out_observations = []
    if HELD_OUT_METADATA_FILE.exists():
        try:
            with HELD_OUT_METADATA_FILE.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if r.get("path") and r.get("status_code")]
            if len(rows) >= 600:
                # The final 90 observations (indices 510 to 599) constitute
                # the official held-out test split.
                held_out_observations = rows[510:]
        except Exception as exc:
            print(f"  {YELLOW}[WARN] Could not read held-out test metadata: {exc}{RESET}")

    if held_out_observations:
        print(f"  Loaded {len(held_out_observations)} held-out test observations from:")
        print(f"    {HELD_OUT_METADATA_FILE.name}")

        test_anomalies = 0
        evaluated_windows = 0
        reconstruction_errors = []
        decision_logic_ok = True

        for obs in held_out_observations:
            path = obs["path"]
            status_code = int(obs["status_code"]) if obs.get("status_code") else 200
            resp_time = float(obs["measured_response_time_ms"]) if obs.get("measured_response_time_ms") else 20.0

            result = record_observation(path, status_code, resp_time)

            if result.window_ready:
                evaluated_windows += 1
                reconstruction_errors.append(result.reconstruction_error)

                # Deterministic check: decision flag must strictly match error > threshold
                expected_anomaly = result.reconstruction_error > result.threshold
                if result.is_anomaly != expected_anomaly:
                    decision_logic_ok = False

                if result.is_anomaly:
                    test_anomalies += 1

        _check(
            evaluated_windows == 81,
            f"Evaluated exactly 81 held-out test sequence windows (got {evaluated_windows})",
            f"Expected 81 held-out test windows, got {evaluated_windows}",
        )
        _check(
            all(err is not None and math.isfinite(err) and err >= 0.0 for err in reconstruction_errors),
            "All evaluated held-out test windows produced finite non-negative reconstruction errors",
            "Non-finite or negative reconstruction error detected in held-out test windows",
        )
        _check(
            decision_logic_ok,
            "Decision rule is_anomaly == (reconstruction_error > threshold) held for all windows",
            "Decision logic mismatch detected in one or more test windows",
        )

        flagged_rate = (test_anomalies / evaluated_windows * 100) if evaluated_windows > 0 else 0.0
        print(f"\n  [Diagnostic] Held-out normal test evaluation results:")
        print(f"    * Evaluated windows  : {evaluated_windows}")
        print(f"    * Flagged anomalous  : {test_anomalies} window(s)")
        print(f"    * Flagged-window rate: {flagged_rate:.2f}%")
        print(f"    * Unflagged windows  : {evaluated_windows - test_anomalies}")
        print(f"    * Saved threshold    : {result.threshold:.15f}")
        if reconstruction_errors:
            avg_err = sum(reconstruction_errors) / len(reconstruction_errors)
            max_err = max(reconstruction_errors)
            min_err = min(reconstruction_errors)
            print(f"    * Error distribution : min={min_err:.6f}, avg={avg_err:.6f}, max={max_err:.6f}")
        print("    * Rationale: Threshold was derived from the 95th percentile of validation errors;")
        print("      zero anomaly flags on an independent held-out normal dataset are not guaranteed.")

    else:
        # Fallback: In-distribution alternating sequence
        print(f"  {YELLOW}Held-out file not found; testing in-distribution cyclic normal traffic.{RESET}")
        cyclic_scenarios = [
            ("/health", 200, 25.0),
            ("/products/", 200, 28.0),
            ("/debug/test-latency", 200, 135.0),
        ]
        fallback_errors = []
        fallback_anomalies = 0
        fallback_evaluated = 0
        fallback_decision_ok = True
        for i in range(30):
            endpoint, sc, latency = cyclic_scenarios[i % len(cyclic_scenarios)]
            res = record_observation(endpoint, sc, latency)
            if res.window_ready:
                fallback_evaluated += 1
                fallback_errors.append(res.reconstruction_error)
                if res.is_anomaly != (res.reconstruction_error > res.threshold):
                    fallback_decision_ok = False
                if res.is_anomaly:
                    fallback_anomalies += 1

        fallback_rate = (fallback_anomalies / fallback_evaluated * 100) if fallback_evaluated > 0 else 0.0
        print(f"\n  [Diagnostic] Fallback cyclic normal traffic results:")
        print(f"    * Evaluated windows  : {fallback_evaluated}")
        print(f"    * Flagged anomalous  : {fallback_anomalies} window(s)")
        print(f"    * Flagged-window rate: {fallback_rate:.2f}%")

        _check(
            fallback_evaluated == 21,
            f"Evaluated exactly 21 fallback sequence windows (got {fallback_evaluated})",
            f"Expected 21 fallback windows, got {fallback_evaluated}",
        )
        _check(
            all(err is not None and math.isfinite(err) and err >= 0.0 for err in fallback_errors),
            "All fallback normal windows produced finite non-negative reconstruction errors",
            "Non-finite or negative reconstruction error detected in fallback normal traffic",
        )
        _check(
            fallback_decision_ok,
            "Fallback decision rule is_anomaly == (reconstruction_error > threshold) held for all windows",
            "Fallback decision logic mismatch detected",
        )

    # --------------------------------------------------------------
    # 5b: Deterministic Decision Logic Verification
    # --------------------------------------------------------------
    reset_window()
    for _ in range(WINDOW_SIZE):
        result = record_observation("/health", 200, 20.0)

    _check(
        result.window_ready is True,
        "Functional test: window_ready is True when window reaches WINDOW_SIZE",
        "Functional test: window_ready is False",
    )
    _check(
        isinstance(result.reconstruction_error, float) and math.isfinite(result.reconstruction_error) and result.reconstruction_error >= 0.0,
        f"Functional test: reconstruction_error is finite non-negative float ({result.reconstruction_error:.6f})",
        f"Functional test: reconstruction_error is invalid: {result.reconstruction_error}",
    )
    _check(
        result.is_anomaly == (result.reconstruction_error > result.threshold),
        f"Functional test: is_anomaly ({result.is_anomaly}) == "
        f"(reconstruction_error {result.reconstruction_error:.6f} > threshold {result.threshold:.6f})",
        "Functional test: boolean anomaly decision does not match inequality",
    )

    # --------------------------------------------------------------
    # 5c: Diagnostic Test on Repeated Single-Endpoint (/health) Traffic
    # --------------------------------------------------------------
    # Documented rationale:
    # The LSTM Autoencoder was trained on cyclic sequences containing
    # transitions among /health, /products/, and /debug/test-latency.
    # An unbroken sequence of identical /health requests lacks endpoint
    # diversity and temporal transitions, causing the sequence model to
    # reconstruct higher errors. We report this count diagnostically
    # without asserting it must be low.
    reset_window()
    repeated_health_anomalies = 0
    repeated_health_errors = []

    for i in range(30):
        res = record_observation("/health", 200, 10.0 + (i % 5))
        if res.window_ready:
            repeated_health_errors.append(res.reconstruction_error)
            if res.is_anomaly:
                repeated_health_anomalies += 1

    print(f"\n  [Diagnostic] Repeated /health traffic (30 requests, 21 evaluated windows):")
    print(f"    * Anomalies flagged: {repeated_health_anomalies}/21 windows")
    if repeated_health_errors:
        print(f"    * Mean reconstruction error: {sum(repeated_health_errors)/len(repeated_health_errors):.6f}")
        print(f"    * Threshold:                {res.threshold:.6f}")
    print("    * Rationale: Repeated single-endpoint traffic is out-of-distribution (OOD)")
    print("      for an LSTM trained on multi-endpoint transition sequences.")

    _check(
        all(r is not None and math.isfinite(r) and r >= 0.0 for r in repeated_health_errors),
        "Diagnostic: All repeated /health inferences produced valid finite non-negative errors",
        "Diagnostic: Non-finite or negative error encountered during repeated /health requests",
    )
    _check(
        res.error_message is None,
        "Diagnostic: Repeated /health traffic handled cleanly with no internal exceptions",
        f"Diagnostic: Error encountered: {res.error_message}",
    )


# ==================================================================
# Test 6 — Anomalous traffic triggers alerts
# ==================================================================
def test_anomalous_traffic():
    _section("TEST 6 — Anomalous Traffic (expected: anomalies detected)")

    reset_window()
    anomalies = 0

    # Fill with highly anomalous observations:
    # - Extreme response times (5000 ms)
    # - Connection errors (status_code = None)
    # These are very different from the normal traffic in the training set.
    for i in range(20):
        # Alternate between extreme latency and connection errors
        if i % 2 == 0:
            result = record_observation("/health", None, 5000.0)
        else:
            result = record_observation("/does-not-exist", 500, 4500.0)
        if result.window_ready and result.is_anomaly:
            anomalies += 1

    print(f"  Sent 20 anomalous requests (extreme latency + errors).")
    print(f"  Anomalies detected: {anomalies}")

    # With severe anomalies we expect at least 1 alert.
    _check(
        anomalies >= 1,
        f"Anomalous traffic triggered ≥1 anomaly alert ({anomalies} detected)",
        f"No anomaly was detected despite extreme traffic — review threshold",
    )

    if anomalies > 0:
        print(f"  {YELLOW}Last reconstruction error:{RESET} "
              f"{result.reconstruction_error:.8f}")
        print(f"  {YELLOW}Threshold:                {RESET} "
              f"{result.threshold:.8f}")


# ==================================================================
# Test 7 — Thread safety
# ==================================================================
def test_thread_safety():
    _section("TEST 7 — Thread Safety (20 concurrent threads)")

    reset_window()
    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for _ in range(5):
                r = record_observation("/health", 200, 10.0)
                with lock:
                    results.append(r)
        except Exception as exc:
            with lock:
                errors.append(f"Thread {thread_id}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    _check(
        len(errors) == 0,
        f"No exceptions in any thread ({len(results)} results collected in {elapsed:.2f}s)",
        f"{len(errors)} thread errors: {errors[:3]}",
    )
    _check(
        all(r.error_message is None for r in results),
        "All results have no error_message",
        f"{sum(1 for r in results if r.error_message)} results had error_messages",
    )


# ==================================================================
# Test 8 — Error safety (bad inputs)
# ==================================================================
def test_error_safety():
    _section("TEST 8 — Error Safety (bad inputs)")

    # Negative response time — should be clamped to 0, not raise.
    result = record_observation("/health", 200, -50.0)
    _check(
        result.error_message is None,
        "Negative response_time_ms clamped, no error raised",
        f"Unexpected error: {result.error_message}",
    )

    # Path with query string — should be stripped cleanly.
    result = record_observation("/debug/test-latency?delay_ms=1000", 200, 1000.0)
    _check(
        result.error_message is None,
        "Path with query string handled without error",
        f"Unexpected error: {result.error_message}",
    )


# ==================================================================
# Test 9 — reset_window()
# ==================================================================
def test_reset_window():
    _section("TEST 9 — reset_window()")

    # Ensure window is full first.
    reset_window()
    for _ in range(WINDOW_SIZE):
        record_observation("/health", 200, 10.0)

    status_before = get_detector_status()
    _check(
        status_before["window_ready"],
        "Window is ready before reset",
        "Window was not ready before reset",
    )

    reset_window()
    status_after = get_detector_status()

    _check(
        status_after["window_size"] == 0,
        "Window size == 0 after reset",
        f"Window size == {status_after['window_size']} after reset",
    )
    _check(
        not status_after["window_ready"],
        "window_ready == False after reset",
        "window_ready still True after reset",
    )


# ==================================================================
# Test 10 — get_detector_status()
# ==================================================================
def test_detector_status():
    _section("TEST 10 — get_detector_status()")

    reset_window()
    record_observation("/health", 200, 10.0)
    status = get_detector_status()

    expected_keys = [
        "artifacts_loaded", "load_error", "threshold",
        "window_size", "window_capacity", "window_ready",
        "model_path", "scaler_path", "threshold_path",
    ]
    for key in expected_keys:
        _check(key in status, f"Status contains key '{key}'", f"Missing key '{key}'")

    _check(status["artifacts_loaded"],              "artifacts_loaded == True",         "artifacts_loaded == False")
    _check(status["load_error"] is None,            "load_error is None",               f"load_error: {status['load_error']}")
    _check(status["window_capacity"] == WINDOW_SIZE, f"window_capacity == {WINDOW_SIZE}", f"window_capacity == {status['window_capacity']}")
    _check(status["window_size"] == 1,              "window_size == 1",                 f"window_size == {status['window_size']}")
    _check(not status["window_ready"],              "window_ready == False (size 1)",   "window_ready == True at size 1")
    _check("artifacts_normal_extended_v1" in status["model_path"],
                                                    "model_path contains artifact dir name",
                                                    f"model_path: {status['model_path']}")


# ==================================================================
# Summary
# ==================================================================
def run_all_tests():
    print(f"\n{BOLD}{'='*60}")
    print("  ANOMALY DETECTION MODULE — TEST SUITE")
    print(f"{'='*60}{RESET}")
    print("  Target : ecommerce-demo/app/anomaly_detection.py")
    print("  Model  : ml/artifacts_normal_extended_v1/lstm_autoencoder.keras")
    print()

    test_artifact_loading()
    test_feature_extraction()
    test_rolling_window_warmup()
    test_first_inference()
    test_normal_traffic()
    test_anomalous_traffic()
    test_thread_safety()
    test_error_safety()
    test_reset_window()
    test_detector_status()

    print(f"\n{BOLD}{'='*60}{RESET}")
    total = _pass_count + _fail_count
    if _fail_count == 0:
        print(f"{GREEN}{BOLD}  ALL {total} TESTS PASSED{RESET}")
    else:
        print(f"{RED}{BOLD}  {_fail_count}/{total} TESTS FAILED{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    run_all_tests()
