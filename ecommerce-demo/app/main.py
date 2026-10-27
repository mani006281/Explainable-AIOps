import asyncio
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import test_database_connection
from app.logging.logger import get_logger

from app.anomaly_detection import (
    AnomalyResult,
    get_detector_status,
    record_observation,
)

from app.incident_store import (
    record_incident,
    get_incident_history,
)

from app.routes.products import router as products_router
from app.routes.users import router as users_router
from app.routes.orders import router as orders_router
from app.routes.order_items import router as order_items_router
from app.routes.payments import router as payments_router


app = FastAPI(
    title="E-Commerce Demo Application",
    description=(
        "Production-style e-commerce application "
        "for the Explainable AIOps project"
    ),
    version="0.1.0",
)


# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------

# Allow React frontend running on Vite development ports.
# Only explicitly listed local origins are permitted.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = get_logger()


# ---------------------------------------------------------
# Anomaly Detector State & Formatter
# ---------------------------------------------------------

_latest_detection_result: Optional[dict] = None


def _format_anomaly_result(result: AnomalyResult) -> dict:
    """
    Format an AnomalyResult for the public /aiops/status endpoint.
    Excludes internal artifact paths and credentials.
    """

    top_contributor = None

    if result.per_feature_errors:
        top_contributor = result.per_feature_errors[0]

    return {
        "window_ready": result.window_ready,
        "is_anomaly": result.is_anomaly,
        "reconstruction_error": (
            round(result.reconstruction_error, 6)
            if result.reconstruction_error is not None
            else None
        ),
        "threshold": round(result.threshold, 6),
        "window_size": result.window_size,
        "top_feature_contributor": top_contributor,
        "error_message": result.error_message,
    }


# ---------------------------------------------------------
# HTTP Request Logging & AIOps Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    global _latest_detection_result

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)

        status_code = response.status_code

        return response

    finally:
        response_time_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.info(
            "HTTP request completed",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
            },
        )

        # Exclude internal monitoring requests from anomaly observations.
        # Dashboard polling must not contaminate e-commerce traffic windows.
        req_path = request.url.path.rstrip("/") or "/"

        EXCLUDED_OBSERVATION_PATHS = {
            "/health",
            "/health/database",
            "/aiops/status",
            "/aiops/incidents",
        }

        if (
            request.method != "OPTIONS"
            and req_path not in EXCLUDED_OBSERVATION_PATHS
        ):
            try:
                # Offload to worker thread so TensorFlow inference
                # does not block the asyncio event loop.

                detection_result = await asyncio.to_thread(
                    record_observation,
                    request.url.path,
                    status_code,
                    response_time_ms,
                )

                _latest_detection_result = _format_anomaly_result(
                    detection_result
                )

                if detection_result.is_anomaly:

                    logger.warning(
                        "AIOps: Anomaly detected in HTTP request",
                        extra={
                            "event": "anomaly_detected",
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": status_code,
                            "response_time_ms": response_time_ms,
                            "reconstruction_error": (
                                detection_result.reconstruction_error
                            ),
                            "threshold": detection_result.threshold,
                            "top_contributor": (
                                detection_result.per_feature_errors[0]
                                if detection_result.per_feature_errors
                                else None
                            ),
                        },
                    )

                    # Record incident in bounded in-memory history.

                    try:
                        record_incident(
                            path=request.url.path,
                            status_code=status_code,
                            response_time_ms=response_time_ms,
                            reconstruction_error=(
                                detection_result.reconstruction_error
                            ),
                            threshold=detection_result.threshold,
                            per_feature_errors=(
                                detection_result.per_feature_errors
                            ),
                        )

                    except Exception as inc_exc:

                        # Incident recording failure must never
                        # break the HTTP response.

                        logger.error(
                            "AIOps: Failed to record incident in history",
                            extra={
                                "event": "incident_record_error",
                                "error_type": type(inc_exc).__name__,
                                "error_message": str(inc_exc),
                            },
                        )

            except Exception as exc:

                # Anomaly detection failures must never
                # fail an HTTP request.

                logger.error(
                    "AIOps: Anomaly detection middleware processing error",
                    extra={
                        "event": "anomaly_detection_error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )


# ---------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled application error",
        extra={
            "event": "application_error",
            "context": {
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
        },
    )


# ---------------------------------------------------------
# API Routers
# ---------------------------------------------------------

app.include_router(products_router)
app.include_router(users_router)
app.include_router(orders_router)
app.include_router(order_items_router)
app.include_router(payments_router)


# ---------------------------------------------------------
# Basic Health Endpoints
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "E-Commerce Demo Application is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
    }


@app.get("/health/database")
def database_health():

    database = test_database_connection()

    return {
        "database": database,
        "status": "connected",
    }


# ---------------------------------------------------------
# AIOps Status Endpoint
# ---------------------------------------------------------

@app.get("/aiops/status", tags=["AIOps"])
def aiops_status():
    """
    Read-only status endpoint for the Explainable AIOps anomaly detector.

    Exposes readiness, window size, saved threshold, and latest result.

    Does NOT expose credentials, database settings,
    or internal filesystem paths.
    """

    raw_status = get_detector_status()

    return {
        "service": "ecommerce-aiops",
        "detector_ready": raw_status.get(
            "window_ready",
            False,
        ),
        "window_size": raw_status.get(
            "window_size",
            0,
        ),
        "window_capacity": raw_status.get(
            "window_capacity",
            10,
        ),
        "saved_threshold": raw_status.get(
            "threshold",
            0.0,
        ),
        "artifacts_loaded": raw_status.get(
            "artifacts_loaded",
            False,
        ),
        "load_error": raw_status.get(
            "load_error"
        ),
        "latest_detection_result": _latest_detection_result,
    }


# ---------------------------------------------------------
# AIOps Incidents Endpoint
# ---------------------------------------------------------

@app.get("/aiops/incidents", tags=["AIOps"])
def aiops_incidents(limit: int = 100):
    """
    Read-only endpoint returning bounded in-memory incident history.

    Maximum 100 records, newest first.

    Does NOT expose credentials, database configuration,
    or internal file paths.
    """

    incidents = get_incident_history(limit=limit)

    return {
        "service": "ecommerce-aiops",
        "incident_count": len(incidents),
        "max_capacity": 100,
        "explanation_disclaimer": (
            "Feature contributions represent reconstruction-error "
            "contributions from the LSTM Autoencoder, "
            "not proven causal root causes."
        ),
        "incidents": incidents,
    }


# ---------------------------------------------------------
# Development-Only Latency Test Endpoint
# ---------------------------------------------------------

@app.get(
    "/debug/test-latency",
    include_in_schema=False,
)
async def debug_test_latency(delay_ms: int = 1000):

    # Require explicit development configuration.

    if os.getenv("APP_ENV") != "development":

        raise HTTPException(
            status_code=404,
            detail="Not found",
        )

    # Accept delays from 0 to 5000 milliseconds.

    if not 0 <= delay_ms <= 5000:

        raise HTTPException(
            status_code=400,
            detail="delay_ms must be between 0 and 5000",
        )

    await asyncio.sleep(delay_ms / 1000)

    return {
        "message": "Controlled latency test completed",
        "simulated_delay_ms": delay_ms,
    }