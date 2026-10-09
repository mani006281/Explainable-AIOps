import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.database import test_database_connection
from app.logging.logger import get_logger

from app.routes.products import router as products_router
from app.routes.users import router as users_router
from app.routes.orders import router as orders_router
from app.routes.order_items import router as order_items_router
from app.routes.payments import router as payments_router


app = FastAPI(
    title="E-Commerce Demo Application",
    description="Production-style e-commerce application for the Explainable AIOps project",
    version="0.1.0",
)


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------

logger = get_logger()


# ---------------------------------------------------------
# HTTP Request Logging Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
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