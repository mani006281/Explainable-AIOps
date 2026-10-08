from fastapi import FastAPI

from app.database import test_database_connection

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
app.include_router(products_router)
app.include_router(users_router)
app.include_router(orders_router)
app.include_router(order_items_router)
app.include_router(payments_router)


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