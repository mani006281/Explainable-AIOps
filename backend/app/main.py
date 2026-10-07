from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.database import test_database_connection


app = FastAPI(
    title="Explainable AIOps Framework",
    description=(
        "Real-Time Cloud Log Anomaly Detection, "
        "Root Cause Analysis and Incident Response"
    ),
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Explainable AIOps Framework is running",
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