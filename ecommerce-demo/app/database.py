from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# Load ecommerce-demo/.env
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)


DATABASE_USER = os.getenv("ECOMMERCE_DB_USER")
DATABASE_PASSWORD = os.getenv("ECOMMERCE_DB_PASSWORD")
DATABASE_HOST = os.getenv("ECOMMERCE_DB_HOST", "localhost")
DATABASE_PORT = os.getenv("ECOMMERCE_DB_PORT", "5432")
DATABASE_NAME = os.getenv("ECOMMERCE_DB_NAME", "ecommerce_db")


if not DATABASE_USER or not DATABASE_PASSWORD:
    raise RuntimeError("E-commerce database credentials are not configured")


engine = create_engine(
    "postgresql+psycopg://",
    connect_args={
        "user": DATABASE_USER,
        "password": DATABASE_PASSWORD,
        "host": DATABASE_HOST,
        "port": DATABASE_PORT,
        "dbname": DATABASE_NAME,
    },
)


def test_database_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database();")
        )
        return result.scalar()