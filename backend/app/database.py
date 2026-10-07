import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "aiops_db")

if not DATABASE_USER or not DATABASE_PASSWORD:
    raise RuntimeError("Database credentials are not configured")

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