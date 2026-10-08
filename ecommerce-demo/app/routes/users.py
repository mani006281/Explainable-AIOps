from fastapi import APIRouter
from sqlalchemy import text

from app.database import engine


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.get("/")
def get_users():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT id, name, email, created_at
                FROM users
                ORDER BY id;
            """)
        )

        users = [
            {
                "id": row.id,
                "name": row.name,
                "email": row.email,
                "created_at": row.created_at.isoformat(),
            }
            for row in result
        ]

    return users