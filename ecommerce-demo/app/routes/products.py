from fastapi import APIRouter
from sqlalchemy import text

from app.database import engine


router = APIRouter(
    prefix="/products",
    tags=["Products"],
)


@router.get("/")
def get_products():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT id, name, description, price, stock
                FROM products
                ORDER BY id;
            """)
        )

        products = [
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "price": float(row.price),
                "stock": row.stock,
            }
            for row in result
        ]

    return products