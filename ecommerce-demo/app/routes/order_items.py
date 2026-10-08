from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.database import engine

router = APIRouter(prefix="/orders", tags=["Order Items"])


class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int
    price: float


@router.post("/{order_id}/items")
def create_order_item(order_id: int, item: OrderItemCreate):

    with engine.begin() as connection:

        # Check whether the order exists
        order = connection.execute(
            text("""
                SELECT id
                FROM orders
                WHERE id = :order_id
            """),
            {"order_id": order_id},
        ).fetchone()

        if not order:
            raise HTTPException(
                status_code=404,
                detail="Order not found",
            )

        # Check whether the product exists
        product = connection.execute(
            text("""
                SELECT id
                FROM products
                WHERE id = :product_id
            """),
            {"product_id": item.product_id},
        ).fetchone()

        if not product:
            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )

        # Insert order item
        result = connection.execute(
            text("""
                INSERT INTO order_items
                (order_id, product_id, quantity, price)
                VALUES
                (:order_id, :product_id, :quantity, :price)
                RETURNING id, order_id, product_id, quantity, price
            """),
            {
                "order_id": order_id,
                "product_id": item.product_id,
                "quantity": item.quantity,
                "price": item.price,
            },
        )

        row = result.fetchone()

        return {
            "id": row.id,
            "order_id": row.order_id,
            "product_id": row.product_id,
            "quantity": row.quantity,
            "price": float(row.price),
        }