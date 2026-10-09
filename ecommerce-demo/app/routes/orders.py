from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.database import engine
from app.logging.logger import get_logger


logger = get_logger()


router = APIRouter(
    prefix="/orders",
    tags=["Orders"],
)


class OrderCreate(BaseModel):
    user_id: int
    total_amount: float


@router.post("/")
def create_order(order: OrderCreate):
    with engine.begin() as connection:

        # Check whether the user exists
        user = connection.execute(
            text("""
                SELECT id
                FROM users
                WHERE id = :user_id;
            """),
            {"user_id": order.user_id},
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        # Create the order
        result = connection.execute(
            text("""
                INSERT INTO orders (user_id, total_amount)
                VALUES (:user_id, :total_amount)
                RETURNING id, user_id, total_amount, status, created_at;
            """),
            {
                "user_id": order.user_id,
                "total_amount": order.total_amount,
            },
        )

        row = result.fetchone()

    # Structured business event log
    logger.info(
        "Order created successfully",
        extra={
            "event": "order_created",
            "context": {
                "order_id": row.id,
                "user_id": row.user_id,
                "total_amount": float(row.total_amount),
                "status": row.status,
            },
        },
    )

    return {
        "id": row.id,
        "user_id": row.user_id,
        "total_amount": float(row.total_amount),
        "status": row.status,
        "created_at": row.created_at.isoformat(),
    }


@router.get("/{order_id}")
def get_order(order_id: int):

    with engine.connect() as connection:

        order = connection.execute(
            text("""
                SELECT
                    o.id,
                    o.user_id,
                    u.name AS user_name,
                    o.total_amount,
                    o.status,
                    o.created_at
                FROM orders o
                JOIN users u ON o.user_id = u.id
                WHERE o.id = :order_id;
            """),
            {"order_id": order_id},
        ).fetchone()

        if order is None:
            raise HTTPException(
                status_code=404,
                detail="Order not found",
            )

        items_result = connection.execute(
            text("""
                SELECT
                    oi.id,
                    oi.product_id,
                    p.name AS product_name,
                    oi.quantity,
                    oi.price
                FROM order_items oi
                JOIN products p ON oi.product_id = p.id
                WHERE oi.order_id = :order_id
                ORDER BY oi.id;
            """),
            {"order_id": order_id},
        )

        items = [
            {
                "id": row.id,
                "product_id": row.product_id,
                "product_name": row.product_name,
                "quantity": row.quantity,
                "price": float(row.price),
            }
            for row in items_result
        ]

    return {
        "id": order.id,
        "user_id": order.user_id,
        "user_name": order.user_name,
        "total_amount": float(order.total_amount),
        "status": order.status,
        "created_at": order.created_at.isoformat(),
        "items": items,
    }