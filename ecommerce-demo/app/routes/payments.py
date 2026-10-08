from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.database import engine


router = APIRouter(
    prefix="/payments",
    tags=["Payments"],
)


class PaymentCreate(BaseModel):
    order_id: int


@router.post("/")
def create_payment(payment: PaymentCreate):

    with engine.begin() as connection:

        # Get the order and its total amount
        order = connection.execute(
            text("""
                SELECT id, total_amount, status
                FROM orders
                WHERE id = :order_id;
            """),
            {"order_id": payment.order_id},
        ).fetchone()

        if order is None:
            raise HTTPException(
                status_code=404,
                detail="Order not found",
            )

        # Create a simulated successful payment
        result = connection.execute(
            text("""
                INSERT INTO payments
                (order_id, amount, status)
                VALUES
                (:order_id, :amount, 'SUCCESS')
                RETURNING id, order_id, amount, status, created_at;
            """),
            {
                "order_id": order.id,
                "amount": order.total_amount,
            },
        )

        row = result.fetchone()

        # Update the order after successful payment
        connection.execute(
            text("""
                UPDATE orders
                SET status = 'CONFIRMED'
                WHERE id = :order_id;
            """),
            {"order_id": order.id},
        )

    return {
        "id": row.id,
        "order_id": row.order_id,
        "amount": float(row.amount),
        "status": row.status,
        "created_at": row.created_at.isoformat(),
    }