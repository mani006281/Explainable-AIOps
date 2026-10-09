from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from app.database import engine
from app.logging.logger import get_logger


logger = get_logger()


router = APIRouter(
    prefix="/payments",
    tags=["Payments"],
)


class PaymentCreate(BaseModel):
    order_id: int


@router.post("/")
def create_payment(payment: PaymentCreate):

    with engine.begin() as connection:

        # Get the order
        order = connection.execute(
            text("""
                SELECT id, user_id, total_amount, status
                FROM orders
                WHERE id = :order_id;
            """),
            {"order_id": payment.order_id},
        ).fetchone()

        # -------------------------------------------------
        # Failure: Order does not exist
        # -------------------------------------------------
        if order is None:

            logger.warning(
                "Payment failed because order was not found",
                extra={
                    "event": "payment_failed",
                    "context": {
                        "order_id": payment.order_id,
                        "reason": "Order not found",
                    },
                },
            )

            raise HTTPException(
                status_code=404,
                detail="Order not found",
            )

        # -------------------------------------------------
        # Failure: Order already confirmed
        # -------------------------------------------------
        if order.status == "CONFIRMED":

            logger.warning(
                "Payment failed because order is already confirmed",
                extra={
                    "event": "payment_failed",
                    "context": {
                        "order_id": order.id,
                        "user_id": order.user_id,
                        "reason": "Order is already confirmed",
                        "order_status": order.status,
                    },
                },
            )

            raise HTTPException(
                status_code=400,
                detail="Order is already confirmed",
            )

        # -------------------------------------------------
        # Create simulated successful payment
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Update order status after successful payment
        # -------------------------------------------------
        connection.execute(
            text("""
                UPDATE orders
                SET status = 'CONFIRMED'
                WHERE id = :order_id;
            """),
            {"order_id": order.id},
        )

    # -----------------------------------------------------
    # Structured business event: payment success
    # -----------------------------------------------------
    logger.info(
        "Payment completed successfully",
        extra={
            "event": "payment_success",
            "context": {
                "payment_id": row.id,
                "order_id": row.order_id,
                "user_id": order.user_id,
                "amount": float(row.amount),
                "payment_status": row.status,
                "order_status": "CONFIRMED",
            },
        },
    )

    return {
        "id": row.id,
        "order_id": row.order_id,
        "amount": float(row.amount),
        "status": row.status,
        "created_at": row.created_at.isoformat(),
    }