from uuid import UUID

from pydantic import BaseModel

from ..transaction import Transaction


class TransactionAnomalyJob(BaseModel):
    """
    Transaction anomaly model.
    """

    id: UUID
    transaction: Transaction
