from uuid import UUID

from pydantic import BaseModel

from ..loan import Loan
from ..transaction import Transaction


class TransactionAnomalyJob(BaseModel):
    """
    Transaction anomaly model.
    """

    id: UUID
    transaction: Transaction


class CreditRiskJob(BaseModel):
    """
    Credit risk model.
    """

    id: UUID
    loan: Loan


class FraudNetworkJob(BaseModel):
    """
    Fraud network model.
    """

    id: UUID
    transactions: list[Transaction]
