from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ..transaction import Transaction
from .enums import JobStatus


class TransactionAnomalyRequest(BaseModel):
    """
    A request for transaction anomaly detection service to detect anomalies in a transaction.
    """

    transaction: Transaction


class TransactionAnomalyResult(BaseModel):
    """
    Result of transaction anomaly detection service.
    """

    id: UUID = Field(alias="job_id", description="Unique identifier for the job")
    status: JobStatus = Field(alias="job_status", description="Status of the job")
    is_anomaly: bool | None = Field(
        description="Indicates whether the transaction is an anomaly or not",
    )
    created_at: datetime = Field(
        description="Timestamp when the job was created", example="2023-10-01T12:00:00Z"
    )
