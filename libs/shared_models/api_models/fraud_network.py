from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ..online_transaction import OnlineTransaction
from .enums import JobStatus


class FraudNetworkRequest(BaseModel):
    """
    A request for fraud network detection service to detect fraud networks with transactions.
    """

    transactions: list[OnlineTransaction]


class FraudNetworkResult(BaseModel):
    """
    Result of fraud network detection service.
    """

    id: UUID = Field(alias="job_id", description="Unique identifier for the job")
    status: JobStatus = Field(alias="job_status", description="Status of the job")
    fraud_networks: list[list[str]] | None
    created_at: datetime = Field(
        description="Timestamp when the job was created", example="2023-10-01T12:00:00Z"
    )
