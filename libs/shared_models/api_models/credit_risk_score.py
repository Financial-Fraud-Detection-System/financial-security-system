from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ..loan import Loan
from .enums import CreditRiskType, JobStatus


class CreditRiskRequest(BaseModel):
    """
    A request for transaction anomaly detection service to detect anomalies in a transaction.
    """

    loan: Loan


class CreditRiskResult(BaseModel):
    """
    Result of a credit risk evaluation job.
    """

    id: UUID = Field(description="Unique identifier for the job")
    status: JobStatus = Field(description="Status of the job")
    risk_type: CreditRiskType | None = Field(
        description="Indicates type of credit risk evaluated or not"
    )
    created_at: datetime = Field(
        description="Timestamp when the job was created", example="2023-10-01T12:00:00Z"
    )
