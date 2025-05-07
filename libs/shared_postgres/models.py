import uuid

from sqlalchemy import Boolean, Column, DateTime, Enum, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

from .enums import JobStatus

Base = declarative_base()


class TransactionAnomalyJob(Base):  # type: ignore
    __tablename__ = "transaction_anomaly_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.queued)
    is_anomaly = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
