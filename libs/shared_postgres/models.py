import uuid

from sqlalchemy import Boolean, Column, DateTime, Enum, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

from .enums import CreditRiskType, JobStatus

Base = declarative_base()


class TransactionAnomalyJob(Base):  # type: ignore
    __tablename__ = "transaction_anomaly_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.queued)
    is_anomaly = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CreditRiskJob(Base):  # type: ignore
    __tablename__ = "credit_risk_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.queued)
    risk_type = Column(Enum(CreditRiskType), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FraudNetworkJob(Base):  # type: ignore
    __tablename__ = "fraud_network_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.queued)
    fraud_networks = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
