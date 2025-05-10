import os


class Config:
    SERVICE_NAME = os.getenv("SERVICE_NAME", "backend")
    KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
    KAFKA_TRANSACTION_ANOMALY_JOB_TOPIC = os.getenv(
        "KAFKA_TRANSACTION_ANOMALY_JOB_TOPIC", "transaction-anomaly-job"
    )
    KAFKA_CREDIT_RISK_JOB_TOPIC = os.getenv(
        "KAFKA_CREDIT_RISK_JOB_TOPIC", "credit-risk-job"
    )
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/financial_db"
    )
