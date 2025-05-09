import os


class Config:
    SERVICE_NAME = os.getenv("SERVICE_NAME", "credit-risk-score")
    KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
    KAFKA_CREDIT_RISK_JOB_TOPIC = os.getenv(
        "KAFKA_CREDIT_RISK_JOB_TOPIC", "credit-risk-job"
    )
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "credit-risk-score")
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/financial_db"
    )
