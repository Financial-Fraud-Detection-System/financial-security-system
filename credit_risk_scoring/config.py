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
    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8123")
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "password")
    CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default")
    CLICKHOUSE_LOGS_TABLE = os.getenv("CLICKHOUSE_LOGS_TABLE", "logs")
