import os


class Config:
    SERVICE_NAME = os.getenv("SERVICE_NAME", "fraud-network-detection")
    KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
    KAFKA_FRAUD_NETWORK_DETECTION_TOPIC = os.getenv(
        "KAFKA_FRAUD_NETWORK_DETECTION_TOPIC", "fraud-network-detection"
    )
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-network-detection")
    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8123")
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "password")
    CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "fraud_detection")
    CLICKHOUSE_LOGS_TABLE = os.getenv("CLICKHOUSE_LOGS_TABLE", "logs")
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/financial_db"
    )
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
