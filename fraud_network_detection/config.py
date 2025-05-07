import os


class Config:
    SERVICE_NAME = os.getenv("SERVICE_NAME", "fraud-network-detection")
    KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
    KAFKA_DUMMY_TOPIC = os.getenv("KAFKA_DUMMY_TOPIC", "dummy")
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-network-detection")
