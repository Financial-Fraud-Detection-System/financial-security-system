import logging

from config import Config
from handlers.credit_risk_job_handler import credit_risk_job_handler

from libs.shared_kafka.kafka_client import KafkaClient
from libs.shared_postgres.db import initialize_database
from libs.shared_postgres.models import Base

initialize_database(Base, Config.DATABASE_URL)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Initialize Kafka client
    kafka_client = KafkaClient(
        service_name=Config.SERVICE_NAME, brokers=Config.KAFKA_BROKERS
    )

    # Consuming messages
    try:
        kafka_client.consume(
            Config.KAFKA_TRANSACTION_ANOMALY_JOB_TOPIC,
            group_id=Config.KAFKA_GROUP_ID,
            on_message=credit_risk_job_handler,
        )
    except KeyboardInterrupt:
        logging.info("Stopping Kafka client...")
    finally:
        kafka_client.stop()
