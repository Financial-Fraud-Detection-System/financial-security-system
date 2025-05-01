import logging

from config import Config
from handlers.dummy_handler import dummy_handler

from libs.shared_kafka.kafka_client import KafkaClient

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
            Config.KAFKA_DUMMY_TOPIC,
            group_id=Config.KAFKA_GROUP_ID,
            on_message=dummy_handler,
        )
    except KeyboardInterrupt:
        logging.info("Stopping Kafka client...")
    finally:
        kafka_client.stop()
