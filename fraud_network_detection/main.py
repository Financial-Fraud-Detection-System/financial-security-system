import logging

from config import Config
from handlers.fraud_network_job import fraud_network_job_handler

from libs.shared_clickhouse.db import ClickHouseDatabase
from libs.shared_clickhouse.log_handlers import ClickHouseLogHandler
from libs.shared_kafka.kafka_client import KafkaClient
from libs.shared_postgres.db import initialize_database as initialize_postgres_database
from libs.shared_postgres.models import Base as PostgresBase

initialize_postgres_database(PostgresBase, Config.DATABASE_URL)

clickhouse_db = ClickHouseDatabase(
    host=Config.CLICKHOUSE_HOST,
    port=Config.CLICKHOUSE_PORT,
    user=Config.CLICKHOUSE_USER,
    password=Config.CLICKHOUSE_PASSWORD,
    database=Config.CLICKHOUSE_DATABASE,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            ClickHouseLogHandler(
                clickhouse_db, Config.SERVICE_NAME, table=Config.CLICKHOUSE_LOGS_TABLE
            ),
        ],
    )

    # Initialize Kafka client
    kafka_client = KafkaClient(
        service_name=Config.SERVICE_NAME, brokers=Config.KAFKA_BROKERS
    )

    # Consuming messages
    try:
        kafka_client.consume(
            Config.KAFKA_FRAUD_NETWORK_DETECTION_TOPIC,
            group_id=Config.KAFKA_GROUP_ID,
            on_message=fraud_network_job_handler,
        )
    except KeyboardInterrupt:
        logging.info("Stopping Kafka client...")
    finally:
        kafka_client.stop()
