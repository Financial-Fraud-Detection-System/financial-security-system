import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, List

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord


class KafkaClient:
    """
    A simple Kafka client for sending and consuming messages.
    This client also sends periodic health metrics to a specified topic.

    Example usage::

        from kafka_client import KafkaClient

        # Initialize Kafka client
        kafka_client = KafkaClient(service_name="example_service")

        # Sending a message
        kafka_client.send("example-topic", "value", "key")

        # Consuming messages
        def print_message(key, value):
            print(f"Received message: key={key}, value={value}")
        kafka_client.consume("example-topic", group_id="example-group", on_message=print_message)
    """

    def __init__(
        self,
        service_name: str,
        brokers: str | List[str] = "localhost:9092",
        health_topic: str = "health-metrics",
        health_interval: int = 30,
    ):
        """
        Initializes the Kafka client.

        :param service_name: Name of the service using this client.
        :param brokers: Kafka broker addresses.
        :param health_topic: Topic for sending health metrics.
        :param health_interval: Interval (in seconds) for sending health metrics.
        """
        self._logger = logging.getLogger("kafka_client").getChild(service_name)
        self._logger.debug("Initializing")
        self.service_name = service_name
        self.brokers = brokers
        self.health_topic = health_topic
        self.health_interval = health_interval
        self.producer = KafkaProducer(
            bootstrap_servers=self.brokers,
        )
        self._stop_event = threading.Event()

        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def send(
        self,
        topic: str,
        value,
        key=None,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    ):
        """
        Sends a message to the specified Kafka topic.

        :param topic: Topic to send the message to.
        :param value: Message value.
        :param key: Message key (optional).
        :param key_serializer: Function to serialize the key (default is UTF-8 encoding).
        :param value_serializer: Function to serialize the value (default is JSON encoding).
        :return: None
        """
        self._logger.getChild(topic).debug(f"Sending message: key={key}, value={value}")
        message = {}
        if key is not None:
            try:
                message["key"] = key_serializer(key)
            except Exception as e:
                self._logger.warning(f"Error serializing key: {e}")
                return
        try:
            message["value"] = value_serializer(value)
        except Exception as e:
            self._logger.warning(f"Error serializing value: {e}")
            return
        self.producer.send(topic, **message)

    def consume(
        self,
        topic: str,
        group_id: str,
        on_message: Callable[[str, dict], None],
        key_deserializer=lambda k: k.decode("utf-8"),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    ):
        """
        Consumes messages from the specified Kafka topic.
        This method blocks until the consumer is stopped.
        To shadow consume, set the `enable_auto_commit` parameter to False.

        :param topic: Topic to consume messages from.
        :param group_id: Consumer group ID.
        :param on_message: Callback function to handle messages.
        :param key_deserializer: Function to deserialize the key (default is UTF-8 decoding).
        :param value_deserializer: Function to deserialize the value (default is JSON decoding).
        :param auto_offset_reset: Offset reset policy (default is "latest").
        :param enable_auto_commit: Whether to enable auto commit (default is True).
        :return: None
        """
        self._logger.debug(f"Starting consumer for topic: {topic}")
        topicLogger = self._logger.getChild(topic)
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.brokers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
        )
        while not self._stop_event.is_set():
            message: ConsumerRecord = next(consumer)
            topicLogger.debug(f"Received message: {message}")
            try:
                key = key_deserializer(message.key)
            except Exception as e:
                topicLogger.warning(f"Error deserializing key: {e}")
                continue
            try:
                value = value_deserializer(message.value)
            except Exception as e:
                topicLogger.warning(f"Error deserializing value: {e}")
                continue
            on_message(key, value)
        self._logger.debug(f"Stopped consumer for topic: {topic}")

    def _health_loop(self):
        self._logger.debug("Starting health check thread")
        while not self._stop_event.is_set():
            health_data = {
                "service": self.service_name,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
            }
            self.send(self.health_topic, health_data)
            time.sleep(self.health_interval)
        self._logger.debug("Stopping health check thread")

    def stop(self, timeout: int = 5):
        """
        Stops the Kafka client and its health check thread.

        :return: None
        """
        self._stop_event.set()
        self._health_thread.join(timeout)
        if self._health_thread.is_alive():
            self._logger.warning(
                "Health check thread did not fully terminate during close"
            )
        self.producer.close(timeout=timeout)
