import json
import time
from unittest.mock import MagicMock, patch

import pytest
from kafka_client import KafkaClient


@pytest.fixture
def kafka_client():
    with patch("kafka_client.KafkaProducer") as MockProducer:
        mock_producer = MagicMock()
        MockProducer.return_value = mock_producer
        client = KafkaClient(service_name="test_service", health_interval=1)
        yield client
        client.stop()


def test_send_message(kafka_client):
    kafka_client.producer = MagicMock()
    kafka_client.producer.send = MagicMock()

    kafka_client.send("test-topic", {"foo": "bar"}, key="test-key")

    kafka_client.producer.send.assert_called_once()
    (topic_arg,), kwargs = kafka_client.producer.send.call_args
    assert topic_arg == "test-topic"
    assert kwargs["key"] == b"test-key"
    assert kwargs["value"] == json.dumps({"foo": "bar"}).encode("utf-8")


def test_send_serialization_errors(kafka_client, caplog):
    kafka_client.producer.send = MagicMock()

    # Cause key serialization failure
    kafka_client.send("test-topic", {"val": 1}, key=object())
    assert "Error serializing key" in caplog.text

    # Cause value serialization failure
    kafka_client.send("test-topic", object())
    assert "Error serializing value" in caplog.text


def test_health_check_runs(kafka_client):
    kafka_client.producer.send = MagicMock()

    time.sleep(2.2)  # Wait for health check to run at least once
    kafka_client.stop()

    # Health check should have sent at least once
    assert kafka_client.producer.send.call_count >= 1


def test_consume_logic():
    with (
        patch("kafka_client.KafkaProducer"),
        patch("kafka_client.KafkaConsumer") as MockConsumer,
    ):
        fake_record = MagicMock()
        fake_record.key = b"key"
        fake_record.value = json.dumps({"val": 42}).encode("utf-8")

        mock_consumer = MagicMock()
        mock_consumer.__next__.side_effect = [fake_record]
        MockConsumer.return_value = mock_consumer

        client = KafkaClient(service_name="consume_test", health_interval=1)
        received = []

        def handle_message(k, v):
            received.append((k, v))
            client.stop()

        try:
            client.consume(
                "test-topic",
                "test-group",
                on_message=handle_message,
            )
        except StopIteration:
            pass

        assert received[0][0] == "key"
        assert received[0][1] == {"val": 42}
