"""
This module defines a FastAPI application for the backend of the Financial Security System.
Running as a script will start a development uvicorn server serving the app.
"""

import uvicorn
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from libs.shared_kafka.kafka_client import KafkaClient
from libs.shared_models.api_models.transaction_anomaly import (
    TransactionAnomalyRequest,
    TransactionAnomalyResult,
)
from libs.shared_models.kafka_models.jobs import (
    TransactionAnomalyJob as KafkaTransactionAnomalyJob,
)
from libs.shared_postgres.db import get_session_maker, initialize_database
from libs.shared_postgres.models import Base, TransactionAnomalyJob

from .config import Config

# Initialize the database models
initialize_database(Base, Config.DATABASE_URL)

# Initialize the database session maker
SessionLocal = get_session_maker(Config.DATABASE_URL)


# Initialize Kafka client
kafka_client = KafkaClient(
    service_name=Config.SERVICE_NAME, brokers=Config.KAFKA_BROKERS
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()


@app.get("/")
def read_root():
    """
    Handles GET requests to the root endpoint.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {
        "message": "Welcome to backend FastAPI server for Financial Security System!"
    }


@app.post("/transaction_anomaly", response_model=TransactionAnomalyResult)
def queue_transaction_anomaly_detection_job(
    request: TransactionAnomalyRequest, db: Session = Depends(get_db)
):
    """
    Queues transaction anomaly detection jobs for background processing.
    """

    # Create a new transaction anomaly job
    transaction_anomaly_job = TransactionAnomalyJob()

    # Add the job to the session
    db.add(transaction_anomaly_job)
    db.commit()

    # Create a Kafka job object
    kafka_job = KafkaTransactionAnomalyJob(
        id=transaction_anomaly_job.id,
        transaction=request.transaction,
    )

    # Send the job to Kafka
    kafka_client.send(
        Config.KAFKA_TRANSACTION_ANOMALY_JOB_TOPIC,
        kafka_job.model_dump(mode="json"),
        key=str(transaction_anomaly_job.id),
    )

    return TransactionAnomalyResult(
        job_id=transaction_anomaly_job.id,
        job_status=transaction_anomaly_job.status,
        is_anomaly=transaction_anomaly_job.is_anomaly,
        created_at=transaction_anomaly_job.created_at,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
