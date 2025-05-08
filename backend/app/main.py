"""
This module defines a FastAPI application for the backend of the Financial Security System.
Running as a script will start a development uvicorn server serving the app.
"""

import uvicorn
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from libs.shared_kafka.kafka_client import KafkaClient
from libs.shared_models.api_models.credit_risk_score import (
    CreditRiskRequest,
    CreditRiskResult,
)
from libs.shared_models.api_models.transaction_anomaly import (
    TransactionAnomalyRequest,
    TransactionAnomalyResult,
)
from libs.shared_models.kafka_models.jobs import CreditRiskJob as KafkaCreditRiskJob
from libs.shared_models.kafka_models.jobs import (
    TransactionAnomalyJob as KafkaTransactionAnomalyJob,
)
from libs.shared_postgres.db import get_session_maker, initialize_database
from libs.shared_postgres.models import Base, CreditRiskJob, TransactionAnomalyJob

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

    Note:
        The incoming transaction data is not persisted in the database. Instead, it is forwarded
        to Kafka for processing. This design decision ensures that the database only stores
        metadata about the job, while the complete transaction details are handled externally.
    """

    # Create a new transaction anomaly job
    # The transaction data is not stored in the database to keep it lightweight. Instead, it is
    # forwarded to Kafka for processing.
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


@app.post("/credit_risk", response_model=CreditRiskResult)
def queue_credit_score_prediction_job(
    request: CreditRiskRequest, db: Session = Depends(get_db)
):
    """
    Queues credit risk score prediction jobs for background processing.

    Note:
        The incoming transaction data is not persisted in the database. Instead, it is forwarded
        to Kafka for processing. This design decision ensures that the database only stores
        metadata about the job, while the complete transaction details are handled externally.
    """

    # Create a new credit risk score prediction job
    # The loan data is not stored in the database to keep it lightweight. Instead, it is
    # forwarded to Kafka for processing.
    credit_risk_job = CreditRiskJob()

    # Add the job to the session
    db.add(credit_risk_job)
    db.commit()

    # Create a Kafka job object
    kafka_job = KafkaCreditRiskJob(
        id=credit_risk_job.id,
        laon=request.loan,
    )

    # Send the job to Kafka
    kafka_client.send(
        Config.KAFKA_CREDIT_RISK_JOB_TOPIC,
        kafka_job.model_dump(mode="json"),
        key=str(credit_risk_job.id),
    )

    return CreditRiskResult(
        job_id=credit_risk_job.id,
        job_status=credit_risk_job.status,
        is_anomaly=credit_risk_job.is_anomaly,
        created_at=credit_risk_job.created_at,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
