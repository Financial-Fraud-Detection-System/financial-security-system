import json
import logging
from uuid import UUID

from config import Config
from models.predict_fraud import process_and_predict

from libs.shared_models.kafka_models.jobs import FraudNetworkJob as KafkaJob
from libs.shared_postgres.db import get_session_maker
from libs.shared_postgres.enums import JobStatus
from libs.shared_postgres.models import FraudNetworkJob as Job

session_maker = get_session_maker(Config.DATABASE_URL)


def fraud_network_job_handler(_: str, message: dict) -> None:
    logger = logging.getLogger(__name__)

    try:
        kafka_job = KafkaJob(**message)
        logger.info(f"Processing job: {kafka_job}")
        transaction_dict_list = (
            [t.model_dump(mode="json") for t in kafka_job.transactions]
            if kafka_job.transactions
            else []
        )

        # Step 1: Mark job as processing
        mark_job_as_processing(kafka_job.id)

        # Step 2: Run model
        result = process_and_predict(
            transaction_dict_list,
        )

        # Step 3: Save results
        mark_job_as_done(kafka_job.id, [ring[1] for ring in result])

        logger.info(
            f"Job processed successfully: {kafka_job.id} - {len(result)} new fraud rings detected"
        )

    except Exception as e:
        logger.error(f"Error processing job: {e}")
        try:
            mark_job_as_failed(kafka_job.id)
        except Exception as rollback_err:
            logger.error(f"Failed to update job status to failed: {rollback_err}")


def mark_job_as_processing(job_id: UUID) -> Job:
    """
    Mark the job as processing in the database.
    Args:
        job_id (UUID): The ID of the job to mark as processing.
    Returns:
        Job: The updated job object.
    Raises:
        ValueError: If the job ID is not found in the database.
    """
    with session_maker() as db:
        job: Job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError("Job id not found")
        job.status = JobStatus.processing
        db.commit()
        return job


def mark_job_as_done(job_id: UUID, fraud_networks: list[list[str]]) -> Job:
    """
    Mark the job as done in the database.
    Args:
        job_id (UUID): The ID of the job to mark as done.
        is_anomaly (bool): Whether the transaction is an anomaly or not.
    Returns:
        Job: The updated job object.
    Raises:
        ValueError: If the job ID is not found in the database.
    """
    with session_maker() as db:
        job: Job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError("Job id not found")
        job.status = JobStatus.done
        job.fraud_networks = json.dumps(fraud_networks)
        db.commit()
        return job


def mark_job_as_failed(job_id: UUID) -> Job:
    """
    Mark the job as failed in the database.
    Args:
        job_id (UUID): The ID of the job to mark as failed.
    Returns:
        Job: The updated job object.
    Raises:
        ValueError: If the job ID is not found in the database.
    """
    with session_maker() as db:
        job: Job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError("Job id not found")
        job.status = JobStatus.failed
        db.commit()
        return job
