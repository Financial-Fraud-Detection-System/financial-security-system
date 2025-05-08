import logging

from config import Config
from models.predict_anomaly import anomaly_detection_wrapper

from libs.shared_models.kafka_models.jobs import TransactionAnomalyJob as KafkaJob
from libs.shared_postgres.db import get_session_maker
from libs.shared_postgres.enums import JobStatus
from libs.shared_postgres.models import TransactionAnomalyJob as Job

session_maker = get_session_maker(Config.DATABASE_URL)


def transaction_anomaly_job_handler(_: str, message: dict) -> None:
    logger = logging.getLogger(__name__)

    try:
        kafka_job = KafkaJob(**message)
        logger.info(f"Processing job: {kafka_job}")
        transaction_dict = kafka_job.transaction.model_dump(mode="json")

        # Step 1: Mark job as processing
        mark_job_as_processing(kafka_job.id)

        # Step 2: Run model
        result = anomaly_detection_wrapper(transaction_dict)

        # Step 3: Save results
        mark_job_as_done(kafka_job.id, result.get("is_anomaly"))

        logger.info(
            f"Job processed successfully: {kafka_job.id} - {result.get('is_anomaly')}"
        )

    except Exception as e:
        logger.error(f"Error processing job: {e}")
        try:
            mark_job_as_failed(kafka_job.id)
        except Exception as rollback_err:
            logger.error(f"Failed to update job status to failed: {rollback_err}")


def mark_job_as_processing(job_id: str) -> Job:
    """
    Mark the job as processing in the database.
    Args:
        job_id (str): The ID of the job to mark as processing.
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


def mark_job_as_done(job_id: str, is_anomaly: bool) -> Job:
    """
    Mark the job as done in the database.
    Args:
        job_id (str): The ID of the job to mark as done.
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
        job.is_anomaly = is_anomaly
        db.commit()
        return job


def mark_job_as_failed(job_id: str) -> Job:
    """
    Mark the job as failed in the database.
    Args:
        job_id (str): The ID of the job to mark as failed.
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
