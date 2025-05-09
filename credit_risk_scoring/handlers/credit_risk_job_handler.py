import logging
from uuid import UUID

from config import Config
from models.predict_score import predict_credit_score_combined

from libs.shared_models.kafka_models.jobs import CreditRiskJob as KafkaJob
from libs.shared_postgres.db import get_session_maker
from libs.shared_postgres.enums import JobStatus
from libs.shared_postgres.models import CreditRiskJob as Job

session_maker = get_session_maker(Config.DATABASE_URL)


def credit_risk_job_handler(_: str, message: dict) -> None:
    """
    Handler for credit risk job messages.
    """
    logger = logging.getLogger(__name__)

    try:
        kafka_job = KafkaJob(**message)
        logger.info(f"Processing job: {kafka_job}")
        loan_dict = kafka_job.loan.model_dump(mode="json")

        # Step 1: Mark job as processing
        mark_job_as_processing(kafka_job.id)

        # Step 2: Run model
        result = predict_credit_score_combined(loan_dict)

        # Step 3: Save results
        mark_job_as_done(kafka_job.id, result.get("Credit_Score"))

        logger.info(
            f"Job processed successfully: {kafka_job.id} - {result.get('Credit_Score')}"
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


def mark_job_as_done(job_id: UUID, risk_type: str) -> Job:
    """
    Mark the job as done in the database.
    Args:
        job_id (str): The ID of the job to mark as done.
        risk_type (str): The type of risk associated with the job.
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
        try:
            job.risk_type = CreditRiskType[risk_type]
        except KeyError:
            raise ValueError(f"Invalid risk type: {risk_type}")
        db.commit()
        return job


def mark_job_as_failed(job_id: UUID) -> Job:
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
