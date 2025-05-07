import logging

from config import Config

from libs.shared_models.kafka_models.jobs import TransactionAnomalyJob as KafkaJob
from libs.shared_postgres.db import get_session_maker
from libs.shared_postgres.enums import JobStatus
from libs.shared_postgres.models import TransactionAnomalyJob as Job

session_maker = get_session_maker(Config.DATABASE_URL)


def transaction_anomaly_job_handler(_: str, message: dict) -> None:
    """
    Handler for transaction anomaly detection job.
    """
    logger = logging.getLogger(__name__)
    # Process the message
    try:
        db = session_maker()
        logger.info(f"Processing message: {message}")
        kafka_job = KafkaJob(**message)
        job: Job = db.query(Job).filter(Job.id == kafka_job.id).first()
        if not job:
            raise ValueError("Job id not found")
        job.status = JobStatus.processing
        db.commit()
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return
    finally:
        db.close()
    # Processing the job
    try:
        db = session_maker()
        logger.info(f"Processing job: {kafka_job.id}")
        job.status = JobStatus.done
        db.commit()
        logger.info(f"Job processed successfully: {kafka_job.id}")
    except Exception as e:
        logger.error(f"Error processing job: {e}")
        job.status = JobStatus.failed
        db.commit()
    finally:
        db.close()
