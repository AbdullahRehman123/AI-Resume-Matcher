"""Background task: run structured extraction on one candidate's CV.

Called once per candidate at upload time. Retries on transient failures
(e.g. a flaky Inference API response) instead of silently failing.
"""

import logging

from app.celery_app import celery_app
from app.database import SessionLocal
from app.orm_models import Candidate
from app.extractor import extract_structured_data

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=15)
def extract_candidate_task(self, candidate_id: int):
    db = SessionLocal()
    try:
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if not candidate:
            logger.warning("Candidate %d not found, skipping", candidate_id)
            return

        try:
            cv_data = extract_structured_data(candidate.raw_text)
        except Exception as exc:
            logger.exception("Extraction failed for candidate %d, retrying", candidate_id)
            candidate.status = "failed"
            db.commit()
            raise self.retry(exc=exc)

        if not cv_data.is_resume:
            candidate.status = "rejected"
            candidate.rejection_reason = cv_data.rejection_reason
            logger.warning(
                "Candidate %d rejected as non-resume: %s", candidate_id, cv_data.rejection_reason
            )
        else:
            candidate.name = cv_data.name
            candidate.email = cv_data.email
            candidate.phone = cv_data.phone
            candidate.skills = cv_data.skills
            candidate.years_experience = cv_data.years_experience
            candidate.job_titles = cv_data.job_titles
            candidate.education = cv_data.education
            candidate.summary = cv_data.summary
            candidate.status = "extracted"
            logger.info("Candidate %d extracted: name=%s", candidate_id, cv_data.name)

        db.commit()
    finally:
        db.close()