"""Background task: run structured extraction on one candidate's CV.

Called once per candidate at upload time.

No owner_id check needed here: this only mutates a specific candidate_id
that was already created (with owner_id set) by the authenticated
/candidates/batch upload that queued it - it doesn't take requests from a
caller that could ask for someone else's candidate.

run_extraction() is the actual work, deliberately decoupled from Celery so
it can also run under FastAPI's BackgroundTasks (see TASK_BACKEND in
main.py). extract_candidate_task is a thin Celery wrapper around it that
adds retry-on-failure - BackgroundTasks has no equivalent, so that backend
just doesn't retry (see the TASK_BACKEND comment in .env.example).
"""

import logging

from app.celery_app import celery_app
from app.database import SessionLocal
from app.orm_models import Candidate
from app.extractor import extract_structured_data

logger = logging.getLogger(__name__)


def run_extraction(candidate_id: int) -> None:
    """Look up the candidate, run structured extraction, and update its
    fields/status. Re-raises on extraction failure (after marking the
    candidate "failed" and committing) so callers can decide what to do -
    the Celery wrapper retries, FastAPI's BackgroundTasks just logs it."""
    db = SessionLocal()
    try:
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if not candidate:
            logger.warning("Candidate %d not found, skipping", candidate_id)
            return

        try:
            cv_data = extract_structured_data(candidate.raw_text)
        except Exception:
            logger.exception("Extraction failed for candidate %d", candidate_id)
            candidate.status = "failed"
            db.commit()
            raise

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


@celery_app.task(bind=True, max_retries=3, default_retry_delay=15)
def extract_candidate_task(self, candidate_id: int):
    try:
        run_extraction(candidate_id)
    except Exception as exc:
        raise self.retry(exc=exc)