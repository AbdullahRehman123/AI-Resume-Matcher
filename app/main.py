"""FastAPI app: upload a CV, match it against one or more jobs."""

import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.logging_config import setup_logging
from app.database import get_db, init_db
from app.orm_models import JobPosting as JobPostingORM
from app.parser import extract_text
from app.extractor import extract_structured_data
from app.matcher import match_cv_to_job
from app.schemas import JobRequirements, JobCreate, JobUpdate, JobOut, MatchResult

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="CV-Job Matcher", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so unexpected errors are logged with a full traceback
    instead of surfacing only a generic 500 with nothing in the logs."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
async def on_startup():
    init_db()


@app.post("/jobs", response_model=JobOut, status_code=201)
async def create_job(job: JobCreate, db: Session = Depends(get_db)):
    db_job = JobPostingORM(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    logger.info("Created job posting id=%d: %s", db_job.id, db_job.title)
    return db_job


@app.get("/jobs", response_model=List[JobOut])
async def list_jobs(db: Session = Depends(get_db)):
    return db.query(JobPostingORM).all()


@app.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(JobPostingORM).filter(JobPostingORM.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.put("/jobs/{job_id}", response_model=JobOut)
async def update_job(job_id: int, job_update: JobUpdate, db: Session = Depends(get_db)):
    job = db.query(JobPostingORM).filter(JobPostingORM.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    for field, value in job_update.model_dump(exclude_unset=True).items():
        setattr(job, field, value)
    db.commit()
    db.refresh(job)
    logger.info("Updated job posting id=%d", job_id)
    return job


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(JobPostingORM).filter(JobPostingORM.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    db.delete(job)
    db.commit()
    logger.info("Deleted job posting id=%d", job_id)
    return None


@app.post("/match", response_model=List[MatchResult])
async def match_cv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info("Received CV upload: %s", file.filename)
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        logger.warning("Rejected upload with unsupported extension: %s", suffix)
        raise HTTPException(400, "Only PDF and DOCX CVs are supported right now.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_text = extract_text(tmp_path)
    except ValueError as e:
        logger.warning("Text extraction failed for %s: %s", file.filename, e)
        raise HTTPException(422, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    cv_data = extract_structured_data(raw_text)

    if not cv_data.is_resume:
        logger.warning("Rejected non-resume upload %s: %s", file.filename, cv_data.rejection_reason)
        raise HTTPException(422, cv_data.rejection_reason or "This doesn't look like a CV/resume.")

    jobs_orm = db.query(JobPostingORM).all()
    if not jobs_orm:
        raise HTTPException(400, "No jobs in the database yet - add one via POST /jobs first.")

    jobs = [
        JobRequirements(
            title=j.title,
            description=j.description,
            required_skills=j.required_skills or [],
            min_years_experience=j.min_years_experience,
        )
        for j in jobs_orm
    ]

    results = [match_cv_to_job(cv_data, job) for job in jobs]
    results.sort(key=lambda r: r.similarity_score, reverse=True)
    logger.info("Completed matching for %s against %d jobs", file.filename, len(jobs))
    return results


@app.get("/health")
async def health():
    return {"status": "ok"}