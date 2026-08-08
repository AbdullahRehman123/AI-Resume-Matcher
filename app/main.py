"""FastAPI app: upload a CV, match it against one or more jobs."""

import logging
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.logging_config import setup_logging
from app.database import get_db, init_db
from app.orm_models import JobPosting as JobPostingORM, Candidate, User
from app.parser import extract_text
from app.extractor import extract_structured_data
from app.matcher import match_cv_to_job, rank_candidates_by_similarity, explain_match, compute_skill_overlap
from app.tasks import extract_candidate_task
from app.auth import hash_password, verify_password, create_access_token, get_current_user
from app.schemas import (
    JobRequirements, JobCreate, JobUpdate, JobOut, MatchResult,
    CVData, BatchUploadResponse, BatchStatusResponse, CandidateMatchResult,
    UserCreate, Token,
)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="CV-Job Matcher", version="0.1.0")

# Deliberate scale tradeoff: capped at 5 so match-batch can run the LLM
# explanation on every candidate instead of shortlisting (see below). If
# batch size ever needs to grow past a handful, bring shortlisting back
# before the LLM step - running the LLM on hundreds/thousands of CVs
# won't hold up.
MAX_BATCH_SIZE = 5

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


@app.post("/auth/signup", response_model=Token, status_code=201)
async def signup(body: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(400, "An account with this email already exists.")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Signed up new user id=%d", user.id)

    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@app.post("/auth/login", response_model=Token)
async def login(body: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password.")

    logger.info("User id=%d logged in", user.id)
    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@app.post("/jobs", response_model=JobOut, status_code=201)
async def create_job(job: JobCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_job = JobPostingORM(**job.model_dump(), owner_id=current_user.id)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    logger.info("Created job posting id=%d: %s", db_job.id, db_job.title)
    return db_job


@app.get("/jobs", response_model=List[JobOut])
async def list_jobs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(JobPostingORM).filter(JobPostingORM.owner_id == current_user.id).all()


@app.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(JobPostingORM).filter(
        JobPostingORM.id == job_id, JobPostingORM.owner_id == current_user.id
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.put("/jobs/{job_id}", response_model=JobOut)
async def update_job(job_id: int, job_update: JobUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(JobPostingORM).filter(
        JobPostingORM.id == job_id, JobPostingORM.owner_id == current_user.id
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    for field, value in job_update.model_dump(exclude_unset=True).items():
        setattr(job, field, value)
    db.commit()
    db.refresh(job)
    logger.info("Updated job posting id=%d", job_id)
    return job


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(JobPostingORM).filter(
        JobPostingORM.id == job_id, JobPostingORM.owner_id == current_user.id
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    db.delete(job)
    db.commit()
    logger.info("Deleted job posting id=%d", job_id)
    return None


@app.post("/match", response_model=List[MatchResult])
async def match_cv(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
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

    jobs_orm = db.query(JobPostingORM).filter(JobPostingORM.owner_id == current_user.id).all()
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


@app.post("/candidates/batch", response_model=BatchUploadResponse)
async def upload_batch(files: List[UploadFile] = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Upload many CVs at once. Text is parsed synchronously (fast, local),
    but LLM extraction is queued as a background task per candidate -
    this endpoint returns immediately, it doesn't wait for extraction."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            400,
            f"Batch too large: {len(files)} files submitted, max is {MAX_BATCH_SIZE} per batch.",
        )

    batch_id = str(uuid.uuid4())
    queued, skipped = 0, 0

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".pdf", ".docx"):
            logger.warning("Skipping unsupported file in batch: %s", file.filename)
            skipped += 1
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            raw_text = extract_text(tmp_path)
        except ValueError as e:
            logger.warning("Skipping %s in batch - %s", file.filename, e)
            skipped += 1
            continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        candidate = Candidate(
            batch_id=batch_id, filename=file.filename, raw_text=raw_text, status="pending",
            owner_id=current_user.id,
        )
        db.add(candidate)
        db.commit()
        db.refresh(candidate)

        extract_candidate_task.delay(candidate.id)
        queued += 1

    logger.info("Batch %s: queued %d candidates, skipped %d", batch_id, queued, skipped)
    return BatchUploadResponse(batch_id=batch_id, queued=queued, skipped=skipped)


@app.get("/candidates/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Poll this while extraction runs in the background."""
    candidates = db.query(Candidate).filter(
        Candidate.batch_id == batch_id, Candidate.owner_id == current_user.id
    ).all()
    if not candidates:
        raise HTTPException(404, "Batch not found")

    status_counts: dict = {}
    for c in candidates:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1

    return BatchStatusResponse(batch_id=batch_id, total=len(candidates), status_counts=status_counts)


@app.get("/jobs/{job_id}/match-batch/{batch_id}", response_model=List[CandidateMatchResult])
async def match_batch_to_job(
    job_id: int, batch_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Rank every extracted candidate in a batch against a job using cheap
    local embeddings, then run the LLM explanation on every candidate - no
    shortlisting. Deliberate scale tradeoff: batch size is capped at
    MAX_BATCH_SIZE (5), so there's no meaningful cost savings from
    shortlisting, and silently dropping low-ranked candidates loses
    information a recruiter might want. The embedding score is still used
    for sort order. If batch size ever grows past a handful, shortlisting
    should come back before the LLM step - this won't hold up at real
    scale (hundreds/thousands of CVs)."""
    job_orm = db.query(JobPostingORM).filter(
        JobPostingORM.id == job_id, JobPostingORM.owner_id == current_user.id
    ).first()
    if not job_orm:
        raise HTTPException(404, "Job not found")

    job = JobRequirements(
        title=job_orm.title,
        description=job_orm.description,
        required_skills=job_orm.required_skills or [],
        min_years_experience=job_orm.min_years_experience,
    )

    # Ownership check on the whole batch (any status) first, separate from the
    # "extracted" filter below - so a batch that isn't the caller's own gets a
    # 404 ("Batch not found"), while a batch that IS theirs but still mid-
    # extraction gets the more specific 400 "not ready yet" message.
    batch_candidates = db.query(Candidate).filter(
        Candidate.batch_id == batch_id, Candidate.owner_id == current_user.id
    ).all()
    if not batch_candidates:
        raise HTTPException(404, "Batch not found")

    candidates_orm = [c for c in batch_candidates if c.status == "extracted"]
    if not candidates_orm:
        raise HTTPException(400, "No extracted candidates in this batch yet - check /candidates/batch/{batch_id} for status.")

    cv_data_list = [
        CVData(
            name=c.name, email=c.email, phone=c.phone,
            skills=c.skills or [], years_experience=c.years_experience,
            job_titles=c.job_titles or [], education=c.education or [],
            summary=c.summary, raw_text=c.raw_text,
        )
        for c in candidates_orm
    ]

    scores = rank_candidates_by_similarity(cv_data_list, job)

    # Pair each candidate/CVData/score explicitly by index, then sort together -
    # avoids relying on object identity to map results back to a candidate.
    ranked = sorted(
        zip(candidates_orm, cv_data_list, scores),
        key=lambda triple: triple[2],
        reverse=True,
    )
    logger.info("Explaining all %d ranked candidates for job '%s'", len(ranked), job.title)

    results = []
    for candidate_orm, cv_data, score in ranked:
        details = explain_match(cv_data, job)
        matched, missing = compute_skill_overlap(cv_data.skills, job.required_skills)
        results.append(CandidateMatchResult(
            job_title=job.title,
            similarity_score=score,
            verdict=details["verdict"],
            explanation=details["explanation"],
            matched_skills=matched,
            missing_skills=missing,
            candidate_id=candidate_orm.id,
            candidate_name=cv_data.name,
            filename=candidate_orm.filename,
        ))

    return results


@app.get("/health")
async def health():
    return {"status": "ok"}