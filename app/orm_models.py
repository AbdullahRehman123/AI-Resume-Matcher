"""SQLAlchemy ORM models - kept separate from the Pydantic schemas in
schemas.py, which describe API request/response shapes, not DB tables."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ARRAY
from sqlalchemy.sql import func

from app.database import Base


class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    required_skills = Column(ARRAY(String), default=list)
    min_years_experience = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Candidate(Base):
    """A CV that's been uploaded as part of a batch. Extraction happens
    once here and gets reused for matching against any number of jobs -
    it doesn't get re-run per job."""
 
    __tablename__ = "candidates"
 
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, index=True, nullable=False)
    filename = Column(String, nullable=False)
    raw_text = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, extracted, rejected, failed
    rejection_reason = Column(String, nullable=True)
 
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    skills = Column(ARRAY(String), default=list)
    years_experience = Column(Float, nullable=True)
    job_titles = Column(ARRAY(String), default=list)
    education = Column(ARRAY(String), default=list)
    summary = Column(String, nullable=True)
 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())