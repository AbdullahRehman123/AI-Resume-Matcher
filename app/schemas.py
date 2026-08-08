"""Data models shared across the pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CVData(BaseModel):
    """Structured fields extracted from a raw CV."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    job_titles: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    is_resume: bool = True
    rejection_reason: Optional[str] = None
    raw_text: str = ""


class JobRequirements(BaseModel):
    """A job posting, either pasted as free text or pre-structured."""

    title: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None


class JobCreate(BaseModel):
    """Request body for creating a job posting."""

    title: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None


class JobUpdate(BaseModel):
    """Request body for updating a job posting - all fields optional,
    only the ones provided get changed."""

    title: Optional[str] = None
    description: Optional[str] = None
    required_skills: Optional[List[str]] = None
    min_years_experience: Optional[float] = None


class JobOut(BaseModel):
    """Response shape for a job posting read from the DB."""

    model_config = {"from_attributes": True}

    id: int
    title: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None


class MatchResult(BaseModel):
    """Output of matching one CV against one job."""

    job_title: str
    similarity_score: float  # 0-1, from embeddings
    verdict: str  # "strong fit" / "partial fit" / "weak fit"
    explanation: str
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list) 


class BatchUploadResponse(BaseModel):
    """Returned immediately after a batch upload - extraction happens
    in the background, so this doesn't include results yet."""
 
    batch_id: str
    queued: int
    skipped: int
 
 
class BatchStatusResponse(BaseModel):
    """Poll this to check whether background extraction has finished."""
 
    batch_id: str
    total: int
    status_counts: dict
 
 
class CandidateMatchResult(MatchResult):
    """A MatchResult with the candidate identity attached, for batch results."""

    candidate_id: int
    candidate_name: Optional[str] = None
    filename: str


class UserCreate(BaseModel):
    """Request body for signup/login - same shape for both."""

    email: str
    password: str


class UserOut(BaseModel):
    """Response shape for a user - never includes hashed_password."""

    model_config = {"from_attributes": True}

    id: int
    email: str


class Token(BaseModel):
    """Response shape for signup/login - the client stores access_token."""

    access_token: str
    token_type: str = "bearer"
