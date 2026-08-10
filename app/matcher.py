"""Match a structured CV against a job posting.

Stage 1: sentence embeddings give a fast, cheap similarity score - this is
what scales to many CVs / many jobs. Computed via HF's hosted Inference
API (same InferenceClient used for LLM calls below) rather than a local
sentence-transformers model - PyTorch's memory footprint alone doesn't fit
Render's 512MB free tier, and at MAX_BATCH_SIZE=5 there's no real
performance case for computing embeddings locally anyway.
Stage 2: the LLM only runs on the shortlist to explain the fit and
call out gaps a plain similarity score would miss.
"""

import json
import logging
import os
from huggingface_hub import InferenceClient
from app.schemas import CVData, JobRequirements, MatchResult

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("MATCHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

_client = InferenceClient(api_key=os.getenv("HF_TOKEN"), provider="auto")

EXPLAIN_PROMPT = """You are screening a candidate for a job. Compare the
candidate profile against the job requirements and return ONLY valid JSON:

{{
  "verdict": "strong fit" | "partial fit" | "weak fit",
  "explanation": string (2-3 sentences, specific and concrete)
}}

Candidate skills: {skills}
Candidate experience: {years} years, past titles: {titles}
Candidate summary: {summary}

Job title: {job_title}
Job description: {job_description}
Required skills: {required_skills}
Minimum experience required: {min_years} years

JSON:"""


def compute_skill_overlap(cv_skills: list, required_skills: list) -> tuple:
    """Deterministic matched/missing skills - substring match, case-insensitive,
    so 'Sql server' satisfies a required skill of 'SQL' either direction.
    Don't rely on the LLM for this; free-form JSON generation isn't a
    reliable way to do exact set comparison."""
    normalized_cv = [s.strip().lower() for s in cv_skills]
    matched, missing = [], []
    for req in required_skills:
        req_norm = req.strip().lower()
        if any(req_norm in cv_s or cv_s in req_norm for cv_s in normalized_cv):
            matched.append(req)
        else:
            missing.append(req)
    return matched, missing


def get_embedding(text: str) -> list:
    """Get a sentence embedding via HF's hosted Inference API. Same
    try/except-log-reraise treatment as the LLM calls below (explain_match,
    extractor.py's extract_structured_data) - a flaky/unavailable Inference
    API is a real failure mode here now, not a local, near-infallible call."""
    try:
        embedding = _client.feature_extraction(text, model=EMBED_MODEL)
    except Exception:
        logger.exception("Embedding call failed for model %s", EMBED_MODEL)
        raise

    vector = embedding.tolist()
    # sentence-transformers models return one pooled vector per input, but
    # stay defensive about a nested [[...]] shape rather than assume it.
    if vector and isinstance(vector[0], list):
        vector = vector[0]
    return vector


def cosine_similarity(a: list, b: list) -> float:
    """Pure-Python cosine similarity - no numpy needed, these vectors are
    small (384-dim for all-MiniLM-L6-v2)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarity(cv: CVData, job: JobRequirements) -> float:
    cv_text = f"{cv.summary or ''} Skills: {', '.join(cv.skills)}"
    job_text = f"{job.description} Required skills: {', '.join(job.required_skills)}"
    logger.debug("Similarity input - CV: %s | Job: %s", cv_text, job_text)

    cv_embedding = get_embedding(cv_text)
    job_embedding = get_embedding(job_text)
    score = cosine_similarity(cv_embedding, job_embedding)
    score = round(max(0.0, min(1.0, score)), 3)
    logger.info("Similarity score for job '%s': %.3f", job.title, score)
    return score


def explain_match(cv: CVData, job: JobRequirements) -> dict:
    prompt = EXPLAIN_PROMPT.format(
        skills=", ".join(cv.skills) or "none listed",
        years=cv.years_experience or "unknown",
        titles=", ".join(cv.job_titles) or "none listed",
        summary=cv.summary or "none provided",
        job_title=job.title,
        job_description=job.description,
        required_skills=", ".join(job.required_skills) or "none specified",
        min_years=job.min_years_experience or "not specified",
    )

    logger.info("Calling explanation model %s for job '%s'", LLM_MODEL, job.title)
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0,
        )
    except Exception:
        logger.exception("Explanation model call failed for job '%s'", job.title)
        raise

    content = response.choices[0].message.content.strip().strip("`")
    content = content.removeprefix("json").strip()
    logger.debug("Raw explanation model response:\n%s", content)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.exception("Could not parse explanation model output as JSON: %.200s", content)
        raise

    logger.info("Verdict for job '%s': %s", job.title, result.get("verdict"))
    logger.debug("Full explanation result for job '%s':\n%s", job.title, json.dumps(result, indent=2))
    return result


def rank_candidates_by_similarity(candidates: list, job: JobRequirements) -> list:
    """Batch version of compute_similarity - fetches the job embedding ONCE
    and reuses it across every candidate, rather than re-fetching it per
    candidate. Each candidate still needs its own Inference API call
    (get_embedding takes one text at a time), so this is N+1 calls rather
    than the single local-encode batch call it used to be - an inherent
    cost of moving embeddings off this process, acceptable at
    MAX_BATCH_SIZE=5.

    Returns scores in the SAME order as `candidates` (not sorted) -
    callers that need to pair scores back to a candidate ID should zip
    their own id list against this rather than relying on object identity.
    """
    job_text = f"{job.description} Required skills: {', '.join(job.required_skills)}"
    logger.info("Fetching job embedding once for '%s', reusing across %d candidates", job.title, len(candidates))
    job_embedding = get_embedding(job_text)

    scores = []
    for c in candidates:
        cv_text = f"{c.summary or ''} Skills: {', '.join(c.skills)}"
        cv_embedding = get_embedding(cv_text)
        score = cosine_similarity(job_embedding, cv_embedding)
        scores.append(round(max(0.0, min(1.0, score)), 3))
    return scores


def match_cv_to_job(cv: CVData, job: JobRequirements) -> MatchResult:
    logger.info("Matching CV '%s' against job '%s'", cv.name or "unknown", job.title)
    similarity = compute_similarity(cv, job)
    details = explain_match(cv, job)
    matched_skills, missing_skills = compute_skill_overlap(cv.skills, job.required_skills)
    logger.debug(
        "Final match for job '%s': score=%.3f, details=%s, matched=%s, missing=%s",
        job.title, similarity, details, matched_skills, missing_skills,
    )

    return MatchResult(
        job_title=job.title,
        similarity_score=similarity,
        verdict=details["verdict"],
        explanation=details["explanation"],
        matched_skills=matched_skills,
        missing_skills=missing_skills,
    )
