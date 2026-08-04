"""Match a structured CV against a job posting.

Stage 1: sentence-transformer embeddings give a fast, cheap similarity
score - this is what scales to many CVs / many jobs.
Stage 2: the LLM only runs on the shortlist to explain the fit and
call out gaps a plain similarity score would miss.
"""

import json
import logging
import os
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from app.schemas import CVData, JobRequirements, MatchResult

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("MATCHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

_embedder = SentenceTransformer(EMBED_MODEL)
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


def compute_similarity(cv: CVData, job: JobRequirements) -> float:
    cv_text = f"{cv.summary or ''} Skills: {', '.join(cv.skills)}"
    job_text = f"{job.description} Required skills: {', '.join(job.required_skills)}"
    logger.debug("Similarity input - CV: %s | Job: %s", cv_text, job_text)

    embeddings = _embedder.encode([cv_text, job_text], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
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
    """Batch version of compute_similarity - encodes every candidate plus
    the job in ONE call instead of one call per candidate. This is the
    step that makes thousands of CVs cheap: no LLM involved, just local
    embedding math, so it scales freely regardless of batch size.
 
    Returns scores in the SAME order as `candidates` (not sorted) -
    callers that need to pair scores back to a candidate ID should zip
    their own id list against this rather than relying on object identity.
    """
    job_text = f"{job.description} Required skills: {', '.join(job.required_skills)}"
    cv_texts = [f"{c.summary or ''} Skills: {', '.join(c.skills)}" for c in candidates]
 
    logger.info("Bulk-encoding %d candidates against job '%s'", len(candidates), job.title)
    all_embeddings = _embedder.encode([job_text] + cv_texts, convert_to_tensor=True)
    job_embedding, cv_embeddings = all_embeddings[0], all_embeddings[1:]
 
    scores = util.cos_sim(job_embedding, cv_embeddings)[0]
    return [round(max(0.0, min(1.0, s.item())), 3) for s in scores]

 
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