"""Turn raw CV text into structured fields using an instruct LLM.

Uses the Hugging Face Inference API so no local GPU is required to
get started. Swap MODEL_ID for any instruct model you have access to.
"""

import json
import logging
import os
from huggingface_hub import InferenceClient
from app.json_utils import extract_json_object
from app.schemas import CVData

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("EXTRACTOR_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

_client = InferenceClient(api_key=os.getenv("HF_TOKEN"), provider="auto")

# Cheap first-pass filter, no LLM call needed - catches obviously-wrong
# uploads (invoices, articles, random PDFs) before spending a request on them.
RESUME_SIGNAL_WORDS = (
    "experience", "education", "skills", "resume", "curriculum vitae",
    "employment", "objective", "career", "qualification", "certification",
)


def looks_like_resume(text: str) -> bool:
    if len(text.strip()) < 100:
        return False
    lowered = text.lower()
    hits = sum(1 for word in RESUME_SIGNAL_WORDS if word in lowered)
    return hits >= 2

EXTRACTION_PROMPT = """You extract structured data from a CV/resume.
Return ONLY valid JSON, no prose, no markdown fences, matching this schema:

{{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "skills": list of strings (technical and soft skills, deduplicated),
  "years_experience": number or null (total years of professional experience),
  "job_titles": list of strings (past job titles held),
  "education": list of strings (degree + institution, one entry each),
  "summary": string or null (2-3 sentence summary of the candidate),
  "is_resume": boolean (false if this document is clearly not a CV/resume -
    e.g. it's an invoice, article, job posting, or unrelated document),
  "rejection_reason": string or null (if is_resume is false, one sentence
    saying what the document actually appears to be)
}}

CV text:
---
{cv_text}
---

JSON:"""


def extract_structured_data(cv_text: str) -> CVData:
    if not looks_like_resume(cv_text):
        logger.warning("Document failed resume heuristic check, skipping LLM call")
        return CVData(
            is_resume=False,
            rejection_reason="Document doesn't contain typical resume content "
            "(no experience/education/skills sections detected).",
            raw_text=cv_text,
        )

    prompt = EXTRACTION_PROMPT.format(cv_text=cv_text[:8000])  # guard context length
    logger.info("Calling extraction model %s on %d chars of CV text", MODEL_ID, len(cv_text))

    try:
        response = _client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0,
        )
    except Exception:
        logger.exception("Extraction model call failed for model %s", MODEL_ID)
        raise

    content = response.choices[0].message.content.strip()
    logger.debug("Raw extraction model response:\n%s", content)

    try:
        data = extract_json_object(content)
    except json.JSONDecodeError:
        logger.exception("Could not parse extraction model output as JSON: %.200s", content)
        raise

    logger.info(
        "Extracted fields: name=%s, %d skills, %s years experience",
        data.get("name"), len(data.get("skills", [])), data.get("years_experience"),
    )
    logger.debug("Full structured extraction:\n%s", json.dumps(data, indent=2))
    return CVData(**data, raw_text=cv_text)