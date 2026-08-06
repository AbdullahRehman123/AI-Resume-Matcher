# AI Resume Matcher (English, v0.1)

Upload a CV (PDF or DOCX), get it matched against job postings with a
similarity score and a plain-language explanation of fit.

## Pipeline

1. **Parse** — extract raw text from the CV (`app/parser.py`).
   PDFs without a text layer (scanned CVs) raise an error for now —
   that's where an OCR step would plug in later.
2. **Extract** — an instruct LLM turns raw text into structured fields:
   name, skills, experience, titles, education (`app/extractor.py`).
3. **Match** — sentence-transformer embeddings give a fast similarity
   score between the CV and each job; an LLM then explains the fit and
   lists matched/missing skills for the explanation layer
   (`app/matcher.py`).
4. **Serve** — a FastAPI endpoint ties it together (`app/main.py`).

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
uvicorn app.main:app --reload
```

Then POST a CV file to `http://localhost:8000/match`:

```bash
curl -X POST http://localhost:8000/match -F "file=@resume.pdf"
```

## Notes / next steps

- `JOB_BOARD` in `main.py` is a hardcoded list for now — replace with
  a real job source (a DB, or scraped postings) once the core loop works.
- `EXTRACTOR_MODEL` and `MATCHER_MODEL` env vars let you swap the LLM
  without touching code — try a smaller model first to keep latency down.
- English-only for this version. Roman Urdu / Urdu support is a
  natural next step, reusing the NLP work already done elsewhere.
- No auth, no persistence, no batch processing yet — this is the
  thinnest possible working slice, meant to be extended.
