# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CV-Job Matcher

AI-powered CV screening tool. Long-term direction: this is stage 1 of a
bigger recruitment screening system (see "Bigger picture" below) - stay
mindful of that when suggesting architecture, but don't over-build for it.

## Stack
- FastAPI + Pydantic v2 (`app/schemas.py` for API shapes, `app/orm_models.py` for DB tables - kept separate on purpose)
- PostgreSQL via SQLAlchemy (`app/database.py`), tables created via `Base.metadata.create_all()` - no Alembic yet
- Celery + Redis for background CV extraction (`app/celery_app.py`, `app/tasks.py`)
- HuggingFace Inference Providers for LLM calls (`InferenceClient(api_key=..., provider="auto")` - NOT the old `api-inference.huggingface.co` endpoint, which is retired)
- `sentence-transformers` (`all-MiniLM-L6-v2`) for local embedding similarity - no API cost, batches freely
- Plain HTML/CSS/JS frontend (`static/index.html`) - intentionally basic, polish comes later once features are locked down

## Pipeline (single-CV path, `/match`)
1. **Parse** — extract raw text from the CV (`app/parser.py`). PDFs without a
   text layer (scanned CVs) raise a `ValueError` for now - that's where an
   OCR step would plug in later.
2. **Extract** — an instruct LLM turns raw text into structured fields: name,
   skills, experience, titles, education (`app/extractor.py`). A cheap
   keyword heuristic (`looks_like_resume`) runs first to reject obviously
   non-CV documents without spending an LLM call.
3. **Match** — sentence-transformer embeddings give a fast similarity score
   between the CV and each job; an LLM then explains the fit and lists
   matched/missing skills (`app/matcher.py`).
4. **Serve** — a FastAPI endpoint ties it together (`app/main.py`).

The batch path (`/candidates/batch` → Celery → `/jobs/{id}/match-batch/{batch_id}`)
reuses the same `extractor.py`/`matcher.py` functions but persists extraction
results to the `candidates` table instead of returning them inline - see
"Two-stage matching funnel" below.

## Architecture decisions (don't relitigate these without reason)
- **LLM vs deterministic code**: LLM only used for judgment calls (verdict, explanation). Anything that's exact set/list comparison (matched/missing skills) is computed in plain Python (`compute_skill_overlap` in `matcher.py`), not asked of the LLM - LLMs aren't reliable at exact list operations even at temperature=0.
- **Two-stage matching funnel**: cheap local embeddings rank ALL candidates first (`rank_candidates_by_similarity`); the expensive LLM explanation call (`explain_match`) only runs on the top-N shortlist. This is the core cost-control pattern for batch mode - don't add LLM calls before the embedding filter.
- **Candidates extracted once, reused across jobs**: `Candidate` table stores structured extraction permanently. Never re-run extraction just because a new job posting comes in - only re-run the (cheap) ranking step.
- **Batch processing is async by necessity**: `/candidates/batch` returns immediately with a `batch_id`; Celery worker does extraction in the background; client polls `/candidates/batch/{batch_id}` for status. Don't try to make batch synchronous.

## API endpoints (`app/main.py`)
- `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `PUT /jobs/{id}`, `DELETE /jobs/{id}` — job posting CRUD.
- `POST /match` — upload one CV, match synchronously against every job in the DB, return results sorted by similarity.
- `POST /candidates/batch` — upload many CVs; parses text synchronously, queues LLM extraction per candidate via Celery, returns a `batch_id` immediately.
- `GET /candidates/batch/{batch_id}` — poll extraction status counts for a batch.
- `GET /jobs/{job_id}/match-batch/{batch_id}?top_n=20` — rank a batch's extracted candidates against a job via embeddings, then run the LLM explanation only on the top `top_n`.
- `GET /health`
- `GET /` — serves `static/index.html`, which already exercises all of the above endpoints.

## Known gotchas (already hit these once)
- Windows: env vars need `set VAR=value` (cmd) or `$env:VAR="value"` (PowerShell) - no spaces around `=`, and a running process won't pick up a var set after it started (must restart).
- `pip` dependency pins matter: `transformers` needs to be pinned explicitly alongside `huggingface-hub`, or pip's resolver backtracks through dozens of versions.
- `pdfminer`/`pdfplumber`/`PIL` are extremely chatty at DEBUG level - already muted in `logging_config.py`, don't unmute without reason.
- `LOG_LEVEL=DEBUG` logs full CV text and raw model responses (PII) - fine for local dev, never for anything shared or committed. Already gitignored (`logs/`).
- **Celery on Windows**: the default `prefork` pool crashes worker subprocesses with `PermissionError`/`WinError 6` (billiard/multiprocessing isn't Windows-safe). Run the worker with `celery -A app.celery_app worker --pool=solo --loglevel=info` instead - no `--concurrency` flag with `solo` (it's inherently one task at a time). For real concurrency on Windows, use `-P eventlet -c 5` (requires `pip install eventlet`).
- `app/database.py`'s `DATABASE_URL` fallback default is a **real, already-committed Postgres credential** (it's been in git history since the initial commit, even though `.env` itself is gitignored). Don't add more secrets as hardcoded defaults anywhere in `app/` - always require them via env var instead. This existing one needs to be rotated and replaced with a non-secret default (or made to fail loudly if unset).

## Environment variables
| Var | Default | Used in |
|---|---|---|
| `HF_TOKEN` | — (required) | `extractor.py`, `matcher.py` — HF Inference Providers auth |
| `DATABASE_URL` | hardcoded fallback (see gotcha above) | `database.py` |
| `REDIS_URL` | `redis://localhost:6379/0` | `celery_app.py` — Celery broker/backend |
| `EXTRACTOR_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | `extractor.py` — structured extraction LLM |
| `MATCHER_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | `matcher.py` — explanation LLM |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `matcher.py` — local embedding model |
| `LOG_LEVEL` | `INFO` | `logging_config.py` |

## Commands
- Run API: `uvicorn app.main:app --reload`
- Run worker: `celery -A app.celery_app worker --concurrency=5 --loglevel=info` (concurrency caps HF API load, not just performance)
- Postgres/Redis must both be running before the API or worker will work (no docker-compose in this repo yet - run/install them natively or via your own container setup).
- No test suite or linter is configured in this repo yet.

## Current status / not yet built
- No auth on any endpoint
- English-only extraction/matching (Roman Urdu/Urdu is a deliberate future step, not an oversight)
- No Alembic migrations - schema changes go through `create_all()` for now
- No OCR fallback for scanned PDFs (`parser.py` raises instead)

## Bigger picture (context, not a todo list)
This CV matcher is module 1 of a 3-module recruitment screening system inspired by a real Upwork job post: (1) CV screening/matching - built, (2) timed psychometric testing - not started, (3) AI voice interview agent - not started, but reuses an existing voice agent stack (Speechmatics STT + Gemini LLM + SIP/RTP) from a prior project. Don't assume modules 2/3 unless explicitly asked to build them.
