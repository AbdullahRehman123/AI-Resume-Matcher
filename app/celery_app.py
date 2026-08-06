"""Celery app for background CV extraction.

Concurrency is controlled by the worker's --concurrency flag, not in
code - that's what caps how many HF Inference API calls happen at once
across the whole batch. Start it with:

    celery -A app.celery_app worker --concurrency=5 --loglevel=info

Keep --concurrency modest (5-10) to stay comfortably under Inference
Providers rate limits - raising it doesn't help once you're rate limited,
it just means more requests fail and get retried.
"""

import os
from dotenv import load_dotenv
from celery import Celery

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("cv_job_matcher", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,  # re-queue a task if the worker dies mid-extraction
    worker_prefetch_multiplier=1,  # don't hand out more tasks than a worker can run
)

# Make sure tasks.py gets registered with this app
import app.tasks  # noqa: E402,F401 